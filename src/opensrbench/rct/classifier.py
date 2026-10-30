"""RCT classifiers.

CochraneRctClassifier
  A faithful re-implementation of the Cochrane RCT Classifier recipe
  (Thomas, McDonald, Noel-Storr et al., J Clin Epidemiol 2021;133:140-151):

  * input: title + abstract text only (no publication-type metadata);
  * features: binary bag-of-words with English stop words removed;
  * SVM1: linear SVM over uni-, bi- and tri-grams;
  * SVM2: linear SVM over unigrams, trained with RCTs oversampled;
  * ensemble of the two (mean of standardised margins), Platt-calibrated
    (logistic regression) on a held-out calibration split;
  * operating cut-off chosen to retrieve 99% of RCTs on calibration data
    (here the exact split-conformal order statistic);
  * records with abstract < 400 characters or title < 15 characters are not
    machine-classified (probability NaN -> always kept by filters).

  The original was trained on Cochrane Crowd labels (Embase records); here
  the default training labels are MEDLINE publication types assigned by NLM
  indexers (see rct/corpus.py). A Cochrane Crowd export can be supplied
  instead. The model is ours, trained only on decontaminated data, so it
  introduces no leakage into the benchmark.

PublicationTypeScorer
  1.0 if MEDLINE indexed the record as a Randomized Controlled Trial, else 0
  (indexing lags for recent records; best used as a feature).

HfRctScorer
  Optional transformer classifier from the Hugging Face hub (e.g.
  system-technologies/BiomedNLP-BiomedBERT-abstract-RCT-classifier-public,
  Open RAIL-M). Its training data are unknown, so it cannot be audited for
  benchmark contamination; it is opt-in and reported separately.
"""

import json
import logging
import os
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.feature_extraction import text as sk_text

from opensrbench.calibration import thresholds
from opensrbench.text import design

MIN_ABSTRACT_CHARS = 400
MIN_TITLE_CHARS = 15
MODEL_FILE = "cochrane_rct_classifier.joblib"
CARD_FILE = "model_card.json"


def _texts(titles: Sequence[str], abstracts: Sequence[str]) -> list[str]:
    return [f"{t or ''} {a or ''}" for t, a in zip(titles, abstracts)]


def classifiable(titles: Sequence[str], abstracts: Sequence[str]) -> np.ndarray:
    """Cochrane's machine-classification eligibility rule."""
    return np.array(
        [
            len(a or "") >= MIN_ABSTRACT_CHARS
            and len(t or "") >= MIN_TITLE_CHARS
            for t, a in zip(titles, abstracts)
        ]
    )


class CochraneRctClassifier:
    """Two-SVM ensemble with Platt calibration (see module docstring)."""

    def __init__(
        self,
        c1: float = 0.1,
        c2: float = 0.1,
        min_df: int = 3,
        max_features: int = 1_000_000,
        oversample_ratio: float = 1.0,
        seed: int = 0,
    ):
        self.params = {
            "c1": c1,
            "c2": c2,
            "min_df": min_df,
            "max_features": max_features,
            "oversample_ratio": oversample_ratio,
            "seed": seed,
        }
        self.vec1 = sk_text.CountVectorizer(
            binary=True,
            ngram_range=(1, 3),
            stop_words="english",
            min_df=min_df,
            max_features=max_features,
            dtype=np.float32,
        )
        self.vec2 = sk_text.CountVectorizer(
            binary=True,
            ngram_range=(1, 1),
            stop_words="english",
            min_df=min_df,
            dtype=np.float32,
        )
        self.svm1 = svm.LinearSVC(C=c1)
        self.svm2 = svm.LinearSVC(C=c2)
        self.margin_stats: dict[str, tuple[float, float]] = {}
        self.platt = linear_model.LogisticRegression()
        self.cutoff: float | None = None
        self.card: dict[str, Any] = {}

    def _margins(self, texts: list[str]) -> np.ndarray:
        m1 = self.svm1.decision_function(self.vec1.transform(texts))
        m2 = self.svm2.decision_function(self.vec2.transform(texts))
        (mu1, sd1), (mu2, sd2) = (
            self.margin_stats["svm1"],
            self.margin_stats["svm2"],
        )
        return 0.5 * ((m1 - mu1) / sd1 + (m2 - mu2) / sd2)

    def fit(
        self,
        titles: Sequence[str],
        abstracts: Sequence[str],
        labels: np.ndarray,
        calib_titles: Sequence[str],
        calib_abstracts: Sequence[str],
        calib_labels: np.ndarray,
        target_recall: float = 0.99,
    ) -> "CochraneRctClassifier":
        """Fits both SVMs, standardises margins, calibrates and sets cut-off.

        Args:
          titles, abstracts, labels: Training data (1 = RCT).
          calib_titles, calib_abstracts, calib_labels: Disjoint calibration
            data for Platt scaling and the recall cut-off.
          target_recall: Recall the cut-off must retain on calibration RCTs.
        """
        labels = np.asarray(labels).astype(int)
        texts = _texts(titles, abstracts)
        x1 = self.vec1.fit_transform(texts)
        self.svm1.fit(x1, labels)
        rng = np.random.default_rng(self.params["seed"])
        pos = np.flatnonzero(labels == 1)
        neg = np.flatnonzero(labels == 0)
        extra = max(
            0, int(self.params["oversample_ratio"] * len(neg)) - len(pos)
        )
        boot = (
            rng.choice(pos, size=extra, replace=True)
            if (extra and len(pos))
            else np.array([], dtype=int)
        )
        idx = np.concatenate([np.arange(len(labels)), boot])
        x2 = self.vec2.fit_transform([texts[i] for i in idx])
        self.svm2.fit(x2, labels[idx])
        calib = _texts(calib_titles, calib_abstracts)
        m1 = self.svm1.decision_function(self.vec1.transform(calib))
        m2 = self.svm2.decision_function(self.vec2.transform(calib))
        self.margin_stats = {
            "svm1": (float(m1.mean()), float(m1.std() + 1e-9)),
            "svm2": (float(m2.mean()), float(m2.std() + 1e-9)),
        }
        margins = self._margins(calib)
        calib_labels = np.asarray(calib_labels).astype(int)
        self.platt.fit(margins.reshape(-1, 1), calib_labels)
        probs = self.platt.predict_proba(margins.reshape(-1, 1))[:, 1]
        self.cutoff = thresholds.pooled_conformal(
            probs[calib_labels == 1], target_recall
        )
        self.card = {
            "recipe": (
                "Thomas et al. 2021 Cochrane RCT Classifier "
                "(SVM 1-3 gram + SVM unigram oversampled, Platt)"
            ),
            "params": self.params,
            "target_recall": target_recall,
            "cutoff": self.cutoff,
            "n_train": int(len(labels)),
            "n_train_rct": int(labels.sum()),
            "n_calib": int(len(calib_labels)),
            "n_calib_rct": int(calib_labels.sum()),
        }
        return self

    def predict_proba(
        self,
        titles: Sequence[str],
        abstracts: Sequence[str],
        apply_length_rule: bool = True,
    ) -> np.ndarray:
        """P(RCT); NaN for records the Cochrane rule leaves unclassified."""
        if not len(titles):
            return np.zeros(0)
        margins = self._margins(_texts(titles, abstracts))
        probs = self.platt.predict_proba(margins.reshape(-1, 1))[:, 1]
        if apply_length_rule:
            probs = np.where(classifiable(titles, abstracts), probs, np.nan)
        return probs

    def evaluate(self, titles, abstracts, labels) -> dict[str, float]:
        """Recall / precision / specificity at the cut-off and AUC."""
        from opensrbench.evaluation import \
            ranking  # pylint: disable=import-outside-toplevel

        labels = np.asarray(labels).astype(int)
        probs = self.predict_proba(titles, abstracts, apply_length_rule=False)
        keep = probs >= self.cutoff
        tp = int((keep & (labels == 1)).sum())
        return {
            "recall": tp / max(1, int(labels.sum())),
            "precision": tp / max(1, int(keep.sum())),
            "specificity": (
                float((~keep)[labels == 0].mean())
                if (labels == 0).any()
                else float("nan")
            ),
            "auc": ranking.auc(labels, probs) or float("nan"),
            "n": int(len(labels)),
        }

    def save(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, MODEL_FILE)
        joblib.dump(self, path, compress=3)
        with open(
            os.path.join(out_dir, CARD_FILE), "w", encoding="utf-8"
        ) as fh:
            json.dump(self.card, fh, indent=2, default=str)
        return path

    @staticmethod
    def load(model_dir: str) -> "CochraneRctClassifier":
        """Loads a classifier *this pipeline* trained (trusted local file)."""
        return joblib.load(os.path.join(model_dir, MODEL_FILE))


class PublicationTypeScorer:
    """MEDLINE publication-type indicator."""

    def predict_proba(self, publication_types: Sequence[str]) -> np.ndarray:
        return np.array(
            [
                float(design.has_rct_publication_type(p))
                for p in publication_types
            ]
        )


class HfRctScorer:
    """Optional Hugging Face sequence classifier (not decontaminated)."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 64,
        positive_label: str | None = None,
    ):
        import torch  # pylint: disable=import-outside-toplevel
        import transformers  # pylint: disable=import-outside-toplevel

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = (
            transformers.AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
            .to(self.device)
            .eval()
        )
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.positive = labels.get(
            (positive_label or "").lower(),
            labels.get("rct", max(labels.values())),
        )
        self.batch_size = batch_size
        logging.warning(
            "HF RCT scorer %s: training data unknown; results "
            "are not contamination-audited.",
            model_name,
        )

    def predict_proba(
        self, titles: Sequence[str], abstracts: Sequence[str]
    ) -> np.ndarray:
        torch = self._torch
        out = []
        texts = _texts(titles, abstracts)
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(
                    texts[i : i + self.batch_size],
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits.float()
                out.append(
                    torch.softmax(logits, -1)[:, self.positive].cpu().numpy()
                )
        return np.concatenate(out) if out else np.zeros(0)
