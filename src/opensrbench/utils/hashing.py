"""Deterministic hashing and text fingerprinting.

All randomness that affects which records enter the benchmark is replaced by
SHA-256 orderings so the benchmark can be rebuilt bit-for-bit by anyone.
"""

import hashlib
import re
import unicodedata

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def sha256_hex(payload: str) -> str:
    """Returns the hex SHA-256 digest of a UTF-8 string."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_for_fingerprint(text: object) -> str:
    """Lower-cases, strips accents and removes every non [a-z0-9] character.

    Args:
      text: Any value; non-strings (e.g. NaN/None) normalize to "".

    Returns:
      The normalized string (possibly empty).
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return _NON_ALNUM.sub("", text.lower())


def text_fingerprint(text: object, max_chars: int = 0) -> str:
    """Computes the decontamination fingerprint of a title or abstract.

    The text is normalized first and *then* truncated, so whitespace or
    punctuation differences between sources cannot shift the window.

    Unlike the v1.0 protocol snippet, empty text maps to "" (not to
    SHA256("")), so records with missing abstracts never collide.

    Args:
      text: Title or abstract.
      max_chars: If > 0, keep only this many normalized characters.

    Returns:
      Hex digest, or "" when the normalized text is empty.
    """
    cleaned = normalize_for_fingerprint(text)
    if max_chars > 0:
        cleaned = cleaned[:max_chars]
    if not cleaned:
        return ""
    return sha256_hex(cleaned)


def stable_int(payload: str, modulus: int) -> int:
    """Maps a string to a stable integer in [0, modulus)."""
    return int(sha256_hex(payload)[:15], 16) % modulus


def stable_unit(payload: str) -> float:
    """Maps a string to a stable float in [0, 1)."""
    return int(sha256_hex(payload)[:15], 16) / float(16**15)


def short_id(payload: str, length: int = 16) -> str:
    """Returns a short stable identifier derived from SHA-256."""
    return sha256_hex(payload)[:length]
