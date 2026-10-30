"""`osb doctor`: environment readiness checks (exit code 1 on failure)."""

import importlib
import os
import shutil
import sys

from opensrbench import config


def _check(
    label: str, ok: bool, detail: str = "", required: bool = True
) -> bool:
    mark = "OK  " if ok else ("FAIL" if required else "WARN")
    print(f"[{mark}] {label}{': ' + detail if detail else ''}")
    return ok or not required


def run(args) -> int:
    good = True
    good &= _check(
        "python >= 3.11", sys.version_info >= (3, 11), sys.version.split()[0]
    )
    for module, required in (
        ("numpy", True),
        ("pandas", True),
        ("pyarrow", True),
        ("lxml", True),
        ("sklearn", True),
        ("torch", False),
        ("transformers", False),
        ("peft", False),
        ("httpx", False),
        ("openpyxl", False),
        ("flash_attn", False),
    ):
        try:
            mod = importlib.import_module(module)
            good &= _check(
                f"import {module}", True, getattr(mod, "__version__", "")
            )
        except ImportError as exc:
            good &= _check(f"import {module}", False, str(exc), required)
    try:
        import torch  # pylint: disable=import-outside-toplevel

        n = torch.cuda.device_count()
        names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
        _check(
            "CUDA devices",
            n > 0,
            f"{n} [{names}]" if n else "none",
            required=False,
        )
        if n:
            _check(
                "bf16 supported", torch.cuda.is_bf16_supported(), required=False
            )
    except ImportError:
        pass
    cfg = config.Config.load(args.config)
    paths = config.Paths.from_config(cfg)
    _check("data root", True, paths.root)
    _check("runs root", True, paths.runs)
    free = shutil.disk_usage(os.path.dirname(paths.root) or ".").free / 1e9
    _check(
        "free disk >= 200 GB at data root",
        free >= 200,
        f"{free:.0f} GB",
        required=False,
    )
    _check(
        "NCBI_EMAIL set",
        bool(cfg.get("network.ncbi_email", "")),
        required=False,
    )
    _check(
        "NCBI_API_KEY set (10 req/s)",
        bool(cfg.get("network.ncbi_api_key", "")),
        required=False,
    )
    _check(
        "HF_HOME set",
        bool(os.environ.get("HF_HOME")),
        os.environ.get("HF_HOME", ""),
        required=False,
    )
    _check(
        "HF_HUB_OFFLINE (compute nodes)",
        os.environ.get("HF_HUB_OFFLINE") == "1",
        os.environ.get("HF_HUB_OFFLINE", "unset"),
        required=False,
    )
    if getattr(args, "network", False):
        import requests  # pylint: disable=import-outside-toplevel

        for url in (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi",
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            "?query=cancer&format=json&pageSize=1",
            "https://huggingface.co/api/models/bert-base-uncased",
        ):
            try:
                ok = requests.get(url, timeout=20).status_code < 500
            except requests.RequestException:
                ok = False
            _check(f"reach {url.split('/')[2]}", ok, required=False)
    print("Ready." if good else "Fix the FAIL items above.")
    return 0 if good else 1
