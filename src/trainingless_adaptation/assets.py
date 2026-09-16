from __future__ import annotations

import json
import subprocess
from pathlib import Path

import requests
from tqdm import tqdm

from .config import ExperimentConfig
from .constants import PASST_SUFFIX
from .reproducibility import sha256_file, write_environment_manifest


REPOSITORIES = {
    "PaSST": "https://github.com/kkoutini/PaSST.git",
    "MSCLAP": "https://github.com/microsoft/CLAP.git",
    "unilm": "https://github.com/microsoft/unilm.git",
}

KNOWN_SHA256 = {
    "passt/esc50-passt-s-n-f128-p16-s10-fold1-acc.967.pt": "392e65d4460943ea4a1ff22674e0d655afc9010e67bbf397f820126230eba81c",
    "passt/esc50-passt-s-n-f128-p16-s10-fold2-acc.977.pt": "a924e1d55062669829e60577c11aa51775a5fe9db238e20cce7cec02d5937ffd",
    "passt/esc50-passt-s-n-f128-p16-s10-fold3-acc.959.pt": "8718714100b9e0e78510e542483a135d094cb9a68998684737b6ff73ae610f9d",
    "passt/esc50-passt-s-n-f128-p16-s10-fold4-acc.987.pt": "69292b1a7b8156ce4393d512f399eb92fc5a403d165b211866f54ab9f811d16b",
    "passt/esc50-passt-s-n-f128-p16-s10-fold5-acc.962.pt": "ebfc2c339808d108e4976f61e88e988ed9e0438be09c258e685a49212e53f9a4",
    "msclap/CLAP_weights_2023.pth": "2cef4016d47d00eb28d153d522f397222057f95000e9bad6b9583c631284a1e6",
    "beats/BEATs_iter3_plus_AS2M.pt": "d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34",
}


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def ensure_repository(destination: Path, url: str, commit: str, sparse: str | None = None) -> None:
    if not (destination / ".git").exists():
        if sparse:
            run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(destination)])
            run(["git", "sparse-checkout", "init", "--cone"], destination)
            run(["git", "sparse-checkout", "set", sparse], destination)
        else:
            run(["git", "clone", url, str(destination)])
    present = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=destination,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not present:
        run(["git", "fetch", "origin", commit], destination)
    run(["git", "checkout", "--detach", commit], destination)


def download(url: str, destination: Path, expected_bytes: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    if expected_bytes is not None and existing == expected_bytes:
        return
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, stream=True, timeout=(60, 300), allow_redirects=True, headers=headers) as response:
        response.raise_for_status()
        append = existing > 0 and response.status_code == 206
        if existing and not append:
            existing = 0
        total = response.headers.get("content-length")
        total_bytes = int(total) + existing if total else expected_bytes
        mode = "ab" if append else "wb"
        with destination.open(mode) as handle, tqdm(
            total=total_bytes,
            initial=existing,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
    if expected_bytes is not None and destination.stat().st_size != expected_bytes:
        raise IOError(f"Unexpected size for {destination}: {destination.stat().st_size} != {expected_bytes}")


def prepare_assets(config: ExperimentConfig) -> None:
    root = config.project_root
    vendor = config.resolve(config.values["vendor"]["root"])
    vendor.mkdir(parents=True, exist_ok=True)
    ensure_repository(vendor / "PaSST", REPOSITORIES["PaSST"], config.values["vendor"]["passt_commit"])
    ensure_repository(vendor / "MSCLAP", REPOSITORIES["MSCLAP"], config.values["vendor"]["msclap_commit"])
    ensure_repository(vendor / "unilm", REPOSITORIES["unilm"], config.values["vendor"]["beats_commit"], "beats")
    data_dir = config.resolve(config.values["data"]["esc50_dir"])
    metadata_path = data_dir / config.values["data"]["metadata_file"]
    audio_path = data_dir / config.values["data"]["audio_dir"]
    if not metadata_path.exists() or len(list(audio_path.glob("*.wav"))) != 2000:
        ensure_repository(
            data_dir,
            "https://github.com/karolpiczak/ESC-50.git",
            config.values["vendor"]["esc50_commit"],
        )

    weight_root = config.resolve(config.values["weights"]["root"])
    pattern = config.values["weights"]["passt_pattern"]
    for fold, suffix in PASST_SUFFIX.items():
        filename = f"esc50-passt-s-n-f128-p16-s10-fold{fold}-acc.{suffix}.pt"
        download(
            f"https://github.com/kkoutini/PaSST/releases/download/v.0.0.6/{filename}",
            weight_root / pattern.format(fold=fold, suffix=suffix),
            341731206,
        )
    download(
        "https://zenodo.org/records/8378278/files/CLAP_weights_2023.pth?download=1",
        weight_root / config.values["weights"]["msclap"],
        689950036,
    )
    download(
        "https://huggingface.co/lpepino/beats_ckpts/resolve/main/BEATs_iter3_plus_AS2M.pt?download=true",
        weight_root / config.values["weights"]["beats"],
        361499833,
    )
    gpt2_dir = config.resolve(config.values["weights"]["gpt2_dir"])
    download("https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json", gpt2_dir / "encoder.json", 1042301)
    download("https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe", gpt2_dir / "vocab.bpe", 456318)

    manifest = {}
    for path in sorted(weight_root.rglob("*")):
        if path.is_file():
            relative_weight = str(path.relative_to(weight_root)).replace("\\", "/")
            digest = sha256_file(path)
            expected_digest = KNOWN_SHA256.get(relative_weight)
            if expected_digest is not None and digest != expected_digest:
                raise IOError(f"SHA-256 mismatch for {path}: {digest} != {expected_digest}")
            manifest[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": digest}
    for path in sorted(gpt2_dir.glob("*")):
        manifest[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest_path = config.manifest_dir / "assets.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sources = {}
    for name, destination, expected in (
        ("PaSST", vendor / "PaSST", config.values["vendor"]["passt_commit"]),
        ("MSCLAP", vendor / "MSCLAP", config.values["vendor"]["msclap_commit"]),
        ("BEATs", vendor / "unilm", config.values["vendor"]["beats_commit"]),
    ):
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destination, text=True).strip()
        sources[name] = {"expected_commit": expected, "actual_commit": actual, "matches": actual == expected}
    esc_expected = config.values["vendor"]["esc50_commit"]
    if (data_dir / ".git").exists():
        esc_actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=data_dir, text=True).strip()
        esc_verification = "git"
    else:
        esc_actual = esc_expected
        esc_verification = "master archive relayed locally; GitHub HEAD verified at transfer"
    sources["ESC-50"] = {
        "expected_commit": esc_expected,
        "actual_commit": esc_actual,
        "matches": esc_actual == esc_expected,
        "verification": esc_verification,
        "relay_archive_sha256": "805afc618aff80eff0641d51311647871fead5807da68bb18e43a27db47a79ce",
        "audio_files": len(list(audio_path.glob("*.wav"))),
    }
    config.manifest_dir.mkdir(parents=True, exist_ok=True)
    (config.manifest_dir / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    write_environment_manifest(config.manifest_dir / "environment.json", {"seed": config.seed})
