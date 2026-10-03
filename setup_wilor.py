"""Install the full WiLoR + AnyHand tracker for local research/prototyping.

Model assets have their own noncommercial terms; they stay in ignored data/.
WiLoR: https://github.com/rolpotamias/WiLoR#license
AnyHand: https://github.com/chen-si-cs/AnyHand
MANO: https://mano.is.tue.mpg.de/license.html
No camera or desktop inputs are used during installation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import venv

ROOT = Path(__file__).resolve().parent
ENV = ROOT / "data" / "wilor-env"
MODELS = ROOT / "data" / "wilor-models"
MINI_COMMIT = "ebec42f94c389070cdd7dda6fd1bf0b4a659c960"
ANYHAND_REVISION = "7e62890787a701f41c80587de0b92f653d01a7fe"
MINI_REVISION = "b00adea9a6843bbb4c9042109c5eb29ab2a59dea"
ASSETS = {
    "anyhand_wilor.ckpt": (
        f"https://huggingface.co/chen-si-02/AnyHand-Models/resolve/{ANYHAND_REVISION}/anyhand_wilor.ckpt",
        "9709eca6e77fb77d2cfcf6ee641660f98dca6941e9fc1cf7a5be19fe62164b77"),
    "detector.pt": (f"https://huggingface.co/warmshao/WiLoR-mini/resolve/{MINI_REVISION}/pretrained_models/detector.pt",
                    "5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033"),
    "MANO_RIGHT.pkl": (f"https://huggingface.co/warmshao/WiLoR-mini/resolve/{MINI_REVISION}/pretrained_models/MANO_RIGHT.pkl",
                       "45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767"),
    "mano_mean_params.npz": (f"https://huggingface.co/warmshao/WiLoR-mini/resolve/{MINI_REVISION}/pretrained_models/mano_mean_params.npz",
                             "efc0ec58e4a5cef78f3abfb4e8f91623b8950be9eff8b8e0dbb0d036ebc63988"),
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download_models():
    MODELS.mkdir(parents=True, exist_ok=True)
    manifest = {"model": "WiLoR + AnyHand", "assets": {}}
    for name, (url, expected) in ASSETS.items():
        target = MODELS / name
        if not target.exists():
            print(f"Downloading {name}...", flush=True)
            staged = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(url, timeout=90) as response, staged.open("wb") as output:
                count, next_report = 0, 256 * 1024 * 1024
                while block := response.read(4 * 1024 * 1024):
                    output.write(block)
                    count += len(block)
                    if count >= next_report:
                        print(f"  {count / 1024**3:.2f} GiB", flush=True)
                        next_report += 256 * 1024 * 1024
            if digest(staged) != expected:
                raise RuntimeError(f"Checksum mismatch: {name}")
            staged.replace(target)
        if digest(target) != expected:
            raise RuntimeError(f"Unexpected {target}. Remove that file and rerun setup.")
        manifest["assets"][name] = {"url": url, "sha256": expected, "bytes": target.stat().st_size}
    (MODELS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--models-only", action="store_true")
    modes.add_argument("--runtime-only", action="store_true")
    args = parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("This installer is for Windows with an NVIDIA GPU.")
    python = ENV / "Scripts" / "python.exe"
    if not args.models_only:
        if not python.is_file():
            venv.EnvBuilder(with_pip=True).create(ENV)
        def pip(*arguments):
            subprocess.run([str(python), "-m", "pip", *arguments], check=True)
        pip("install", "--upgrade", "pip")
        pip("install", "torch==2.9.1", "torchvision==0.24.1",
            "--index-url", "https://download.pytorch.org/whl/cu130")
        pip("install", "numpy==1.26.4", "opencv-python==4.11.0.86", "smplx==0.1.28",
            "timm==0.9.12", "einops==0.8.1", "ultralytics==8.1.34", "scikit-image==0.24.0",
            "roma==1.5.4", "omegaconf==2.3.0", "yacs==0.1.8", "dill==0.4.1", "wheel", "setuptools")
        pip("install", "--no-build-isolation", "chumpy==0.70")
        # Upstream's torch<=2.5 restriction predates RTX 5090 support. Install
        # inference code only and use the explicitly verified modern runtime.
        pip("install", "--no-deps", f"git+https://github.com/warmshao/WiLoR-mini@{MINI_COMMIT}")
    if not args.runtime_only:
        download_models()
        print("WiLoR + AnyHand installed. Select it with the camera off, then Apply.")
    else:
        print("WiLoR runtime installed.")


if __name__ == "__main__":
    main()
