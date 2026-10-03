"""Install the optional Windows GPU hand tracker, without changing the app venv.

Run with the app's Python. Downloads weights only from OpenMMLab's model zoo.
The runtime never downloads models or opens the camera during setup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import venv
import zipfile

ROOT = Path(__file__).resolve().parent
ENV = ROOT / "data" / "tracking-env"
MODELS = ROOT / "data" / "hand-models"
BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
MODELS_SPEC = {
    "detector.onnx": "rtmdet_nano_8xb32-300e_hand-267f9c8f.zip",
    "pose.onnx": "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip",
}
MODEL_SHA256 = {
    "detector.onnx": "568d3ea97a5b142488366b67e036b6a5cb0a1fef9087a710cb8e66b6979fbac2",
    "pose.onnx": "39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29",
}


def download_models():
    MODELS.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "OpenMMLab RTMPose Hand5", "files": {}}
    for name, archive in MODELS_SPEC.items():
        target = MODELS / name
        if not target.exists():
            print(f"Downloading {name} from OpenMMLab...", flush=True)
            with tempfile.TemporaryDirectory(dir=MODELS) as temporary:
                local = Path(temporary) / "model.zip"
                with urllib.request.urlopen(BASE + archive, timeout=60) as response, local.open("wb") as output:
                    shutil.copyfileobj(response, output)
                with zipfile.ZipFile(local) as bundle:
                    members = [m for m in bundle.infolist() if m.filename.endswith(".onnx")]
                    if len(members) != 1 or members[0].file_size > 200_000_000:
                        raise RuntimeError("Unexpected model archive contents")
                    # Copy just the model; never extract paths supplied by an archive.
                    staged = Path(temporary) / name
                    with bundle.open(members[0]) as source, staged.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    if hashlib.sha256(staged.read_bytes()).hexdigest() != MODEL_SHA256[name]:
                        raise RuntimeError(f"Downloaded {name} does not match the verified model checksum")
                    staged.replace(target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != MODEL_SHA256[name]:
            raise RuntimeError(f"{target} is not the expected model. Remove that file and rerun setup.")
        manifest["files"][name] = {
            "url": BASE + archive,
            "sha256": digest,
            "bytes": target.stat().st_size,
        }
    (MODELS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-only", action="store_true")
    args = parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("This optional DirectML tracker requires Windows.")
    python = ENV / "Scripts" / "python.exe"
    if not args.models_only:
        if not python.exists():
            venv.EnvBuilder(with_pip=True).create(ENV)
        subprocess.run([str(python), "-m", "pip", "install",
                        "numpy==1.26.4", "opencv-contrib-python==4.11.0.86",
                        "onnxruntime-directml==1.23.0", "tqdm==4.67.1"], check=True)
        # rtmlib lists both mutually overlapping cv2 packages and CPU ORT.
        # Their APIs are provided by contrib and DirectML in this isolated env.
        subprocess.run([str(python), "-m", "pip", "install", "--no-deps",
                        "rtmlib==0.0.16"], check=True)
    download_models()
    print("RTMPose installed. Select RTMPose Hand5 in the Hand guide with the camera off.")


if __name__ == "__main__":
    main()
