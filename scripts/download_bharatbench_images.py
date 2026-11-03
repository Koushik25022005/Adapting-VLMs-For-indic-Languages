"""Download the images BharatBench questions refer to (BharatBench ships questions only).

  POPE        -> COCO val2014 images, taken from the HF dataset lmms-lab/POPE
                 (or pass --coco-val2014 <dir> if you already have COCO val2014)
  LLaVA-Bench -> MBZUAI/multilingual-llava-bench-in-the-wild  images/
  MM-Vet      -> official mm-vet.zip release (github.com/yuweihao/MM-Vet)

    python scripts/download_bharatbench_images.py --out data/bharatbench_images
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

MMVET_ZIP = "https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip"


def pope(out: Path, coco_dir: str | None):
    dst = out / "pope"
    dst.mkdir(parents=True, exist_ok=True)
    if coco_dir:
        print(f"POPE: using local COCO val2014 at {coco_dir}")
        os.symlink(os.path.abspath(coco_dir), dst / "val2014", target_is_directory=True)
        return
    from datasets import load_dataset

    ds = load_dataset("lmms-lab/POPE", split="test")
    seen = set()
    for row in ds:
        name = row["image_source"]
        if name in seen:
            continue
        seen.add(name)
        fn = name if name.lower().endswith((".jpg", ".png")) else f"{name}.jpg"
        row["image"].convert("RGB").save(dst / fn, quality=95)
    print(f"POPE: {len(seen)} images -> {dst}")


def llavabench(out: Path):
    from huggingface_hub import snapshot_download

    local = snapshot_download("MBZUAI/multilingual-llava-bench-in-the-wild", repo_type="dataset",
                              allow_patterns=["images/*"])
    dst = out / "llavabench"
    shutil.copytree(Path(local) / "images", dst, dirs_exist_ok=True)
    print(f"LLaVA-Bench: {len(list(dst.iterdir()))} images -> {dst}")


def mmvet(out: Path):
    dst = out / "mmvet"
    dst.mkdir(parents=True, exist_ok=True)
    print(f"MM-Vet: downloading {MMVET_ZIP}")
    with urllib.request.urlopen(MMVET_ZIP) as r:
        data = r.read()
    n = 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for m in z.namelist():
            if "/images/" in m and not m.endswith("/") and "__MACOSX" not in m:
                (dst / Path(m).name).write_bytes(z.read(m))
                n += 1
    print(f"MM-Vet: {n} images -> {dst}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/bharatbench_images")
    p.add_argument("--tasks", nargs="+", default=["pope", "llavabench", "mmvet"])
    p.add_argument("--coco-val2014", default=None)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if "pope" in a.tasks:
        pope(out, a.coco_val2014)
    if "llavabench" in a.tasks:
        llavabench(out)
    if "mmvet" in a.tasks:
        mmvet(out)


if __name__ == "__main__":
    main()
