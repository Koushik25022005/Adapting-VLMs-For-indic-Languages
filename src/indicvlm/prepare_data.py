"""Materialise a balanced, local subset of Chitrakshara-Cap for QLoRA training.

Chitrakshara-Cap is large (tens of millions of image-caption pairs), so we stream it
from the Hugging Face Hub, keep up to ``--per-language`` valid samples per language,
save the images to disk and write ``train.jsonl`` / ``val.jsonl`` manifests:

    {"image": "images/hi/000001.jpg", "caption": "...", "language": "Hindi"}

Column names are auto-detected (image / image_url / url ..., caption / text ...,
language / lang ...). Override them with ``--image-column`` etc. if the dataset
card uses different names.

Example:
    python -m indicvlm.prepare_data --dataset krutrim-ai-labs/Chitrakshara-Cap \
        --out data/chitrakshara_cap --per-language 5000
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from .languages import LANGUAGES, canonical_language

IMAGE_COLS = ["image", "img", "image_url", "url", "URL", "image_path", "img_url"]
CAPTION_COLS = ["caption", "text", "TEXT", "description", "alt_text", "captions"]
LANG_COLS = ["language", "lang", "lang_code", "language_code", "LANGUAGE"]


def pick_column(columns, candidates, override=None):
    if override:
        if override not in columns:
            raise KeyError(f"Column '{override}' not in dataset columns {list(columns)}")
        return override
    for c in candidates:
        if c in columns:
            return c
    return None


def load_image(value, timeout: float = 10.0):
    """Return a PIL RGB image from a PIL image, HF image dict, bytes, local path or URL."""
    from PIL import Image

    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):  # HF Image feature in non-decoded form
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        value = value.get("path")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            import requests

            r = requests.get(value, timeout=timeout, headers={"User-Agent": "indicvlm-data-prep"})
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        if os.path.exists(value):
            return Image.open(value).convert("RGB")
    return None


def valid_caption(text, min_chars: int, max_chars: int) -> bool:
    return isinstance(text, str) and min_chars <= len(text.strip()) <= max_chars


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="krutrim-ai-labs/Chitrakshara-Cap")
    p.add_argument("--config", default=None, help="HF dataset config/subset name, if any")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="data/chitrakshara_cap")
    p.add_argument("--languages", nargs="*", default=[l for l in LANGUAGES if l != "Sanskrit"])
    p.add_argument("--per-language", type=int, default=5000)
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument("--image-column", default=None)
    p.add_argument("--caption-column", default=None)
    p.add_argument("--language-column", default=None)
    p.add_argument("--min-caption-chars", type=int, default=5)
    p.add_argument("--max-caption-chars", type=int, default=1000)
    p.add_argument("--min-image-side", type=int, default=64)
    p.add_argument("--max-scan", type=int, default=5_000_000, help="Stop after scanning this many rows")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    from datasets import load_dataset
    from tqdm import tqdm

    random.seed(args.seed)
    wanted = {canonical_language(l) for l in args.languages}
    wanted.discard(None)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    first = next(iter(ds))
    cols = list(first.keys())
    img_col = pick_column(cols, IMAGE_COLS, args.image_column)
    cap_col = pick_column(cols, CAPTION_COLS, args.caption_column)
    lang_col = pick_column(cols, LANG_COLS, args.language_column)
    print(f"Columns: {cols}\n -> image={img_col} caption={cap_col} language={lang_col}")
    if img_col is None or cap_col is None:
        raise SystemExit("Could not detect image/caption columns; pass --image-column/--caption-column.")
    if lang_col is None:
        print("WARNING: no language column detected; use --config to select a per-language subset "
              "and --languages with a single language.")
        if len(wanted) != 1:
            raise SystemExit("Without a language column, pass exactly one --languages value.")

    counts: dict[str, int] = defaultdict(int)
    records = []
    bar = tqdm(total=args.per_language * len(wanted), desc="kept")
    for i, row in enumerate(ds):
        if i >= args.max_scan or all(counts[l] >= args.per_language for l in wanted):
            break
        lang = canonical_language(row.get(lang_col)) if lang_col else next(iter(wanted))
        if lang not in wanted or counts[lang] >= args.per_language:
            continue
        cap = row.get(cap_col)
        if isinstance(cap, list):
            cap = cap[0] if cap else None
        if not valid_caption(cap, args.min_caption_chars, args.max_caption_chars):
            continue
        try:
            img = load_image(row.get(img_col))
        except Exception:
            img = None
        if img is None or min(img.size) < args.min_image_side:
            continue
        code = LANGUAGES[lang]["codes"][0]
        rel = Path("images") / code / f"{counts[lang]:07d}.jpg"
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        img.save(out / rel, format="JPEG", quality=92)
        records.append({"image": str(rel), "caption": cap.strip(), "language": lang})
        counts[lang] += 1
        bar.update(1)
    bar.close()

    random.shuffle(records)
    n_val = max(1, int(len(records) * args.val_fraction)) if records else 0
    for name, part in (("val", records[:n_val]), ("train", records[n_val:])):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"dataset": args.dataset, "split": args.split, "per_language": dict(counts),
             "train": len(records) - n_val, "val": n_val}
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    missing = [l for l in wanted if counts[l] < args.per_language]
    if missing:
        print(f"WARNING: fewer than {args.per_language} samples for: {missing}")


if __name__ == "__main__":
    main()
