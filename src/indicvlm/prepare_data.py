"""Materialise a balanced, local image-caption subset of Chitrakshara for QLoRA training.

Two row layouts are supported (auto-detected, or force with ``--format``):

* **interleaved** (Chitrakshara web documents, OBELICS-style)::

    {"images": [None, "https://.../a.jpg", None],
     "texts":  ["caption-like text", None, "following paragraph"],
     "metadata": '[null, {"alt_text": "...", "original_width": 789, ...}, null]', ...}

  ``images[i]`` and ``texts[i]`` are aligned positions in the page. Every image becomes
  an (image, caption) pair whose caption is the text right before it, the text right
  after it, or its alt text — tried in ``--caption-source`` order, first valid one wins.

* **pairs**: one image (PIL / bytes / path / URL) and one caption per row.

The datasets carry no language label, so unless a language column exists or
``--assume-language`` is given, the language is detected from the caption's Unicode
script (see ``indicvlm.languages.detect_language``). Images are downloaded in parallel.

Output (``--out``)::

    images/<lang-code>/0000000.jpg
    train.jsonl / val.jsonl   {"image": "images/hi/0000000.jpg", "caption": "...", "language": "Hindi"}
    stats.json

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
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .languages import LANGUAGES, canonical_language, detect_language

IMAGE_COLS = ["image", "img", "image_url", "url", "URL", "image_path", "img_url"]
CAPTION_COLS = ["caption", "text", "TEXT", "description", "alt_text", "captions"]
LANG_COLS = ["language", "lang", "lang_code", "language_code", "LANGUAGE"]
_SENT_END = re.compile(r"[।॥.!?|]")


def pick_column(columns, candidates, override=None):
    if override:
        if override not in columns:
            raise KeyError(f"Column '{override}' not in dataset columns {list(columns)}")
        return override
    for c in candidates:
        if c in columns:
            return c
    return None


# --------------------------------------------------------------------------- captions
def clip_caption(text: str, max_chars: int) -> str:
    """Trim to at most ``max_chars``, cutting at the last sentence end (।, ., ?, ! ...) if possible."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    ends = [m.end() for m in _SENT_END.finditer(head)]
    if ends and ends[-1] >= max_chars // 3:
        return head[: ends[-1]].strip()
    cut = head.rsplit(" ", 1)[0]
    return (cut if cut else head).strip()


def clean_caption(text, min_chars: int, max_chars: int, min_words: int) -> str | None:
    """Return a normalised caption, or None if it is unusable (too short / no letters)."""
    if not isinstance(text, str):
        return None
    text = clip_caption(text, max_chars)
    if len(text) < min_chars or len(text.split()) < min_words:
        return None
    if text.lower().startswith(("http://", "https://", "www.")):
        return None
    return text


def valid_caption(text, min_chars: int, max_chars: int) -> bool:
    return isinstance(text, str) and min_chars <= len(text.strip()) <= max_chars


def _parse_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _neighbour_text(texts, images, i, step):
    j = i + step
    if 0 <= j < len(texts) and texts[j] and not (j < len(images) and images[j]):
        return texts[j]
    return None


def interleaved_pairs(row: dict, sources=("before", "after", "alt"), min_chars=5, max_chars=500,
                      min_words=3, min_side=64):
    """Yield candidate (image, caption, known_size) dicts from one interleaved document row."""
    images = row.get("images") or []
    texts = row.get("texts") or []
    meta = _parse_json(row.get("metadata"))
    meta = meta if isinstance(meta, list) else []
    for i, img in enumerate(images):
        if not img:
            continue
        m = meta[i] if i < len(meta) and isinstance(meta[i], dict) else {}
        w, h = m.get("original_width"), m.get("original_height")
        if isinstance(w, (int, float)) and isinstance(h, (int, float)) and min(w, h) < min_side:
            continue  # skip known-tiny images (icons, logos) without downloading
        for src in sources:
            raw = {"before": lambda: _neighbour_text(texts, images, i, -1),
                   "after": lambda: _neighbour_text(texts, images, i, +1),
                   "alt": lambda: m.get("alt_text")}[src]()
            cap = clean_caption(raw, min_chars, max_chars, min_words)
            if cap:
                yield {"image": img, "caption": cap, "source": src}
                break


def pair_rows(row: dict, img_col: str, cap_col: str, min_chars=5, max_chars=500, min_words=3):
    cap = row.get(cap_col)
    if isinstance(cap, list):
        cap = next((c for c in cap if c), None)
    cap = clean_caption(cap, min_chars, max_chars, min_words)
    img = row.get(img_col)
    if cap and img is not None:
        yield {"image": img, "caption": cap, "source": "pair"}


# --------------------------------------------------------------------------- images
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

            r = requests.get(value, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (indicvlm-data-prep)"})
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        if os.path.exists(value):
            return Image.open(value).convert("RGB")
    return None


def _fetch(value, timeout, min_side):
    try:
        img = load_image(value, timeout)
    except Exception:
        return None
    if img is None or min(img.size) < min_side:
        return None
    return img


# --------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="krutrim-ai-labs/Chitrakshara-Cap")
    p.add_argument("--config", default=None, help="HF dataset config/subset name, if any")
    p.add_argument("--split", default="train")
    p.add_argument("--data-dir", default=None, help="HF data_dir (e.g. a per-language folder)")
    p.add_argument("--out", default="data/chitrakshara_cap")
    p.add_argument("--languages", nargs="*", default=[l for l in LANGUAGES if l != "Sanskrit"])
    p.add_argument("--per-language", type=int, default=5000)
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument("--format", choices=["auto", "interleaved", "pairs"], default="auto")
    p.add_argument("--caption-source", default="before,after,alt",
                   help="interleaved rows: order in which to take the caption (before/after/alt)")
    p.add_argument("--image-column", default=None)
    p.add_argument("--caption-column", default=None)
    p.add_argument("--language-column", default=None)
    p.add_argument("--assume-language", default=None,
                   help="label every sample with this language instead of detecting it")
    p.add_argument("--min-caption-chars", type=int, default=5)
    p.add_argument("--max-caption-chars", type=int, default=500)
    p.add_argument("--min-caption-words", type=int, default=3)
    p.add_argument("--min-image-side", type=int, default=64)
    p.add_argument("--workers", type=int, default=32, help="parallel image downloads")
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--max-scan", type=int, default=5_000_000, help="stop after scanning this many rows")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    from datasets import load_dataset
    from tqdm import tqdm

    random.seed(args.seed)
    wanted = {canonical_language(l) for l in args.languages}
    wanted.discard(None)
    assumed = canonical_language(args.assume_language) if args.assume_language else None
    if args.assume_language and assumed is None:
        raise SystemExit(f"Unknown --assume-language {args.assume_language!r}")
    sources = [s.strip() for s in args.caption_source.split(",") if s.strip()]
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True, data_dir=args.data_dir)
    first = next(iter(ds))
    cols = list(first.keys())
    fmt = args.format
    if fmt == "auto":
        fmt = "interleaved" if {"images", "texts"} <= set(cols) else "pairs"
    img_col = cap_col = None
    if fmt == "pairs":
        img_col = pick_column(cols, IMAGE_COLS, args.image_column)
        cap_col = pick_column(cols, CAPTION_COLS, args.caption_column)
        if img_col is None or cap_col is None:
            raise SystemExit("Could not detect image/caption columns; pass --image-column/--caption-column.")
    lang_col = pick_column(cols, LANG_COLS, args.language_column)
    lang_mode = ("assumed: " + assumed) if assumed else (f"column '{lang_col}'" if lang_col else "script detection")
    print(f"Columns: {cols}\n -> format={fmt}"
          + (f" image={img_col} caption={cap_col}" if fmt == "pairs" else f" caption sources={sources}")
          + f" | language from {lang_mode}")

    def candidates(row):
        if fmt == "interleaved":
            yield from interleaved_pairs(row, sources, args.min_caption_chars, args.max_caption_chars,
                                         args.min_caption_words, args.min_image_side)
        else:
            yield from pair_rows(row, img_col, cap_col, args.min_caption_chars, args.max_caption_chars,
                                 args.min_caption_words)

    def language_of(row, cap):
        if assumed:
            return assumed
        if lang_col:
            return canonical_language(row.get(lang_col))
        return detect_language(cap)

    counts: dict[str, int] = defaultdict(int)
    stats = Counter()
    records, seen_urls, batch = [], set(), []
    bar = tqdm(total=args.per_language * len(wanted), desc="kept")

    def flush(executor):
        imgs = executor.map(lambda c: _fetch(c["image"], args.timeout, args.min_image_side), batch)
        for cand, img in zip(batch, imgs):
            lang = cand["language"]
            if img is None:
                stats["image_failed"] += 1
                continue
            if counts[lang] >= args.per_language:
                continue
            rel = Path("images") / LANGUAGES[lang]["codes"][0] / f"{counts[lang]:07d}.jpg"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            img.save(out / rel, format="JPEG", quality=92)
            records.append({"image": str(rel), "caption": cand["caption"], "language": lang,
                            "caption_source": cand["source"]})
            counts[lang] += 1
            stats["kept_from_" + cand["source"]] += 1
            bar.update(1)
        batch.clear()

    def full():
        return all(counts[l] >= args.per_language for l in wanted)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        in_batch: Counter = Counter()
        for i, row in enumerate(ds):
            if i >= args.max_scan or full():
                break
            stats["rows_scanned"] += 1
            for cand in candidates(row):
                stats["candidates"] += 1
                lang = language_of(row, cand["caption"])
                if lang is None:
                    stats["language_unknown"] += 1
                    continue
                if lang not in wanted:
                    stats["language_not_wanted"] += 1
                    continue
                need = args.per_language - counts[lang]
                # over-fetch 2x to absorb dead links, but don't flood the batch with one language
                if need <= 0 or in_batch[lang] >= max(2 * need, 1):
                    continue
                key = cand["image"] if isinstance(cand["image"], str) else None
                if key is not None:
                    if key in seen_urls:
                        stats["duplicate_url"] += 1
                        continue
                    seen_urls.add(key)
                cand["language"] = lang
                batch.append(cand)
                in_batch[lang] += 1
            if len(batch) >= args.workers * 4:
                flush(ex)
                in_batch.clear()
        if batch:
            flush(ex)
    bar.close()

    random.shuffle(records)
    n_val = max(1, int(len(records) * args.val_fraction)) if records else 0
    for name, part in (("val", records[:n_val]), ("train", records[n_val:])):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"dataset": args.dataset, "config": args.config, "split": args.split, "format": fmt,
               "language_from": lang_mode, "per_language": dict(counts),
               "train": len(records) - n_val, "val": n_val, "counters": dict(stats)}
    (out / "stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    missing = sorted(l for l in wanted if counts[l] < args.per_language)
    if missing:
        print(f"WARNING: fewer than {args.per_language} samples for: {missing} "
              f"(raise --max-scan, or that language is rare in this dataset/config)")


if __name__ == "__main__":
    main()
