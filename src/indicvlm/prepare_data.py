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

The datasets carry no language label, so the language of every pair is detected from
the caption's Unicode script (see ``indicvlm.languages.detect_language``). Images are downloaded in parallel.

Output (``--out``)::

    images/<lang-code>/0000000.jpg
    train.jsonl / val.jsonl   {"image": "images/hi/0000000.jpg", "caption": "...", "language": "Hindi"}
    stats.json

The Hub dataset is ``krutrim-ai-labs/chitrakshara`` with two configs: ``Chitrakshara-Cap``
(image-caption pairs, the default here) and ``Chitrakshara-IL`` (interleaved documents).
Config names are matched case-insensitively.

Example:
    python -m indicvlm.prepare_data --out data/chitrakshara_cap --per-language 5000
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
from collections import Counter, defaultdict
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .languages import LANGUAGES, canonical_language, detect_language

IMAGE_COLS = ["image", "images", "img", "image_url", "url", "URL", "image_path", "img_url"]
CAPTION_COLS = ["caption", "text", "texts", "TEXT", "description", "alt_text", "captions"]
_SENT_END = re.compile(r"[।॥.!?|]")


def resolve_config(dataset: str, config: str | None, lister=None) -> str | None:
    """Return the dataset's config name matching ``config`` case-insensitively (as given if unknown)."""
    if not config:
        return config
    try:
        if lister is None:
            from datasets import get_dataset_config_names as lister
        names = list(lister(dataset))
    except Exception:
        return config
    for n in names:
        if n.lower() == config.lower():
            return n
    raise SystemExit(f"Config {config!r} not found in {dataset}; available: {names}")


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
    if isinstance(img, list):  # e.g. an "images" column holding a one-element list
        img = next((i for i in img if i is not None), None)
    if cap and img is not None:
        yield {"image": img, "caption": cap, "source": "pair"}


# --------------------------------------------------------------------------- images
class SlowDownload(Exception):
    """Raised when a download exceeds its total time budget or size cap."""


_thread_local = threading.local()
USER_AGENT = "Mozilla/5.0 (indicvlm-data-prep)"


def _session():
    """One keep-alive HTTP session per worker thread (no automatic retries)."""
    s = getattr(_thread_local, "session", None)
    if s is None:
        import requests
        from requests.adapters import HTTPAdapter

        s = requests.Session()
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        s.headers["User-Agent"] = USER_AGENT
        _thread_local.session = s
    return s


def http_get(url: str, timeout: float = 5.0, deadline: float = 15.0, max_bytes: int = 20_000_000) -> bytes:
    """Download ``url`` with a connect/read ``timeout`` AND a hard total ``deadline``.

    ``requests``' own timeout only bounds each socket operation, so a server that trickles
    bytes can otherwise hold a worker for minutes. Here the body is read in chunks and the
    download is abandoned once ``deadline`` seconds have passed or ``max_bytes`` is exceeded.
    """
    t0 = time.monotonic()
    with _session().get(url, timeout=(timeout, timeout), stream=True) as r:
        r.raise_for_status()
        if r.headers.get("Content-Type", "").startswith("text/"):
            raise ValueError("not an image")
        buf = bytearray()
        # read1() returns whatever bytes have arrived (one socket read), so the deadline is
        # checked even when a server drips data; iter_content() would block until a full
        # chunk or the end of the body arrives.
        read = getattr(r.raw, "read1", None)
        chunks = (iter(lambda: read(65536, decode_content=True), b"") if read
                  else r.iter_content(chunk_size=65536))
        for chunk in chunks:
            buf += chunk
            if len(buf) > max_bytes:
                raise SlowDownload("too large")
            if time.monotonic() - t0 > deadline:
                raise SlowDownload("deadline exceeded")
    return bytes(buf)


def to_rgb(img):
    """Convert any PIL mode to RGB; transparent pixels become white (not black)."""
    from PIL import Image

    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")  # avoids PIL's palette-transparency warning
    if img.mode in ("RGBA", "LA", "PA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.convert("RGBA").getchannel("A"))
        return bg
    return img.convert("RGB")


def _open(data: bytes):
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    img.load()
    return to_rgb(img)


def load_image(value, timeout: float = 5.0, deadline: float = 15.0):
    """Return a PIL RGB image from a PIL image, HF image dict, bytes, local path or URL."""
    from PIL import Image

    if value is None:
        return None
    if isinstance(value, Image.Image):
        return to_rgb(value)
    if isinstance(value, dict):  # HF Image feature in non-decoded form
        if value.get("bytes"):
            return _open(value["bytes"])
        value = value.get("path")
    if isinstance(value, (bytes, bytearray)):
        return _open(bytes(value))
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            return _open(http_get(value, timeout, deadline))
        if os.path.exists(value):
            with open(value, "rb") as f:
                return _open(f.read())
    return None


def _failure_reason(exc: Exception) -> str:
    name = type(exc).__name__
    if isinstance(exc, SlowDownload):
        return "slow"
    if "Timeout" in name:
        return "timeout"
    if name in ("HTTPError",):
        return "http_error"
    if "Connection" in name or "SSL" in name or name in ("InvalidURL", "MissingSchema", "TooManyRedirects"):
        return "connection"
    if name in ("UnidentifiedImageError", "DecompressionBombError", "OSError", "SyntaxError", "ValueError"):
        return "not_image"
    return "other"


def fetch_image(value, timeout: float = 5.0, min_side: int = 64, deadline: float = 15.0,
                max_side: int | None = 1024):
    """Return ``(image, None)`` or ``(None, reason)``. Large images are downscaled to ``max_side``
    (Qwen2.5-VL resizes to <= max_pixels anyway), which saves disk space and encode time."""
    try:
        img = load_image(value, timeout, deadline=deadline)
    except Exception as e:  # noqa: BLE001 - every failure just means "skip this image"
        return None, _failure_reason(e)
    if img is None:
        return None, "missing"
    if min(img.size) < min_side:
        return None, "too_small"
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    return img, None


def _fetch(value, timeout, min_side, deadline: float = 15.0):
    return fetch_image(value, timeout, min_side, deadline)[0]


# --------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="krutrim-ai-labs/chitrakshara")
    p.add_argument("--config", default="Chitrakshara-Cap",
                   help="HF config: Chitrakshara-Cap (pairs) or Chitrakshara-IL (interleaved); case-insensitive")
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
    p.add_argument("--min-caption-chars", type=int, default=5)
    p.add_argument("--max-caption-chars", type=int, default=500)
    p.add_argument("--min-caption-words", type=int, default=3)
    p.add_argument("--min-image-side", type=int, default=64)
    p.add_argument("--workers", type=int, default=64, help="parallel image downloads")
    p.add_argument("--timeout", type=float, default=5.0, help="connect/read timeout per socket operation (s)")
    p.add_argument("--deadline", type=float, default=15.0, help="hard limit on total time per image (s)")
    p.add_argument("--max-side", type=int, default=1024, help="downscale saved images to this longest side")
    p.add_argument("--max-scan", type=int, default=5_000_000, help="stop after scanning this many rows")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    from datasets import load_dataset
    from tqdm import tqdm

    random.seed(args.seed)
    wanted = {canonical_language(l) for l in args.languages}
    wanted.discard(None)
    sources = [s.strip() for s in args.caption_source.split(",") if s.strip()]
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    args.config = resolve_config(args.dataset, args.config)
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
    print(f"Columns: {cols}\n -> format={fmt}"
          + (f" image={img_col} caption={cap_col}" if fmt == "pairs" else f" caption sources={sources}"))

    def candidates(row):
        if fmt == "interleaved":
            yield from interleaved_pairs(row, sources, args.min_caption_chars, args.max_caption_chars,
                                         args.min_caption_words, args.min_image_side)
        else:
            yield from pair_rows(row, img_col, cap_col, args.min_caption_chars, args.max_caption_chars,
                                 args.min_caption_words)

    counts: dict[str, int] = defaultdict(int)
    stats = Counter()
    records, seen_urls = [], set()
    bar = tqdm(total=args.per_language * len(wanted), desc="kept")

    # Downloads run continuously: up to `max_inflight` are in flight at any time and each
    # result is handled as soon as it finishes, so one slow site never stalls the others
    # (the old version waited for a whole batch, i.e. for its slowest URL).
    ex = ThreadPoolExecutor(max_workers=args.workers)
    inflight: dict = {}  # future -> candidate
    pending: Counter = Counter()  # in-flight downloads per language
    max_inflight = args.workers * 2
    last_postfix = 0.0

    def job(value):
        return fetch_image(value, args.timeout, args.min_image_side, args.deadline, args.max_side)

    def handle(fut):
        cand = inflight.pop(fut)
        lang = cand["language"]
        pending[lang] -= 1
        img, reason = fut.result()
        if img is None:
            stats["image_failed"] += 1
            stats["failed_" + reason] += 1
            return
        if counts[lang] >= args.per_language:
            stats["surplus_downloads"] += 1
            return
        rel = Path("images") / LANGUAGES[lang]["codes"][0] / f"{counts[lang]:07d}.jpg"
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        img.save(out / rel, format="JPEG", quality=90)
        records.append({"image": str(rel), "caption": cand["caption"], "language": lang,
                        "caption_source": cand["source"]})
        counts[lang] += 1
        stats["kept_from_" + cand["source"]] += 1
        bar.update(1)

    def harvest(block: bool):
        nonlocal last_postfix
        if inflight:
            done, _ = wait(list(inflight), timeout=None if block else 0, return_when=FIRST_COMPLETED)
            for f in done:
                handle(f)
        now = time.monotonic()
        if now - last_postfix > 2:
            last_postfix = now
            tried = stats["image_failed"] + sum(counts.values())
            bar.set_postfix(rows=stats["rows_scanned"], inflight=len(inflight),
                            fail=f"{stats['image_failed'] / max(tried, 1):.0%}",
                            lang_skip=stats["language_unknown"] + stats["language_not_wanted"], refresh=False)

    def full():
        return all(counts[l] >= args.per_language for l in wanted)

    try:
        for i, row in enumerate(ds):
            if i >= args.max_scan or full():
                break
            stats["rows_scanned"] += 1
            for cand in candidates(row):
                stats["candidates"] += 1
                lang = detect_language(cand["caption"])
                if lang is None:
                    stats["language_unknown"] += 1
                    continue
                if lang not in wanted:
                    stats["language_not_wanted"] += 1
                    continue
                need = args.per_language - counts[lang]
                # over-fetch up to 2x what is still needed, to absorb dead links
                if need <= 0 or pending[lang] >= max(2 * need, 1):
                    continue
                key = cand["image"] if isinstance(cand["image"], str) else None
                if key is not None:
                    if key in seen_urls:
                        stats["duplicate_url"] += 1
                        continue
                    seen_urls.add(key)
                cand["language"] = lang
                inflight[ex.submit(job, cand["image"])] = cand
                pending[lang] += 1
                while len(inflight) >= max_inflight:
                    harvest(block=True)
            harvest(block=False)
        while inflight and not full():
            harvest(block=True)
    finally:
        # don't wait for stragglers once every quota is met
        ex.shutdown(wait=False, cancel_futures=True)
    bar.close()

    random.shuffle(records)
    n_val = max(1, int(len(records) * args.val_fraction)) if records else 0
    for name, part in (("val", records[:n_val]), ("train", records[n_val:])):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"dataset": args.dataset, "config": args.config, "split": args.split, "format": fmt,
               "per_language": dict(counts),
               "train": len(records) - n_val, "val": n_val, "counters": dict(stats)}
    (out / "stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    missing = sorted(l for l in wanted if counts[l] < args.per_language)
    if missing:
        print(f"WARNING: fewer than {args.per_language} samples for: {missing} "
              f"(raise --max-scan, or that language is rare in this dataset/config)")


if __name__ == "__main__":
    main()
