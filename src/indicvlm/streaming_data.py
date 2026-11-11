"""Stream Chitrakshara straight from the Hugging Face Hub into QLoRA training.

This is the streaming alternative to ``prepare_data.py`` (which saves a local subset first):

    1. load & stream   load_dataset("krutrim-ai-labs/chitrakshara", "Chitrakshara-Cap", streaming=True)
    2. format          clean the caption, detect its language, build the chat example
                       user:      <image> + "Describe this image in {language}."  (the "question")
                       assistant: the Chitrakshara caption                          (the "answer")
    3. images          download image URLs in parallel inside a batched map; failed ones are dropped
    4. collate & mask  ``QwenVLCaptionCollator`` applies the Qwen2.5-VL chat template and masks
                       everything except the assistant answer (prompt, image tokens, padding -> -100)

Used by ``indicvlm.train`` when the config has ``streaming: true``.

Quick look at the stream (no GPU needed):
    python -m indicvlm.streaming_data --take 5
"""

from __future__ import annotations

import argparse
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .languages import LANGUAGES, canonical_language, detect_language
from .prepare_data import (CAPTION_COLS, IMAGE_COLS, _fetch, clean_caption, interleaved_pairs,
                           pick_column, resolve_config)

DEFAULT_LANGUAGES = [l for l in LANGUAGES if l != "Sanskrit"]


def _flatten_rows(batch: dict, fmt: str, img_col: str | None, cap_col: str | None, wanted: set,
                  min_chars: int, max_chars: int, min_words: int, min_side: int, sources) -> dict:
    """Batched map: raw rows -> {"image": url|PIL, "caption": str, "language": str} candidates."""
    out = {"image": [], "caption": [], "language": []}
    n = len(next(iter(batch.values())))
    for i in range(n):
        row = {k: v[i] for k, v in batch.items()}
        if fmt == "interleaved":
            cands = interleaved_pairs(row, sources, min_chars, max_chars, min_words, min_side)
        else:
            img = row.get(img_col)
            if isinstance(img, list):
                img = next((x for x in img if x is not None), None)
            cap = row.get(cap_col)
            if isinstance(cap, list):
                cap = next((c for c in cap if c), None)
            cap = clean_caption(cap, min_chars, max_chars, min_words)
            cands = [{"image": img, "caption": cap}] if cap and img is not None else []
        for c in cands:
            lang = detect_language(c["caption"])
            if lang in wanted:
                out["image"].append(c["image"])
                out["caption"].append(c["caption"])
                out["language"].append(lang)
    return out


def in_holdout(image, caption: str, fraction: float) -> bool:
    """Deterministic split by content hash: the same pair is always train or always eval,
    regardless of shuffling, shards or dataloader workers."""
    if fraction <= 0:
        return False
    key = (image if isinstance(image, str) else "") + "\x00" + caption
    return (zlib.crc32(key.encode("utf-8")) % 1_000_000) < fraction * 1_000_000


class _Holdout:
    def __init__(self, fraction: float, keep_eval: bool):
        self.fraction, self.keep_eval = fraction, keep_eval

    def __call__(self, example) -> bool:
        return in_holdout(example["image"], example["caption"], self.fraction) == self.keep_eval


class _PerLanguageCap:
    """Stateful filter keeping at most ``cap`` samples per language (per dataloader worker)."""

    def __init__(self, cap: int | None):
        self.cap, self.counts = cap, Counter()

    def __call__(self, example) -> bool:
        if not self.cap:
            return True
        if self.counts[example["language"]] >= self.cap:
            return False
        self.counts[example["language"]] += 1
        return True


def _download(batch: dict, workers: int, timeout: float, min_side: int) -> dict:
    """Batched map: fetch images in parallel, drop rows whose image fails."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        imgs = list(ex.map(lambda v: _fetch(v, timeout, min_side), batch["image"]))
    keep = [i for i, im in enumerate(imgs) if im is not None]
    return {"image": [imgs[i] for i in keep],
            "caption": [batch["caption"][i] for i in keep],
            "language": [batch["language"][i] for i in keep]}


def stream_chitrakshara(dataset: str = "krutrim-ai-labs/chitrakshara", config: str | None = "Chitrakshara-Cap",
                        split: str = "train", languages=None, max_per_language: int | None = None,
                        shuffle_buffer: int = 10_000, seed: int = 42,
                        holdout_fraction: float = 0.0, part: str = "train",
                        image_column: str | None = None, caption_column: str | None = None,
                        fmt: str = "auto", caption_source=("before", "after", "alt"),
                        min_caption_chars: int = 5, max_caption_chars: int = 500, min_caption_words: int = 3,
                        min_image_side: int = 64, download_workers: int = 32, download_batch: int = 64,
                        timeout: float = 10.0, load_dataset_fn=None):
    """Return a ``datasets.IterableDataset`` yielding {"image": PIL.Image, "caption", "language"}.

    With ``holdout_fraction`` > 0, pairs are split by a content hash: ``part="train"`` yields
    only non-held-out pairs and ``part="eval"`` only held-out ones (the two never overlap)."""
    if part not in ("train", "eval"):
        raise ValueError("part must be 'train' or 'eval'")
    if load_dataset_fn is None:
        from datasets import load_dataset as load_dataset_fn

    config = resolve_config(dataset, config)
    wanted = {canonical_language(l) for l in (languages or DEFAULT_LANGUAGES)} - {None}

    # 1. Load & stream
    raw = load_dataset_fn(dataset, config, split=split, streaming=True)
    cols = list(next(iter(raw)).keys())
    if fmt == "auto":
        fmt = "interleaved" if {"images", "texts"} <= set(cols) else "pairs"
    img_col = cap_col = None
    if fmt == "pairs":
        img_col = pick_column(cols, IMAGE_COLS, image_column)
        cap_col = pick_column(cols, CAPTION_COLS, caption_column)
        if img_col is None or cap_col is None:
            raise ValueError(f"Could not detect image/caption columns in {cols}")
    if shuffle_buffer:
        raw = raw.shuffle(seed=seed, buffer_size=shuffle_buffer)

    # 2. Format: caption + language (rows with no usable caption / unknown language are dropped)
    ds = raw.map(_flatten_rows, batched=True, batch_size=256, remove_columns=cols,
                 fn_kwargs=dict(fmt=fmt, img_col=img_col, cap_col=cap_col, wanted=wanted,
                                min_chars=min_caption_chars, max_chars=max_caption_chars,
                                min_words=min_caption_words, min_side=min_image_side,
                                sources=tuple(caption_source)))
    if holdout_fraction > 0:
        ds = ds.filter(_Holdout(holdout_fraction, keep_eval=(part == "eval")))
    # Balance: per-language cap before downloading, so capped languages cost no bandwidth
    ds = ds.filter(_PerLanguageCap(max_per_language))
    # 3. Images: parallel download, failed rows dropped
    ds = ds.map(_download, batched=True, batch_size=download_batch,
                fn_kwargs=dict(workers=download_workers, timeout=timeout, min_side=min_image_side))
    return ds


def from_config(cfg: dict, eval_split: bool = False):
    """Build the training (or held-out eval) stream from the training YAML config."""
    s = cfg.get("stream", {}) or {}
    kw = dict(dataset=s.get("dataset", "krutrim-ai-labs/chitrakshara"),
              config=s.get("config", "Chitrakshara-Cap"), split=s.get("split", "train"),
              languages=s.get("languages"), shuffle_buffer=s.get("shuffle_buffer", 10_000),
              seed=cfg.get("seed", 42), image_column=s.get("image_column"),
              caption_column=s.get("caption_column"), max_caption_chars=s.get("max_caption_chars", 500),
              min_image_side=s.get("min_image_side", 64), download_workers=s.get("download_workers", 32),
              timeout=s.get("timeout", 10.0))
    frac = float(s.get("holdout_fraction", 0.0))
    if eval_split:
        # Collected once into memory (a fixed list), so every evaluation sees the same
        # samples and does not re-download them.
        n = int(s.get("max_eval_samples", 0))
        if frac <= 0 or n <= 0:
            return None
        ds = stream_chitrakshara(**{**kw, "shuffle_buffer": 0}, holdout_fraction=frac, part="eval")
        return list(ds.take(n))
    return stream_chitrakshara(**kw, holdout_fraction=frac, part="train",
                               max_per_language=s.get("max_per_language"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="krutrim-ai-labs/chitrakshara")
    p.add_argument("--config", default="Chitrakshara-Cap")
    p.add_argument("--languages", nargs="*", default=None)
    p.add_argument("--take", type=int, default=5)
    a = p.parse_args(argv)
    ds = stream_chitrakshara(a.dataset, a.config, languages=a.languages, shuffle_buffer=0)
    counts = Counter()
    for ex in ds.take(a.take):
        counts[ex["language"]] += 1
        print(f"[{ex['language']}] {ex['image'].size} {ex['caption'][:100]}")
    print(dict(counts))


if __name__ == "__main__":
    main()
