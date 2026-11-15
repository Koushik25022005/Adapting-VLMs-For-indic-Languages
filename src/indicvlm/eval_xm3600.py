"""Evaluate image captioning on XM3600 (Crossmodal-3600; Thapliyal et al., EMNLP 2022).

3,600 images, each with human captions in 36 languages. The Indic ones are Hindi, Bengali
and Telugu. This is the task the adapter was trained for (captioning), so it is the
in-domain check. The metric is CIDEr-D (``indicvlm.cider``), which needs no judge model.

    python -m indicvlm.eval_xm3600 --load-4bit --run-name qwen25vl7b-base
    python -m indicvlm.eval_xm3600 --load-4bit --run-name qwen25vl7b-qlora \
        --adapter outputs/qwen25vl7b-chitrakshara-qlora/final_adapter

Writes ``results/<run-name>/xm3600_scores.json`` and ``xm3600_predictions.jsonl``.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from .cider import cider_d, tokenize
from .languages import detect_language

# XM3600 split (language code) -> language name used in the prompt.
LANGUAGES = {"hi": "Hindi", "bn": "Bengali", "te": "Telugu", "en": "English"}
DEFAULT_PROMPT = "Describe this image in one short sentence in {language}."


def to_image(value):
    """The XM3600 copies on the Hub store the image as a PIL image, raw bytes or {'bytes': ...}."""
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        value = value.get("bytes") or value.get("path")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value))
    return Image.open(value)


def references(row: dict) -> list[str]:
    caps = row.get("captions")
    if isinstance(caps, str):
        caps = [caps]
    return [c for c in (caps or []) if isinstance(c, str) and c.strip()]


def score(records: list[dict]) -> dict:
    """Records hold language, references and prediction. Returns {language: {metric: value}}."""
    out = {}
    for lang in dict.fromkeys(r["language"] for r in records):
        rows = [r for r in records if r["language"] == lang]
        cider, _ = cider_d([r["references"] for r in rows], [r["prediction"] for r in rows])
        in_lang = sum(detect_language(r["prediction"]) == lang for r in rows)
        out[lang] = {
            "CIDEr": round(cider, 2),
            # Share of captions written in the requested language (script-based detection).
            "In-language %": round(100 * in_lang / len(rows), 2),
            "Avg words": round(sum(len(tokenize(r["prediction"])) for r in rows) / len(rows), 1),
            "n": len(rows),
        }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="floschne/xm3600")
    p.add_argument("--languages", nargs="+", default=list(LANGUAGES), help="XM3600 language codes")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="must contain {language}")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--adapter", default=None, help="PEFT LoRA adapter dir; omit for the base model")
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--load-4bit", action="store_true", help="evaluate with the same NF4 base used in training")
    p.add_argument("--limit", type=int, default=None, help="debug: only the first N images per language")
    args = p.parse_args(argv)

    from datasets import Image as HFImage
    from datasets import load_dataset
    from tqdm import tqdm

    from .infer_bharatbench import QwenVL

    name = args.run_name or ("qwen25vl7b-qlora" if args.adapter else "qwen25vl7b-base")
    model = QwenVL(args.base_model, args.adapter, args.load_4bit, args.max_pixels)

    records = []
    for code in args.languages:
        language = LANGUAGES.get(code, code)
        ds = load_dataset(args.dataset, split=code)
        if isinstance(ds.features.get("image"), HFImage):
            ds = ds.cast_column("image", HFImage(decode=False))  # decode ourselves, one batch at a time
        n = min(len(ds), args.limit) if args.limit else len(ds)
        prompt = args.prompt.format(language=language)
        for start in tqdm(range(0, n, args.batch_size), desc=f"xm3600/{code}/{name}"):
            rows = [ds[i] for i in range(start, min(start + args.batch_size, n))]
            preds = model.generate([(to_image(r["image"]), prompt) for r in rows], args.max_new_tokens)
            for r, pred in zip(rows, preds):
                records.append({"image_id": r.get("image_id"), "language": language,
                                "references": references(r), "prediction": pred})

    out = Path(args.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "xm3600_predictions.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    res = {"model": name, "benchmark": "XM3600", "dataset": args.dataset, "prompt": args.prompt,
           "scores": score(records)}
    (out / "xm3600_scores.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
