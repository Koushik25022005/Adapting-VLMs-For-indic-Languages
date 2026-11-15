"""Evaluate on CVQA (Romero et al., NeurIPS 2024): culturally grounded multiple-choice VQA.

Each question has an image, four options and one correct option. It is asked in the local
language and in English. Scoring is plain accuracy on the option letter, so no judge model
is involved.

    python -m indicvlm.eval_cvqa --load-4bit --run-name qwen25vl7b-base
    python -m indicvlm.eval_cvqa --load-4bit --run-name qwen25vl7b-qlora \
        --adapter outputs/qwen25vl7b-chitrakshara-qlora/final_adapter

Writes ``results/<run-name>/cvqa_scores.json`` and ``cvqa_predictions.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LETTERS = "ABCD"
DEFAULT_COUNTRIES = ["India"]
PROMPT_TAIL = "Answer with the letter of the correct option only."
VARIANTS = {"local": ("Question", "Options"), "english": ("Translated Question", "Translated Options")}
METRIC = {"local": "Accuracy (local)", "english": "Accuracy (English)"}


def parse_subset(value) -> tuple[str, str]:
    """``"('Hindi', 'India')"`` (also a real tuple/list, or ``Hindi-India``) -> (language, country)."""
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return str(value[0]).strip(), str(value[1]).strip()
    parts = re.findall(r"['\"]([^'\"]+)['\"]", str(value))
    if len(parts) < 2:
        parts = [p for p in re.split(r"[-_,/()]+", str(value)) if p.strip()]
    if len(parts) < 2:
        return str(value).strip(), ""
    return parts[0].strip(), parts[1].strip()


def build_prompt(question: str, options: list[str]) -> str:
    lines = [str(question).strip()] + [f"{LETTERS[i]}. {str(o).strip()}" for i, o in enumerate(options)]
    return "\n".join(lines) + "\n" + PROMPT_TAIL


def parse_answer(response: str, options: list[str]) -> int | None:
    """Index of the chosen option, or None when the response names no option."""
    text = (response or "").strip()
    # "B", "(b)", "B. Rose", "B: ..." -- but not a sentence that starts with the article "A".
    m = re.match(r"^[\s(\[\"'*]*([A-Da-d])(?:[\s\"'*]*$|[.):\],])", text)
    if not m:  # "The answer is B", "Option C", "उत्तर: D"
        m = re.search(r"(?:answer|option|उत्तर|উত্তর|பதில்|సమాధానం)(?:\s+is)?\W{0,6}([A-D])(?![A-Za-z])", text, flags=re.I)
    if m:
        return LETTERS.index(m.group(1).upper())
    low = text.lower()
    hits = [i for i, o in enumerate(options) if str(o).strip() and str(o).strip().lower() in low]
    return hits[0] if len(hits) == 1 else None


def has_label(label) -> bool:
    return isinstance(label, int) and not isinstance(label, bool) and 0 <= label < len(LETTERS)


def score(records: list[dict]) -> dict:
    """Records hold language, variant, label and pred. Returns {language: {metric: value}}."""
    out: dict[str, dict] = {}
    for variant, metric in METRIC.items():
        by_lang: dict[str, list[dict]] = {}
        for r in records:
            if r["variant"] == variant:
                by_lang.setdefault(r["language"], []).append(r)
        for lang, rows in by_lang.items():
            labelled = [r for r in rows if has_label(r["label"])]
            d = out.setdefault(lang, {})
            d["n"] = len(rows)
            if labelled:
                d[metric] = round(100 * sum(r["pred"] == r["label"] for r in labelled) / len(labelled), 2)
            d[f"Unparsed % ({variant})"] = round(100 * sum(r["pred"] is None for r in rows) / len(rows), 2)
    return out


def select_rows(ds, countries, languages, limit):
    """Indices to evaluate, chosen from the Subset column alone (no image decoding)."""
    want_c = {c.lower() for c in countries} if countries else None
    want_l = {l.lower() for l in languages} if languages else None
    picked, per_lang = [], {}
    for i, subset in enumerate(ds["Subset"]):
        lang, country = parse_subset(subset)
        if want_c is not None and country.lower() not in want_c:
            continue
        if want_l is not None and lang.lower() not in want_l:
            continue
        if limit and per_lang.get(lang, 0) >= limit:
            continue
        per_lang[lang] = per_lang.get(lang, 0) + 1
        picked.append(i)
    return picked


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="afaji/cvqa")
    p.add_argument("--split", default="test")
    p.add_argument("--countries", nargs="*", default=DEFAULT_COUNTRIES,
                   help="keep subsets from these countries (pass none to keep every country)")
    p.add_argument("--languages", nargs="*", default=None, help="keep only these subset languages")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS),
                   help="ask the question in the local language, in English, or both")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--adapter", default=None, help="PEFT LoRA adapter dir; omit for the base model")
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--load-4bit", action="store_true", help="evaluate with the same NF4 base used in training")
    p.add_argument("--limit", type=int, default=None, help="debug: only the first N questions per language")
    args = p.parse_args(argv)

    from datasets import load_dataset
    from tqdm import tqdm

    from .infer_bharatbench import QwenVL

    name = args.run_name or ("qwen25vl7b-qlora" if args.adapter else "qwen25vl7b-base")
    ds = load_dataset(args.dataset, split=args.split)
    idx = select_rows(ds, args.countries, args.languages, args.limit)
    if not idx:
        raise SystemExit(f"no CVQA rows match countries={args.countries} languages={args.languages}; "
                         f"Subset values look like {sorted(set(map(str, ds['Subset'])))[:5]}")
    ds = ds.select(idx)
    model = QwenVL(args.base_model, args.adapter, args.load_4bit, args.max_pixels)

    records = []
    for start in tqdm(range(0, len(ds), args.batch_size), desc=f"cvqa/{name}"):
        rows = [ds[i] for i in range(start, min(start + args.batch_size, len(ds)))]
        for variant in args.variants:
            qcol, ocol = VARIANTS[variant]
            preds = model.generate([(r["image"], build_prompt(r[qcol], r[ocol])) for r in rows],
                                   args.max_new_tokens)
            for r, resp in zip(rows, preds):
                lang, country = parse_subset(r["Subset"])
                records.append({"id": r.get("ID"), "language": lang, "country": country, "variant": variant,
                                "label": r.get("Label"), "pred": parse_answer(resp, r[ocol]), "response": resp})

    out = Path(args.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cvqa_predictions.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    scores = score(records)
    res = {"model": name, "benchmark": "CVQA", "dataset": args.dataset, "countries": args.countries,
           "scores": scores}
    (out / "cvqa_scores.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if not any(has_label(r["label"]) for r in records):
        print(f"\nThis copy of CVQA has no answer labels (Label is {records[0]['label']!r}), so no accuracy "
              f"was computed. The predictions are in {out / 'cvqa_predictions.jsonl'}.")


if __name__ == "__main__":
    main()
