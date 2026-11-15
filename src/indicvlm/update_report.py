"""Write measured numbers into the research report (between the RESULTS markers).

BharatBench (ours vs Chitrarth, optional zero-shot column):

    python -m indicvlm.update_report \
        --ours results/qwen25vl7b-qlora/bharatbench_scores.json \
        --base results/qwen25vl7b-base/bharatbench_scores.json

CVQA and XM3600 (zero-shot base model vs QLoRA):

    python -m indicvlm.update_report \
        --cvqa-base results/qwen25vl7b-base/cvqa_scores.json \
        --cvqa-ours results/qwen25vl7b-qlora/cvqa_scores.json \
        --xm3600-base results/qwen25vl7b-base/xm3600_scores.json \
        --xm3600-ours results/qwen25vl7b-qlora/xm3600_scores.json

Only the sections whose score files are given are rewritten. Missing values are shown
as "—" so the report is valid before every run has finished.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .languages import BHARATBENCH_REPORT_LANGUAGES

METRICS = ["POPE", "LLaVA-Bench", "MMVet"]
BEGIN, END = "<!-- BEGIN:BHARATBENCH_RESULTS -->", "<!-- END:BHARATBENCH_RESULTS -->"
BASE_NAME, OURS_NAME = "Qwen2.5-VL-7B (zero-shot)", "Qwen2.5-VL-7B + QLoRA (ours)"
# Benchmarks reported as base model vs QLoRA: marker name -> metrics shown, in order.
COMPARISONS = {
    "CVQA": ["Accuracy (local)", "Accuracy (English)", "Unparsed % (local)"],
    "XM3600": ["CIDEr", "In-language %", "Avg words"],
}


def markers(name: str) -> tuple[str, str]:
    return f"<!-- BEGIN:{name}_RESULTS -->", f"<!-- END:{name}_RESULTS -->"


def _fmt(v, signed=False):
    if v is None:
        return "—"
    return f"{v:+.2f}" if signed else f"{v:.2f}"


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def render_tables(chitrarth: dict, ours: dict | None, base: dict | None = None,
                  languages=BHARATBENCH_REPORT_LANGUAGES) -> str:
    c, o = chitrarth["scores"], (ours or {}).get("scores", {})
    b = (base or {}).get("scores", {}) if base else None
    get = lambda s, l, m: (s.get(l) or {}).get(m)  # noqa: E731
    indic = [l for l in languages if l != "English"]

    parts = []
    for m in METRICS:
        hdr = ["Language", "Chitrarth"] + (["Qwen2.5-VL-7B (zero-shot)"] if b is not None else []) \
              + ["Qwen2.5-VL-7B + QLoRA (ours)", "Δ (ours − Chitrarth)"]
        rows = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] + ["---:"] * (len(hdr) - 1)) + "|"]

        def row(name, cv, bv, ov):
            d = ov - cv if ov is not None and cv is not None else None
            cells = [name, _fmt(cv)] + ([_fmt(bv)] if b is not None else []) + [_fmt(ov), _fmt(d, True)]
            return "| " + " | ".join(cells) + " |"

        for l in languages:
            rows.append(row(l, get(c, l, m), get(b, l, m) if b is not None else None, get(o, l, m)))
        for label, langs in (("**Indic avg (10)**", indic), ("**All avg (11)**", languages)):
            cv = _mean([get(c, l, m) for l in langs])
            bv = _mean([get(b, l, m) for l in langs]) if b is not None else None
            ovs = [get(o, l, m) for l in langs]
            ov = _mean(ovs) if all(v is not None for v in ovs) else None  # only average complete rows
            rows.append(row(label, cv, bv, ov))
        parts.append(f"#### {m}\n\n" + "\n".join(rows))
    return "\n\n".join(parts)


def render_comparison(base: dict | None, ours: dict | None, metrics: list[str]) -> str:
    """One table per metric: language | base | QLoRA | Δ (QLoRA − base), plus an Indic average."""
    b, o = (base or {}).get("scores", {}), (ours or {}).get("scores", {})
    get = lambda s, l, m: (s.get(l) or {}).get(m)  # noqa: E731
    langs = list(dict.fromkeys([*b, *o]))
    indic = [l for l in langs if l != "English"]
    parts = []
    for m in metrics:
        if not any(get(s, l, m) is not None for s in (b, o) for l in langs):
            continue  # e.g. no accuracy when the dataset copy has no labels
        rows = [f"| Language | n | {BASE_NAME} | {OURS_NAME} | Δ (QLoRA − base) |", "|---|---:|---:|---:|---:|"]

        def row(name, n, bv, ov):
            d = ov - bv if ov is not None and bv is not None else None
            return f"| {name} | {n if n is not None else '—'} | {_fmt(bv)} | {_fmt(ov)} | {_fmt(d, True)} |"

        for l in langs:
            n = get(o, l, "n") if get(o, l, "n") is not None else get(b, l, "n")
            rows.append(row(l, n, get(b, l, m), get(o, l, m)))
        if len(indic) > 1:
            cols = [[get(s, l, m) for l in indic] for s in (b, o)]
            bv, ov = (_mean(c) if all(v is not None for v in c) else None for c in cols)  # complete rows only
            rows.append(row(f"**Indic avg ({len(indic)})**", None, bv, ov))
        parts.append(f"#### {m}\n\n" + "\n".join(rows))
    return "\n\n".join(parts) if parts else "_No scores yet._"


def update_report(report: Path, tables: str, begin: str = BEGIN, end: str = END) -> None:
    text = report.read_text(encoding="utf-8")
    if begin not in text or end not in text:
        raise SystemExit(f"{report} has no {begin} ... {end} markers")
    new = re.sub(re.escape(begin) + r".*?" + re.escape(end), lambda _: f"{begin}\n{tables}\n{end}",
                 text, flags=re.S)
    report.write_text(new, encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="report/research_report.md")
    p.add_argument("--chitrarth", default="results/chitrarth_bharatbench.json")
    p.add_argument("--ours", default=None)
    p.add_argument("--base", default=None)
    for name in COMPARISONS:
        p.add_argument(f"--{name.lower()}-base", default=None, help=f"{name} scores of the zero-shot base model")
        p.add_argument(f"--{name.lower()}-ours", default=None, help=f"{name} scores of the QLoRA model")
    a = p.parse_args(argv)
    load = lambda f: json.loads(Path(f).read_text(encoding="utf-8")) if f else None  # noqa: E731
    done = []
    if a.ours or a.base:
        update_report(Path(a.report), render_tables(load(a.chitrarth), load(a.ours), load(a.base)))
        done.append("BharatBench")
    for name, metrics in COMPARISONS.items():
        base, ours = getattr(a, f"{name.lower()}_base"), getattr(a, f"{name.lower()}_ours")
        if base or ours:
            update_report(Path(a.report), render_comparison(load(base), load(ours), metrics), *markers(name))
            done.append(name)
    print(f"updated {a.report}: {', '.join(done) or 'nothing (no score files given)'}")


if __name__ == "__main__":
    main()
