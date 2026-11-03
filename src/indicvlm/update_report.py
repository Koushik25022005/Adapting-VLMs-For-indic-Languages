"""Write BharatBench numbers into the research report (between the RESULTS markers).

    python -m indicvlm.update_report \
        --ours results/qwen25vl7b-qlora/bharatbench_scores.json \
        --base results/qwen25vl7b-base/bharatbench_scores.json      # optional zero-shot row

Missing values are shown as "—" so the report is valid before every run has finished.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .languages import BHARATBENCH_REPORT_LANGUAGES

METRICS = ["POPE", "LLaVA-Bench", "MMVet"]
BEGIN, END = "<!-- BEGIN:BHARATBENCH_RESULTS -->", "<!-- END:BHARATBENCH_RESULTS -->"


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


def update_report(report: Path, tables: str) -> None:
    text = report.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{report} has no {BEGIN} ... {END} markers")
    new = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: f"{BEGIN}\n{tables}\n{END}",
                 text, flags=re.S)
    report.write_text(new, encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="report/research_report.md")
    p.add_argument("--chitrarth", default="results/chitrarth_bharatbench.json")
    p.add_argument("--ours", default=None)
    p.add_argument("--base", default=None)
    a = p.parse_args(argv)
    load = lambda f: json.loads(Path(f).read_text(encoding="utf-8")) if f else None  # noqa: E731
    update_report(Path(a.report), render_tables(load(a.chitrarth), load(a.ours), load(a.base)))
    print(f"updated {a.report}")


if __name__ == "__main__":
    main()
