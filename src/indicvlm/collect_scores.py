"""Parse the score files written by the BharatBench evaluators into one JSON file.

Metrics (same as Chitrarth's BharatBench table):
  POPE        overall accuracy x 100 (mean over adversarial / popular / random)
  LLaVA-Bench relative score: model / GPT-4-reference judge score x 100 ("all" row)
  MM-Vet      total score (0-100)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .languages import BHARATBENCH_REPORT_LANGUAGES


def parse_pope(text: str) -> float | None:
    m = re.findall(r"Overall Accuracy:\s*([0-9.]+)", text)
    return round(float(m[-1]) * 100, 2) if m else None


def parse_llavabench(text: str) -> dict[str, float]:
    """``results.txt`` holds blocks 'Processing: review_<Lang>' followed by '<cat> rel gpt4 model'."""
    out, current = {}, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Processing:"):
            current = line.split(":", 1)[1].strip().replace("review_", "")
        elif current and line.startswith("all "):
            out[current] = float(line.split()[1])  # last block for a language wins
    return out


def parse_mmvet(text: str) -> dict[str, float]:
    """``grades.txt`` holds pandas tables: a header (rec ocr ... total std runs) + 'mm-vet_<Lang>' rows."""
    out, header = {}, None
    for line in text.splitlines():
        toks = line.split()
        if not toks:
            continue
        if "total" in toks and not toks[0].startswith("mm-vet_"):
            header = toks
        elif toks[0].startswith("mm-vet_") and header:
            lang = toks[0][len("mm-vet_"):]
            out[lang] = round(float(toks[1 + header.index("total")]), 2)
    return out


def collect(workdir: Path, pope_scorer: str = "gpt", languages=BHARATBENCH_REPORT_LANGUAGES) -> dict:
    scores: dict[str, dict] = {l: {} for l in languages}
    pope_dir = workdir / "Indic-POPE-Evaluator/scores"
    for lang in languages:
        fn = f"results_{lang.lower()}.txt" if pope_scorer == "gpt" else f"{pope_scorer}_results_{lang.lower()}.txt"
        if (pope_dir / fn).exists():
            v = parse_pope((pope_dir / fn).read_text(encoding="utf-8"))
            if v is not None:
                scores[lang]["POPE"] = v
    lb = workdir / "Indic-LLaVABench-Evaluator/scores/results.txt"
    if lb.exists():
        for lang, v in parse_llavabench(lb.read_text(encoding="utf-8")).items():
            scores.setdefault(lang, {})["LLaVA-Bench"] = v
    mv = workdir / "Indic-MMVet-Evaluator/grades/grades.txt"
    if mv.exists():
        for lang, v in parse_mmvet(mv.read_text(encoding="utf-8")).items():
            scores.setdefault(lang, {})["MMVet"] = v
    return scores


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True, help="BharatBench working copy the evaluators ran in")
    p.add_argument("--out", required=True)
    p.add_argument("--model-name", default="model")
    p.add_argument("--pope-scorer", default="gpt")
    a = p.parse_args(argv)
    res = {"model": a.model_name, "pope_scorer": a.pope_scorer, "scores": collect(Path(a.workdir), a.pope_scorer)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
