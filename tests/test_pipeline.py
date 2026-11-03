import json
import os
import random
from pathlib import Path

import pytest

from indicvlm.collect_scores import collect, parse_llavabench, parse_mmvet, parse_pope
from indicvlm.data import IGNORE_INDEX, build_messages, mask_labels
from indicvlm.infer_bharatbench import (LOADERS, ImageIndex, response_path, write_responses)
from indicvlm.languages import BHARATBENCH_REPORT_LANGUAGES, canonical_language
from indicvlm.prepare_data import pick_column, valid_caption
from indicvlm.update_report import BEGIN, END, render_tables, update_report

ROOT = Path(__file__).resolve().parents[1]
BB = Path(os.environ.get("BHARATBENCH_DIR", ROOT / "third_party/BharatBench"))

# Excerpts of real BharatBench evaluator outputs (format reference).
POPE_TXT = """Category: adversarial, Accuracy: 0.7607
Category: popular, Accuracy: 0.8197
Category: random, Accuracy: 0.5124
Overall Accuracy: 0.6976
"""
LLAVA_TXT = """Processing: review_Telugu
all 9.5 94.5 9.0
llava_bench_complex 9.7 91.8 8.9
llava_bench_conv 9.9 94.7 9.4
=================================
Processing: review_Hindi
all 55.1 80.0 44.1
llava_bench_conv 50.0 80.0 40.0
=================================
"""
MMVET_TXT = """                   rec      ocr      know   gen  spat  math     total  std        runs
mm-vet_Hindi  35.53333  30.3125  28.09524  33.5  28.0  15.0  31.74312  0.0  [31.74312]

                   rec      ocr      know   gen  spat  math     total  std        runs
mm-vet_Odia  20.0  10.0  10.0  10.0  10.0  5.0  21.5  0.0  [21.5]
"""


def test_canonical_language():
    assert canonical_language("hi") == "Hindi"
    assert canonical_language("hin_Deva") == "Hindi"
    assert canonical_language("ORIYA") == "Odia"
    assert canonical_language("as") == "Assamese"
    assert canonical_language("xx") is None
    assert len(BHARATBENCH_REPORT_LANGUAGES) == 11


def test_prepare_helpers():
    assert pick_column(["url", "caption", "lang"], ["image", "url"]) == "url"
    assert pick_column(["a"], ["b"]) is None
    with pytest.raises(KeyError):
        pick_column(["a"], ["b"], override="c")
    assert valid_caption("नमस्ते दुनिया", 5, 100)
    assert not valid_caption("ab", 5, 100)
    assert not valid_caption(None, 5, 100)


def test_build_messages():
    m = build_messages("ఒక పిల్లి", "Telugu", random.Random(0))
    assert m[0]["role"] == "user" and m[0]["content"][0] == {"type": "image"}
    assert "Telugu" in m[0]["content"][1]["text"]
    assert m[1] == {"role": "assistant", "content": [{"type": "text", "text": "ఒక పిల్లి"}]}


def test_mask_labels_only_assistant_tokens():
    H, END_T, PAD = [7, 8, 9], 2, 0
    #       system/user + image tokens   | header | caption | end | pad
    ids = [1, 5, 5, 6, 6, 6, 2, 3] + H + [40, 41, 42] + [END_T] + [PAD, PAD]
    labels = mask_labels(ids, H, END_T)
    s = 8 + len(H)
    assert labels[:s] == [IGNORE_INDEX] * s
    assert labels[s:s + 4] == [40, 41, 42, END_T]
    assert labels[s + 4:] == [IGNORE_INDEX, IGNORE_INDEX]
    # no assistant turn -> nothing supervised
    assert set(mask_labels([1, 2, 3], H, END_T)) == {IGNORE_INDEX}


def test_parsers():
    assert parse_pope(POPE_TXT) == 69.76
    assert parse_llavabench(LLAVA_TXT) == {"Telugu": 9.5, "Hindi": 55.1}
    assert parse_mmvet(MMVET_TXT) == {"Hindi": 31.74, "Odia": 21.5}


def test_collect(tmp_path):
    (tmp_path / "Indic-POPE-Evaluator/scores").mkdir(parents=True)
    (tmp_path / "Indic-POPE-Evaluator/scores/results_hindi.txt").write_text(POPE_TXT)
    (tmp_path / "Indic-LLaVABench-Evaluator/scores").mkdir(parents=True)
    (tmp_path / "Indic-LLaVABench-Evaluator/scores/results.txt").write_text(LLAVA_TXT)
    (tmp_path / "Indic-MMVet-Evaluator/grades").mkdir(parents=True)
    (tmp_path / "Indic-MMVet-Evaluator/grades/grades.txt").write_text(MMVET_TXT)
    s = collect(tmp_path)
    assert s["Hindi"] == {"POPE": 69.76, "LLaVA-Bench": 55.1, "MMVet": 31.74}
    assert s["Tamil"] == {}


def test_report_rendering(tmp_path):
    chit = json.loads((ROOT / "results/chitrarth_bharatbench.json").read_text())
    assert chit["scores"]["Telugu"] == {"POPE": 79.9, "LLaVA-Bench": 54.8, "MMVet": 43.76}
    ours = {"scores": {"Hindi": {"POPE": 80.0}}}
    t = render_tables(chit, ours)
    assert "| Hindi | 78.68 | 80.00 | +1.32 |" in t
    assert "| Tamil | 83.28 | — | — |" in t
    # averages are only shown for complete rows
    assert "| **Indic avg (10)** | 79.7" in t and "| **Indic avg (10)** |" in t.split("#### LLaVA")[0]
    assert t.split("#### LLaVA")[0].split("**Indic avg (10)** |")[1].split("|")[1].strip() == "—"
    rep = tmp_path / "r.md"
    rep.write_text(f"# R\n{BEGIN}\nold\n{END}\ntail\n")
    update_report(rep, t)
    txt = rep.read_text()
    assert "old" not in txt and "tail" in txt and "#### MMVet" in txt
    update_report(rep, "x")  # idempotent re-run
    assert rep.read_text().count(BEGIN) == 1


def test_write_responses_formats(tmp_path):
    pope_items = [{"qid": 1, "image": "a.jpg", "prompt": "q1", "raw": {}},
                  {"qid": 2, "image": "a.jpg", "prompt": "q2", "raw": {}}]
    p = write_responses(tmp_path, "pope", "Hindi", pope_items, {1: "हाँ", 2: "नहीं"}, "m")
    assert p == tmp_path / "Indic-POPE-Evaluator/model_responses/hindi_results.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert [r["question_id"] for r in rows] == [1, 2] and rows[0]["text"] == "हाँ"
    assert set(rows[0]) == {"question_id", "prompt", "text", "answer_id", "model_id", "metadata"}

    raw = {"image": "001.jpg", "text": "q", "category": "conv", "question_id": 0}
    p = write_responses(tmp_path, "llavabench", "Hindi", [{"qid": 0, "image": "001.jpg", "prompt": "q", "raw": raw}],
                        {0: "ans"}, "m")
    r = json.loads(p.read_text())
    assert p.name == "responses_Hindi.jsonl"
    assert r["response"] == "ans" and r["model_name"] == "m" and r["question_id"] == 0

    p = write_responses(tmp_path, "mmvet", "Hindi", [{"qid": "v1_0", "image": "v1_0.png", "prompt": "q", "raw": {}}],
                        {"v1_0": "4"}, "m")
    assert p.name == "mm-vet_Hindi.json" and json.loads(p.read_text()) == {"v1_0": "4"}
    assert response_path(tmp_path, "mmvet", "Odia").parent.name == "model_responses"


def test_image_index(tmp_path):
    (tmp_path / "pope").mkdir()
    (tmp_path / "pope/COCO_val2014_000000310196.jpg").write_bytes(b"x")
    (tmp_path / "v1_0.png").write_bytes(b"x")
    idx = ImageIndex([tmp_path])
    assert idx.find("COCO_val2014_000000310196.jpg").endswith("COCO_val2014_000000310196.jpg")
    assert idx.find("v1_0.jpg").endswith("v1_0.png")  # stem fallback
    assert idx.find("missing.jpg") is None


@pytest.mark.skipif(not BB.exists(), reason="BharatBench clone not present")
@pytest.mark.parametrize("task", ["pope", "llavabench", "mmvet"])
def test_loaders_on_real_bharatbench(task):
    for lang in BHARATBENCH_REPORT_LANGUAGES:
        items = LOADERS[task](BB, lang)
        assert items and all(it["image"] and it["prompt"] for it in items)
    if task == "pope":
        assert len(LOADERS["pope"](BB, "Hindi")) == 8910
