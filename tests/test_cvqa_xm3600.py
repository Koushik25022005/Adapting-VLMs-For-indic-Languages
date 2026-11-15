import json

import pytest

from indicvlm.cider import cider_d, tokenize
from indicvlm.eval_cvqa import build_prompt, has_label, parse_answer, parse_subset, select_rows
from indicvlm.eval_cvqa import score as cvqa_score
from indicvlm.eval_xm3600 import references, to_image
from indicvlm.eval_xm3600 import score as xm_score
from indicvlm.update_report import COMPARISONS, markers, render_comparison, update_report


def test_parse_subset():
    assert parse_subset("('Hindi', 'India')") == ("Hindi", "India")
    assert parse_subset('("Sinhala", "Sri Lanka")') == ("Sinhala", "Sri Lanka")
    assert parse_subset(("Tamil", "India")) == ("Tamil", "India")
    assert parse_subset("Urdu-Pakistan") == ("Urdu", "Pakistan")
    assert parse_subset("Hindi") == ("Hindi", "")


def test_build_prompt():
    p = build_prompt("यह कौन सा फूल है?", ["गुलाब", "कमल", "गेंदा", "चमेली"])
    assert p.splitlines()[0] == "यह कौन सा फूल है?"
    assert p.splitlines()[1:5] == ["A. गुलाब", "B. कमल", "C. गेंदा", "D. चमेली"]
    assert p.endswith("only.")


@pytest.mark.parametrize("response,expected", [
    ("B", 1), ("b", 1), ("(C)", 2), ("D.", 3), ("A. Rose", 0), ("B: कमल", 1), ("**C**", 2),
    ("The answer is D", 3), ("Option b", 1), ("उत्तर: C", 2),
    ("Lotus", 1),                         # option text, no letter
    ("यह एक कमल है", None),               # no option named ("कमल" != any English option)
    ("A man is standing near a lake", None),  # the article "A" is not an answer
    ("Rose or Lotus", None),              # ambiguous
    ("", None), (None, None),
])
def test_parse_answer(response, expected):
    assert parse_answer(response, ["Rose", "Lotus", "Marigold", "Jasmine"]) == expected


def test_cvqa_score():
    recs = [
        {"language": "Hindi", "variant": "local", "label": 1, "pred": 1},
        {"language": "Hindi", "variant": "local", "label": 2, "pred": None},
        {"language": "Hindi", "variant": "english", "label": 1, "pred": 1},
        {"language": "Hindi", "variant": "english", "label": 2, "pred": 2},
        {"language": "Tamil", "variant": "local", "label": 0, "pred": 3},
    ]
    s = cvqa_score(recs)
    assert s["Hindi"] == {"n": 2, "Accuracy (local)": 50.0, "Unparsed % (local)": 50.0,
                          "Accuracy (English)": 100.0, "Unparsed % (english)": 0.0}
    assert s["Tamil"]["Accuracy (local)"] == 0.0 and "Accuracy (English)" not in s["Tamil"]


def test_cvqa_score_without_labels():
    assert not has_label(-1) and not has_label(None) and not has_label(True) and has_label(3)
    s = cvqa_score([{"language": "Hindi", "variant": "local", "label": -1, "pred": 1}])
    assert s["Hindi"] == {"n": 1, "Unparsed % (local)": 0.0}


def test_select_rows():
    ds = {"Subset": ["('Hindi', 'India')", "('Spanish', 'Spain')", "('Tamil', 'India')",
                     "('Hindi', 'India')", "('Urdu', 'Pakistan')"]}
    assert select_rows(ds, ["India"], None, None) == [0, 2, 3]
    assert select_rows(ds, ["india", "pakistan"], None, 1) == [0, 2, 4]
    assert select_rows(ds, None, ["hindi"], None) == [0, 3]
    assert select_rows(ds, [], None, None) == [0, 1, 2, 3, 4]  # no countries -> keep all


def test_tokenize_keeps_indic_marks():
    assert tokenize("एक बिल्ली, सोफ़े पर बैठी है।") == ["एक", "बिल्ली", "सोफ़े", "पर", "बैठी", "है"]
    assert tokenize("A Cat! (on a sofa)") == ["a", "cat", "on", "a", "sofa"]
    assert tokenize("") == [] and tokenize(None) == []


def test_cider():
    refs = [["एक बिल्ली सोफ़े पर बैठी है", "सोफ़े पर एक बिल्ली"],
            ["दो लोग समुद्र तट पर चल रहे हैं"],
            ["लाल रंग की एक पुरानी कार सड़क पर खड़ी है"]]
    exact = [r[0] for r in refs]
    shuffled = [exact[1], exact[2], exact[0]]
    good, per = cider_d(refs, exact)
    bad, _ = cider_d(refs, shuffled)
    assert good > 500 and bad < 20 and len(per) == 3
    # punctuation and case do not matter; an empty prediction scores 0
    assert cider_d(refs, [e + "।" for e in exact])[0] == pytest.approx(good)
    assert cider_d(refs, ["", "", ""])[0] == 0.0
    # partial overlap lands in between
    part, _ = cider_d(refs, ["एक बिल्ली", "दो लोग चल रहे हैं", "एक पुरानी कार"])
    assert bad < part < good
    with pytest.raises(ValueError):
        cider_d(refs, ["x"])


def test_xm3600_helpers(tmp_path):
    from PIL import Image

    assert references({"captions": ["a", " ", None, "b"]}) == ["a", "b"]
    assert references({"captions": "one"}) == ["one"] and references({}) == []
    path = tmp_path / "i.png"
    Image.new("RGB", (4, 4)).save(path)
    img = Image.open(path)
    assert to_image(img) is img
    assert to_image({"bytes": path.read_bytes(), "path": None}).size == (4, 4)
    assert to_image(path.read_bytes()).size == (4, 4)
    assert to_image(str(path)).size == (4, 4)


def test_xm3600_score():
    recs = [
        {"language": "Hindi", "references": ["एक बिल्ली सोफ़े पर बैठी है"], "prediction": "एक बिल्ली सोफ़े पर बैठी है"},
        {"language": "Hindi", "references": ["दो लोग समुद्र तट पर चल रहे हैं"], "prediction": "Two people walking"},
        {"language": "Telugu", "references": ["ఒక పిల్లి"], "prediction": "ఒక పిల్లి"},
    ]
    s = xm_score(recs)
    assert list(s) == ["Hindi", "Telugu"]
    assert s["Hindi"]["n"] == 2 and s["Hindi"]["In-language %"] == 50.0 and s["Hindi"]["CIDEr"] > 0
    assert s["Hindi"]["Avg words"] == 4.5
    assert s["Telugu"]["In-language %"] == 100.0


def test_render_comparison_and_report(tmp_path):
    base = {"scores": {"Hindi": {"Accuracy (local)": 60.0, "Accuracy (English)": 70.0, "n": 200},
                       "Tamil": {"Accuracy (local)": 50.0, "n": 210}}}
    ours = {"scores": {"Hindi": {"Accuracy (local)": 62.5, "Accuracy (English)": 65.0, "n": 200},
                       "Tamil": {"Accuracy (local)": 45.0, "n": 210}}}
    t = render_comparison(base, ours, COMPARISONS["CVQA"])
    assert "| Hindi | 200 | 60.00 | 62.50 | +2.50 |" in t
    assert "| Tamil | 210 | 50.00 | 45.00 | -5.00 |" in t
    assert "| **Indic avg (2)** | — | 55.00 | 53.75 | -1.25 |" in t
    english = t.split("#### Accuracy (English)")[1]
    assert "| Tamil | 210 | — | — | — |" in english
    assert "| **Indic avg (2)** | — | — | — | — |" in english  # incomplete rows are not averaged
    assert "Unparsed" not in t  # metric absent everywhere -> no table
    # only one model scored so far
    t1 = render_comparison(None, ours, COMPARISONS["CVQA"])
    assert "| Hindi | 200 | — | 62.50 | — |" in t1
    assert render_comparison(None, None, COMPARISONS["XM3600"]) == "_No scores yet._"

    b, e = markers("CVQA")
    rep = tmp_path / "r.md"
    rep.write_text(f"# R\n{b}\nold\n{e}\nkeep\n<!-- BEGIN:XM3600_RESULTS -->\nx\n<!-- END:XM3600_RESULTS -->\n")
    update_report(rep, t, b, e)
    txt = rep.read_text()
    assert "old" not in txt and "keep" in txt and "\nx\n" in txt and "+2.50" in txt


def test_update_report_cli(tmp_path):
    from indicvlm.update_report import main

    rep = tmp_path / "r.md"
    rep.write_text("".join(f"{b}\nold\n{e}\n" for b, e in map(markers, ["BHARATBENCH", "CVQA", "XM3600"])))
    xm = tmp_path / "xm.json"
    xm.write_text(json.dumps({"scores": {"Hindi": {"CIDEr": 31.2, "In-language %": 99.0, "Avg words": 9.1, "n": 3600}}}))
    main(["--report", str(rep), "--xm3600-ours", str(xm)])
    txt = rep.read_text()
    assert "| Hindi | 3600 | — | 31.20 | — |" in txt and "#### In-language %" in txt
    assert txt.count("old") == 2  # BharatBench and CVQA sections untouched


# --------------------------------------------------------------------------- end to end, fake model
class FakeModel:
    """Stands in for QwenVL: answers "B" to multiple-choice prompts, else a fixed Hindi caption."""

    def __init__(self, *a, **k):
        pass

    def generate(self, batch, max_new_tokens):
        from PIL import Image

        assert all(isinstance(img, Image.Image) for img, _ in batch)
        return ["B" if "A. " in prompt else "एक बिल्ली सोफ़े पर बैठी है" for _, prompt in batch]


def _patch(monkeypatch, splits):
    import datasets

    import indicvlm.infer_bharatbench as ib

    monkeypatch.setattr(ib, "QwenVL", FakeModel)
    monkeypatch.setattr(datasets, "load_dataset", lambda name, split=None, **k: splits[split])


def test_eval_cvqa_end_to_end(tmp_path, monkeypatch):
    import datasets
    from PIL import Image

    from indicvlm import eval_cvqa

    img = Image.new("RGB", (32, 32), "red")
    rows = [("('Hindi', 'India')", 1), ("('Hindi', 'India')", 0), ("('Tamil', 'India')", 1),
            ("('Spanish', 'Spain')", 1)]
    ds = datasets.Dataset.from_dict({
        "image": [img] * 4, "ID": [f"q{i}" for i in range(4)], "Subset": [s for s, _ in rows],
        "Question": ["प्रश्न?"] * 4, "Translated Question": ["Question?"] * 4,
        "Options": [["क", "ख", "ग", "घ"]] * 4, "Translated Options": [["a", "b", "c", "d"]] * 4,
        "Label": [l for _, l in rows]})
    _patch(monkeypatch, {"test": ds})
    eval_cvqa.main(["--out-dir", str(tmp_path), "--run-name", "m", "--batch-size", "2"])
    res = json.loads((tmp_path / "m/cvqa_scores.json").read_text())
    assert res["scores"] == {
        "Hindi": {"n": 2, "Accuracy (local)": 50.0, "Unparsed % (local)": 0.0,
                  "Accuracy (English)": 50.0, "Unparsed % (english)": 0.0},
        "Tamil": {"n": 1, "Accuracy (local)": 100.0, "Unparsed % (local)": 0.0,
                  "Accuracy (English)": 100.0, "Unparsed % (english)": 0.0}}
    preds = [json.loads(l) for l in (tmp_path / "m/cvqa_predictions.jsonl").read_text().splitlines()]
    assert len(preds) == 6 and {p["country"] for p in preds} == {"India"}  # Spain filtered out
    with pytest.raises(SystemExit):
        eval_cvqa.main(["--out-dir", str(tmp_path), "--countries", "Atlantis"])


def test_eval_xm3600_end_to_end(tmp_path, monkeypatch):
    import datasets
    from PIL import Image

    from indicvlm import eval_xm3600

    path = tmp_path / "i.png"
    Image.new("RGB", (32, 32), "blue").save(path)
    raw = {"bytes": path.read_bytes(), "path": None}  # floschne/xm3600 stores undecoded bytes
    hi = datasets.Dataset.from_dict({
        "image_id": ["a", "b", "c"], "image": [raw] * 3,
        "captions": [["एक बिल्ली सोफ़े पर बैठी है"], ["दो लोग चल रहे हैं", "लोग"], ["एक लाल कार"]]})
    decoded = datasets.Dataset.from_dict({"image_id": ["a"], "image": [Image.open(path)],
                                          "captions": [["a cat"]]})  # a copy with a real Image feature
    _patch(monkeypatch, {"hi": hi, "en": decoded})
    eval_xm3600.main(["--out-dir", str(tmp_path), "--run-name", "m", "--languages", "hi", "en",
                      "--batch-size", "2"])
    res = json.loads((tmp_path / "m/xm3600_scores.json").read_text())
    assert res["scores"]["Hindi"]["n"] == 3 and res["scores"]["Hindi"]["In-language %"] == 100.0
    assert res["scores"]["Hindi"]["CIDEr"] > 0
    assert res["scores"]["English"] == {"CIDEr": 0.0, "In-language %": 0.0, "Avg words": 6.0, "n": 1}
