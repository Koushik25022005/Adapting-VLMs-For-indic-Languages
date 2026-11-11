import json

import pytest

from indicvlm import prepare_data as pdm
from indicvlm.languages import detect_language

# The row shape reported from Chitrakshara (interleaved web document).
HINDI_ROW = {
    "images": [None, "https://ananttvlive.com/static/c1e/client/82638/uploaded/bd327ae26fd0ec3bed3b6ed250c343d1.jpg"
                     "?width=789&height=605&resizemode=4", None],
    "metadata": '[null, { "alt_text": "sdf", "document_url": "https://ananttvlive.com/Country/x.htm", "format": "jpeg",'
                ' "original_height": 605, "original_width": 789, "rendered_height": 153, "rendered_width": 200,'
                ' "src": "https://ananttvlive.com/a.jpg" }, null]',
    "general_metadata": '{"url": "https://ananttvlive.com/Country/x.html"}',
    "texts": [
        "मेला समिति ने इस बार गरबा स्थल पर फूड जोन बनाया है। यहां बड़ी संख्या में युवाओं ने चटपटे व्यंजनों का लुत्फ उठाया।",
        None,
        "गुजराती गरबा धुन तारा बिना श्याम मने एकालडू चलो के साथ युवाओं के हुजूम ने गुजराती लय में गरबा किया तो सिंधी गीत"
        " मुंहिंजो खटी आयो खैर सां होजमालो की धुन पर सिंधी स्टाइल में डांडिया करते हुए युवाओं को हुजूम झूम उठा।",
    ],
}

SAMPLES = {
    "Hindi": HINDI_ROW["texts"][0],
    "Marathi": "मुंबईत आज मोठा पाऊस झाला आणि अनेक रस्ते पाण्याखाली गेले आहेत.",
    "Assamese": "অসমৰ মুখ্যমন্ত্ৰীয়ে আজি নতুন আঁচনি ঘোষণা কৰে।",
    "Bengali": "কলকাতায় আজ প্রবল বৃষ্টি হয়েছে।",
    "Tamil": "சென்னையில் இன்று கனமழை பெய்தது.",
    "Telugu": "హైదరాబాద్‌లో ఈరోజు భారీ వర్షం కురిసింది.",
    "Kannada": "ಬೆಂಗಳೂರಿನಲ್ಲಿ ಇಂದು ಭಾರಿ ಮಳೆಯಾಗಿದೆ.",
    "Malayalam": "കൊച്ചിയിൽ ഇന്ന് കനത്ത മഴ പെയ്തു.",
    "Gujarati": "અમદાવાદમાં આજે ભારે વરસાદ પડ્યો.",
    "Odia": "ଭୁବନେଶ୍ୱରରେ ଆଜି ପ୍ରବଳ ବର୍ଷା ହେଲା।",
    "English": "Heavy rain lashed the city today.",
}


@pytest.mark.parametrize("lang,text", list(SAMPLES.items()))
def test_detect_language(lang, text):
    assert detect_language(text) == lang


def test_detect_language_unsure():
    assert detect_language("ਅੰਮ੍ਰਿਤਸਰ ਵਿੱਚ ਮੀਂਹ") is None  # Punjabi: not a target language
    assert detect_language("12345 !!") is None
    assert detect_language(None) is None
    assert detect_language("राम श्याम") is None  # Devanagari without Hindi/Marathi cues


def test_clip_caption_cuts_at_sentence_end():
    t = HINDI_ROW["texts"][0]
    first = t.split("।")[0] + "।"
    assert pdm.clip_caption(t, len(first) + 5) == first
    assert pdm.clip_caption("short text", 100) == "short text"
    assert len(pdm.clip_caption("word " * 100, 50)) <= 50


def test_interleaved_pairs_on_reported_row():
    pairs = list(pdm.interleaved_pairs(HINDI_ROW))
    assert len(pairs) == 1
    p = pairs[0]
    assert p["image"].startswith("https://ananttvlive.com/") and p["source"] == "before"
    assert p["caption"] == HINDI_ROW["texts"][0]
    # "after" first -> the following paragraph (clipped to max chars at a sentence boundary)
    p2 = next(pdm.interleaved_pairs(HINDI_ROW, sources=("after",), max_chars=500))
    assert p2["source"] == "after" and p2["caption"].startswith("गुजराती")
    # alt text "sdf" is junk -> no pair when it is the only source
    assert list(pdm.interleaved_pairs(HINDI_ROW, sources=("alt",))) == []


def test_interleaved_skips_known_tiny_images():
    row = dict(HINDI_ROW, metadata='[null, {"original_width": 32, "original_height": 32}, null]')
    assert list(pdm.interleaved_pairs(row)) == []


def _run(tmp_path, monkeypatch, rows, extra=()):
    from PIL import Image

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: rows)
    monkeypatch.setattr(datasets, "get_dataset_config_names", lambda name: ["Chitrakshara-Cap", "Chitrakshara-IL"])
    monkeypatch.setattr(pdm, "load_image",
                        lambda v, timeout=10: None if "dead" in str(v) else Image.new("RGB", (128, 96)))
    out = tmp_path / "out"
    pdm.main(["--out", str(out), "--per-language", "2", "--workers", "2", "--val-fraction", "0.2", *extra])
    rows_out = [json.loads(l) for f in ("train.jsonl", "val.jsonl") for l in (out / f).read_text().splitlines()]
    return out, rows_out, json.loads((out / "stats.json").read_text())


def test_main_interleaved_end_to_end(tmp_path, monkeypatch):
    def doc(url, text):
        return {"images": [url, None], "texts": [None, text], "metadata": "[{}, null]"}

    rows = [HINDI_ROW,
            doc("https://x/1.jpg", SAMPLES["Tamil"]),
            doc("https://x/dead.jpg", SAMPLES["Tamil"]),  # download fails
            doc("https://x/1.jpg", SAMPLES["Tamil"]),  # duplicate URL
            doc("https://x/2.jpg", SAMPLES["Marathi"]),
            doc("https://x/3.jpg", "ਅੰਮ੍ਰਿਤਸਰ ਵਿੱਚ ਅੱਜ ਮੀਂਹ ਪਿਆ"),  # Punjabi -> unknown
            doc("https://x/4.jpg", SAMPLES["Hindi"] + " दूसरा"),
            doc("https://x/5.jpg", SAMPLES["Hindi"] + " तीसरा")]  # Hindi quota (2) already full
    out, recs, stats = _run(tmp_path, monkeypatch, rows, ["--languages", "Hindi", "Tamil", "Marathi"])
    by_lang = {}
    for r in recs:
        by_lang[r["language"]] = by_lang.get(r["language"], 0) + 1
        assert (out / r["image"]).exists()
    assert by_lang == {"Hindi": 2, "Tamil": 1, "Marathi": 1}
    assert stats["format"] == "interleaved" and stats["language_from"] == "script detection"
    assert stats["counters"]["image_failed"] == 1
    assert stats["counters"]["duplicate_url"] == 1
    assert stats["counters"]["language_unknown"] == 1


def test_main_pairs_with_assumed_language(tmp_path, monkeypatch):
    rows = [{"text": SAMPLES["Hindi"], "image": "https://x/a.jpg"},
            {"text": "ab", "image": "https://x/b.jpg"},  # too short
            {"text": SAMPLES["Hindi"], "image": "https://x/c.jpg"}]
    _, recs, stats = _run(tmp_path, monkeypatch, rows, ["--languages", "Hindi", "--assume-language", "hi"])
    assert stats["format"] == "pairs" and len(recs) == 2
    assert {r["language"] for r in recs} == {"Hindi"}


def test_resolve_config_case_insensitive():
    lister = lambda name: ["Chitrakshara-Cap", "Chitrakshara-IL"]  # noqa: E731
    assert pdm.resolve_config("krutrim-ai-labs/chitrakshara", "chitrakshara-cap", lister) == "Chitrakshara-Cap"
    assert pdm.resolve_config("krutrim-ai-labs/chitrakshara", "CHITRAKSHARA-IL", lister) == "Chitrakshara-IL"
    assert pdm.resolve_config("x", None, lister) is None
    with pytest.raises(SystemExit):
        pdm.resolve_config("krutrim-ai-labs/chitrakshara", "nope", lister)

    def offline(name):
        raise ConnectionError
    assert pdm.resolve_config("x", "chitrakshara-cap", offline) == "chitrakshara-cap"


def test_defaults_point_at_chitrakshara_cap(tmp_path, monkeypatch):
    import datasets
    seen = {}

    def fake_load(name, config, **k):
        seen.update(name=name, config=config)
        return [{"text": SAMPLES["Tamil"], "image": "https://x/a.jpg"}]
    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    monkeypatch.setattr(datasets, "get_dataset_config_names", lambda name: ["Chitrakshara-Cap", "Chitrakshara-IL"])
    from PIL import Image
    monkeypatch.setattr(pdm, "load_image", lambda v, timeout=10: Image.new("RGB", (128, 96)))
    pdm.main(["--out", str(tmp_path / "o"), "--per-language", "1", "--languages", "Tamil"])
    assert seen == {"name": "krutrim-ai-labs/chitrakshara", "config": "Chitrakshara-Cap"}
