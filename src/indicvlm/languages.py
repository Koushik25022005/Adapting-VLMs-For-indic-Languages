"""Language metadata shared by data prep, inference and reporting."""

from __future__ import annotations

# Canonical name -> ISO 639-1 / 639-3 codes and common aliases that may appear in
# the Chitrakshara-Cap language column.
LANGUAGES: dict[str, dict] = {
    "Hindi": {"codes": ["hi", "hin"]},
    "Bengali": {"codes": ["bn", "ben", "bangla"]},
    "Telugu": {"codes": ["te", "tel"]},
    "Tamil": {"codes": ["ta", "tam"]},
    "Marathi": {"codes": ["mr", "mar"]},
    "Gujarati": {"codes": ["gu", "guj"]},
    "Kannada": {"codes": ["kn", "kan"]},
    "Malayalam": {"codes": ["ml", "mal"]},
    "Odia": {"codes": ["or", "ori", "ory", "oriya"]},
    "Assamese": {"codes": ["as", "asm"]},
    "Sanskrit": {"codes": ["sa", "san"]},
    "English": {"codes": ["en", "eng"]},
}

# The 10 Indic languages + English that Chitrarth reports on BharatBench.
BHARATBENCH_REPORT_LANGUAGES = [
    "Telugu", "Hindi", "Bengali", "Malayalam", "Kannada", "Assamese",
    "Tamil", "Marathi", "Gujarati", "Odia", "English",
]

_ALIAS = {}
for _name, _meta in LANGUAGES.items():
    _ALIAS[_name.lower()] = _name
    for _c in _meta["codes"]:
        _ALIAS[_c.lower()] = _name


def canonical_language(value: str | None) -> str | None:
    """Map a language name/code (any case, e.g. 'hi', 'hin_Deva', 'Hindi') to a canonical name."""
    if not value:
        return None
    v = str(value).strip().lower().replace("-", "_")
    if v in _ALIAS:
        return _ALIAS[v]
    head = v.split("_")[0]  # e.g. 'hin_deva' -> 'hin'
    return _ALIAS.get(head)


# --------------------------------------------------------------------------- script-based language ID
# Chitrakshara rows carry no language label, so we infer it from the caption's Unicode script.
# Scripts shared by two languages are split with simple, high-precision cues:
#   Devanagari -> Hindi vs Marathi (function words, the Marathi letter ळ)
#   Bengali    -> Assamese (uses ৰ / ৱ) vs Bengali (uses র)
_SCRIPT_RANGES = [
    ("Devanagari", 0x0900, 0x097F),
    ("Bengali", 0x0980, 0x09FF),
    ("Gurmukhi", 0x0A00, 0x0A7F),
    ("Gujarati", 0x0A80, 0x0AFF),
    ("Odia", 0x0B00, 0x0B7F),
    ("Tamil", 0x0B80, 0x0BFF),
    ("Telugu", 0x0C00, 0x0C7F),
    ("Kannada", 0x0C80, 0x0CFF),
    ("Malayalam", 0x0D00, 0x0D7F),
]
_SCRIPT_TO_LANGUAGE = {"Gujarati": "Gujarati", "Odia": "Odia", "Tamil": "Tamil", "Telugu": "Telugu",
                       "Kannada": "Kannada", "Malayalam": "Malayalam"}
_HINDI_WORDS = {"है", "हैं", "और", "के", "की", "का", "में", "से", "को", "ने", "था", "थे", "थी",
                "यह", "वह", "भी", "पर", "नहीं", "लिए", "किया", "गया", "रहे", "एक"}
_MARATHI_WORDS = {"आहे", "आहेत", "आणि", "नाही", "मध्ये", "केले", "होते", "होता", "होती", "असे",
                  "झाले", "करण्यात", "त्यांनी", "हे", "व", "या", "तर", "म्हणून", "येथे", "एका"}
_PUNCT = str.maketrans({c: " " for c in "।॥,.!?;:\"'()[]{}-–—|/"})


def dominant_script(text: str, min_share: float = 0.6) -> str | None:
    """Return the script making up at least ``min_share`` of the letters in ``text``."""
    counts: dict[str, int] = {}
    total = 0
    for ch in text:
        o = ord(ch)
        if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            counts["Latin"] = counts.get("Latin", 0) + 1
            total += 1
            continue
        for name, lo, hi in _SCRIPT_RANGES:
            if lo <= o <= hi:
                counts[name] = counts.get(name, 0) + 1
                total += 1
                break
    if not total:
        return None
    script, n = max(counts.items(), key=lambda kv: kv[1])
    return script if n / total >= min_share else None


def detect_language(text: str | None) -> str | None:
    """Best-effort language ID for captions; returns a canonical name or None when unsure."""
    if not text:
        return None
    script = dominant_script(text)
    if script is None:
        return None
    if script == "Latin":
        return "English"
    if script in _SCRIPT_TO_LANGUAGE:
        return _SCRIPT_TO_LANGUAGE[script]
    if script == "Bengali":
        if "ৰ" in text or "ৱ" in text:  # ৰ ৱ: Assamese only
            return "Assamese"
        if "র" in text:  # র: Bengali only
            return "Bengali"
        return None
    if script == "Devanagari":
        words = text.translate(_PUNCT).split()
        hi = sum(w in _HINDI_WORDS for w in words)
        mr = sum(w in _MARATHI_WORDS for w in words) + 2 * text.count("ळ")
        if hi > mr:
            return "Hindi"
        if mr > hi:
            return "Marathi"
        return None
    return None  # e.g. Gurmukhi (Punjabi) is not a target language
