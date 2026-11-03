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
