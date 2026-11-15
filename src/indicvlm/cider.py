"""CIDEr-D (Vedantam et al., 2015) in pure Python, for captions in any script.

The reference implementation (pycocoevalcap) needs Java for its English PTB tokenizer, which
does not suit Indic text. Here both references and predictions go through the same simple
tokenizer: lower-case, punctuation removed (including the danda), split on whitespace.
Scores are therefore comparable between models scored here, but not with published
numbers that used another tokenizer.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter

N = 4
SIGMA = 6.0


def tokenize(text: str) -> list[str]:
    out = []
    for ch in unicodedata.normalize("NFC", text or "").lower():
        cat = unicodedata.category(ch)
        # Keep letters, digits and combining marks (vowel signs, virama); drop punctuation/symbols.
        out.append(ch if cat[0] in "LNM" or ch in "‌‍" else " ")
    return "".join(out).split()


def _ngrams(tokens: list[str]) -> Counter:
    c = Counter()
    for n in range(1, N + 1):
        for i in range(len(tokens) - n + 1):
            c[tuple(tokens[i:i + n])] += 1
    return c


def _vec(counts: Counter, df: Counter, log_docs: float):
    vec = [dict() for _ in range(N)]
    norm = [0.0] * N
    for ng, tf in counts.items():
        n = len(ng) - 1
        w = tf * (log_docs - math.log(max(1.0, df[ng])))
        vec[n][ng] = w
        norm[n] += w * w
    return vec, [math.sqrt(x) for x in norm]


def _sim(hv, rv, hn, rn, len_h: int, len_r: int) -> list[float]:
    penalty = math.exp(-((len_h - len_r) ** 2) / (2 * SIGMA ** 2))
    vals = []
    for n in range(N):
        v = sum(min(w, rv[n].get(ng, 0.0)) * rv[n].get(ng, 0.0) for ng, w in hv[n].items())
        if hn[n] and rn[n]:
            v /= hn[n] * rn[n]
        vals.append(v * penalty)
    return vals


def cider_d(references: list[list[str]], predictions: list[str]) -> tuple[float, list[float]]:
    """``references[i]`` are the reference captions of image ``i``; ``predictions[i]`` its
    predicted caption. Returns (corpus score, per-image scores), scaled as papers report it:
    100 x the pycocoevalcap value, so a prediction equal to its only reference scores 1000."""
    if len(references) != len(predictions):
        raise ValueError("references and predictions differ in length")
    ref_tok = [[tokenize(r) for r in refs if r and r.strip()] for refs in references]
    hyp_tok = [tokenize(p) for p in predictions]
    df = Counter()
    for refs in ref_tok:
        seen = set()
        for r in refs:
            seen.update(_ngrams(r))
        df.update(seen)
    log_docs = math.log(max(1, len(ref_tok)))

    scores = []
    for refs, hyp in zip(ref_tok, hyp_tok):
        if not refs:
            scores.append(0.0)
            continue
        hv, hn = _vec(_ngrams(hyp), df, log_docs)
        total = [0.0] * N
        for r in refs:
            rv, rn = _vec(_ngrams(r), df, log_docs)
            for n, v in enumerate(_sim(hv, rv, hn, rn, len(hyp), len(r))):
                total[n] += v
        scores.append(10.0 * sum(total) / N / len(refs) * 100.0)
    return (sum(scores) / len(scores) if scores else 0.0), scores
