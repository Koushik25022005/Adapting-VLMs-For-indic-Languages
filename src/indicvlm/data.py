"""Dataset + collator turning Chitrakshara-Cap captions into Qwen2.5-VL chat SFT examples."""

from __future__ import annotations

import json
import random
from pathlib import Path

IGNORE_INDEX = -100

# Captioning instructions. The target-language name is inserted so the model learns to
# produce captions in the requested Indic language (the captions themselves are in that
# language). Keep these in English so every language shares the same instruction set.
CAPTION_TEMPLATES = [
    "Describe this image in {language}.",
    "Write a caption for this image in {language}.",
    "What is shown in this picture? Answer in {language}.",
    "Give a short description of the image in {language}.",
    "Provide a {language} caption for this image.",
]


def build_messages(caption: str, language: str, rng: random.Random | None = None,
                   templates=CAPTION_TEMPLATES) -> list[dict]:
    rng = rng or random
    instruction = rng.choice(templates).format(language=language)
    return [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]},
        {"role": "assistant", "content": [{"type": "text", "text": caption}]},
    ]


def find_subsequence(seq: list[int], sub: list[int], start: int = 0) -> int:
    n = len(sub)
    for i in range(start, len(seq) - n + 1):
        if seq[i:i + n] == sub:
            return i
    return -1


def mask_labels(input_ids: list[int], assistant_header: list[int], end_token: int) -> list[int]:
    """Supervise only assistant turns: tokens after ``<|im_start|>assistant\\n`` up to and
    including ``<|im_end|>``. Everything else (system, user, image tokens, padding) is ignored."""
    labels = [IGNORE_INDEX] * len(input_ids)
    pos = 0
    while True:
        h = find_subsequence(input_ids, assistant_header, pos)
        if h < 0:
            break
        s = h + len(assistant_header)
        e = s
        while e < len(input_ids) and input_ids[e] != end_token:
            e += 1
        stop = min(e + 1, len(input_ids))  # include <|im_end|> so the model learns to stop
        labels[s:stop] = input_ids[s:stop]
        pos = stop
    return labels


class CaptionDataset:
    """Reads a manifest written by ``prepare_data.py``."""

    def __init__(self, manifest: str | Path, max_samples: int | None = None):
        self.root = Path(manifest).parent
        with open(manifest, encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        if max_samples:
            self.rows = self.rows[:max_samples]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = r["image"]
        path = img if Path(img).is_absolute() else str(self.root / img)
        return {"image": path, "caption": r["caption"], "language": r["language"]}


class QwenVLCaptionCollator:
    """Batches samples for ``Qwen2_5_VLForConditionalGeneration`` with assistant-only loss."""

    def __init__(self, processor, max_caption_tokens: int = 256, seed: int = 0):
        from PIL import Image  # noqa: F401  (fail early if pillow missing)

        self.processor = processor
        self.max_caption_tokens = max_caption_tokens
        self.rng = random.Random(seed)
        tok = processor.tokenizer
        self.assistant_header = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.end_token = tok.convert_tokens_to_ids("<|im_end|>")

    def _truncate(self, caption: str) -> str:
        # Truncate the caption (never the prompt) so image placeholder tokens are never cut.
        tok = self.processor.tokenizer
        ids = tok.encode(caption, add_special_tokens=False)
        if len(ids) <= self.max_caption_tokens:
            return caption
        return tok.decode(ids[: self.max_caption_tokens], skip_special_tokens=True)

    def __call__(self, batch):
        import torch
        from PIL import Image

        texts, images = [], []
        for ex in batch:
            msgs = build_messages(self._truncate(ex["caption"]), ex["language"], self.rng)
            texts.append(self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
            images.append(Image.open(ex["image"]).convert("RGB"))
        enc = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = [mask_labels(ids.tolist(), self.assistant_header, self.end_token)
                  for ids in enc["input_ids"]]
        labels = torch.tensor(labels, dtype=torch.long)
        labels[enc["attention_mask"] == 0] = IGNORE_INDEX
        enc["labels"] = labels
        return enc
