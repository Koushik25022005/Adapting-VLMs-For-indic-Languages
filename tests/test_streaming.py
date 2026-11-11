import pickle

import pytest

datasets = pytest.importorskip("datasets")
from PIL import Image  # noqa: E402

from indicvlm import prepare_data as pdm  # noqa: E402
from indicvlm import streaming_data as sd  # noqa: E402
from indicvlm.data import mask_labels  # noqa: E402

HI = "मेला समिति ने इस बार गरबा स्थल पर फूड जोन बनाया है।"
TA = "சென்னையில் இன்று கனமழை பெய்தது."
PA = "ਅੰਮ੍ਰਿਤਸਰ ਵਿੱਚ ਅੱਜ ਮੀਂਹ ਪਿਆ"  # Punjabi: dropped


def _rows():
    return [{"image": "https://x/1.jpg", "text": HI},
            {"image": "https://x/dead.jpg", "text": HI},     # download fails -> dropped
            {"image": "https://x/2.jpg", "text": TA},
            {"image": "https://x/3.jpg", "text": PA},
            {"image": "https://x/4.jpg", "text": "ab"},      # caption too short
            {"image": "https://x/5.jpg", "text": HI + " फिर"},
            {"image": "https://x/6.jpg", "text": HI + " तीसरा"}]


@pytest.fixture
def fake_hub(monkeypatch):
    monkeypatch.setattr(pdm, "load_image",
                        lambda v, timeout=10: None if "dead" in str(v) else Image.new("RGB", (128, 96)))
    monkeypatch.setattr(datasets, "get_dataset_config_names", lambda n: ["Chitrakshara-Cap", "Chitrakshara-IL"],
                        raising=False)
    calls = {}

    def load(name, config, split, streaming):
        calls.update(name=name, config=config, streaming=streaming)
        return datasets.Dataset.from_list(_rows()).to_iterable_dataset()
    return load, calls


def test_stream_formats_downloads_and_filters(fake_hub):
    load, calls = fake_hub
    ds = sd.stream_chitrakshara(config="chitrakshara-cap", shuffle_buffer=0, load_dataset_fn=load,
                                download_workers=2, download_batch=3)
    out = list(ds)
    assert calls == {"name": "krutrim-ai-labs/chitrakshara", "config": "Chitrakshara-Cap", "streaming": True}
    assert [e["language"] for e in out] == ["Hindi", "Tamil", "Hindi", "Hindi"]
    assert all(isinstance(e["image"], Image.Image) and e["image"].size == (128, 96) for e in out)
    assert set(out[0]) == {"image", "caption", "language"}


def test_stream_per_language_cap_and_languages(fake_hub):
    load, _ = fake_hub
    ds = sd.stream_chitrakshara(shuffle_buffer=0, load_dataset_fn=load, max_per_language=1)
    assert sorted(e["language"] for e in ds) == ["Hindi", "Tamil"]
    ds = sd.stream_chitrakshara(shuffle_buffer=0, load_dataset_fn=load, languages=["Tamil"])
    assert [e["language"] for e in ds] == ["Tamil"]


def test_holdout_split_is_disjoint_and_complete(fake_hub):
    load, _ = fake_hub
    full = {e["caption"] for e in sd.stream_chitrakshara(shuffle_buffer=0, load_dataset_fn=load)}
    held = {e["caption"] for e in sd.stream_chitrakshara(shuffle_buffer=0, load_dataset_fn=load,
                                                           holdout_fraction=0.5, part="eval")}
    train = {e["caption"] for e in sd.stream_chitrakshara(shuffle_buffer=3, load_dataset_fn=load,
                                                            holdout_fraction=0.5, part="train")}
    assert not held & train and held | train == full


def test_in_holdout_fraction():
    n = sum(sd.in_holdout(f"https://x/{i}.jpg", "cap", 0.1) for i in range(20000))
    assert 1600 < n < 2400
    assert sd.in_holdout("u", "c", 0.1) == sd.in_holdout("u", "c", 0.1)
    assert not sd.in_holdout("u", "c", 0.0)


def test_stream_shuffle_and_picklable(fake_hub):
    load, _ = fake_hub
    ds = sd.stream_chitrakshara(shuffle_buffer=4, seed=1, load_dataset_fn=load)
    assert sorted(e["language"] for e in ds) == ["Hindi", "Hindi", "Hindi", "Tamil"]
    pickle.dumps(sd._PerLanguageCap(3))  # sent to DataLoader workers


class _Tok:
    """Tiny char-level stand-in for the Qwen tokenizer, enough to exercise the collator."""
    pad_token_id = 0

    def encode(self, s, add_special_tokens=False):
        if s == "<|im_start|>assistant\n":
            return [7, 8, 9]
        return [100 + (ord(c) % 50) for c in s]

    def decode(self, ids, skip_special_tokens=True):
        return "x" * len(ids)

    def convert_tokens_to_ids(self, t):
        return 2


class _Proc:
    tokenizer = _Tok()

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        return msgs[0]["content"][1]["text"] + "|" + msgs[1]["content"][0]["text"]

    def __call__(self, text, images, padding, return_tensors):
        import torch
        seqs = []
        for t in text:
            prompt, ans = t.split("|")
            seqs.append([1] * 4 + [5] * 3 + [7, 8, 9] + self.tokenizer.encode(ans) + [2])
        L = max(map(len, seqs))
        ids = [s + [0] * (L - len(s)) for s in seqs]
        att = [[1] * len(s) + [0] * (L - len(s)) for s in seqs]
        assert all(isinstance(im, Image.Image) for im in images)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(att)}


def test_collator_accepts_streamed_pil_images(fake_hub):
    from indicvlm.data import IGNORE_INDEX, QwenVLCaptionCollator
    load, _ = fake_hub
    batch = list(sd.stream_chitrakshara(shuffle_buffer=0, load_dataset_fn=load))[:2]
    enc = QwenVLCaptionCollator(_Proc())(batch)
    labels = enc["labels"].tolist()
    for row, ids, att in zip(labels, enc["input_ids"].tolist(), enc["attention_mask"].tolist()):
        expected = [l if a else IGNORE_INDEX for l, a in zip(mask_labels(ids, [7, 8, 9], 2), att)]
        assert row == expected
        assert row[:10] == [IGNORE_INDEX] * 10  # prompt + image + assistant header are masked
        assert any(l != IGNORE_INDEX for l in row)  # the caption is supervised
