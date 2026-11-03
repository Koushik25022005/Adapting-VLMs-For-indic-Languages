"""Generate BharatBench (Indic-POPE / Indic-LLaVA-Bench / Indic-MMVet) responses.

Writes model responses in exactly the layout the official BharatBench evaluators
(https://github.com/ola-krutrim/BharatBench) read:

  Indic-POPE-Evaluator/model_responses/{language}_results.jsonl        (lower-case language)
  Indic-LLaVABench-Evaluator/model_responses/responses_{Language}.jsonl
  Indic-MMVet-Evaluator/model_responses/mm-vet_{Language}.json

Example (fine-tuned model):
    python -m indicvlm.infer_bharatbench --bharatbench third_party/BharatBench \
        --images data/bharatbench_images --adapter outputs/qwen25vl7b-chitrakshara-qlora/final_adapter \
        --out results/responses/qwen25vl7b-qlora
Pass no ``--adapter`` to evaluate the zero-shot base model.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from .languages import BHARATBENCH_REPORT_LANGUAGES

TASKS = ("pope", "llavabench", "mmvet")
MAX_NEW_TOKENS = {"pope": 16, "llavabench": 1024, "mmvet": 512}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


# --------------------------------------------------------------------------- inputs
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def pope_questions(bb: Path, lang: str):
    rows = read_jsonl(bb / "Indic-POPE-Evaluator/question_files" / f"pope_test_{lang.lower()}.jsonl")
    return [{"qid": r["question_id"], "image": r["image"], "prompt": r["text"], "raw": r} for r in rows]


def llavabench_questions(bb: Path, lang: str):
    rows = read_jsonl(bb / "Indic-LLaVABench-Evaluator/inputs" / lang / "questions.jsonl")
    return [{"qid": r["question_id"], "image": r["image"], "prompt": r["text"], "raw": r} for r in rows]


def mmvet_questions(bb: Path, lang: str):
    with open(bb / "Indic-MMVet-Evaluator/mm-vet_qa" / f"mm-vet_{lang}.json", encoding="utf-8") as f:
        d = json.load(f)
    return [{"qid": k, "image": v["imagename"], "prompt": v["question"], "raw": v} for k, v in d.items()]


LOADERS = {"pope": pope_questions, "llavabench": llavabench_questions, "mmvet": mmvet_questions}


def response_path(out: Path, task: str, lang: str) -> Path:
    if task == "pope":
        return out / "Indic-POPE-Evaluator/model_responses" / f"{lang.lower()}_results.jsonl"
    if task == "llavabench":
        return out / "Indic-LLaVABench-Evaluator/model_responses" / f"responses_{lang}.jsonl"
    return out / "Indic-MMVet-Evaluator/model_responses" / f"mm-vet_{lang}.json"


def write_responses(out: Path, task: str, lang: str, items, answers: dict, model_name: str) -> Path:
    path = response_path(out, task, lang)
    path.parent.mkdir(parents=True, exist_ok=True)
    if task == "mmvet":
        path.write_text(json.dumps({it["qid"]: answers[it["qid"]] for it in items}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        return path
    with open(path, "w", encoding="utf-8") as f:
        for it in items:  # keep question-file order: the POPE scorer aligns labels by position
            a = answers[it["qid"]]
            if task == "pope":
                rec = {"question_id": it["qid"], "prompt": it["prompt"], "text": a,
                       "answer_id": uuid.uuid4().hex[:22], "model_id": model_name, "metadata": {}}
            else:
                rec = {**it["raw"], "response": a, "model_name": model_name}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


class ImageIndex:
    """Resolve benchmark image names (e.g. 'COCO_val2014_000000310196.jpg', 'v1_0.png', '001.jpg')
    by basename/stem anywhere under one or more image roots."""

    def __init__(self, roots):
        self.by_name, self.by_stem = {}, {}
        for root in roots:
            for dirpath, _, files in os.walk(root):
                for fn in files:
                    if Path(fn).suffix.lower() in IMAGE_EXTS:
                        full = os.path.join(dirpath, fn)
                        self.by_name.setdefault(fn, full)
                        self.by_stem.setdefault(Path(fn).stem, full)

    def find(self, name: str):
        return self.by_name.get(name) or self.by_stem.get(Path(name).stem)


# --------------------------------------------------------------------------- model
class QwenVL:
    def __init__(self, base: str, adapter: str | None, load_4bit: bool, max_pixels: int):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        kw = {"torch_dtype": dtype, "device_map": "auto"}
        if load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=["visual", "lm_head"])
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, **kw)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(base, min_pixels=256 * 28 * 28, max_pixels=max_pixels)
        self.processor.tokenizer.padding_side = "left"

    def generate(self, batch, max_new_tokens: int):
        import torch
        from PIL import Image

        texts, images = [], []
        for img_path, prompt in batch:
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            texts.append(self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            images.append(Image.open(img_path).convert("RGB"))
        enc = self.processor(text=texts, images=images, padding=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      repetition_penalty=1.05)
        gen = out[:, enc["input_ids"].shape[1]:]
        return [t.strip() for t in self.processor.batch_decode(gen, skip_special_tokens=True)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bharatbench", default="third_party/BharatBench", help="clone of ola-krutrim/BharatBench")
    p.add_argument("--images", nargs="+", default=["data/bharatbench_images"],
                   help="dir(s) holding COCO val2014 (POPE), LLaVA-Bench and MM-Vet images")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--adapter", default=None, help="PEFT LoRA adapter dir; omit for the base model")
    p.add_argument("--model-name", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    p.add_argument("--languages", nargs="+", default=BHARATBENCH_REPORT_LANGUAGES)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--load-4bit", action="store_true", help="evaluate with the same NF4 base used in training")
    p.add_argument("--limit", type=int, default=None, help="debug: only the first N questions per file")
    args = p.parse_args(argv)

    from tqdm import tqdm

    bb, out = Path(args.bharatbench), Path(args.out)
    name = args.model_name or ("qwen25vl7b-chitrakshara-qlora" if args.adapter else "qwen25vl7b-instruct")
    index = ImageIndex(args.images)
    model = QwenVL(args.base_model, args.adapter, args.load_4bit, args.max_pixels)

    for task in args.tasks:
        for lang in args.languages:
            try:
                items = LOADERS[task](bb, lang)
            except FileNotFoundError:
                print(f"[skip] {task}/{lang}: no question file")
                continue
            items = items[: args.limit] if args.limit else items
            missing = [it["image"] for it in items if index.find(it["image"]) is None]
            if missing:
                raise SystemExit(f"{task}/{lang}: {len(missing)} images not found, e.g. {missing[:3]}. "
                                 "Run scripts/download_bharatbench_images.py first.")
            answers = {}
            for i in tqdm(range(0, len(items), args.batch_size), desc=f"{task}/{lang}"):
                chunk = items[i:i + args.batch_size]
                preds = model.generate([(index.find(it["image"]), it["prompt"]) for it in chunk],
                                       MAX_NEW_TOKENS[task])
                answers.update({it["qid"]: a for it, a in zip(chunk, preds)})
            print("wrote", write_responses(out, task, lang, items, answers, name))


if __name__ == "__main__":
    main()
