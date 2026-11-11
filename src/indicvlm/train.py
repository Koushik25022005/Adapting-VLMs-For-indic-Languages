"""QLoRA fine-tuning of Qwen2.5-VL-7B-Instruct on Chitrakshara-Cap.

    python -m indicvlm.train --config configs/qlora_qwen25vl_chitrakshara.yaml

* The language model is loaded in 4-bit NF4 with double quantisation (bitsandbytes);
  the SigLIP-style vision tower and the merger are kept in bf16 and frozen.
* LoRA adapters (PEFT) are attached to all attention + MLP projections of the LLM only.
* Loss is computed on assistant (caption) tokens only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Regex for PEFT ``target_modules``: every linear projection of the LLM decoder,
# excluding the vision encoder (module names under ``visual.``).
LLM_LORA_TARGETS = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def load_config(path: str | None, overrides: list[str]) -> dict:
    import yaml

    cfg = {}
    if path:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    for o in overrides:  # key=value, value parsed as YAML
        k, v = o.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def build_model_and_processor(cfg: dict):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise SystemExit("QLoRA (bitsandbytes 4-bit) needs a CUDA GPU; none was found.")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
        # Keep the vision tower/merger and LM head in full precision.
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["model_name_or_path"],
        quantization_config=bnb,
        torch_dtype=compute_dtype,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
    )
    processor = AutoProcessor.from_pretrained(
        cfg["model_name_or_path"],
        min_pixels=cfg.get("min_pixels", 256 * 28 * 28),
        max_pixels=cfg.get("max_pixels", 512 * 28 * 28),
    )
    processor.tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    for n, p in model.named_parameters():  # vision tower stays frozen
        if "visual" in n:
            p.requires_grad_(False)

    lora = LoraConfig(
        r=cfg.get("lora_r", 64),
        lora_alpha=cfg.get("lora_alpha", 16),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        target_modules=cfg.get("lora_target_modules", LLM_LORA_TARGETS),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, processor


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/qlora_qwen25vl_chitrakshara.yaml")
    p.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.overrides)

    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    from .data import CaptionDataset, QwenVLCaptionCollator

    set_seed(cfg.get("seed", 42))
    model, processor = build_model_and_processor(cfg)
    if cfg.get("streaming"):
        # Stream Chitrakshara from the Hub (no local copy). A stream has no length, so the
        # number of optimiser steps must be given explicitly.
        from .streaming_data import from_config

        if int(cfg.get("max_steps", -1)) <= 0:
            raise SystemExit("streaming: true needs max_steps > 0 in the config")
        train_ds = from_config(cfg)
        eval_ds = from_config(cfg, eval_split=True)
    else:
        train_ds = CaptionDataset(cfg["train_file"], cfg.get("max_train_samples"))
        eval_ds = CaptionDataset(cfg["eval_file"], cfg.get("max_eval_samples")) if cfg.get("eval_file") else None
    collator = QwenVLCaptionCollator(processor, cfg.get("max_caption_tokens", 256), cfg.get("seed", 42))

    use_bf16 = torch.cuda.is_bf16_supported()
    out = cfg.get("output_dir", "outputs/qwen25vl7b-chitrakshara-qlora")
    targs = TrainingArguments(
        output_dir=out,
        num_train_epochs=cfg.get("num_train_epochs", 1),
        max_steps=cfg.get("max_steps", -1),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.0),
        max_grad_norm=cfg.get("max_grad_norm", 0.3),
        optim=cfg.get("optim", "paged_adamw_8bit"),
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=cfg.get("logging_steps", 10),
        save_strategy="steps",
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 3),
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=cfg.get("eval_steps", 500),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
        remove_unused_columns=False,  # the collator needs the raw fields
        report_to=cfg.get("report_to", "none"),
        seed=cfg.get("seed", 42),
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=collator)
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    adapter_dir = Path(out) / "final_adapter"
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    metrics = trainer.state.log_history
    (Path(out) / "train_log.json").write_text(json.dumps(metrics, indent=2))
    (Path(out) / "train_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"Saved LoRA adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
