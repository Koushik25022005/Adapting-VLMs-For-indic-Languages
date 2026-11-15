# Adapting VLMs for Indic Languages

This repository fine-tunes **Qwen2.5-VL-7B-Instruct** with **QLoRA** (PEFT + bitsandbytes 4-bit)
on **Chitrakshara-Cap** and evaluates it in two ways, both written up in
[`report/research_report.md`](report/research_report.md):

* **CVQA** and **XM3600**: the QLoRA model against the zero-shot base model. No judge model or
  API key is needed. This is the default evaluation.
* **BharatBench** (Indic-POPE, Indic-LLaVA-Bench and Indic-MM-Vet, official evaluators): the
  QLoRA model against **Chitrarth**. Optional, GPT-judged.

> **Results status:** the QLoRA model has BharatBench scores
> (`results/qwen25vl7b-qlora/bharatbench_scores.json`). CVQA and XM3600 have not been run yet;
> their tables in the report are empty until `scripts/run_cvqa_xm3600.sh` has been run on a GPU
> machine. The Chitrarth numbers in `results/chitrarth_bharatbench.json` are copied from the
> [Chitrarth README](https://github.com/ola-krutrim/Chitrarth#5-evaluation-results).

## Requirements

* **Python 3.13.5** (pinned in `.python-version` and `pyproject.toml`). `scripts/setup_env.sh` finds
  it automatically, and installs it with `uv python install 3.13.5` if `uv` is available; otherwise
  set `PYTHON=/path/to/python3.13`.
* An NVIDIA GPU with CUDA. QLoRA (bitsandbytes 4-bit) does not run on CPU. Training needs
  ≥ 24 GB with the default config, and 40–80 GB is comfortable.
* `scripts/setup_env.sh` reads the CUDA version your NVIDIA driver supports from `nvidia-smi`. On a
  CUDA 12.4–12.9 driver it installs the CUDA 12.4 build of PyTorch (2.6.0), because the newest
  PyTorch needs a CUDA 13 driver and otherwise reports `CUDA False`. Override with
  `TORCH_CUDA=cu124` or `TORCH_CUDA=default`.
* On a 24 GB card (e.g. RTX 4090) train with
  `per_device_train_batch_size=2 gradient_accumulation_steps=8` appended to the train command.
* Access to the Hugging Face Hub (`Qwen/Qwen2.5-VL-7B-Instruct`, `krutrim-ai-labs/chitrakshara`
  (config `Chitrakshara-Cap`, the default), `afaji/cvqa`, `floschne/xm3600`; for BharatBench
  also `lmms-lab/POPE` and `MBZUAI/multilingual-llava-bench-in-the-wild`). Set `HF_TOKEN` if a
  dataset is gated.
* `OPENAI_API_KEY`, only for the BharatBench GPT judges.

## Quick start

```bash
bash scripts/setup_env.sh
```

```bash
bash scripts/run_pipeline.sh
```

`run_pipeline.sh` runs the steps below in order. Each step can also be run on its own.

| Step | Command | Output |
|---|---|---|
| 1. Sample Chitrakshara-Cap (balanced per language) | `python -m indicvlm.prepare_data --per-language 5000` | `data/chitrakshara_cap/{train,val}.jsonl`, `stats.json` |
| 2. QLoRA training | `python -m indicvlm.train --config configs/qlora_qwen25vl_chitrakshara.yaml` | `outputs/.../final_adapter` |
| 3. CVQA, base and QLoRA | `python -m indicvlm.eval_cvqa --load-4bit [--adapter <dir>]` | `results/<run>/cvqa_scores.json`, `cvqa_predictions.jsonl` |
| 4. XM3600, base and QLoRA | `python -m indicvlm.eval_xm3600 --load-4bit [--adapter <dir>]` | `results/<run>/xm3600_scores.json`, `xm3600_predictions.jsonl` |
| 5. Fill the report | `python -m indicvlm.update_report --cvqa-base ... --cvqa-ours ... --xm3600-base ... --xm3600-ours ...` | `report/research_report.md` |

Steps 3–5 score the zero-shot base model and the adapter (run names `qwen25vl7b-base` and
`qwen25vl7b-qlora`), so the report shows the effect of the QLoRA adaptation alone.

### CVQA and XM3600 with an adapter you already trained

```bash
ADAPTER=outputs/qwen25vl7b-chitrakshara-qlora/final_adapter bash scripts/run_cvqa_xm3600.sh
```

This runs steps 3–5. `BENCHMARKS=cvqa` or `BENCHMARKS=xm3600` runs one benchmark, `SKIP_BASE=1`
reuses the base model's existing score files, and `LIMIT=20` is a quick smoke test.

| Benchmark | What it measures | Languages | Metric |
|---|---|---|---|
| **CVQA** (`afaji/cvqa`) | Multiple-choice VQA about local culture. Not what the adapter was trained on, so it shows whether caption tuning hurt question answering. | The India subsets by default (`--countries India`; pass `--countries` with no value for every country, or `--languages Hindi Tamil`) | Accuracy on the option letter, asked in the local language and in English. "Unparsed %" is the share of responses that name no option. |
| **XM3600** (`floschne/xm3600`) | Captioning, the task the adapter was trained on. | Hindi, Bengali, Telugu, English (`--languages hi bn te en`) | CIDEr-D against the human captions, plus the share of captions in the requested language and the average length. |

CIDEr-D is computed by `indicvlm.cider` (pure Python, one tokenizer for references and
predictions), so it compares the two models fairly but is not comparable with published XM3600
numbers. If your copy of CVQA has no answer labels, `eval_cvqa` still writes the predictions and
says that no accuracy was computed.

### BharatBench (optional)

```bash
bash scripts/setup_bharatbench.sh
```

```bash
export OPENAI_API_KEY=sk-...
BENCHMARKS="cvqa xm3600 bharatbench" bash scripts/run_pipeline.sh
```

| Step | Command | Output |
|---|---|---|
| Download benchmark images | `python scripts/download_bharatbench_images.py` | `data/bharatbench_images/` |
| Generation | `python -m indicvlm.infer_bharatbench --adapter <dir> --load-4bit --out results/responses/qwen25vl7b-qlora` | evaluator-ready response files |
| Official scoring | `bash scripts/run_bharatbench.sh results/responses/qwen25vl7b-qlora qwen25vl7b-qlora` | `results/qwen25vl7b-qlora/bharatbench_scores.json` |
| Fill the report | `python -m indicvlm.update_report --ours results/qwen25vl7b-qlora/bharatbench_scores.json [--base ...]` | `report/research_report.md` |

### Streaming instead of a local copy

`STREAMING=1 bash scripts/run_pipeline.sh` skips step 1 and trains with
`configs/qlora_qwen25vl_chitrakshara_stream.yaml`. Rows are read from
`krutrim-ai-labs/chitrakshara` (config `Chitrakshara-Cap`) during training
(`indicvlm/streaming_data.py`):

1. **Load and stream** with `load_dataset(..., streaming=True)`, through a shuffle buffer.
2. **Format:** clean the caption and detect its language. The user turn is the image plus
   "Describe this image in {language}.", and the assistant turn is the caption.
3. **Images:** URLs are downloaded in parallel, and rows whose image fails are dropped.
4. **Collate and mask:** apply the Qwen2.5-VL chat template. The loss is computed only on the
   caption tokens; the prompt, image tokens and padding are set to -100.

A stream has no length, so set `max_steps` (default 3000, i.e. about 48k samples at an
effective batch of 16). `stream.max_per_language` caps each language, so Hindi doesn't
dominate. `stream.holdout_fraction` (1%) of pairs, chosen by a content hash, is never trained on, and
`stream.max_eval_samples` of them are collected once for evaluation. To preview the
stream without a GPU:
`python -m indicvlm.streaming_data --take 5`.

### Useful knobs

* Config overrides: `python -m indicvlm.train --config ... max_steps=200 lora_r=32`
* Smoke test of generation: `python -m indicvlm.infer_bharatbench ... --limit 20 --tasks pope`
* `POPE_SCORER=muril bash scripts/run_bharatbench.sh ...` uses BharatBench's embedding-based POPE
  scorer, which needs no API calls. The default GPT scorer makes about 18k judge calls per
  language, because BharatBench's script classifies both the answers and the labels with GPT.
* `JUDGE_MODEL=gpt-4o ...` replaces the GPT judge (BharatBench default `gpt-4-0613`) if that model
  is no longer served. Record this in the report, because the scores are then not strictly
  comparable with Chitrarth's.
* Chitrakshara rows are interleaved web documents (`images` / `texts` lists plus a `metadata`
  JSON string). `prepare_data` pairs each image with the text just before it, else the text just
  after it, else its alt text (`--caption-source before,after,alt`). It clips captions at a
  sentence end (`--max-caption-chars 500`) and downloads images in parallel. Downloads run continuously
  (`--workers 64`), each with a hard total time limit (`--deadline 15` s, plus `--timeout 5` s
  per socket operation), so slow or dead sites never hold up the rest. Saved images are
  downscaled to `--max-side 1024`. The progress bar shows rows scanned, downloads in flight,
  the image failure rate (`fail`) and captions skipped for language (`lang_skip`); `stats.json`
  breaks failures down by reason (`failed_http_error`, `failed_timeout`, `failed_slow` ...).
  Plain one-image-one-caption datasets work too (`--format pairs`).
* The rows have no language label, so the language is detected from the caption's script.
  Hindi and Marathi are separated by common words, and Assamese and Bengali by the letters
  ৰ/ৱ vs র. Captions it can't place, such as Punjabi, are skipped.
  `stats.json` records how many images failed to download and how many captions had an
  unknown language.

## Layout

```
configs/                 training config (QLoRA hyper-parameters)
src/indicvlm/            prepare_data, data (collator), train, eval_cvqa, eval_xm3600, cider,
                         infer_bharatbench, collect_scores, update_report
scripts/                 env setup, CVQA/XM3600 run, BharatBench setup/scoring, image download, end-to-end pipeline
results/                 chitrarth_bharatbench.json (published baseline) + per-run scores
report/research_report.md
tests/                   CPU unit tests (pytest)
```

## Tests

```bash
.venv/bin/python -m pytest
```
