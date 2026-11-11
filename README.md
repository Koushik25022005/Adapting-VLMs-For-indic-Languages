# Adapting VLMs for Indic Languages

This repository fine-tunes **Qwen2.5-VL-7B-Instruct** with **QLoRA** (PEFT + bitsandbytes 4-bit)
on **Chitrakshara-Cap**, evaluates it with the official **BharatBench** evaluators (Indic-POPE,
Indic-LLaVA-Bench and Indic-MM-Vet), and compares it with **Chitrarth** in
[`report/research_report.md`](report/research_report.md).

> **Results status:** no training or evaluation run has been done yet. The report's results
> tables show "—" for our model until the pipeline below has been run on a GPU machine. The
> Chitrarth numbers in `results/chitrarth_bharatbench.json` are copied from the
> [Chitrarth README](https://github.com/ola-krutrim/Chitrarth#5-evaluation-results).

## Requirements

* **Python 3.13.5** (pinned in `.python-version` and `pyproject.toml`). `scripts/setup_env.sh` finds
  it automatically, and installs it with `uv python install 3.13.5` if `uv` is available; otherwise
  set `PYTHON=/path/to/python3.13`.
* An NVIDIA GPU with CUDA. QLoRA (bitsandbytes 4-bit) does not run on CPU. Training needs
  ≥ 24 GB with the default config, and 40–80 GB is comfortable.
* Access to the Hugging Face Hub (`Qwen/Qwen2.5-VL-7B-Instruct`, `krutrim-ai-labs/chitrakshara`
  (config `Chitrakshara-Cap`, the default), `lmms-lab/POPE`,
  `MBZUAI/multilingual-llava-bench-in-the-wild`). Set `HF_TOKEN` if a dataset is gated.
* `OPENAI_API_KEY` for the BharatBench GPT judges.

## Quick start

```bash
bash scripts/setup_env.sh
```

```bash
bash scripts/setup_bharatbench.sh
```

```bash
export OPENAI_API_KEY=sk-...
bash scripts/run_pipeline.sh
```

`run_pipeline.sh` runs the steps below in order. Each step can also be run on its own.

| Step | Command | Output |
|---|---|---|
| 1. Sample Chitrakshara-Cap (balanced per language) | `python -m indicvlm.prepare_data --per-language 5000` | `data/chitrakshara_cap/{train,val}.jsonl`, `stats.json` |
| 2. Download benchmark images | `python scripts/download_bharatbench_images.py` | `data/bharatbench_images/` |
| 3. QLoRA training | `python -m indicvlm.train --config configs/qlora_qwen25vl_chitrakshara.yaml` | `outputs/.../final_adapter` |
| 4. BharatBench generation | `python -m indicvlm.infer_bharatbench --adapter <dir> --load-4bit --out results/responses/qwen25vl7b-qlora` | evaluator-ready response files |
| 5. Official scoring | `bash scripts/run_bharatbench.sh results/responses/qwen25vl7b-qlora qwen25vl7b-qlora` | `results/qwen25vl7b-qlora/bharatbench_scores.json` |
| 6. Fill the report | `python -m indicvlm.update_report --ours results/qwen25vl7b-qlora/bharatbench_scores.json [--base ...]` | `report/research_report.md` |

The pipeline also scores the zero-shot base model (set `SKIP_BASE=1` to skip it). This lets the
report separate the effect of the QLoRA adaptation from the base model's own Indic ability.

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
  sentence end (`--max-caption-chars 500`) and downloads images in parallel (`--workers 32`).
  Plain one-image-one-caption datasets work too (`--format pairs`).
* The rows have no language label, so the language is detected from the caption's script.
  Hindi and Marathi are separated by common words, and Assamese and Bengali by the letters
  ৰ/ৱ vs র. Captions it can't place, such as Punjabi, are skipped. If you load a subset known
  to be one language, pass `--assume-language Hindi` instead.
  `stats.json` records how many images failed to download and how many captions had an
  unknown language.

## Layout

```
configs/                 training config (QLoRA hyper-parameters)
src/indicvlm/            prepare_data, data (collator), train, infer_bharatbench, collect_scores, update_report
scripts/                 env setup, BharatBench setup/scoring, image download, end-to-end pipeline
results/                 chitrarth_bharatbench.json (published baseline) + per-run scores
report/research_report.md
tests/                   CPU unit tests (pytest)
```

## Tests

```bash
.venv/bin/python -m pytest
```
