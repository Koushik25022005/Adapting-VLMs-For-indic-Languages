# Adapting Qwen2.5-VL-7B-Instruct to Indic Languages with QLoRA on Chitrakshara-Cap: A BharatBench Comparison with Chitrarth

> **Status:** the pipeline is implemented, but training and evaluation have **not been run yet**.
> Every "ours" / "zero-shot" cell below shows "—" until `scripts/run_pipeline.sh` finishes and
> `python -m indicvlm.update_report` writes the measured numbers in. The Chitrarth column holds the
> published numbers from the Chitrarth README.

## Abstract

We study whether parameter-efficient fine-tuning can give a strong general-purpose
vision-language model competitive Indic-language abilities. We fine-tune
**Qwen2.5-VL-7B-Instruct** with **QLoRA** (4-bit NF4 base model, LoRA adapters on the language
model) on a language-balanced subset of **Chitrakshara-Cap**, a multilingual image-caption corpus for
Indian languages. We evaluate on **BharatBench** (Indic-POPE, Indic-LLaVA-Bench and Indic-MM-Vet in
10 Indic languages plus English) and compare against **Chitrarth**, a VLM built for Indic languages
(Krutrim-1 LLM + SigLIP encoder), and against the zero-shot Qwen2.5-VL-7B-Instruct base model.

## 1. Introduction

Most open VLMs are trained mainly on English data. Chitrarth (Khan et al., 2025) showed that
multilingual pre-training and instruction tuning on Indic data produces a model that works across
10 Indian languages, and it introduced BharatBench to measure this. Here we ask a cheaper
question: *how far can a strong multilingual base VLM get with only caption-level QLoRA adaptation
on Indic data?*

## 2. Experimental Setup

### 2.1 Model

| Component | Setting |
|---|---|
| Base model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Quantisation | bitsandbytes 4-bit NF4, double quantisation, bf16 compute; vision tower and `lm_head` not quantised |
| Adapters | PEFT LoRA, r = 64, α = 16, dropout 0.05, on `q,k,v,o,gate,up,down` projections of the LLM decoder only |
| Frozen | Vision encoder and merger |
| Image resolution | 256–512 × 28² pixels (≤ 512 visual tokens) during training |

### 2.2 Training data: Chitrakshara-Cap

Source: Hugging Face dataset `krutrim-ai-labs/chitrakshara`, config `Chitrakshara-Cap`.

Chitrakshara rows are interleaved web documents: aligned lists of images (URLs) and text
paragraphs, plus per-image metadata. We stream the corpus and turn every image into an
image–caption pair. The caption is the paragraph just before the image, else the one just after it,
else its alt text. Captions are clipped at a sentence end to at most 500 characters. We drop
captions with fewer than 3 words or 5 characters, images that fail to download, and images whose
shorter side is under 64 px. The rows carry no language label, so we assign the language from the
caption's Unicode script. Hindi and Marathi (both Devanagari) are separated by frequent
function words and the Marathi letter ळ, and Assamese and Bengali by ৰ/ৱ versus র. Pairs whose
language cannot be decided are discarded. We keep up to *N* pairs per language (Hindi, Bengali,
Telugu, Tamil, Marathi, Gujarati, Kannada, Malayalam, Odia, Assamese, English).
Each pair becomes a single-turn chat sample: the user turn holds the image and an instruction such
as *"Describe this image in {language}."*, and the assistant turn holds the caption. The loss is
computed on the assistant tokens only. The real per-language counts are recorded in
`data/chitrakshara_cap/stats.json` after data preparation.

### 2.3 Optimisation

1 epoch, AdamW (paged, 8-bit), learning rate 2e-4 with cosine decay and 3 % warm-up, effective
batch size 16 per GPU, max grad-norm 0.3, gradient checkpointing, seed 42. The full configuration
is in `configs/qlora_qwen25vl_chitrakshara.yaml`. The training logs are saved to `outputs/.../train_log.json`.

### 2.4 Evaluation: BharatBench

We use the official evaluators from `ola-krutrim/BharatBench` (commit `de5aa6b`) without changes,
with greedy decoding:

| Task | Metric | Judge |
|---|---|---|
| Indic-POPE (COCO adversarial / popular / random, 8,910 questions per language) | Accuracy × 100, averaged over the 3 splits | GPT (`gpt-4-0613`), which maps each free-form answer to yes/no |
| Indic-LLaVA-Bench (in-the-wild, 60 questions) | Relative score: model ÷ GPT-4 reference × 100 | `gpt-4-0613` |
| Indic-MM-Vet (218 questions) | Total score (0–100) | `gpt-4o-mini-2024-07-18` |

The Chitrarth numbers come from its public README. We did not re-run them. If a different judge
model has to be used (for example, because `gpt-4-0613` is no longer available), we report it here,
because a different judge makes the numbers not strictly comparable.

## 3. Results

<!-- BEGIN:BHARATBENCH_RESULTS -->
#### POPE

| Language | Chitrarth | Qwen2.5-VL-7B + QLoRA (ours) | Δ (ours − Chitrarth) |
|---|---:|---:|---:|
| Telugu | 79.90 | 64.68 | -15.22 |
| Hindi | 78.68 | 52.50 | -26.18 |
| Bengali | 83.24 | 51.70 | -31.54 |
| Malayalam | 85.29 | 99.97 | +14.68 |
| Kannada | 85.52 | 100.00 | +14.48 |
| Assamese | 55.59 | 49.36 | -6.23 |
| Tamil | 83.28 | 5.86 | -77.42 |
| Marathi | 79.17 | 100.00 | +20.83 |
| Gujarati | 84.75 | 97.51 | +12.76 |
| Odia | 82.03 | 26.46 | -55.57 |
| English | 87.63 | 100.00 | +12.37 |
| **Indic avg (10)** | 79.75 | 64.80 | -14.94 |
| **All avg (11)** | 80.46 | 68.00 | -12.46 |

#### LLaVA-Bench

| Language | Chitrarth | Qwen2.5-VL-7B + QLoRA (ours) | Δ (ours − Chitrarth) |
|---|---:|---:|---:|
| Telugu | 54.80 | 18.70 | -36.10 |
| Hindi | 51.50 | 20.70 | -30.80 |
| Bengali | 53.70 | 21.30 | -32.40 |
| Malayalam | 55.50 | 17.80 | -37.70 |
| Kannada | 58.10 | 20.30 | -37.80 |
| Assamese | 59.10 | 18.50 | -40.60 |
| Tamil | 58.30 | 19.20 | -39.10 |
| Marathi | 52.80 | 17.00 | -35.80 |
| Gujarati | 55.90 | 20.90 | -35.00 |
| Odia | 62.80 | 17.00 | -45.80 |
| English | 67.90 | 30.70 | -37.20 |
| **Indic avg (10)** | 56.25 | 19.14 | -37.11 |
| **All avg (11)** | 57.31 | 20.19 | -37.12 |

#### MMVet

| Language | Chitrarth | Qwen2.5-VL-7B + QLoRA (ours) | Δ (ours − Chitrarth) |
|---|---:|---:|---:|
| Telugu | 43.76 | 10.69 | -33.07 |
| Hindi | 38.85 | 11.83 | -27.02 |
| Bengali | 33.24 | 11.97 | -21.27 |
| Malayalam | 25.36 | 16.65 | -8.71 |
| Kannada | 46.19 | 12.43 | -33.76 |
| Assamese | 37.29 | 10.28 | -27.01 |
| Tamil | 34.31 | 16.01 | -18.30 |
| Marathi | 40.96 | 11.61 | -29.35 |
| Gujarati | 39.03 | 17.34 | -21.69 |
| Odia | 19.67 | 19.59 | -0.08 |
| English | 30.49 | 37.75 | +7.26 |
| **Indic avg (10)** | 35.87 | 13.84 | -22.03 |
| **All avg (11)** | 35.38 | 16.01 | -19.36 |
<!-- END:BHARATBENCH_RESULTS -->

## 4. Discussion

_To be written once the measured results are available. Points to cover: (i) per-language gains over
the zero-shot base, which isolate the effect of Chitrakshara-Cap QLoRA; (ii) where the adapted model
beats or trails Chitrarth, particularly on low-resource Assamese and Odia; (iii) whether caption-only
supervision helps hallucination (POPE) more than open-ended reasoning (LLaVA-Bench, MM-Vet);
(iv) any regression on English._

## 5. Limitations

* Training uses captions only, with no multilingual instruction or VQA data. Chitrarth was
  instruction-tuned, so gains on LLaVA-Bench and MM-Vet may be limited.
* BharatBench uses GPT judges. Scores depend on the judge model and version, and on API
  non-determinism.
* We train on a subset of Chitrakshara-Cap because of compute limits.
* The Chitrarth numbers are copied from the published README, not reproduced.

## References

* S. Khan et al. *Chitrarth: Bridging Vision and Language for a Billion People.* arXiv:2502.15392, 2025.
* Krutrim AI Labs. *BharatBench.* https://github.com/ola-krutrim/BharatBench
* Krutrim AI Labs. *Chitrakshara* multilingual multimodal dataset (Chitrakshara-Cap).
* Qwen Team. *Qwen2.5-VL Technical Report.* 2025.
* T. Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
* E. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
