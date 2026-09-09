# I2D-Benchmark

**[English](README.md) | [中文](README.zh.md)**

[![Paper](https://img.shields.io/badge/arXiv-2603.24430-b31b1b.svg)](https://arxiv.org/abs/2603.24430)
[![Demo Page](https://img.shields.io/badge/Project-Demo%20Page-blue.svg)](https://ssf-1103.github.io/I2D-Bench/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow.svg)](https://huggingface.co/datasets/ssf1103/I2D-Bench)

**Iterate to Differentiate**: Enhancing Discriminability and Reliability in Zero-Shot TTS Evaluation

## Overview

I2D-Benchmark is an evaluation benchmark for zero-shot TTS models. The same text is synthesized by a model **N times in succession** and scored independently for each round, exposing stability and variance issues that a single-pass evaluation can hide, and producing more reliable model rankings.

**Key Features**

- **Iterative synthesis evaluation** — each sample is synthesized for N rounds with per-iteration metrics
- **Multi-dimensional metrics** — WER/CER, speaker similarity (SIM), UTMOSv2, DNSMOS, emotion F1, etc.
- **Pluggable models** — wrap any TTS behind the unified `BaseTTSProcessor` interface
- **Three ready-to-use test sets**:
  - **LibriTTS / test_clean** — English, audiobook-style reading
  - **Seed-Eval / test_zh** — Chinese, Chinese zero-shot
  - **CV3-Eval / emotion_zeroshot** — Chinese, emotion zero-shot

## Repository Layout

```
.
├── configs/                  # Dataset & model configurations
│   ├── TTS_configs.yaml      # TTS registry: venv / inference script / checkpoint paths
│   ├── LibriTTS/             # English test-set configs
│   ├── Seed-Eval/            # Chinese test-set configs
│   ├── CV3-Eval/             # Emotion test-set configs
│   └── test_set/             # Smoke-test mini config
├── scripts/                  # Entry points & data preparation
│   ├── run.sh                # Evaluation pipeline entry (stages 1–6)
│   ├── reproduce.sh          # Environment setup + smoke test
│   ├── prepare_dataset.sh    # Dataset + dependent evaluator prep
│   ├── cal_sim.sh            # WavLM speaker similarity
│   ├── cal_wer.sh            # CER / WER computation
│   └── merge_results.sh      # Merge multi-worker results
├── src/                      # TTS Processor implementations
│   ├── tts_common.py         # BaseTTSProcessor base class
│   ├── qwen3_tts.py          # Example: Qwen3-TTS adapter
│   ├── cosyvoice_tts.py      # Others: f5tts / index_tts / …
│   └── …
├── utils/                    # Shared utilities
│   ├── comprehensive_analysis.py  # Streamlit visualization dashboard
│   ├── calculate_score.py    # Score aggregation
│   └── …
├── versa/                    # General-purpose metric dependency (WER / UTMOSv2 / DNSMOS / SIM)
├── seed-tts-eval/            # Chinese CER / speaker similarity dependency
├── CV3-Eval/                 # Emotion evaluation dependency
├── data/                     # Test-set data and per-model synthesized audio
├── results/                  # Evaluation output
└── checkpoints/              # Third-party weights (wavlm / emotion2vec / rawnet3 / …)
```

## Quick Start: Smoke Test

The repo ships with 3 English samples and can run the full metrics pipeline without preparing any TTS weights:

```bash
bash scripts/reproduce.sh
```

The script proceeds in order: create venv → install dependencies → install `versa` → install UTMOSv2 → download the RawNet3 speaker-verification weights → invoke `scripts/run.sh --stage 2-6`.

The first run needs to install dependencies and download models, so please be patient. Idempotent steps are skipped on re-runs.

## Full Evaluation Workflow

### 1. Prepare the test set and its evaluator dependencies

The prep script downloads the test-set tarball and, per dataset, installs/pulls the matching evaluator dependencies:

```bash
bash scripts/prepare_dataset.sh LibriTTS   # English
bash scripts/prepare_dataset.sh Seed-Eval  # Chinese: pulls wavlm + seed-tts-eval
bash scripts/prepare_dataset.sh CV3-Eval   # Emotion: pulls emotion2vec_plus_large
```

### 2. Run the evaluation

```bash
bash scripts/run.sh \
    --TTS_Models Qwen3-TTS \
    --config configs/LibriTTS/test_clean/config.yaml \
    --stage 1-6 --max_iterations 3 --max_entries 10
```

`--TTS_Models` accepts a comma-separated list; each model is evaluated independently, with isolated outputs.

| Argument | Purpose |
|---|---|
| `--TTS_Models` | Model name, must be registered in `configs/TTS_configs.yaml`; comma-separated for multiple |
| `--config` | Path to the dataset config file |
| `--stage N` or `--stage N-M` | Pipeline stages to run (e.g. `--stage 2-6` skips stage 1 synthesis) |
| `--max_iterations N` | Override the iteration count in the config |
| `--max_entries N` | Evaluate only the first N entries (useful for quick checks) |
| `--num_workers N` | Override the worker count for all stages (default: number of GPUs) |

### Pipeline Stages

| Stage | What it does | Dependencies |
|---|---|---|
| 1 | TTS inference, synthesizing audio round-by-round | `src/<model>_tts.py` + matching `venv` (from `TTS_configs.yaml`) |
| 2 | Generate `iter_*.scp` index files | — |
| 3 | General metrics (WER, UTMOSv2, DNSMOS, SIM/RawNet3, …) controlled by `metrics_config` | `versa/` |
| 4 | CER + speaker similarity (only for `language: zh` datasets) | `seed-tts-eval/` + `checkpoints/wavlm_large_finetune.pth` |
| 5 | Emotion F1 (only when `run_eval_emotion: true`) | `CV3-Eval/` + `checkpoints/emotion2vec_plus_large/` |
| 6 | Aggregate final scores into `results/<dataset>/<model>/score_summary.csv` | — |

## `src/` Processor Adaptation

`src/<model>.py` wraps each TTS as a `BaseTTSProcessor` (see `src/tts_common.py` for the base class). When porting this repo to a new environment, pay particular attention to the TTS repo paths referenced in each Processor's `setup_environment()`.

Install each TTS's own dependencies following its official README; once installed, update `venv_path` / `script_path` / `model_path` in `configs/TTS_configs.yaml` to point at your local setup.

## Adding a New TTS Model

Adding a new TTS model takes two steps: write a Processor, then register it.

### 1. Create `your_tts.py` under `src/`

```python
from tts_common import BaseTTSProcessor

class YourTTSProcessor(BaseTTSProcessor):
    model_name = "YourTTS"

    def setup_environment(self):
        import sys
        sys.path.insert(0, "/path/to/your-tts")

    def load_model(self, device: str):
        from your_tts import Model
        return Model(device=device)

    def synthesize(self, model, prompt_wav, prompt_text, text, output_path):
        model.generate(text=text, prompt_wav=prompt_wav, output_path=output_path)
        return True
```

### 2. Register it in `configs/TTS_configs.yaml`

```yaml
YourTTS:
  venv_path: "/path/to/your-tts/.venv/bin/activate"
  script_path: "src/your_tts.py"
  model_path: "/path/to/your-tts/checkpoints"
```

Once both steps are done, you can trigger evaluation on any prepared test set with `bash scripts/run.sh --TTS_Models YourTTS …`.

## Visualization (`utils/comprehensive_analysis.py`)

`utils/comprehensive_analysis.py` ships a Streamlit dashboard for browsing and comparing per-iteration metric trajectories across TTS models:

```bash
# Run from the repo root
streamlit run utils/comprehensive_analysis.py
```

On launch it scans `results/Seed-Eval/test_zh`, `results/LibriTTS/test_clean`, and `results/CV3-Eval/emotion_zeroshot`, reads each model's `score_summary.csv`, and presents:

- Per-iteration metric trajectories for each model
- Overall rankings aggregated by mean
- Cross-test-set comparisons

## Acknowledgments

This benchmark stands on the shoulders of the following open-source projects. We thank their authors for releasing the code and models:

- [versa](https://github.com/wavlab-speech/versa)
- [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval)
- [CV3-Eval](https://github.com/QwenAudio/CV3-Eval)

