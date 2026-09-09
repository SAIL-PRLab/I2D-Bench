# I2D-Benchmark

**[English](README.md) | [中文](README.zh.md)**

[![Paper](https://img.shields.io/badge/arXiv-2603.24430-b31b1b.svg)](https://arxiv.org/abs/2603.24430)
[![Demo Page](https://img.shields.io/badge/Project-Demo%20Page-blue.svg)](https://ssf-1103.github.io/I2D-Bench/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow.svg)](https://huggingface.co/datasets/ssf1103/I2D-Bench)

**Iterate to Differentiate**：通过迭代合成提升零样本 TTS 评估的区分性与可靠性。

## 项目简介

I2D-Benchmark 是一个面向零样本 TTS 模型的评测基准。同一文本让模型**迭代合成 N 次**并分别打分，从而更可靠地区分不同模型。

**核心特性**

- **迭代合成评估**：每个样本运行 N 轮合成，逐轮统计指标
- **多维度指标**：WER/CER、说话人相似度（SIM）、UTMOSv2、DNSMOS、情感 F1 等
- **可插拔模型**：通过统一接口 `BaseTTSProcessor` 接入任意 TTS
- **三个开箱即用测试集**：
  - **LibriTTS / test_clean** —— 英文，英文新闻朗读
  - **Seed-Eval / test_zh** —— 中文，中文零样本
  - **CV3-Eval / emotion_zeroshot** —— 中文，情感零样本

## 仓库结构

```
.
├── configs/                  # 数据集与模型配置
│   ├── TTS_configs.yaml      # TTS 模型注册表：venv / 推理脚本 / 权重路径
│   ├── LibriTTS/             # 英文测试集配置
│   ├── Seed-Eval/            # 中文测试集配置
│   ├── CV3-Eval/             # 情感测试集配置
│   └── test_set/             # 冒烟测试用的小样本配置
├── scripts/                  # 评测入口与数据准备
│   ├── run.sh                # 评测流水线入口（stage 1–6）
│   ├── reproduce.sh          # 环境准备 + 冒烟测试
│   ├── prepare_dataset.sh    # 数据集与对应评测模型准备
│   ├── cal_sim.sh            # WavLM 说话人相似度
│   ├── cal_wer.sh            # CER / WER 计算
│   └── merge_results.sh      # 多 worker 结果合并
├── src/                      # TTS Processor 实现
│   ├── tts_common.py         # BaseTTSProcessor 基类
│   ├── qwen3_tts.py          # 示例：Qwen3-TTS 适配
│   ├── cosyvoice_tts.py      # 其余：f5tts / index_tts / …
│   └── …
├── utils/                    # 通用工具
│   ├── comprehensive_analysis.py  # Streamlit 可视化仪表盘
│   ├── calculate_score.py    # 分数聚合
│   └── …
├── versa/                    # 通用指标依赖项目（WER / UTMOSv2 / DNSMOS / SIM）
├── seed-tts-eval/            # 中文 CER / 说话人相似度依赖项目
├── CV3-Eval/                 # 情感评估依赖项目
├── data/                     # 测试集数据与各模型合成音频
├── results/                  # 评测结果输出
└── checkpoints/              # 第三方权重（wavlm / emotion2vec / rawnet3 / …）
```

## 快速开始：冒烟测试

仓库内置 3 条英文样本，无需准备任何 TTS 权重即可跑完度量流水线：

```bash
bash scripts/reproduce.sh
```

脚本依次：创建 venv → 安装依赖 → 安装 `versa` → 安装 UTMOSv2 → 下载 RawNet3 说话人识别权重 → 调用 `scripts/run.sh --stage 2-6`。

首次运行需要安装依赖与下载模型，请耐心等待。

## 完整评测流程

### 1. 准备测试集与对应评测所需依赖

准备脚本会下载测试集 tarball，并按需安装/拉取对应的评测依赖：

```bash
bash scripts/prepare_dataset.sh LibriTTS   # 英文
bash scripts/prepare_dataset.sh Seed-Eval  # 中文：拉取 wavlm + seed-tts-eval
bash scripts/prepare_dataset.sh CV3-Eval   # 情感：拉取 emotion2vec_plus_large
```

### 2. 运行评测

```bash
bash scripts/run.sh \
    --TTS_Models Qwen3-TTS \
    --config configs/LibriTTS/test_clean/config.yaml \
    --stage 1-6 --max_iterations 3 --max_entries 10
```

`--TTS_Models` 支持逗号分隔多个模型，每个模型独立评测，产物互不干扰。

| 参数 | 作用 |
|---|---|
| `--TTS_Models` | 模型名，需在 `configs/TTS_configs.yaml` 注册，多个用逗号分隔 |
| `--config` | 数据集配置文件路径 |
| `--stage N` 或 `--stage N-M` | 运行的流水线阶段（如 `--stage 2-6` 跳过 stage 1 的合成） |
| `--max_iterations N` | 覆盖配置中的迭代轮数 |
| `--max_entries N` | 仅取数据集前 N 条（快速验证用） |
| `--num_workers N` | 覆盖所有阶段的并行 worker 数（默认 GPU 数） |

### 流水线阶段

| 阶段 | 内容 | 依赖 |
|---|---|---|
| 1 | TTS 模型推理，逐轮生成音频 | `src/<model>_tts.py` + 对应 `venv`（由 `TTS_configs.yaml` 指定） |
| 2 | 生成 `iter_*.scp` 索引文件 | — |
| 3 | 通用指标（WER、UTMOSv2、DNSMOS、SIM/RawNet3 等），由 `metrics_config` 控制 | `versa/` |
| 4 | CER + 说话人相似度（仅 `language: zh` 的数据集） | `seed-tts-eval/` + `checkpoints/wavlm_large_finetune.pth` |
| 5 | 情感 F1（仅 `run_eval_emotion: true`） | `CV3-Eval/` + `checkpoints/emotion2vec_plus_large/` |
| 6 | 汇总最终分数到 `results/<数据集>/<模型>/score_summary.csv` | — |

## src/ 处理器适配说明

`src/<model>.py` 把每个 TTS 包成 `BaseTTSProcessor`（基类见 `src/tts_common.py`）。将本仓库适配到新环境时，请重点检查每个 Processor 的 `setup_environment()` 中 TTS 仓库的本地路径。

各 TTS 自身的依赖请按其官方 README 安装；安装完成后，在 `configs/TTS_configs.yaml` 中把对应模型的 `venv_path` / `script_path` / `model_path` 改成本地路径即可。

## 接入新 TTS 模型

新增一个 TTS 模型只需两步：写 Processor + 在注册表中登记。

### 1. 在 `src/` 创建 `your_tts.py`

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

### 2. 在 `configs/TTS_configs.yaml` 注册

```yaml
YourTTS:
  venv_path: "/path/to/your-tts/.venv/bin/activate"
  script_path: "src/your_tts.py"
  model_path: "/path/to/your-tts/checkpoints"
```

完成上述两步后，即可在任意已准备好的测试集上使用 `bash scripts/run.sh --TTS_Models YourTTS …` 触发评测。

## 可视化分析（utils/comprehensive_analysis.py）

`utils/comprehensive_analysis.py` 提供一个基于 Streamlit 的可视化仪表盘，用于浏览与对比各 TTS 模型在不同迭代轮次下的指标轨迹：

```bash
# 在仓库根目录启动
streamlit run utils/comprehensive_analysis.py
```

启动后默认扫描 `results/Seed-Eval/test_zh`、`results/LibriTTS/test_clean`、`results/CV3-Eval/emotion_zeroshot` 三个目录，读取每个模型下的 `score_summary.csv`，并以表格与图表展示：

- 各模型的 per-iteration 指标轨迹
- 按均值聚合后的整体排名
- 多测试集横向对比

## 致谢

本基准离不开以下开源项目的支持，感谢作者们开源的代码与模型：

- [versa](https://github.com/wavlab-speech/versa)
- [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval) 
- [CV3-Eval](https://github.com/QwenAudio/CV3-Eval)
