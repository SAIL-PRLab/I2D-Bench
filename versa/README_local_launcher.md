# VERSA 本地并行启动脚本说明（launch_local.sh）

该脚本用于在本地将评估任务切分为多个块，并在多块 GPU 上按最大并行度进行并行执行。脚本仅使用 GPU 进行处理，并将任务均匀轮询分配到各个 GPU。

## 功能概述
- 对输入的 `wav.scp`（预测/参考）和可选文本文件进行样本对齐，确保 Key 一致。
- 将对齐后的文件根据 `split_size` 分割为多个子块。
- 根据 `--max-parallel` 控制总体并行度，并按 **轮询(round-robin)** 将任务均分到每块 GPU。
- 记录每个任务的运行日志与状态，并最终提供结果合并脚本。

## 使用方法

```
./versa/launch_local.sh <pred_wavscp> <gt_wavscp|None> <score_dir> <split_size> [--config=FILE] [--max-parallel=N] [--text=FILE]
```

- 必选参数：
  - `pred_wavscp`：预测音频列表文件（形如：`utt_id /path/to/audio.wav`）
  - `gt_wavscp|None`：参考音频列表文件或 `None`（无参考时设置为 `None`）
  - `score_dir`：结果输出目录
  - `split_size`：切分块数（建议与数据规模和并发能力匹配）

- 可选参数：
  - `--config=FILE`：评估指标配置文件（默认：仓库根目录下的 `metrics.yaml`）
  - `--max-parallel=N`：最大并行任务数，默认使用机器 CPU 核心数（仅作为并发上限）
  - `--text=FILE`：可选文本文件（例如转录），将与音频按样本 Key 对齐与分块。

### 示例

- 参考评估，8 并发，显式传入配置文件：
```
./versa/launch_local.sh data/pred.scp data/gt.scp results/exp1 16 --config=metrics.yaml --max-parallel=8
```

- 无参考评估，同时传入文本：
```
./versa/launch_local.sh data/pred.scp None results/exp2 32 --config=metrics.yaml --text=data/transcripts.txt
```

## 执行流程
1. 检查参数与输入文件；自动探测 GPU 数量（若 `nvidia-smi` 不可用或未报告 GPU，则默认使用 1 块 GPU）。
2. 使用内嵌 Python 脚本对 `pred/gt/text` 进行样本 Key 对齐，生成过滤后的文件。
3. 将过滤后的文件按 `split_size` 分块到 `score_dir/pred`, `score_dir/gt`, `score_dir/text`。
4. 逐块提交 GPU 任务：
   - 按 `--max-parallel` 控制总体并发；达到上限时等待任意任务完成。
   - 按 **轮询** 分配 GPU 编号（`rank = 块序号 % GPU_COUNT`），确保长期均衡。
   - 调用 `egs/run_gpu.sh` 执行实际评估，并将结果写到 `score_dir/result/*.result.gpu.txt`。
5. 等待所有任务完成后，输出成功/失败统计与各 GPU 的任务完成数，并生成 `merge_results.sh` 以合并结果。

## 输出结构
```
score_dir/
├─ pred/              # 分块后的预测列表
├─ gt/                # 分块后的参考列表（可选）
├─ text/              # 分块后的文本（可选）
├─ result/            # 各块生成的结果文件（*.result.gpu.txt）
├─ logs/              # 每个块的运行日志
├─ running_jobs.txt   # 运行中的任务记录
├─ completed_jobs.txt # 成功完成的任务记录
└─ failed_jobs.txt    # 失败任务记录
```

## 合并结果
脚本在结束时会生成 `merge_results.sh`，可用于合并所有分块结果：
```
./<score_dir>/merge_results.sh <score_dir>/result <score_dir>/final_results.txt
```
- 将生成 `<score_dir>/final_results.gpu.txt`（包含所有 `*.result.gpu.txt` 的拼接）。

## 注意事项
- `--max-parallel` 仅控制总体并发上限；任务分配到 GPU 采用轮询策略，无需等待某块 GPU 完全空闲。
- 如果传入 `--text`，脚本会确保与音频按样本 Key 对齐，否则报错。
- `egs/run_gpu.sh` 的接口需兼容本脚本的调用顺序：
  ```
  run_gpu.sh <pred_scp> <gt_scp|None> <output_file> <config_file> <io_type> <text_file> <gpu_rank>
  ```
- 默认 `IO_TYPE` 为 `soundfile`，如需调整可通过环境变量 `IO_TYPE` 传入。

## 依赖
- Bash
- Python 3（用于对齐过滤）
- `nvidia-smi`（用于探测 GPU；若不可用则默认 1 块 GPU）

## 常见问题
- 报错 “No overlapping entries”：表示 `pred/gt/text` 三者的样本 Key 无交集，请检查输入文件的第一列（Key）是否一致。
- 任务失败可在 `logs/` 中查看对应块的 `.log` 文件排查具体原因。
