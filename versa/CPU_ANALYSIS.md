# Versa 指标 CPU 占用分析报告

## 概述
本报告分析了 `whisper_wer`、`speaker` 和 `pseudo_mos` 三个指标在测评时的 CPU 资源占用情况。

---

## 1. whisper_wer（Whisper 语音识别率）

### 核心操作
- **文件位置**: [versa/corpus_metrics/whisper_wer.py](versa/corpus_metrics/whisper_wer.py)
- **主要函数**: `whisper_wer_setup()` 和 `whisper_levenshtein_metric()`

### CPU 占用关键点

#### a) **音频重采样** (librosa.resample)
```python
# 行 64-65
if fs != TARGET_FS:
    pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=TARGET_FS)
```
**占用程度**: 🔴 **高**
- **原因**: librosa 的 resample 是纯 CPU 操作，使用 scipy 的默认重采样算法
- **影响**: 在非 16kHz 采样率的音频处理中会产生大量 CPU 计算
- **优化建议**:
  - 使用更快的库如 `librosa` 的 `res_type='kaiser_best'` 改为 `res_type='kaiser_fast'`
  - 或使用 `julius` 库（更高效）或 NVIDIA 的 GPU 加速库

#### b) **Whisper 模型推理**
```python
# 行 66-70
with torch.no_grad():
    inf_text = wer_utils["model"].transcribe(
        torch.tensor(pred_x).float(), beam_size=wer_utils["beam_size"]
    )["text"]
```
**占用程度**: 🔴 **超高**
- **原因**: OpenAI Whisper 是大型多层 Transformer 模型，默认使用 large 模型
- **CPU 占用**: 
  - 虽然代码指定了 `device='cuda'` 时使用 GPU，但当 `use_gpu=False` 时完全依赖 CPU
  - 音频长段处理时，长序列 self-attention 的复杂度为 O(n²)
- **具体开销**:
  - Whisper Large：~1.5B 参数
  - 处理 30 秒音频需要多次矩阵乘法运算
  - beam search（默认 beam_size=5）会进行多路径搜索，增加 5 倍计算量
- **优化建议**:
  - **使用 GPU 推理** (强烈推荐)
  - 降低 beam_size（从 5 改为 1 或 2）
  - 使用小型 Whisper 模型（tiny/base 而非 large）
  - 实现音频分块处理（当前已有 CHUNK_SIZE=30，但需要验证是否生效）

#### c) **文本清洗和 Levenshtein 距离计算**
```python
# 行 71-88
ref_text = wer_utils["cleaner"](ref_text).strip()
pred_text = wer_utils["cleaner"](inf_text).strip()
# ... Levenshtein 计算
for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
```
**占用程度**: 🟡 **中等**
- **原因**: Levenshtein 距离的动态规划算法复杂度为 O(mn)，m、n 为词数
- **实际影响**: 相对较小（通常词数 < 1000），但在大量批处理时累积

### 总结
**whisper_wer 最大的 CPU 瓶颈是 Whisper 模型推理**，特别是在 CPU-only 环境下。

---

## 2. speaker（说话人相似度）

### 核心操作
- **文件位置**: [versa/utterance_metrics/speaker.py](versa/utterance_metrics/speaker.py)
- **主要函数**: `speaker_model_setup()` 和 `speaker_metric()`

### CPU 占用关键点

#### a) **音频重采样** (librosa.resample)
```python
# 行 35-37
if fs != 16000:
    gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=16000)
    pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)
```
**占用程度**: 🔴 **高**
- **原因**: 与 whisper_wer 相同，librosa resample 是 CPU 密集操作
- **影响**: 对每个待评估音频都需要重采样两次（gt 和 pred）
- **优化建议**: 同 whisper_wer，使用 `res_type='kaiser_fast'` 或其他加速库

#### b) **说话人嵌入提取**
```python
# 行 39-40
embedding_gen = model(pred_x).squeeze(0).cpu().numpy()
embedding_gt = model(gt_x).squeeze(0).cpu().numpy()
```
**占用程度**: 🔴 **高**
- **模型**: espnet/voxcelebs12_rawnet3（RawNet3 架构）
- **参数量**: 约 300-500M 参数
- **CPU 占用**: 
  - 模型使用 `Speech2Embedding` 进行推理
  - 涉及多层卷积和池化操作
  - 两个音频各需要一次完整前向传播
  - 当 `device='cpu'` 时，所有计算都在 CPU 上
- **优化建议**:
  - **使用 GPU 推理** (强烈推荐)
  - 考虑使用更轻量的说话人提取模型（如果可用）

#### c) **相似度计算**
```python
# 行 41-43
similarity = np.dot(embedding_gen, embedding_gt) / (
    np.linalg.norm(embedding_gen) * np.linalg.norm(embedding_gt)
)
```
**占用程度**: 🟢 **低**
- **原因**: 向量点积和范数计算很快（嵌入维度通常 < 2048）

### 总结
**speaker 的 CPU 瓶颈是模型推理**，需要对两条音频分别进行特征提取。

---

## 3. pseudo_mos（伪 MOS 评分）

### 核心操作
- **文件位置**: [versa/utterance_metrics/pseudo_mos.py](versa/utterance_metrics/pseudo_mos.py)
- **主要函数**: `pseudo_mos_setup()` 和 `pseudo_mos_metric()`

### CPU 占用关键点

#### a) **多个预测器加载** (Setup 阶段)
```python
# 行 43-125
# - utmos (torch.hub.load)
# - utmosv2 (条件加载)
# - dnsmos, plcmos (onnxruntime 推理)
# - singmos_v1, singmos_pro
# - dnsmos_pro_*
```
**占用程度**: 🟡 **中等** (setup 阶段) / 🔴 **高** (推理阶段)
- **问题**: 该函数会加载多个预测器模型，每个模型可能都是百万级参数
- **内存和 CPU 占用**: 累积效应显著

#### b) **音频重采样** (librosa.resample)
```python
# 行 145-147, 190-192, 210-212, 等多处
if fs != predictor_fs["utmos"]:
    pred_utmos = librosa.resample(
        pred, orig_sr=fs, target_sr=predictor_fs["utmos"]
    )
```
**占用程度**: 🔴 **高**
- **原因**: 对 **每个预测器** 都可能需要重采样（如果采样率不匹配）
- **影响**: 如果启用 5 个预测器，最多可能重采样 5 次
- **优化建议**:
  - 在 setup 阶段确保所有预测器使用统一采样率
  - 或在输入时提前统一重采样，而非每次都做

#### c) **UTMOSv2 频谱处理**
```python
# 行 160-168
cfg = predictor_dict["utmosv2"].cfg
spec_info = process_audio_only_versa(pred_utmosv2, cfg)
spec_info = torch.tensor(spec_info).float().unsqueeze(0)
# ... (NUM_REPETITIONS = 5)
for i in range(NUM_REPETITIONS):
    score_info.append(
        predictor_dict["utmosv2"](pred_tensor.float(), spec_info, d)
    )
```
**占用程度**: 🔴 **超高**
- **原因 1**: `process_audio_only_versa()` 涉及频谱处理（STFT、Mel 变换等）
  - STFT 的计算复杂度为 O(n log n)（FFT）
  - 这是 CPU 密集操作
- **原因 2**: **重复推理 5 次**
  ```python
  NUM_REPETITIONS = 5  # 硬编码！
  ```
  - 这意味着同一个音频被推理 5 次然后取平均
  - 相当于推理成本增加 5 倍
  - 这看起来是为了提高稳定性，但代价巨大
- **优化建议**:
  - 将 `NUM_REPETITIONS` 从 5 降低到 1 或 2
  - 或使用条件标志控制是否需要多次推理
  - 使用 GPU 加速频谱处理和模型推理

#### d) **多预测器推理** (整体)
```python
# 伪代码：
for predictor in predictor_dict.keys():
    if predictor == "utmos":
        # 推理 utmos...
    elif predictor == "utmosv2":
        # 推理 utmosv2（5 次）...
    elif predictor == "dnsmos":
        # 推理 dnsmos...
    # ... 还有其他预测器
```
**占用程度**: 🔴 **高**
- **原因**: 串行执行多个预测器
- **每个预测器的成本**:
  - UTMOS: ~200M 参数
  - UTMOSv2: ~300M 参数
  - SingMOS: ~200M 参数
  - DNSMOS: ONNX 模型
  - 等等...
- **并发问题**: 所有预测器共享 CPU，如果不使用 GPU，计算完全序列化

### 总结
**pseudo_mos 的 CPU 瓶颈最多**：
1. **频谱处理** (process_audio_only_versa)
2. **多次推理** (NUM_REPETITIONS = 5)
3. **多个模型加载和推理**
4. **重复重采样**

---

## 整体 CPU 占用排序

| 指标 | CPU 占用程度 | 主要瓶颈 |
|------|-------------|---------|
| **pseudo_mos** | 🔴🔴🔴 **极高** | 多预测器 + 5 次推理 + 频谱处理 |
| **whisper_wer** | 🔴🔴 **高** | Whisper 大模型推理 + 音频重采样 |
| **speaker** | 🔴 **高** | RawNet3 模型推理 + 音频重采样 |

---

## 优化建议总结

### 🎯 即时可实施的优化

#### 1. **启用 GPU 推理** (最有效)
```python
# 在 scorer.py 中确保 use_gpu=True
--use_gpu True
```
- **效果**: 可将这三个指标的速度提升 5-20 倍
- **前提**: 需要 NVIDIA GPU 和 CUDA 环境

#### 2. **降低 Whisper beam_size** (whisper_wer)
```yaml
# 在配置文件中修改
- name: whisper_wer
  beam_size: 1  # 从默认 5 改为 1（牺牲少量准确性换速度）
```
- **效果**: 降低 5 倍推理成本
- **精度影响**: 通常 < 1% WER 增长

#### 3. **降低 UTMOSv2 重复次数** (pseudo_mos)
**创建 patch 修改 pseudo_mos.py**:
```python
# 第 175 行附近
NUM_REPETITIONS = 1  # 改为 1（或加配置参数）
```
- **效果**: 降低 5 倍 pseudo_mos 的成本
- **精度影响**: 可能有 0.1-0.2 的 MOS 波动

#### 4. **使用快速重采样** (所有指标)
```python
# 在 speaker.py, whisper_wer.py 等中修改
pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=TARGET_FS, res_type='kaiser_fast')
```
- **效果**: 降低 2-3 倍重采样开销
- **精度影响**: 通常可接受

### 🔧 中期优化

#### 5. **批量处理和缓存**
- 在 `list_scoring()` 中实现批处理，减少函数调用开销
- 缓存重采样后的音频，避免重复处理

#### 6. **条件化预测器加载**
```python
# pseudo_mos_setup() 中，按需加载预测器而非全部加载
# 当前代码在 setup 阶段就加载全部，可以改为懒加载
```

#### 7. **多进程并行化**
- 使用 scorer.py 中的 `--num_shards` 功能
- 在多 GPU 或多 CPU 环境下并行处理不同的音频

### 🚀 长期优化

#### 8. **模型蒸馏**
- 为 whisper_wer、speaker 提供蒸馏后的小模型版本
- 保留高精度同时减少 50% 参数

#### 9. **混合精度推理**
- 使用 float16 或 int8 量化，利用 GPU 的 Tensor Core
- 需要修改 `speaker_metric()`, `pseudo_mos_metric()` 等

#### 10. **GPU 内存优化**
- 实现梯度检查点（虽然这里是推理而非训练）
- 减少中间激活的存储

---

## 诊断命令

### 查看 CPU/GPU 占用
```bash
# 方法 1：使用 htop
htop

# 方法 2：使用 nvidia-smi（仅 GPU）
watch -n 1 nvidia-smi

# 方法 3：使用 psutil（Python）
pip install psutil
python -c "import psutil; print(psutil.cpu_percent(interval=1))"
```

### 分析代码性能
```bash
# 已在 scorer.py 中加入 cProfile
python -m cProfile -s cumulative versa/bin/scorer.py --pred <pred.scp> --score_config <config.yaml>
```

---

## 结论

**pseudo_mos > whisper_wer > speaker** 的 CPU 占用顺序

最有效的优化是：
1. **启用 GPU** 🎯
2. **降低 pseudo_mos 的 NUM_REPETITIONS** 
3. **降低 whisper_wer 的 beam_size**
4. **使用快速重采样算法**
