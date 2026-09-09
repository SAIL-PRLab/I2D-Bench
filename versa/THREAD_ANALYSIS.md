# pseudo_mos_metric 线程创建详细分析

## 主要线程来源分解

在 `pseudo_mos_metric()` 函数中，以下操作会创建大量线程：

### 1️⃣ **librosa.resample() - 音频重采样 [主要线程来源]**
```python
if fs != predictor_fs["utmosv2"]:
    pred_utmosv2 = librosa.resample(pred, orig_sr=fs, target_sr=...)
```

**线程创建原因**：
- librosa 使用 SciPy，SciPy 使用多线程的 FFTPACK/BLAS
- 每次调用 `librosa.resample()` 会触发 FFT 操作
- 在你的代码中，**如果使用多个预测器（utmosv2 + dnsmos + ...），会调用多次 resample()**

**出现次数**：
- `utmosv2`: 1次
- `dnsmos`: 1次  
- `plcmos`: 1次
- `singmos_v1`: 1次
- `singmos_pro`: 1次
- `dnsmos_pro_*`: 1次
- **总计：最多 6 次调用**

---

### 2️⃣ **librosa.stft() - 短时傅里叶变换**
```python
spec = librosa.stft(
    y=samples,
    win_length=win_length,
    hop_length=hop_length,
    n_fft=n_fft,
)
```

**线程创建原因**：
- STFT 是复杂的 FFT 操作，大量并行线程
- 在 `dnsmos_pro_*` 处理流程中调用

---

### 3️⃣ **process_audio_only_versa() - UTMOS v2 特征提取 [重线程操作]**
```python
spec_info = process_audio_only_versa(pred_utmosv2, cfg)
```

**线程创建原因**：
- 这个函数内部会调用 librosa 的多个信号处理函数
- 包括 STFT、梅尔频谱图等
- 是 **utmosv2 中最耗线程的步骤**

---

### 4️⃣ **model forward pass - 神经网络推理**
```python
predictor_dict["utmosv2"](pred_tensor.float(), spec_info, d)  # NUM_REPETITIONS = 5
```

**线程创建原因**：
- PyTorch 默认使用多线程进行张量操作
- BLAS/LAPACK 后端使用多个线程
- **utmosv2 循环 5 次** 会放大线程使用
- ONNX Runtime (dnsmos) 也创建工作线程

---

### 5️⃣ **torch.from_numpy() + to("cuda")**
```python
pred_tensor = torch.from_numpy(pred_utmosv2).unsqueeze(0)
pred_tensor = pred_tensor.to("cuda")  # GPU 内存分配
```

**线程创建原因**：
- PyTorch CUDA 操作涉及多个后台线程

---

## 线程峰值时刻

**最大线程数通常发生在**：
```
1. librosa.resample() 执行 FFT
   ↓
2. process_audio_only_versa() 提取特征
   ↓  
3. utmosv2 model forward pass (x5)
   ↓
4. dnsmos ONNX 推理
   ↓
5. librosa.stft() for dnsmos_pro
```

多个操作**并发执行或快速切换**，导致线程数堆积。

---

## 代码中的具体行号

| 操作 | 行号 | 线程数影响 |
|-----|------|---------|
| utmosv2 resample | ~145-150 | 中等 |
| process_audio_only_versa | ~153 | **高** |
| utmosv2 forward x5 | ~164-171 | **非常高** |
| dnsmos resample | ~180 | 中等 |
| dnsmos.run() | ~185 | **高** |
| dnsmos_pro stft | ~220-224 | **高** |

---

## 优化建议

### ✅ 方案 1：限制线程数（已实施）
在文件开头设置：
```python
os.environ["OMP_NUM_THREADS"] = "2"  # 降低到 2
os.environ["MKL_NUM_THREADS"] = "2"
torch.set_num_threads(2)
```

### ✅ 方案 2：减少 utmosv2 重复次数
```python
# 改为 1 次而不是 5 次
NUM_REPETITIONS = 1  # 改这里
```

### ✅ 方案 3：使用批处理模式
如果需要处理多个样本，使用批处理而不是循环

### ✅ 方案 4：使用 ONNX CPU 推理代替 PyTorch
- ONNX Runtime 通常比 PyTorch 更高效
- 线程管理也更轻量

---

## 验证修改效果

运行后对比：
```bash
# 修改前
# 线程数: 250+
# CPU: 3470.8%

# 修改后（预期）
# 线程数: 20-50
# CPU: 200-400%
```
