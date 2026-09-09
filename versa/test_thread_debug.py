#!/usr/bin/env python3

# Copyright 2023 Takaaki Saeki
# Copyright 2024 Jiatong Shi
# Copyright 2025 Jionghao Han
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import os
import logging

# 限制并行线程（在任何库导入前生效）
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

logger = logging.getLogger(__name__)

import librosa
import numpy as np
import torch
import requests
from pathlib import Path
from typing import Optional
import psutil
import time

try:
    import utmosv2
    from utmosv2.dataset.multi_spec import process_audio_only_versa
except ImportError:
    logger.info(
        "utmosv2 is not installed, please install via `tools/install_utmosv2.sh`"
    )
    utmosv2 = None


class ThreadMonitor:
    """线程和CPU资源监控工具 - 精确定位线程创建和CPU使用"""
    def __init__(self):
        self.process = psutil.Process()
        self.thread_log = []
        self.start_time = time.time()
        
    def record(self, operation_name, detail=""):
        """记录当前线程数和CPU使用情况"""
        num_threads = self.process.num_threads()
        elapsed = time.time() - self.start_time
        
        # 获取CPU使用率和内存信息
        cpu_percent = self.process.cpu_percent(interval=None)  # 非阻塞模式
        mem_info = self.process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024  # RSS 内存（MB）
        
        entry = {
            'operation': operation_name,
            'detail': detail,
            'threads': num_threads,
            'timestamp': elapsed,
            'cpu_percent': cpu_percent,
            'mem_mb': mem_mb
        }
        self.thread_log.append(entry)
        
        if self.thread_log:
            prev_threads = self.thread_log[-2]['threads'] if len(self.thread_log) > 1 else num_threads
            delta = num_threads - prev_threads
            delta_str = f" [+{delta}]" if delta != 0 else ""
            print(f"[{elapsed:6.2f}s] {operation_name:40s} | 线程: {num_threads:3d}{delta_str} | CPU: {cpu_percent:6.1f}% | 内存: {mem_mb:7.1f}MB")
        else:
            print(f"[{elapsed:6.2f}s] {operation_name:40s} | 线程: {num_threads:3d} | CPU: {cpu_percent:6.1f}% | 内存: {mem_mb:7.1f}MB")
    
    def print_report(self):
        """打印监控报告"""
        print("\n" + "="*120)
        print("线程和CPU资源监控详细报告")
        print("="*120)
        print(f"{'#':>3} | {'操作':40s} | {'线程':>5s} | {'变化':>6s} | {'CPU(%)':>7s} | {'内存(MB)':>10s} | {'耗时(s)':>8s}")
        print("-"*120)
        
        for i, log in enumerate(self.thread_log):
            if i > 0:
                delta = log['threads'] - self.thread_log[i-1]['threads']
                time_delta = log['timestamp'] - self.thread_log[i-1]['timestamp']
                delta_str = f"+{delta}" if delta > 0 else f"{delta}"
            else:
                delta_str = "-"
                time_delta = 0
            
            op = log['operation'][:40]
            print(f"{i+1:3d} | {op:40s} | {log['threads']:5d} | {delta_str:>6s} | {log['cpu_percent']:7.1f} | {log['mem_mb']:10.1f} | {time_delta:8.3f}")
        
        print("="*120)
        if self.thread_log:
            max_threads = max(log['threads'] for log in self.thread_log)
            min_threads = min(log['threads'] for log in self.thread_log)
            max_idx = next(i for i, log in enumerate(self.thread_log) if log['threads'] == max_threads)
            
            max_cpu = max(log['cpu_percent'] for log in self.thread_log)
            max_cpu_idx = next(i for i, log in enumerate(self.thread_log) if log['cpu_percent'] == max_cpu)
            
            max_mem = max(log['mem_mb'] for log in self.thread_log)
            min_mem = min(log['mem_mb'] for log in self.thread_log)
            max_mem_idx = next(i for i, log in enumerate(self.thread_log) if log['mem_mb'] == max_mem)
            
            avg_cpu = sum(log['cpu_percent'] for log in self.thread_log) / len(self.thread_log)
            avg_mem = sum(log['mem_mb'] for log in self.thread_log) / len(self.thread_log)
            
            print(f"\n📊 统计信息:")
            print(f"\n  线程统计:")
            print(f"    初始线程数: {self.thread_log[0]['threads']}")
            print(f"    最大线程数: {max_threads} (在步骤 #{max_idx+1})")
            print(f"    最小线程数: {min_threads}")
            print(f"    增长总数:   {max_threads - min_threads}")
            print(f"\n  CPU统计:")
            print(f"    最大CPU使用率: {max_cpu:.1f}% (在步骤 #{max_cpu_idx+1})")
            print(f"    平均CPU使用率: {avg_cpu:.1f}%")
            print(f"\n  内存统计:")
            print(f"    最大内存占用: {max_mem:.1f}MB (在步骤 #{max_mem_idx+1})")
            print(f"    最小内存占用: {min_mem:.1f}MB")
            print(f"    平均内存占用: {avg_mem:.1f}MB")
            print(f"    内存增长: {max_mem - min_mem:.1f}MB")
            print(f"\n  总耗时: {self.thread_log[-1]['timestamp']:.2f}s")
        print("="*120 + "\n")


def pseudo_mos_setup(
    predictor_types, predictor_args, cache_dir="versa_cache", use_gpu=True
):
    # Supported predictor types: utmos, dnsmos, aecmos, plcmos
    # Predictor args: predictor specific args
    predictor_dict = {}
    predictor_fs = {}
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    logging.info(f"Setting up pseudo MOS predictors on {device}")

    # first import utmos to resolve cross-import from the same model
    if "utmos" in predictor_types:
        torch.hub.set_dir(cache_dir)
        utmos = torch.hub.load("ftshijt/SpeechMOS:main", "utmos22_strong").to(device)
        predictor_dict["utmos"] = utmos.float()
        predictor_fs["utmos"] = 16000
    if "utmosv2" in predictor_types:
        if utmosv2 is None:
            raise RuntimeError(
                "utmosv2 is not installed. Please follow `tools/install_utmosv2.sh` to install"
            )
        # NOTE(jiatong): if you have an error of `_pickle.UnpicklingError: invalid load key, 'v'.`
        # It is likely that you did not have `git lfs` properly setup. Please check
        # https://github.com/sarulab-speech/UTMOSv2?tab=readme-ov-file#---quick-prediction--------
        utmos_v2 = utmosv2.create_model(pretrained=True)
        # _cfg = importlib.import_module(f"utmosv2.config.fusion_stage3")
        # cfg = SimpleNamespace(
        #     **{k: v for k, v in _cfg.__dict__.items() if not k.startswith("__")}
        # )
        # utmosv2._settings.configure_execution(cfg)
        predictor_dict["utmosv2"] = utmos_v2.to(device)
        predictor_fs["utmosv2"] = 16000

    if (
        "aecmos" in predictor_types
        or "dnsmos" in predictor_types
        or "plcmos" in predictor_types
    ):
        try:
            import onnxruntime as ort  # NOTE(jiatong): a requirement of aecmos but not in requirements
            # 强制 ONNX Runtime 使用单线程
            _orig_inference_session = ort.InferenceSession

            def _patched_inference_session(*args, **kwargs):
                if "sess_options" not in kwargs or kwargs["sess_options"] is None:
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 1
                    opts.inter_op_num_threads = 1
                    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    kwargs["sess_options"] = opts
                return _orig_inference_session(*args, **kwargs)

            ort.InferenceSession = _patched_inference_session
            from speechmos import dnsmos, plcmos
        except ImportError:
            raise ImportError(
                "Please install speechmos for dnsmos, and plcmos: pip install speechmos onnxruntime"
            )

    for predictor in predictor_types:
        if predictor == "dnsmos":
            predictor_dict["dnsmos"] = dnsmos
            if "dnsmos" not in predictor_args:
                predictor_fs["dnsmos"] = 16000
            else:
                predictor_fs["dnsmos"] = predictor_args["dnsmos"]["fs"]
        elif predictor == "plcmos":
            predictor_dict["plcmos"] = plcmos
            if "plcmos" not in predictor_args:
                predictor_fs["plcmos"] = 16000
            else:
                predictor_fs["plcmos"] = predictor_args["plcmos"]["fs"]
        elif predictor == "utmos" or predictor == "utmosv2":
            continue  # already initialized
        elif predictor == "singmos_v1":
            torch.hub.set_dir(cache_dir)
            singmos = torch.hub.load(
                "South-Twilight/SingMOS:v1.1.1", "singmos_v1", trust_repo=True
            ).to(device)
            predictor_dict["singmos_v1"] = singmos
            predictor_fs["singmos_v1"] = 16000
        elif predictor == "singmos_pro":
            torch.hub.set_dir(cache_dir)
            singmos = torch.hub.load(
                "South-Twilight/SingMOS:v1.1.1", "singmos_pro", trust_repo=True
            ).to(device)
            predictor_dict["singmos_pro"] = singmos
            predictor_fs["singmos_pro"] = 16000
        elif predictor.startswith("dnsmos_pro_"):
            variant = predictor[len("dnsmos_pro_") :]
            model_path = Path(cache_dir) / f"dnsmos_pro_{variant}.pt"
            if not model_path.exists():
                url = f"https://github.com/fcumlin/DNSMOSPro/raw/refs/heads/main/runs/{variant.upper()}/model_best.pt"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f'Downloading: "{url}" to {model_path}')
                response = requests.get(url)
                with open(model_path, "wb") as f:
                    f.write(response.content)
            else:
                logger.info(f"Using cached model: {model_path}")
            predictor_dict[predictor] = torch.jit.load(model_path, map_location=device)
            predictor_fs[predictor] = 16000
        else:
            raise NotImplementedError("Not supported {}".format(predictor))

    return predictor_dict, predictor_fs


def pseudo_mos_metric_debug(pred, fs, predictor_dict, predictor_fs, use_gpu=True):
    """带详细线程监控的 pseudo_mos_metric"""
    monitor = ThreadMonitor()
    monitor.record("═ 函数开始")
    
    scores = {}
    for predictor in predictor_dict.keys():
        monitor.record(f"→ 处理 {predictor} 预测器")
        if predictor == "utmosv2":
            monitor.record("  ↓ utmosv2 初始化")
            
            if fs != predictor_fs["utmosv2"]:
                monitor.record("    ↓ librosa.resample 前")
                pred_utmosv2 = librosa.resample(
                    pred, orig_sr=fs, target_sr=predictor_fs["utmosv2"]
                )
                monitor.record("    ↑ librosa.resample 后")
            else:
                pred_utmosv2 = pred

            if utmosv2 is not None:
                monitor.record("    ↓ process_audio_only_versa 前 [特征提取]")
                cfg = predictor_dict["utmosv2"].cfg
                spec_info = process_audio_only_versa(pred_utmosv2, cfg)
                monitor.record("    ↑ process_audio_only_versa 后 [特征提取]")
                
                spec_info = torch.tensor(spec_info).float().unsqueeze(0)
                monitor.record("    ↓ 张量转换")

                # magic number of data types
                data_type = np.zeros(10)
                data_type[1] = 0
                d = torch.tensor(data_type, dtype=torch.float32).unsqueeze(0)
                if use_gpu:
                    spec_info = spec_info.to("cuda")
                    d = d.to("cuda")
                monitor.record("    ↑ 张量转换到GPU")
            else:
                raise RuntimeError(
                    "utmosv2 is not installed. Use tools/install_utmosv2.sh to install."
                )

            pred_tensor = torch.from_numpy(pred_utmosv2).unsqueeze(0)
            monitor.record("    ↓ 创建输入张量")
            if use_gpu:
                pred_tensor = pred_tensor.to("cuda")
            monitor.record("    ↑ 输入张量到GPU")

            monitor.record("    ↓ utmosv2 模型推理 前 [最耗线程]")
            NUM_REPETITIONS = 5
            with torch.no_grad():
                score_info = []
                for i in range(NUM_REPETITIONS):
                    print(pred_tensor.device, spec_info.device, d.device)
                    monitor.record(f"      ↓ 推理循环 #{i+1}")
                    score_info.append(
                        predictor_dict["utmosv2"](pred_tensor.float(), spec_info, d)
                        .squeeze(1)
                        .cpu()
                        .numpy()[0]
                    )
                    monitor.record(f"      ↑ 推理循环 #{i+1}")
            monitor.record("    ↑ utmosv2 模型推理 后")
            scores.update(utmosv2=sum(score_info) / NUM_REPETITIONS)

        elif predictor == "dnsmos":
            monitor.record("  ↓ dnsmos 初始化")
            
            if fs != predictor_fs["dnsmos"]:
                monitor.record("    ↓ librosa.resample 前")
                pred_dnsmos = librosa.resample(
                    pred, orig_sr=fs, target_sr=predictor_fs["dnsmos"]
                )
                monitor.record("    ↑ librosa.resample 后")
                fs = predictor_fs["dnsmos"]
            else:
                pred_dnsmos = pred

            monitor.record("    ↓ dnsmos.run() 前 [ONNX推理]")
            max_val = np.max(np.abs(pred_dnsmos))
            score = predictor_dict["dnsmos"].run(pred_dnsmos / max_val, sr=fs)
            monitor.record("    ↑ dnsmos.run() 后 [ONNX推理]")

            scores.update(dns_overall=score["ovrl_mos"], dns_p808=score["p808_mos"])

        else:
            raise NotImplementedError("Not supported {}".format(predictor))
    
    monitor.record("═ 函数结束")
    monitor.print_report()
    return scores


if __name__ == "__main__":
    # 测试数据
    print("准备测试数据...")
    audio = np.random.random(16000 * 10)  # 5秒音频
    
    print("\n初始化预测器...")
    predictor_dict, predictor_fs = pseudo_mos_setup(
        ["utmosv2", "dnsmos"],
        predictor_args={"utmosv2": {"fs": 16000}, "dnsmos": {"fs": 16000}},
        use_gpu=True
    )
    
    print("\n" + "="*80)
    print("开始推理并监控线程")
    print("="*80)
    T = 1
    while T > 0:
        scores = pseudo_mos_metric_debug(
            audio, fs=16000, 
            predictor_dict=predictor_dict, 
            predictor_fs=predictor_fs,
            use_gpu=True
        )
        T -= 1
