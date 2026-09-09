#!/usr/bin/env python3

# Copyright 2023 Takaaki Saeki
# Copyright 2024 Jiatong Shi
# Copyright 2025 Jionghao Han
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import logging

logger = logging.getLogger(__name__)

import librosa
import numpy as np
import torch
import requests
from pathlib import Path
from typing import Optional
import psutil
import time


class ThreadMonitor:
    """线程监控工具"""
    def __init__(self):
        self.process = psutil.Process()
        self.thread_log = []
        
    def record(self, operation_name):
        """记录当前线程数"""
        num_threads = self.process.num_threads()
        self.thread_log.append({
            'operation': operation_name,
            'threads': num_threads,
            'timestamp': time.time()
        })
        print(f"[{operation_name}] 线程数: {num_threads}")
        return num_threads
    
    def print_report(self):
        """打印监控报告"""
        print("\n" + "="*60)
        print("线程创建详细报告")
        print("="*60)
        for i, log in enumerate(self.thread_log):
            if i > 0:
                delta = log['threads'] - self.thread_log[i-1]['threads']
                delta_str = f"(+{delta})" if delta > 0 else f"({delta})"
            else:
                delta_str = ""
            print(f"{i+1:2d}. {log['operation']:30s} | 线程: {log['threads']:3d} {delta_str}")
        
        if self.thread_log:
            max_threads = max(log['threads'] for log in self.thread_log)
            min_threads = min(log['threads'] for log in self.thread_log)
            print("-"*60)
            print(f"最大线程数: {max_threads}")
            print(f"最小线程数: {min_threads}")
            print(f"增长总数: {max_threads - min_threads}")
        print("="*60 + "\n")
try:
    import utmosv2
    from utmosv2.dataset.multi_spec import process_audio_only_versa
except ImportError:
    logger.info(
        "utmosv2 is not installed, please install via `tools/install_utmosv2.sh`"
    )
    utmosv2 = None


def pseudo_mos_setup(
    predictor_types, predictor_args, cache_dir="versa_cache", use_gpu=True
):
    os.environ["OMP_NUM_THREADS"] = "4"  # 限制 OpenMP 线程数
    os.environ["MKL_NUM_THREADS"] = "4"  # 限制 MKL 线程数
    torch.set_num_threads(4)  # 限制 PyTorch 线程数
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
        elif predictor == "utmos" or predictor == "utmosv2":
            continue  # already initialized
        else:
            raise NotImplementedError("Not supported {}".format(predictor))

    return predictor_dict, predictor_fs


def pseudo_mos_metric(pred, fs, predictor_dict, predictor_fs, use_gpu=True):
    scores = {}
    for predictor in predictor_dict.keys():
        if predictor == "utmosv2":
            if fs != predictor_fs["utmosv2"]:
                pred_utmosv2 = librosa.resample(
                    pred, orig_sr=fs, target_sr=predictor_fs["utmosv2"]
                )
            else:
                pred_utmosv2 = pred

            if utmosv2 is not None:
                cfg = predictor_dict["utmosv2"].cfg
                spec_info = process_audio_only_versa(pred_utmosv2, cfg)
                spec_info = torch.tensor(spec_info).float().unsqueeze(0)

                # magic number of data types
                # defined at https://github.com/ftshijt/UTMOSv2/blob/main/utmosv2/dataset/_utils.py#L8
                data_type = np.zeros(10)
                # we use general version dataset label: sarulab (1)
                data_type[1] = 0
                d = torch.tensor(data_type, dtype=torch.float32).unsqueeze(0)
                if use_gpu:
                    spec_info = spec_info.to("cuda")
                    d = d.to("cuda")
            else:
                raise RuntimeError(
                    "utmosv2 is not installed. Use tools/install_utmosv2.sh to install."
                )

            pred_tensor = torch.from_numpy(pred_utmosv2).unsqueeze(0)
            if use_gpu:
                pred_tensor = pred_tensor.to("cuda")

            NUM_REPETITIONS = 1
            with torch.no_grad():
                score_info = []
                for i in range(NUM_REPETITIONS):
                    score_info.append(
                        predictor_dict["utmosv2"](pred_tensor.float(), spec_info, d)
                        .squeeze(1)
                        .cpu()
                        .numpy()[0]
                    )
            scores.update(utmosv2=sum(score_info) / NUM_REPETITIONS)

        elif predictor == "dnsmos":
            if fs != predictor_fs["dnsmos"]:
                pred_dnsmos = librosa.resample(
                    pred, orig_sr=fs, target_sr=predictor_fs["dnsmos"]
                )
                fs = predictor_fs["dnsmos"]
            else:
                pred_dnsmos = pred

            max_val = np.max(np.abs(pred_dnsmos))
            score = predictor_dict["dnsmos"].run(pred_dnsmos / max_val, sr=fs)

            scores.update(dns_overall=score["ovrl_mos"], dns_p808=score["p808_mos"])

        else:
            raise NotImplementedError("Not supported {}".format(predictor))

    return scores

if __name__ == "__main__":
    a = np.random.random(16000)
    print(a)
    predictor_dict, predictor_fs = pseudo_mos_setup(
        [
            "utmosv2",
            "dnsmos",
        ],
        predictor_args={
            "dnsmos": {"fs": 16000},
        },
    )
    scores = pseudo_mos_metric(
        a, fs=16000, predictor_dict=predictor_dict, predictor_fs=predictor_fs
    )
    print("metrics: {}".format(scores))
