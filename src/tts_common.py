# tts_common.py
"""
TTS通用模块 - 提供TTS合成的通用框架和工具函数

主要组件:
- BaseTTSProcessor: TTS处理器基类，封装迭代合成、GPU分配、并行处理等通用逻辑
- 工具函数: read_jsonl_file等
"""
import os
import sys
import json
import argparse
import multiprocessing
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import torch
import torchaudio
from tqdm import tqdm


def read_kaldi_data(ref_wav_scp, ref_text_path, target_text_path):
    """
    读取Kaldi格式的数据文件 (text, wav.scp)
    
    参数:
        ref_wav_scp: 参考音频scp文件
        ref_text_path: 参考文本文件
        target_text_path: 目标文本文件
        
    返回:
        entries: 包含uid, audio_path, ref_text, text的字典列表
    """
    data_map = {}
    
    try:
        # 读取ref_wav.scp
        with open(ref_wav_scp, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    uid, path = parts
                    if uid not in data_map: data_map[uid] = {}
                    data_map[uid]['uid'] = uid
                    data_map[uid]['audio_path'] = path

        # 读取ref_text
        with open(ref_text_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    uid, text = parts
                    if uid in data_map:
                        data_map[uid]['ref_text'] = text

        # 读取target_text
        with open(target_text_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    uid, text = parts
                    if uid in data_map:
                        data_map[uid]['text'] = text
                        
    except Exception as e:
        print(f"读取数据文件失败: {e}")
        return []
        
    # 过滤完整条目
    entries = []
    for uid, item in data_map.items():
        if 'audio_path' in item and 'ref_text' in item and 'text' in item:
            entries.append(item)
            
    return entries


def get_worker_gpu_id(num_gpus: int) -> Tuple[int, int, str]:
    """
    根据当前进程获取分配的GPU ID
    
    参数:
        num_gpus: 可用GPU数量
        
    返回:
        (worker_idx, gpu_id, device_str): 工作进程索引、GPU ID、设备字符串
    """
    worker_info = multiprocessing.current_process()._identity
    worker_idx = worker_info[0] - 1 if worker_info else 0
    
    if num_gpus > 0:
        gpu_id = worker_idx % num_gpus
        device_str = f"cuda:{gpu_id}"
    else:
        gpu_id = -1
        device_str = "cpu"
    
    return worker_idx, gpu_id, device_str


class BaseTTSProcessor(ABC):
    """
    TTS处理器基类，封装通用的并行处理、迭代合成逻辑
    
    子类需要实现:
        - load_model(device: str) -> Any: 加载模型
        - synthesize(model, prompt_wav, prompt_text, text, output_path) -> bool: 单次合成
    """
    
    # 类级别配置（子类可覆盖）
    model_name: str = "BaseTTS"
    def __init__(
        self,
        num_workers: int = 4,
        max_iterations: int = 10,
        model_path: Optional[str] = None,
        **kwargs
    ):
        """
        初始化TTS处理器
        
        参数:
            num_workers: 工作进程数
            max_iterations: 最大迭代次数
            model_path: 模型路径
            **kwargs: 子类特定参数
        """
        self.num_workers = num_workers
        self.max_iterations = max_iterations
        self.model_path = model_path
        
        # 获取可用GPU数量
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        self._print_config()
    
    def _print_config(self):
        """打印配置信息"""
        print(f"=== {self.model_name} 配置 ===")
        print(f"工作进程数: {self.num_workers}")
        print(f"最大迭代次数: {self.max_iterations}")
        print(f"模型路径: {self.model_path}")
        print(f"可用GPU数量: {self.num_gpus}")
        print("=" * 40)
    
    @abstractmethod
    def load_model(self, device: str) -> Any:
        """
        加载模型（子类必须实现）
        
        参数:
            device: 设备字符串，如 "cuda:0" 或 "cpu"
            
        返回:
            加载的模型对象
        """
        pass
    
    @abstractmethod
    def synthesize(
        self,
        model: Any,
        prompt_wav: str,
        prompt_text: str,
        text: str,
        output_path: str
    ) -> bool:
        """
        执行单次TTS合成（子类必须实现）
        
        参数:
            model: 模型对象
            prompt_wav: 参考音频路径
            prompt_text: 参考文本
            text: 待合成文本
            output_path: 输出文件路径
            
        返回:
            bool: 合成是否成功
        """
        pass
    
    def setup_environment(self):
        """
        设置环境（子类可覆盖）
        在加载模型前调用，用于添加路径等
        """
        pass
    
    def process_data(
        self,
        ref_wav_scp: str,
        ref_text_path: str,
        target_text_path: str,
        output_dir: str,
        max_entries: Optional[int] = None
    ) -> int:
        """
        并行处理数据
        
        参数:
            ref_wav_scp: 参考音频scp路径
            ref_text_path: 参考文本路径
            target_text_path: 目标文本路径
            output_dir: 输出目录
            max_entries: 最大处理条目数
            
        返回:
            success_count: 成功处理的条目数
        """
        output_path = Path(output_dir)
        
        # 读取数据
        print("正在读取数据...")
        entries = read_kaldi_data(ref_wav_scp, ref_text_path, target_text_path)
        if not entries:
            print("错误: 无法读取数据文件或为空")
            return 0
        
        if max_entries is not None:
            entries = entries[:max_entries]
        
        total_entries = len(entries)
        print(f"共读取 {total_entries} 个条目")
        
        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 准备数据项
        data_items = self._prepare_data_items(entries, output_dir)
        
        # 准备初始化参数
        init_args = self._build_init_args()
        
        # 创建进程池并处理
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(
            processes=self.num_workers,
            initializer=_static_initializer,
            initargs=init_args
        ) as pool:
            print("开始并行处理...")
            results = []
            with tqdm(total=len(data_items), desc="处理进度") as pbar:
                for result in pool.imap(_static_process_entry, data_items):
                    results.append(result)
                    pbar.update(1)
        
        # 统计结果
        success_count = sum(1 for r in results if (r[0] if isinstance(r, tuple) else r))
        return success_count
    
    def _prepare_data_items(
        self,
        entries: List[Dict],
        output_dir: str
    ) -> List[Tuple]:
        """准备处理数据项"""
        data_items = []
        for entry in entries:
            # entry now contains everything needed
            data_items.append((entry, output_dir))
        return data_items
    
    def _build_init_args(self) -> tuple:
        """构建进程初始化参数"""
        base_args = (
            self.__class__.__module__,
            self.__class__.__name__,
            self.model_path,
            self.max_iterations,
            self.num_gpus,
        )
        return base_args
    
    @classmethod
    def add_common_args(cls, parser: argparse.ArgumentParser):
        """添加通用命令行参数"""
        parser.add_argument('--ref_wav', required=True, help='参考音频scp路径')
        parser.add_argument('--ref_text', required=True, help='参考文本路径')
        parser.add_argument('--target_text', required=True, help='目标文本路径')
        parser.add_argument('--output', required=True, help='输出文件夹路径')
        parser.add_argument('--max_entries', type=int, default=None, help='最大处理条目数')
        parser.add_argument('--max_iterations', type=int, default=10, help='最大迭代次数')
        parser.add_argument('--num_workers', type=int, default=4, help='并行处理的进程数')
        parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    
    @classmethod
    def main(cls):
        """
        通用main函数入口
        """
        parser = argparse.ArgumentParser(description=f'使用 {cls.model_name} 迭代合成语音')
        
        # 添加通用参数
        cls.add_common_args(parser)
        
        args = parser.parse_args()
        
        # 检查输入文件
        if not os.path.exists(args.ref_wav):
            print(f"错误: 参考音频文件 '{args.ref_wav}' 不存在")
            return
        if not os.path.exists(args.ref_text):
            print(f"错误: 参考文本文件 '{args.ref_text}' 不存在")
            return
        if not os.path.exists(args.target_text):
            print(f"错误: 目标文本文件 '{args.target_text}' 不存在")
            return
        
        # 创建输出目录
        os.makedirs(args.output, exist_ok=True)
        
        print(f"开始使用 {cls.model_name} 进行语音合成...")
        print(f"参考音频: {args.ref_wav}")
        print(f"参考文本: {args.ref_text}")
        print(f"目标文本: {args.target_text}")
        print(f"输出目录: {args.output}")
        
        # 构建处理器参数
        processor_kwargs = {
            'num_workers': args.num_workers,
            'max_iterations': args.max_iterations,
            'model_path': args.model_path,
        }
        
        # 创建处理器并执行
        processor = cls(**processor_kwargs)
        
        success_count = processor.process_data(
            ref_wav_scp=args.ref_wav,
            ref_text_path=args.ref_text,
            target_text_path=args.target_text,
            output_dir=args.output,
            max_entries=args.max_entries
        )
        
        print(f"\n处理完成! 共成功处理 {success_count} 个音频文件")


# ============ 全局静态函数（用于multiprocessing） ============

# 全局变量（进程级别）
_tts_model = None
_tts_processor_class = None
_max_iterations = None

def _static_initializer(
    module_name: str,
    class_name: str,
    model_path: str,
    max_iterations: int,
    num_gpus: int,
):
    """
    静态进程初始化函数
    """
    global _tts_model, _tts_processor_class, _max_iterations
    
    try:
        # 动态导入并获取处理器类
        import importlib
        module = importlib.import_module(module_name)
        processor_class = getattr(module, class_name)
        
        # 获取GPU分配
        worker_idx, gpu_id, device = get_worker_gpu_id(num_gpus)
        
        if num_gpus > 0:
            print(f"进程 {os.getpid()} (Worker {worker_idx}) 使用 GPU {gpu_id}")
        else:
            print(f"进程 {os.getpid()} 未检测到GPU，使用CPU")
        
        # 创建临时实例用于加载模型
        temp_processor = processor_class.__new__(processor_class)
        temp_processor.model_path = model_path
        
        # 设置环境
        temp_processor.setup_environment()
        
        # 加载模型
        print(f"进程 {os.getpid()} 正在加载 {processor_class.model_name} 模型...")
        _tts_model = temp_processor.load_model(device)
        print(f"进程 {os.getpid()} {processor_class.model_name} 模型加载完成")
        
        _tts_processor_class = processor_class
        _max_iterations = max_iterations
        
    except Exception as e:
        print(f"进程 {os.getpid()} 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        _tts_model = None


def _static_process_entry(data: Tuple) -> Tuple[bool, Optional[str], str]:
    """
    静态处理单个条目
    """
    global _tts_model, _tts_processor_class, _max_iterations
    
    entry, output_dir = data
    
    if _tts_model is None:
        return (False, None, "模型未初始化")
    
    # 获取条目信息
    entry_id = entry.get('uid', '')
    ref_audio_path = entry.get('audio_path', '')
    ref_text = entry.get('ref_text', '')
    text = entry.get('text', '')
    print(f"Processing {entry_id}: {text}")
    
    if not entry_id or not ref_audio_path or not text:
        return (False, None, "条目缺少必要字段(uid, audio_path, text)")
    
    try:
        # 检查参考音频
        if not os.path.exists(ref_audio_path):
            print(f"参考音频文件不存在: {ref_audio_path}")
            return (False, None, f"参考音频文件不存在: {ref_audio_path}")
        
        # 创建输出目录
        audio_dir = Path(output_dir) / entry_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建临时处理器实例用于调用方法
        temp_processor = _tts_processor_class.__new__(_tts_processor_class)
        
        # 设置首次迭代的参考音频与文本
        prompt_wav = ref_audio_path
        prompt_text = ref_text
        
        # 迭代合成
        for iteration in range(_max_iterations):
            print(f"正在处理条目 {entry_id}, 迭代次数: {iteration + 1}")
            iteration_output_path = audio_dir / f"iter_{iteration + 1}.wav"
            
            if iteration_output_path.exists():
            #if iteration_output_path.exists() and torchaudio.info(str(iteration_output_path)).num_frames / 24000 > 2:
                print(f"跳过已存在的迭代结果: {iteration_output_path}")
                prompt_wav = str(iteration_output_path)
                prompt_text = text
                continue
            
            # 执行合成
            success = temp_processor.synthesize(
                model=_tts_model,
                prompt_wav=str(prompt_wav),
                prompt_text=prompt_text,
                text=text,
                output_path=str(iteration_output_path),
            )
            
            if not success:
                print(f"条目 {entry_id} 第 {iteration + 1} 次迭代合成失败")
                return (False, None, f"迭代 {iteration + 1} 合成失败")
            
            # 准备下一次迭代
            prompt_wav = str(iteration_output_path)
            prompt_text = text
        
        return (True, str(audio_dir / f"iter_{_max_iterations}.wav"), "成功")
        
    except Exception as e:
        print(f"处理条目 {entry_id} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return (False, None, str(e))