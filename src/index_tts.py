"""
Index-TTS 处理器

使用Index-TTS模型进行迭代语音克隆合成
"""
import os
import sys
from typing import Any

import torch

from tts_common import BaseTTSProcessor


class IndexTTSProcessor(BaseTTSProcessor):
    """
    IndexTTS2处理器
    """
    
    model_name = "IndexTTS2"
    
    def setup_environment(self):
        """设置IndexTTS2环境"""
        indextts_path = os.path.expanduser('../TTS_Model/index-tts')
        if indextts_path not in sys.path:
            sys.path.insert(0, indextts_path)
    
    def load_model(self, device: str) -> Any:
        """
        加载IndexTTS2模型
        
        参数:
            device: 设备字符串
            
        返回:
            IndexTTS2模型实例
        """
        self.setup_environment()
        from indextts.infer_v2 import IndexTTS2
        
        model = IndexTTS2(
            cfg_path=f"{self.model_path}/config.yaml",
            model_dir=self.model_path,
            device=torch.device(device)
        )
        return model
    
    def synthesize(
        self,
        model: Any,
        prompt_wav: str,
        prompt_text: str,
        text: str,
        output_path: str,
    ) -> bool:
        """
        执行单次IndexTTS2合成
        
        参数:
            model: IndexTTS2模型
            prompt_wav: 参考音频路径
            prompt_text: 参考文本（IndexTTS2不使用）
            text: 待合成文本
            output_path: 输出文件路径
            
        返回:
            bool: 合成是否成功
        """
        try:
            model.infer(
                spk_audio_prompt=prompt_wav,
                text=text,
                output_path=output_path,
                verbose=False
            )
            return True
        except Exception as e:
            print(f"IndexTTS2合成失败: {e}")
            return False


if __name__ == "__main__":
    IndexTTSProcessor.main()
