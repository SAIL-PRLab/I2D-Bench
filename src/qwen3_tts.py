"""
Qwen3-TTS 处理器

使用Qwen3-TTS模型进行迭代语音克隆合成
"""
import os
import sys
from typing import Any

import torch
import soundfile as sf

from tts_common import BaseTTSProcessor


class Qwen3TTSProcessor(BaseTTSProcessor):
    """
    Qwen3-TTS处理器
    """
    
    model_name = "Qwen3-TTS"
    
    def setup_environment(self):
        """设置Qwen3-TTS环境"""
        # 添加可能的路径
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        sys.path.append(os.path.expanduser('../TTS_Model/Qwen3-TTS'))
    
    def load_model(self, device: str) -> Any:
        """
        加载Qwen3-TTS模型
        
        参数:
            device: 设备字符串
            
        返回:
            Qwen3TTSModel实例
        """
        self.setup_environment()
        from qwen_tts import Qwen3TTSModel
        
        model = Qwen3TTSModel.from_pretrained(
            self.model_path,
            device_map=device,
            dtype=torch.bfloat16,
            #attn_implementation="flash_attention_2",
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
        执行单次Qwen3-TTS合成
        
        参数:
            model: Qwen3TTSModel
            prompt_wav: 参考音频路径
            prompt_text: 参考文本
            text: 待合成文本
            output_path: 输出文件路径
            
        返回:
            bool: 合成是否成功
        """
        try:
            wavs, sr = model.generate_voice_clone(
                text=text,
                ref_audio=prompt_wav,
                ref_text=prompt_text,
            )
            
            sf.write(output_path, wavs[0], sr)
            return True
        except Exception as e:
            print(f"Qwen3-TTS合成失败: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    Qwen3TTSProcessor.main()
