"""
F5-TTS 处理器

使用F5-TTS模型进行迭代语音克隆合成
"""
import os
import sys
from typing import Any

from tts_common import BaseTTSProcessor


class F5TTSProcessor(BaseTTSProcessor):
    """
    F5-TTS处理器
    """
    
    model_name = "F5-TTS"
    
    def setup_environment(self):
        """设置F5-TTS环境"""
        f5tts_path = os.path.expanduser('../TTS_Model/F5-TTS')
        if f5tts_path not in sys.path:
            sys.path.insert(0, f5tts_path)
            sys.path.insert(0, os.path.join(f5tts_path, 'src'))
    
    def load_model(self, device: str) -> Any:
        """
        加载F5-TTS模型
        
        参数:
            device: 设备字符串
            
        返回:
            F5TTS模型实例
        """
        self.setup_environment()
        from f5_tts.api import F5TTS
        
        model = F5TTS()
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
        执行单次F5-TTS合成
        
        参数:
            model: F5TTS模型
            prompt_wav: 参考音频路径
            prompt_text: 参考文本
            text: 待合成文本
            output_path: 输出文件路径
            
        返回:
            bool: 合成是否成功
        """
        try:
            wav, sr, spec = model.infer(
                ref_file=prompt_wav,
                ref_text=prompt_text,
                gen_text=text,
                file_wave=output_path,
                seed=None,
            )
            return True
        except Exception as e:
            print(f"F5-TTS合成失败: {e}")
            return False


if __name__ == "__main__":
    F5TTSProcessor.main()
