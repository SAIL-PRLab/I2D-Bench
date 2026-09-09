"""
CosyVoice TTS 处理器

使用CosyVoice模型进行迭代语音克隆合成
"""
import os
import sys
from typing import Any

import torchaudio

from tts_common import BaseTTSProcessor


class CosyVoiceProcessor(BaseTTSProcessor):
    """
    CosyVoice TTS处理器
    """
    
    model_name = "CosyVoice"
    
    def setup_environment(self):
        """设置CosyVoice环境"""
        cosyvoice_path = os.path.expanduser('../TTS_Model/CosyVoice')
        if cosyvoice_path not in sys.path:
            sys.path.insert(0, cosyvoice_path)
            sys.path.insert(0, os.path.join(cosyvoice_path, 'third_party/Matcha-TTS'))
    
    def load_model(self, device: str) -> Any:
        """
        加载CosyVoice模型
        
        参数:
            device: 设备字符串，如 "cuda:0" 或 "cpu"
            
        返回:
            CosyVoice模型实例
        """
        self.setup_environment()
        from cosyvoice.cli.cosyvoice import AutoModel
        
        model = AutoModel(model_dir=self.model_path, device=device)
        return model
    
    def synthesize(
        self,
        model: Any,
        prompt_wav: str,
        prompt_text: str,
        text: str,
        output_path: str
    ) -> bool:
        """
        执行单次CosyVoice合成
        
        参数:
            model: CosyVoice模型
            prompt_wav: 参考音频路径
            prompt_text: 参考文本（已预处理，包含系统提示）
            text: 待合成文本
            output_path: 输出文件路径
            
        返回:
            bool: 合成是否成功
        """
        try:
            prompt_text = "You are a helpful assistant.<|endofprompt|>" + prompt_text
            # 合成语音
            for i, result in enumerate(model.inference_zero_shot(
                text, prompt_text, prompt_wav, stream=False
            )):
                # 保存结果
                torchaudio.save(
                    output_path,
                    result['tts_speech'],
                    model.sample_rate
                )
                return True
            return False
        except Exception as e:
            print(f"CosyVoice合成失败: {e}")
            return False


if __name__ == "__main__":
    CosyVoiceProcessor.main()
