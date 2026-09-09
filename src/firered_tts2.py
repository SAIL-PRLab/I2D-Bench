"""
FireRedTTS2 处理器

使用FireRedTTS2模型进行迭代语音克隆合成
"""
import os
import sys
from typing import Any
import torchaudio
from tts_common import BaseTTSProcessor


class FireRedTTS2Processor(BaseTTSProcessor):
    """
    FireRedTTS2处理器
    """
    
    model_name = "FireRedTTS2"
    default_sample_rate = 24000
    
    def setup_environment(self):
        """设置FireRedTTS2环境"""
        fireredtts2_path = os.path.expanduser('../TTS_Model/FireRedTTS2')
        sys.path.insert(0, fireredtts2_path)
    
    def load_model(self, device: str) -> Any:
        """
        加载F5-TTS模型
        
        参数:
            device: 设备字符串
            
        返回:
            FireRedTTS2模型实例
        """
        self.setup_environment()
        from fireredtts2.fireredtts2 import FireRedTTS2
        
        model = FireRedTTS2(
            pretrained_dir=self.model_path,
            gen_type="monologue",
            device=device,
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
            audio = model.generate_monologue(
                text=text,
                prompt_wav=str(prompt_wav),
                prompt_text=prompt_text,
            )
                
            # 保存迭代结果
            torchaudio.save(
                str(output_path),
                audio.cpu(),
                self.default_sample_rate
            )
            return True
        except Exception as e:
            print(f"FireRedTTS2合成失败: {e}")
            return False


if __name__ == "__main__":
    FireRedTTS2Processor.main()
