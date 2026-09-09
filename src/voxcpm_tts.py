"""
VoxCPM TTS 处理器

使用VoxCPM模型进行迭代语音克隆合成
"""
import os
from typing import Any

import torch
import soundfile as sf
import numpy as np

from tts_common import BaseTTSProcessor


class VoxCPMProcessor(BaseTTSProcessor):
    """
    VoxCPM处理器
    """
    
    model_name = "VoxCPM"
    default_sample_rate = 16000
    
    def load_model(self, device: str) -> Any:
        """
        加载VoxCPM模型
        
        参数:
            device: 设备字符串
            
        返回:
            包含VoxCPM模型和采样率的字典
        """
        from voxcpm import VoxCPM
        
        # VoxCPM通过环境变量设置GPU
        if device.startswith("cuda:"):
            gpu_id = device.split(":")[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        
        model = VoxCPM.from_pretrained(self.model_path)
        
        # 获取模型采样率
        sample_rate = self.default_sample_rate
        if hasattr(model, "tts_model") and hasattr(model.tts_model, "sample_rate"):
            sample_rate = model.tts_model.sample_rate
        
        return {
            'model': model,
            'sample_rate': sample_rate,
        }
    
    def synthesize(
        self,
        model: Any,
        prompt_wav: str,
        prompt_text: str,
        text: str,
        output_path: str,
    ) -> bool:
        """
        执行单次VoxCPM合成
        
        参数:
            model: VoxCPM模型字典
            prompt_wav: 参考音频路径
            prompt_text: 参考文本
            text: 待合成文本
            output_path: 输出文件路径
            
        返回:
            bool: 合成是否成功
        """
        try:
            vox_model = model['model']
            sample_rate = model['sample_rate']
            
            wav = vox_model.generate(
                text=text,
                prompt_wav_path=prompt_wav,
                prompt_text=prompt_text,
                cfg_value=2.0,
                inference_timesteps=10,
                normalize=False,
                denoise=False,
                retry_badcase=True,
                retry_badcase_max_times=3,
                retry_badcase_ratio_threshold=6.0,
            )
            
            # 规范为numpy数组
            if isinstance(wav, torch.Tensor):
                wav = wav.detach().cpu().numpy()
            elif not isinstance(wav, np.ndarray):
                wav = np.asarray(wav, dtype=np.float32)
            
            sf.write(output_path, wav, sample_rate)
            return True
            
        except Exception as e:
            print(f"VoxCPM合成失败: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    VoxCPMProcessor.main()
