"""
MaskGCT TTS 处理器

使用MaskGCT模型进行迭代语音克隆合成
"""
import os
import sys
from typing import Any

import torch
import soundfile as sf
from huggingface_hub import hf_hub_download
import safetensors

from tts_common import BaseTTSProcessor


class MaskGCTProcessor(BaseTTSProcessor):
    """
    MaskGCT处理器
    """
    
    model_name = "MaskGCT"
    default_sample_rate = 24000
    
    def setup_environment(self):
        """设置MaskGCT环境"""
        maskgct_path = os.path.expanduser('../TTS_Model/Amphion')
        if maskgct_path not in sys.path:
            sys.path.insert(0, maskgct_path)
    
    def load_model(self, device: str) -> Any:
        """
        加载MaskGCT模型
        
        参数:
            device: 设备字符串
            
        返回:
            MaskGCT_Inference_Pipeline实例
        """
        self.setup_environment()
        
        from models.tts.maskgct.maskgct_utils import (
            load_config,
            build_semantic_model,
            build_semantic_codec,
            build_acoustic_codec,
            build_t2s_model,
            build_s2a_model,
            MaskGCT_Inference_Pipeline,
        )
        
        model_root = os.path.abspath(self.model_path)
        device_obj = torch.device(device)
        
        cfg_path = os.path.join(model_root, "config/maskgct.json")
        cfg = load_config(cfg_path)
        
        semantic_model, semantic_mean, semantic_std = build_semantic_model(device_obj)
        semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device_obj)
        codec_encoder, codec_decoder = build_acoustic_codec(cfg.model.acoustic_codec, device_obj)
        t2s_model = build_t2s_model(cfg.model.t2s_model, device_obj)
        s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device_obj)
        s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device_obj)
        
        # 下载并加载权重
        repo_id = "amphion/MaskGCT"
        cache_dir = os.path.join(model_root, "hf_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        semantic_code_ckpt = hf_hub_download(
            repo_id, filename="semantic_codec/model.safetensors", cache_dir=cache_dir
        )
        codec_encoder_ckpt = hf_hub_download(
            repo_id, filename="acoustic_codec/model.safetensors", cache_dir=cache_dir
        )
        codec_decoder_ckpt = hf_hub_download(
            repo_id, filename="acoustic_codec/model_1.safetensors", cache_dir=cache_dir
        )
        t2s_model_ckpt = hf_hub_download(
            repo_id, filename="t2s_model/model.safetensors", cache_dir=cache_dir
        )
        s2a_1layer_ckpt = hf_hub_download(
            repo_id, filename="s2a_model/s2a_model_1layer/model.safetensors", cache_dir=cache_dir
        )
        s2a_full_ckpt = hf_hub_download(
            repo_id, filename="s2a_model/s2a_model_full/model.safetensors", cache_dir=cache_dir
        )
        
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        safetensors.torch.load_model(codec_encoder, codec_encoder_ckpt)
        safetensors.torch.load_model(codec_decoder, codec_decoder_ckpt)
        safetensors.torch.load_model(t2s_model, t2s_model_ckpt)
        safetensors.torch.load_model(s2a_model_1layer, s2a_1layer_ckpt)
        safetensors.torch.load_model(s2a_model_full, s2a_full_ckpt)
        
        pipeline = MaskGCT_Inference_Pipeline(
            semantic_model,
            semantic_codec,
            codec_encoder,
            codec_decoder,
            t2s_model,
            s2a_model_1layer,
            s2a_model_full,
            semantic_mean,
            semantic_std,
            device_obj,
        )
        
        return pipeline
    
    def synthesize(
        self,
        model: Any,
        prompt_wav: str,
        prompt_text: str,
        text: str,
        output_path: str,
    ) -> bool:
        """
        执行单次MaskGCT合成
        
        参数:
            model: MaskGCT_Inference_Pipeline
            prompt_wav: 参考音频路径
            prompt_text: 参考文本
            text: 待合成文本
            output_path: 输出文件路径
            
        返回:
            bool: 合成是否成功
        """
        try:
            
            recovered_audio = model.maskgct_inference(
                prompt_wav,
                prompt_text,
                text
            )
            
            sf.write(output_path, recovered_audio, self.default_sample_rate)
            return True
        except Exception as e:
            print(f"MaskGCT合成失败: {e}")
            return False


if __name__ == "__main__":
    MaskGCTProcessor.main()
