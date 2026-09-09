"""
GLM-TTS 处理器

使用GLM-TTS模型进行迭代语音克隆合成
"""
import os
import sys
from typing import Any

import torch
import torchaudio

from tts_common import BaseTTSProcessor


class GLMTTSProcessor(BaseTTSProcessor):
    """
    GLM-TTS处理器
    """
    
    model_name = "GLM-TTS"
    default_sample_rate = 24000
    
    def setup_environment(self):
        """设置GLM-TTS环境"""
        glmtts_path = os.path.expanduser('../TTS_Model/GLM-TTS')
        if glmtts_path not in sys.path:
            sys.path.insert(0, glmtts_path)
    
    def load_model(self, device: str) -> Any:
        """
        加载GLM-TTS模型
        
        参数:
            device: 设备字符串
            
        返回:
            包含所有GLM-TTS组件的字典
        """
        self.setup_environment()
        from glmtts_inference import load_models
        
        device_obj = torch.device(device)
        use_phoneme = False
        sample_rate = self.default_sample_rate
        
        frontend, text_frontend, speech_tokenizer, llm, flow = load_models(
            use_phoneme=use_phoneme,
            sample_rate=sample_rate,
            ckpt_dir=self.model_path,
            device=device_obj
        )
        
        return {
            'frontend': frontend,
            'text_frontend': text_frontend,
            'speech_tokenizer': speech_tokenizer,
            'llm': llm,
            'flow': flow,
            'device': device_obj,
            'use_phoneme': use_phoneme,
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
        执行单次GLM-TTS合成
        
        参数:
            model: GLM-TTS模型组件字典
            prompt_wav: 参考音频路径
            prompt_text: 参考文本
            text: 待合成文本
            output_path: 输出文件路径
            
        返回:
            bool: 合成是否成功
        """
        try:
            from glmtts_inference import generate_long
            
            frontend = model['frontend']
            text_frontend = model['text_frontend']
            llm = model['llm']
            flow = model['flow']
            device = model['device']
            use_phoneme = model['use_phoneme']
            sample_rate = model['sample_rate']
            
            # 文本归一化
            normalized_prompt_text = text_frontend.text_normalize(prompt_text)
            normalized_synth_text = text_frontend.text_normalize(text)
            
            # 提取各种特征
            prompt_text_token = frontend._extract_text_token(normalized_prompt_text + " ")
            prompt_speech_token = frontend._extract_speech_token([prompt_wav])
            speech_feat = frontend._extract_speech_feat(prompt_wav, sample_rate=sample_rate)
            embedding = frontend._extract_spk_embedding(prompt_wav)
            
            cache_speech_token = [prompt_speech_token.squeeze().tolist()]
            flow_prompt_token = torch.tensor(cache_speech_token, dtype=torch.int32).to(device)
            
            # 初始化缓存
            cache = {
                "cache_text": [normalized_prompt_text],
                "cache_text_token": [prompt_text_token],
                "cache_speech_token": cache_speech_token,
                "use_cache": True,
            }
            
            # 生成语音
            tts_speech, _, _, text_tn_dict = generate_long(
                frontend=frontend,
                text_frontend=text_frontend,
                llm=llm,
                flow=flow,
                text_info=[None, normalized_synth_text],
                cache=cache,
                embedding=embedding,
                seed=0,
                flow_prompt_token=flow_prompt_token,
                speech_feat=speech_feat,
                device=device,
                use_phoneme=use_phoneme,
            )
            
            # 保存音频
            torchaudio.save(output_path, tts_speech, sample_rate)
            return True
            
        except Exception as e:
            print(f"GLM-TTS合成失败: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    GLMTTSProcessor.main()
