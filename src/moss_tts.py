"""
MOSS-TTS 处理器

使用MOSS-TTS模型进行迭代语音克隆合成
"""
import os
import sys
from typing import Any, Dict
from pathlib import Path
import torch
import torchaudio
from modelscope import AutoModel, AutoProcessor, GenerationConfig

# Disable the broken cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
# Keep these enabled as fallbacks
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

from tts_common import BaseTTSProcessor


class DelayGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = kwargs.get("layers", [{} for _ in range(32)])
        self.do_samples = kwargs.get("do_samples", None)
        self.n_vq_for_inference = 32
        

def initial_config(tokenizer, model_name_or_path):
    generation_config = DelayGenerationConfig.from_pretrained(model_name_or_path)
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = 151653
    generation_config.max_new_tokens = 1000000
    generation_config.temperature = 1.0
    generation_config.top_p = 0.95
    generation_config.top_k = 100
    generation_config.repetition_penalty = 1.1
    generation_config.use_cache = True
    generation_config.do_sample = False
    return generation_config


class MOSSTTSProcessor(BaseTTSProcessor):
    """
    MOSS-TTS处理器
    """
    
    model_name = "MOSS-TTS"
    
    def setup_environment(self):
        pass
    
    def load_model(self, device: str) -> Any:
        """
        加载MOSS-TTS模型
        
        参数:
            device: 设备字符串，如 "cuda:0" 或 "cpu"
            
        返回:
            Dict: 包含model, processor, generation_config的字典
        """
        self.setup_environment()
        
        print(f"Loading MOSS-TTS from {self.model_path}")
        
        # Determine dtype
        dtype = torch.bfloat16 if "cuda" in str(device) and torch.cuda.is_available() else torch.float32

        # Resolve tokenizer path
        # Assuming folder structure parallel to model path
        # If model path ends with MOSS-TTS-Local, we replace it.
        audio_tokenizer_path = self.model_path.replace("MOSS-TTS-Local", "MOSS-Audio-Tokenizer")
        
        print(f"Loading Audio Tokenizer from {audio_tokenizer_path}")

        processor = AutoProcessor.from_pretrained(
            self.model_path,
            codec_path=audio_tokenizer_path,
            trust_remote_code=True,
        )
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)
                
        model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=dtype,
        ).to(device)
        model.eval()
        
        generation_config = initial_config(processor.tokenizer, self.model_path)
        generation_config.n_vq_for_inference = model.channels - 1
        generation_config.do_samples = [True] * model.channels
        generation_config.layers = [
            {
                "repetition_penalty": 1.0, 
                "temperature": 1.5, 
                "top_p": 1.0, 
                "top_k": 50
            }
        ] + [ 
            {
                "repetition_penalty": 1.1, 
                "temperature": 1.0, 
                "top_p": 0.95,
                "top_k": 50
            }
        ] * (model.channels - 1) 
        
        return {
            "model": model,
            "processor": processor,
            "generation_config": generation_config,
            "device": device
        }
    
    def synthesize(
        self,
        model_pack: Any,
        prompt_wav: str,
        prompt_text: str,
        text: str,
        output_path: str
    ) -> bool:
        """
        执行单次MOSS-TTS合成
        
        参数:
            model_pack: load_model返回的字典
            prompt_wav: 参考音频路径
            prompt_text: 参考文本 (MOSS-TTS seemingly ignores this in snippet, but we keep signature)
            text: 待合成文本
            output_path: 输出文件路径
            
        返回:
            bool: 合成是否成功
        """
        try:
            model = model_pack["model"]
            processor = model_pack["processor"]
            generation_config = model_pack["generation_config"]
            device = model_pack["device"]
            
            
            # Construct conversation
            # Note: prompt_text is not used in the snippet for build_user_message, only audio.
            conversation = [processor.build_user_message(text=text, reference=[prompt_wav])]
            
            # Prepare inputs
            # Input to processor should be a list of conversations, where each conversation is a list of messages.
            batch = processor([conversation], mode="generation")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config
                )

            # Process outputs
            decoded = list(processor.decode(outputs))
            if not decoded:
                print(f"MOSS-TTS: No decoded output for {output_path}")
                return False
                
            message = decoded[0]
            if not hasattr(message, "audio_codes_list") or not message.audio_codes_list:
                 print(f"MOSS-TTS: No audio codes in output for {output_path}")
                 return False

            # Collect and concatenate audio segments
            audio_segments = []
            for audio in message.audio_codes_list:
                audio_segments.append(audio)
            
            if len(audio_segments) > 1:
                final_audio = torch.cat(audio_segments, dim=0)
            else:
                final_audio = audio_segments[0]
            
            # Save audio
            # Audio is expected to be [Time], save expects [Channels, Time]
            torchaudio.save(output_path, final_audio.unsqueeze(0), processor.model_config.sampling_rate)
            
            return True
            
        except Exception as e:
            print(f"MOSS-TTS合成失败: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    MOSSTTSProcessor.main()
