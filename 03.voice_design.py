import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)


target_text = "오늘 날씨가 정말 좋네요. 산책하기 딱 좋은 날이에요."
instruct = "20대 초반 여성, 밝고 경쾌한 음색, 약간 높은 톤으로 활기차게"

wavs, sr = model.generate_voice_design(
    text=target_text,
    language="Korean",
    instruct=instruct,
)
sf.write("output_designed.wav", wavs[0], sr)