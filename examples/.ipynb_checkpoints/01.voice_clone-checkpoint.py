import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# 3초 분량의 참조 음성과 해당 텍스트
ref_audio = "reference_voice.wav"  # 로컬 파일 또는 URL
ref_text = "안녕하세요? 에이로봇의 김현우입니다. 만나서 반갑습니다."

target_text = "복제된 음성으로 새로운 문장을 합성합니다."

wavs, sr = model.generate_voice_clone(
    text=target_text,
    language="Korean",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
sf.write("output_cloned.wav", wavs[0], sr)
