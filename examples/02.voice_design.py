import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa"            # "flash_attention_2",
)


target_text = "오늘 날씨가 정말 좋네요. 산책하기 딱 좋은 날이에요."
instruct = "20대 초반 여성, 밝고 경쾌한 음색, 약간 높은 톤으로 활기차게"

wavs, sr = model.generate_voice_design(
    text=target_text,
    language="Korean",
    instruct=instruct,
)
sf.write("output_designed.wav", wavs[0], sr)



# -------------------------

# batch inference도 가능 

# # batch inference
# wavs, sr = model.generate_voice_design(
#     text=[
#       "哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
#       "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
#     ],
#     language=["Chinese", "English"],
#     instruct=[
#       "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
#       "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
#     ]
# )
# sf.write("output_voice_design_1.wav", wavs[0], sr)
# sf.write("output_voice_design_2.wav", wavs[1], sr)