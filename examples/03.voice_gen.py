import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)


speakers =["Sohee", "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Uncle_fu", "Vivian"]
instruct = "따뜻하고 친근한 어조로"


target_text = "안녕하세요, Qwen3-TTS로 생성한 한국어 음성입니다."


wavs, sr = model.generate_custom_voice(
    text=target_text,
    language="Korean",
    speaker=speakers[0],  # 한국어 네이티브 여성 화자
    instruct=instruct,
)
sf.write("output_gen.wav", wavs[0], sr)




# -------------------------

# batch inference도 가능 

# # batch inference
# wavs, sr = model.generate_custom_voice(
#     text=[
#         "其实我真的有发现，我是一个特别善于观察别人情绪的人。", 
#         "She said she would be here by noon."
#     ],
#     language=["Chinese", "English"],
#     speaker=["Vivian", "Ryan"],
#     instruct=["", "Very happy."]
# )
# sf.write("output_custom_voice_1.wav", wavs[0], sr)
# sf.write("output_custom_voice_2.wav", wavs[1], sr)
