
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)



txt_path = "/home/khw/Workspace/SenseVoice/dataset/scripts/HAPPY.txt"

instruct = "20대 초반 여성, 밝고 경쾌한 음색, 약간 높은 톤으로 활기차게"
language = "Korean"

target_texts = []
target_instructs = []
target_languages = []
with open(txt_path, "r") as f:
    for idx, line in enumerate(f):

        print(f"{idx}: {line}")
        target_texts.append(line.strip())
        target_instructs.append(instruct)   
        target_languages.append(language)


if __name__ == "__main__":

            
    print(target_texts)
    
    
    # batch inference
    wavs, sr = model.generate_voice_design(
        text=target_texts,
        language=target_languages,
        instruct=target_instructs
    )
    for i, wav in enumerate(wavs):
        sf.write(f"./dataset/output_voice_design_{i+1}.wav", wav, sr)