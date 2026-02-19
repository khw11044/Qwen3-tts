import os
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa"            # "flash_attention_2",
)


"""
Sohee:          한국어    (good)
Ryan:           영어       (bad)
Aiden:          영어       (bad)
Ono_Anna:       일본어     (good)
Vivian:         중국어     (so so)
Serena:         중국어     (bad)
Uncle_Fu:       중국어     (good)
Dylan:          중국어     (bad)
Eric:           중국어     (so so)





"""

"""
ANGRY: 매우 화나고, 소리치는 목소리
DISGUSTED: 징그러워하고 역겨워 하는 목소리, 짜증내는 목소리
FEARFUL: 두려워하고, 겁먹은 목소리
HAPPY: 밝고 경쾌한 음색, 약간 높은 톤으로 활기차고 행복한 목소리
SAD: 낮고 침울한 음색, 약간 낮은 톤으로 울먹이듯 슬픈 목소리
SURPRISED: 매우 놀란 목소리


ANGRY: Speak in a very angry tone, shouting with intense emotion. The voice should sound furious, loud, and aggressive.
"ANGRY": "Speak in an extremely furious and explosive tone, as if you have completely lost control. Shout with raw intensity and burning rage. The voice should sound sharp, strained, aggressive, and forceful, with clipped words and heavy emotional tension.",
DISGUSTED: Speak in a disgusted tone, as if something is revolting and unpleasant. The voice should sound repulsed and uncomfortable.
FEARFUL: Speak in a fearful tone, as if you are scared and anxious. The voice should sound trembling, nervous, and frightened.
HAPPY: Speak in a bright and cheerful tone, with a slightly higher pitch. The voice should sound lively, energetic, and genuinely happy.
SAD: Speak in a low and gloomy tone, with a slightly lower pitch. The voice should sound tearful, heavy, and sorrowful.
SURPRISED: Speak in a very surprised tone, as if something unexpected just happened. The voice should sound shocked and startled.


"""

Emotion_prompt_maps = {
    "ANGRY": "Speak in a violently enraged tone, as if you are shouting in uncontrollable rage. The voice should be loud, harsh, and explosive, filled with intense frustration and fury. Each word should sound forceful, sharp, and emotionally charged.",
    "DISGUSTED": "Speak in a deeply disgusted tone, as if you are reacting to something truly revolting. The voice should sound strained, irritated, and slightly sharp, with a sense of nausea and strong emotional rejection.",
    "FEARFUL": "Speak in a terrified, near-panic tone, as if you are facing something truly horrifying. The voice should shake uncontrollably, with uneven breathing and broken delivery. It should sound fragile, desperate, and overwhelmed by fear.",
    "HAPPY": "Speak in an excited and radiant tone with a slightly higher pitch. The voice should be bright, lively, and full of cheerful energy, as if you are truly delighted. Let the words flow with playful enthusiasm and an expressive, smiling quality that clearly conveys joy. 밝고 경쾌한 음색, 약간 높은 톤으로 활기차게",
    "SAD": "Speak in a low and gloomy tone, with a slightly lower pitch. The voice should sound tearful, heavy, and sorrowful.",
    "SURPRISED": "Speak in a highly shocked and explosive tone, as if something just frightened you unexpectedly. The voice should spike in pitch, sound breathless and startled, with a sudden burst of emotional intensity. Deliver the words with dramatic urgency and uncontrollable surprise.",
    "NEUTRAL": "Speak in a calm, flat, and emotionless tone. The voice should be steady and even, with no emotional inflection or emphasis. Deliver the words plainly and matter-of-factly."

}


speakers = ["Sohee", "Ryan", "Aiden", "Ono_Anna", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"]

speaker_idx = 0
# 대본 파일 읽기
script_file = "dataset/scripts/NEUTRAL.txt"
emotion = os.path.basename(script_file).replace(".txt", "")  # "ANGRY"
instruct = Emotion_prompt_maps[emotion]

with open(script_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

# 출력 디렉토리 생성
output_dir = f"output/{emotion}/{speakers[speaker_idx]}"
os.makedirs(output_dir, exist_ok=True)

# batch inference (batch_size=8)
BATCH_SIZE = 8
SEED = 42
total = len(lines[:16])
idx = 0

for batch_start in range(0, total, BATCH_SIZE):
    batch_lines = lines[batch_start:batch_start + BATCH_SIZE]
    batch_size = len(batch_lines)

    print(f"[{batch_start}/{total}] Generating batch ({batch_size} lines)...")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    wavs, sr = model.generate_custom_voice(
        text=batch_lines,
        language=["Korean"] * batch_size,
        speaker=[speakers[speaker_idx]] * batch_size,
        instruct=[instruct] * batch_size,
    )

    for wav in wavs:
        sf.write(f"{output_dir}/{emotion}_{idx:03d}.wav", wav, sr)
        print(f"  Saved: {output_dir}/{emotion}_{idx:03d}.wav")
        idx += 1

print(f"Done! {idx} files saved to {output_dir}/")
