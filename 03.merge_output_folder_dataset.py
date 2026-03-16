import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
WAV_DATASET_DIR = BASE_DIR / "dataset" / "wav_dataset"

emotion_dirs = [d for d in OUTPUT_DIR.iterdir() if d.is_dir()]
print(f"감정 폴더 {len(emotion_dirs)}개 발견: {[d.name for d in emotion_dirs]}")

for emotion_dir in sorted(emotion_dirs):
    dest_emotion_dir = WAV_DATASET_DIR / emotion_dir.name
    dest_emotion_dir.mkdir(parents=True, exist_ok=True)

    wav_files = [f for f in emotion_dir.rglob("*.wav") if ".ipynb_checkpoints" not in f.parts]
    print(f"\n[{emotion_dir.name}] {len(wav_files)}개 복사 중...")

    for wav_file in sorted(wav_files):
        dest = dest_emotion_dir / wav_file.name
        shutil.copy2(wav_file, dest)

    print(f"[{emotion_dir.name}] 완료 → {dest_emotion_dir}")

print("\n전체 완료.")
