import argparse
import shutil
from pathlib import Path

"""
python 04.move_dataset.py /home/khw/_raid/SenseVoice
python 04.move_dataset.py ../SenseVoice
python 04.move_dataset.py /new/path/to/SenseVoice

"""

parser = argparse.ArgumentParser(description="Qwen3-tts dataset을 SenseVoice 폴더로 이동")
parser.add_argument("sensevoice_dir", type=Path, help="SenseVoice 프로젝트 폴더 경로")
args = parser.parse_args()

QWEN_DIR = Path(__file__).parent          # Qwen3-tts/
SENSEVOICE_DIR = args.sensevoice_dir.resolve()

SRC_SCRIPTS = QWEN_DIR / "dataset" / "scripts"
DST_SCRIPTS = SENSEVOICE_DIR / "dataset" / "scripts"

SRC_WAV = QWEN_DIR / "dataset" / "wav_dataset"
DST_WAV = SENSEVOICE_DIR / "dataset" / "wav_dataset"

# 1. scripts 복사 (overwrite)
print("=== scripts 복사 ===")
DST_SCRIPTS.mkdir(parents=True, exist_ok=True)
for src_file in sorted(SRC_SCRIPTS.glob("*.txt")):
    dst_file = DST_SCRIPTS / src_file.name
    shutil.copy2(src_file, dst_file)
    print(f"복사: {src_file.name} → {dst_file}")

# 2. wav 파일 이동 (감정별)
print("\n=== wav 파일 이동 ===")
for emotion_dir in sorted(SRC_WAV.iterdir()):
    if not emotion_dir.is_dir():
        continue

    dst_emotion_dir = DST_WAV / emotion_dir.name
    dst_emotion_dir.mkdir(parents=True, exist_ok=True)

    wav_files = list(emotion_dir.glob("*.wav"))
    print(f"\n[{emotion_dir.name}] {len(wav_files)}개 이동 중...")
    for wav_file in sorted(wav_files):
        dst = dst_emotion_dir / wav_file.name
        shutil.move(str(wav_file), dst)

    print(f"[{emotion_dir.name}] 완료 → {dst_emotion_dir}")

print("\n전체 완료.")
