#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import wave
from pathlib import Path

# 기본 경로 설정
BASE_DIR = Path(__file__).parent
SCRIPTS_DIR = BASE_DIR / "dataset_tts" / "scripts"
WAV_DATASET_DIR = BASE_DIR / "dataset_tts" / "wav_dataset"
OUTPUT_DIR = BASE_DIR / "dataset_tts" / "dataset"

# 감정 리스트
EMOTIONS = ["ANGRY", "DISGUSTED", "FEARFUL", "HAPPY", "NEUTRAL", "SAD", "SURPRISED"]


def get_wav_duration(wav_path):
    """WAV 파일의 길이를 초 단위로 반환"""
    try:
        with wave.open(str(wav_path), 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return int(round(duration))  # 반올림하여 정수로 반환
    except Exception as e:
        print(f"Error reading WAV file {wav_path}: {e}")
        return 0


def read_transcript_file(filepath):
    """대본 파일을 읽어서 줄 단위로 리스트 반환"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 빈 줄 제거 및 줄바꿈 문자 제거
    lines = [line.strip() for line in lines if line.strip()]
    return lines


def get_sorted_wav_files(emotion_dir):
    """감정 폴더의 WAV 파일을 시간순으로 정렬하여 반환"""
    if not emotion_dir.exists():
        return []
    
    wav_files = sorted(f for f in emotion_dir.glob("*.wav") if "-checkpoint" not in f.name)
    return wav_files


def generate_dataset_files():
    """데이터셋 파일 생성"""
    
    # 출력 디렉토리 생성
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 출력 파일 준비
    scp_data = []
    text_data = []
    language_data = []
    emo_data = []
    event_data = []
    
    total_processed = 0
    
    # 각 감정별로 처리
    for emotion in EMOTIONS:
        print(f"\n처리 중: {emotion}")
        
        # 대본 파일 읽기
        transcript_file = SCRIPTS_DIR / f"{emotion}.txt"
        if not transcript_file.exists():
            print(f"  경고: 대본 파일이 없습니다 - {transcript_file}")
            continue
        
        transcripts = read_transcript_file(transcript_file)
        
        # WAV 파일 목록 가져오기
        emotion_wav_dir = WAV_DATASET_DIR / emotion
        wav_files = get_sorted_wav_files(emotion_wav_dir)
        
        if not wav_files:
            print(f"  경고: WAV 파일이 없습니다 - {emotion_wav_dir}")
            continue
        
        # 데이터 생성 (파일명의 인덱스로 대본 매핑)
        # 파일명 형식: {EMOTION}_{SPEAKER}_{INDEX:03d}.wav
        count = 0
        for wav_path in wav_files:
            # 파일명에서 인덱스 추출 (마지막 _NNN 부분)
            stem = wav_path.stem  # e.g. "HAPPY_Eric_042"
            try:
                transcript_idx = int(stem.rsplit("_", 1)[-1])
            except ValueError:
                print(f"  경고: 인덱스 파싱 실패 - {wav_path.name}, 건너뜀")
                continue

            if transcript_idx >= len(transcripts):
                print(f"  경고: 인덱스 {transcript_idx}가 대본 범위({len(transcripts)}줄)를 벗어남 - {wav_path.name}, 건너뜀")
                continue

            # ID 생성: {감정}_{화자포함파일명}_{파일길이(초):03d}_ko
            duration = get_wav_duration(wav_path)
            data_id = f"{stem}_{duration:03d}_ko"

            # 상대 경로로 변환
            rel_wav_path = wav_path.relative_to(BASE_DIR)

            # 각 데이터 추가
            scp_data.append(f"{data_id} {rel_wav_path}")
            text_data.append(f"{data_id} {transcripts[transcript_idx]}")
            language_data.append(f"{data_id} <|ko|>")
            emo_data.append(f"{data_id} <|{emotion}|>")
            event_data.append(f"{data_id} <|Speech|>")
            count += 1
        
        print(f"  완료: {count}개 항목 처리됨")
        total_processed += count
    
    # 파일 저장
    print(f"\n총 {total_processed}개 항목 처리 완료")
    print("파일 저장 중...")
    
    with open(OUTPUT_DIR / "train_wav.scp", 'w', encoding='utf-8') as f:
        f.write('\n'.join(scp_data) + '\n')
    
    with open(OUTPUT_DIR / "train_text.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_data) + '\n')
    
    with open(OUTPUT_DIR / "train_text_language.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(language_data) + '\n')
    
    with open(OUTPUT_DIR / "train_emo.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(emo_data) + '\n')
    
    with open(OUTPUT_DIR / "train_event.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(event_data) + '\n')
    
    print("\n생성된 파일:")
    print(f"  - {OUTPUT_DIR / 'train_wav.scp'}")
    print(f"  - {OUTPUT_DIR / 'train_text.txt'}")
    print(f"  - {OUTPUT_DIR / 'train_text_language.txt'}")
    print(f"  - {OUTPUT_DIR / 'train_emo.txt'}")
    print(f"  - {OUTPUT_DIR / 'train_event.txt'}")
    print("\n완료!")


if __name__ == "__main__":
    generate_dataset_files()
