"""
DeepFilterNet post-processing for generated TTS wav files.
Reads all wav files from output/ and saves enhanced versions to output_df/.

Usage:
    python postprocess_df.py
"""

import multiprocessing as mp
import os
import time
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# [Configuration]
# ==========================================
NUM_GPUS = 2
WORKERS_PER_GPU = 6
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_df")
# ==========================================


def save_audio_worker(path, audio, sr):
    from df.enhance import save_audio
    try:
        save_audio(path, audio, sr)
    except Exception as e:
        print(f"Error saving {path}: {e}")


def worker(gpu_id, worker_id, entries):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True

    import warnings
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.CRITICAL)

    # suppress DF internal logs
    _stdout_fd = os.dup(1)
    _stderr_fd = os.dup(2)
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 1)
    os.dup2(_devnull_fd, 2)

    from df.enhance import enhance, init_df, load_audio

    try:
        model, df_state, _ = init_df()
        sr = df_state.sr()
        model.eval()
    except Exception as e:
        os.dup2(_stdout_fd, 1)
        os.dup2(_stderr_fd, 2)
        os.close(_devnull_fd); os.close(_stdout_fd); os.close(_stderr_fd)
        print(f"[Worker {gpu_id}-{worker_id}] Init failed: {e}")
        return

    os.dup2(_stdout_fd, 1)
    os.dup2(_stderr_fd, 2)
    os.close(_devnull_fd); os.close(_stdout_fd); os.close(_stderr_fd)

    saver_pool = ThreadPoolExecutor(max_workers=2)
    processed = 0
    skipped = 0
    worker_start = time.time()

    print(f"[Worker {gpu_id}-{worker_id}] Start ({len(entries)} files)")

    for src_path, dst_path in entries:
        if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
            skipped += 1
            continue

        try:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            audio, _ = load_audio(src_path, sr=sr)
            enhanced = enhance(model, df_state, audio)
            saver_pool.submit(save_audio_worker, dst_path, enhanced, sr)
            processed += 1
        except Exception as e:
            print(f"[Worker {gpu_id}-{worker_id}] Error {os.path.basename(src_path)}: {e}")

    saver_pool.shutdown(wait=True)
    elapsed = time.time() - worker_start
    print(f"[Worker {gpu_id}-{worker_id}] Done. Processed: {processed}, Skipped: {skipped} ({elapsed:.1f}s)")


if __name__ == "__main__":
    start_time = time.time()
    mp.set_start_method("spawn", force=True)

    # Collect all wav files
    # Input:  output/<emotion>/<speaker>/<file>.wav
    # Output: output_df/<speaker>/<emotion>/<file>.wav
    entries = []
    for root, _, files in os.walk(INPUT_DIR):
        for fname in files:
            if not fname.endswith(".wav"):
                continue
            src_path = os.path.join(root, fname)
            rel = os.path.relpath(root, INPUT_DIR)
            parts = rel.split(os.sep)
            if len(parts) >= 2:
                emotion, speaker = parts[0], parts[1]
                dst_path = os.path.join(OUTPUT_DIR, speaker, emotion, fname)
            else:
                dst_path = os.path.join(OUTPUT_DIR, fname)
            entries.append((src_path, dst_path))

    entries.sort(key=lambda x: x[0])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_files = len(entries)
    print(f"Total WAV files: {total_files}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Split work across workers
    total_workers = NUM_GPUS * WORKERS_PER_GPU
    chunk_size = total_files // total_workers + 1
    chunks = [entries[i:i + chunk_size] for i in range(0, total_files, chunk_size)]

    print(f"Launching {total_workers} workers ({WORKERS_PER_GPU} per GPU)...")

    processes = []
    for i in range(total_workers):
        gpu_id = i % NUM_GPUS
        worker_chunk = chunks[i] if i < len(chunks) else []
        if not worker_chunk:
            continue
        p = mp.Process(target=worker, args=(gpu_id, i, worker_chunk))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed_min = (time.time() - start_time) / 60
    print(f"\nDone. Total: {total_files} files in {elapsed_min:.1f} min")
