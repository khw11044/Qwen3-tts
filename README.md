

```
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts
pip install flash-attn==2.7.4.post1 --no-build-isolation -v  # GPU 메모리 최적화
```