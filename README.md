

```
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts
pip install flash-attn==2.7.4.post1 --no-build-isolation -v  # GPU 메모리 최적화
```



```
uv venv --python 3.10

source .venv/bin/activate

uv pip install setuptools

uv pip install -U qwen-tts


MAX_JOBS=8 uv pip install -U flash-attn --no-build-isolation

```