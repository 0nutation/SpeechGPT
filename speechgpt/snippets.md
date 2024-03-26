# Install

```bash
conda create -n speechgpt pytorch torchvision torchaudio pytorch-cuda=12.1 python=3.10 -c pytorch -c nvidia

pip install accelerate bitsandbytes datasets deepspeed einops evaluate fairseq gradio librosa peft sentencepiece soundfile tensorboard tokenizers transformers tqdm

pip install flash-attn --no-build-isolation
```
