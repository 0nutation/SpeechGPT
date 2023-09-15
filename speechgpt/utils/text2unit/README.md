# Text to unit
The text-to-unit generator adopts a Transformer encoder-decoder architecture. We trained it on LibriSpeech unit-text pairs. 

## Download
```bash
t2u_dir="uitls/text2unit"
cd ${t2u_dir}
wget https://huggingface.co/fnlp/text2unit/resolve/main/text2unit.pt
```

# Inference
```python
python3 utils/text2unit/text2unit.py --text "Today is a good day."
```