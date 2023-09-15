# Speech2unit
We employ [mHuBERT](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/docs/textless_s2st_real_data.md) as the speech tokenizer to discretize speech data into discrete units and remove the repetitive units of adjacent frames to get reduced units.

## Download
```bash
s2u_dir="uitls/speech2unit"
cd ${s2u_dir}
wget https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt
wget https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin
```

## Discretize
```python
python3 speech2unit.py --wav path/to/wav
```