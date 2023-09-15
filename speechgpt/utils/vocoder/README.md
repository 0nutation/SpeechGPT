# Vocoder
We adopt a [unit-based HiFi-GAN vocoder](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/docs/textless_s2st_real_data.md) to convert discrete units back to speech.

## Download 
You should download the vocoder checkpoint and config files before SpeechGPT inference.
```bash
vocoder_dir="utils/vocoder/"
cd ${vocoder_dir}
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json -O config.json
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000 -O vocoder.pt
```

## Unit to speech
```bash
units="<sosp><991><741><945><944><579><969><901><202><393><946><734><498><889><172><871><877><822><89><194><620><915><143><38><914><445><469><167><655><764><70><828><347><376><975><955><333><198><711><510><700><362><932><148><45><914><119><593><167><655><837><81><852><12><852><336><503><523><506><29><561><326><531><576><822><89><834><705><417><675><237><584><eosp>"
bash vocoder.sh ${units}
```