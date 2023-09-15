#!/bin/bash

UNITS=$1   #units 

VOCODER_DIR="speechgpt/utils/vocoder"
IN_CODE_FILE=${VOCODER_DIR}/in_code_file.txt
VOCODER_CKPT=${VOCODER_DIR}/vocoder.pt
VOCODER_CFG=${VOCODER_DIR}/config.json
RESULTS_PATH="output/wav"

mkdir -p ${VOCODER_DIR}
mkdir -p ${RESULTS_PATH}


if [ ! -f ${VOCODER_CFG} ];then
  wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json -O ${VOCODER_CFG}
  wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000 -O ${VOCODER_CKPT}
fi


echo $UNITS | sed -E 's/[^0-9]+/ /g' > ${IN_CODE_FILE}


#genereate file
python3 ${VOCODER_DIR}/generate_waveform_from_code.py \
  --in-code-file ${IN_CODE_FILE} \
  --vocoder ${VOCODER_CKPT} --vocoder-cfg ${VOCODER_CFG} \
  --results-path ${RESULTS_PATH} --dur-prediction 
  

