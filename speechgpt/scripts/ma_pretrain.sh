#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/mnt/hdd1/zhuxiaosu/hf_home"
export WANDB_PROJECT="LLaVA"
NCCL_P2P_LEVEL=NVL

METAROOT="lmsys/vicuna-7b-v1.5"   #stage1
DATAROOT="data"
OUTROOT="output/stage1"
CACHEROOT="./cache/"



AVAILBLE_PORT=$(shuf -i 29500-65535 -n 1)
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7


mkdir -p ${CACHEROOT}/tokenized/train/
mkdir -p ${CACHEROOT}/tokenized/valid/
mkdir -p ${CACHEROOT}/group/train/
mkdir -p ${CACHEROOT}/group/valid/


#ddp realted
# NNODE=$1
# NODE_RANK=$2
# MASTER_ADDR=$3
# MASTER_PORT=$4


echo "stage1: modality-adaptation pretraining"


deepspeed --include=localhost:${CUDA_VISIBLE_DEVICES} --master_port ${AVAILBLE_PORT}  src/train/ma_pretrain.py \
    --deepspeed ./scripts/zero3_offload.json \
    --bf16 True \
    --lora_enable False \
    --train_embeddings False \
    --block_size 1024 \
    --model_name_or_path "${METAROOT}" \
    --train_file ${DATAROOT}/train.txt \
    --validation_file ${DATAROOT}/dev.txt \
    --do_train \
    --do_eval \
    --output_dir "${OUTROOT}" \
    --preprocessing_num_workers 100 \
    --overwrite_output_dir \
    --per_device_eval_batch_size 64 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 1 \
    --log_level debug \
    --logging_steps 1 \
    --save_steps 2 \
    --cache_dir ${CACHEROOT} \
    --gradient_checkpointing True \
    --tf32 True
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
