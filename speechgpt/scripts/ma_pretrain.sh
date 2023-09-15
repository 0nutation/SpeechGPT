#!/bin/bash

METAROOT="llama/hf/7B"   #stage1
DATAROOT="data/stage1"
OUTROOT="output/stage1"
CACHEROOT="${DATAROOT}/cache/"


mkdir -p ${CACHEROOT}/tokenized/train/
mkdir -p ${CACHEROOT}/tokenized/valid/
mkdir -p ${CACHEROOT}/group/train/
mkdir -p ${CACHEROOT}/group/valid/


#ddp realted
NNODE=$1
NODE_RANK=$2
MASTER_ADDR=$3
MASTER_PORT=$4


echo "stage1: modality-adaptation pretraining"


torchrun \
    --nnode $NNODE \
    --nproc_per_node 8 \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT  \
src/train/ma_pretrain.py \
    --bf16 True \
    --block_size 1024 \
    --model_name_or_path "${METAROOT}" \
    --train_file ${DATAROOT}/train.txt \
    --validation_file ${DATAROOT}/dev.txt \
    --do_train \
    --do_eval \
    --output_dir "${OUTROOT}" \
    --preprocessing_num_workers 100 \
    --overwrite_output_dir \
    --per_device_eval_batch_size 3 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --log_level debug \
    --logging_steps 1 \
    --save_steps 300 \
    --cache_dir ${CACHEROOT} \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \

