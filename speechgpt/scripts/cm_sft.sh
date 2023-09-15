#!/bin/bash

METAROOT="output/stage1"   
DATAROOT="data/stage2"
OUTROOT="output/stage2"
CACHEROOT="${DATAROOT}/cache/"


mkdir -p ${CACHEROOT}/tokenized/train/
mkdir -p ${CACHEROOT}/tokenized/valid/


#ddp realted
NNODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
NODE_RANK=$(($(scontrol show hostnames "$SLURM_JOB_NODELIST" | grep -Fn $(hostname) | cut --delimiter=":" --fields=1)-1))


echo "stage2: cross-modal instruction fine-tuning"


torchrun \
    --nnode $NNODE \
    --nproc_per_node 8 \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port 29501  \
speechgpt/src/train/cm_sft.py \
    --model_name_or_path "${METAROOT}" \
    --data_path "${DATAROOT}/SpeechInstruct_cross_modal.jsonl" \
    --cache_dir ${CACHEROOT} \
    --preprocessing_num_workers 10 \
    --model_max_length 512 \
    --bf16 True \
    --do_train \
    --do_eval \
    --train_on_inputs True \
    --output_dir "${OUTROOT}" \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 12 \
    --num_train_epochs 3 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --log_level debug \
    --logging_steps 1 \
    --overwrite_output_dir \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \

