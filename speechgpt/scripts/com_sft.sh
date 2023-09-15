#!/bin/bash

METAROOT="output/stage2"   
DATAROOT="data/stage3"
OUTROOT="output/stage3"
CACHEROOT="${DATAROOT}/cache/"


mkdir -p ${CACHEROOT}/tokenized/train/
mkdir -p ${CACHEROOT}/tokenized/valid/


#ddp realted
NNODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
NODE_RANK=$(($(scontrol show hostnames "$SLURM_JOB_NODELIST" | grep -Fn $(hostname) | cut --delimiter=":" --fields=1)-1))


echo "stage3: chain-of-modality instruction finetuning"


torchrun \
    --nnode $NNODE \
    --nproc_per_node 8 \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port 29502  \
speechgpt/src/train/com_sft.py \
    --model_name_or_path "${METAROOT}" \
    --data_path "${DATAROOT}/SpeechInstruct_chain_of_modality.jsonl" \
    --cache_dir ${CACHEROOT} \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules 'q_proj,v_proj' \
    --preprocessing_num_workers 10 \
    --model_max_length 1024 \
    --val_set_size 10 \
    --bf16 True \
    --do_train \
    --train_on_inputs True \
    --output_dir "${OUTROOT}" \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 300 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --log_level debug \
    --logging_steps 1 \

