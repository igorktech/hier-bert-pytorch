#!/bin/bash

PROJECT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../")

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
export TOKENIZERS_PARALLELISM=false

MODEL_TYPE='hibial'
LAYOUT='i3'
MODEL_WARMUP_STRATEGY='grouped'
MODEL_MAX_LENGTH=512
MODEL_DIR=$PROJECT_DIR/data/$MODEL_TYPE-model
TOKENIZER_DIR=$PROJECT_DIR/data/tokenizer-sp
OUTPUT_MODEL_DIR=$PROJECT_DIR/data/$MODEL_TYPE-bert-$LAYOUT-mlm
TRAIN_FILE_PATH=$PROJECT_DIR/data/datasets/train/train_test.txt

# Generate model
python3 "$PROJECT_DIR/pre_training/generate_model.py" \
  --output_dir "$MODEL_DIR" \
  --tokenizer_name_or_path "$TOKENIZER_DIR" \
  --model_type_prefix "$MODEL_TYPE"

# Run pre_training
python3 "$PROJECT_DIR/pre_training/run_mlm.py" \
  --model_name_or_path "$MODEL_DIR" \
  --train_file "$TRAIN_FILE_PATH" \
  --do_train \
  --do_eval \
  --output_dir "$OUTPUT_MODEL_DIR" \
  --overwrite_output_dir \
  --logging_steps 500 \
  --evaluation_strategy steps \
  --eval_steps 5000 \
  --save_strategy steps \
  --save_steps 5000 \
  --save_total_limit 2 \
  --max_steps 25 \
  --learning_rate 1e-2 \
  --trust_remote_code \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --eval_accumulation_steps 4 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.10 \
  --weight_decay 0.01 \
  --mlm_probability 0.15 \
  --max_seq_length $MODEL_MAX_LENGTH \
  --line_by_line \
  --pad_to_max_length \
#  --torch_compile
