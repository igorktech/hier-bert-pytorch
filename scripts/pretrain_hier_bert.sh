export PYTHONPATH=.

LAYOUT='i3'
MODEL_WARMUP_STRATEGY='grouped'
MODEL_MAX_LENGTH=512
MODEL_DIR=/data/model
TOKENIZER_DIR=/data/tokenizer-sp
OUTPUT_MODEL_DIR=/data/hier-bert-$LAYOUT-mlm
TRAIN_FILE_PATH=/data/dataset_mlm.txt

python3 pretraining/generate_model.py \
  --output_dir $MODEL_DIR \
  --tokenizer_name_or_path $TOKENIZER_DIR

python3 pretraining/run_mlm.py \
  --model_name_or_path $MODEL_DIR \
  --train_file $TRAIN_FILE_PATH \
  --do_train \
  --do_eval \
  --output_dir OUTPUT_MODEL_DIR \
  --overwrite_output_dir \
  --logging_steps 500 \
  --evaluation_strategy steps \
  --eval_steps 5000 \
  --save_strategy steps \
  --save_steps 5000 \
  --save_total_limit 2 \
  --max_steps 25000 \
  --learning_rate 1e-4 \
  --torch_compile \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --eval_accumulation_steps 4 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.10 \
  --weight_decay 0.01 \
  --mlm_probability 0.15 \
  --max_seq_length $MODEL_MAX_LENGTH \
  --line_by_line \
  --pad_to_max_length
