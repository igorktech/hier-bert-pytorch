LEARNING_RATE = 1e-4
BETA1 = 0.9
BETA2 = 0.999
EPS =1e-6
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100

USE_MLM = True
MLM_PROB = 0.15
DATASET_PATH =
OUTPUT_DIR = 
OVERWRITE_OUTPUT_DIR = True
NUM_TRAIN_EPOCHS=10
PER_DEVICE_TRAIN_BATCH_SIZE = 256
SAVE_STEPS = 20000
SAVE_ON_EACH_NODE = True
PREDICTION_LOSS_ONLY = True
SAVE_TOTAL_LIMIT = 3 # Save only the last 3 checkpoints
SAVE_STRATEGY = 'steps'

import torch
from transformers import LineByLineTextDataset,  TextDatasetForNextSentencePrediction, DataCollatorForLanguageModeling

# Create the dataset for next sentence prediction
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=DATASET_PATH,
    # overwrite_cache= True,
    block_size=D_WORD_VEC,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=USE_MLM,mlm_probability=MLM_PROB)


import torch_optimizer as torch_optim
optimizer = torch_optim.Lamb(params = hier_bert_model.parameters(), lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY, betas=(BETA1, BETA2),eps=EPS)

optimizer = torch.optim.AdamW(hier_bert_model.parameters(), lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY, betas=(BETA1, BETA2),eps=EPS)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min(step / WARMUP_STEPS, 1.0)
)


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=OVERWRITE_OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    learning_rate = LEARNING_RATE,
    save_steps=SAVE_STEPS,
    # save_on_each_node=SAVE_ON_EACH_NODE,
    # prediction_loss_only=PREDICTION_LOSS_ONLY,
    save_total_limit=SAVE_TOTAL_LIMIT,  # Save only the last n checkpoints
    save_strategy=SAVE_STRATEGY,  # Save checkpoints at the end of each epoch
    logging_dir=OUTPUT_DIR+"logs/",
    logging_steps=1000,
    # fp16=True,  # Enable mixed precision training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,

    optimizers=(optimizer, scheduler)
)
model = model,
args = training_args,
train_dataset = train_dataset if training_args.do_train else None,
eval_dataset = eval_dataset if training_args.do_eval else None,
compute_metrics = compute_metrics,
tokenizer = tokenizer,
data_collator = data_collator,

trainer.train()