import torch
from transformers import BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig

model_id = "EleutherAI/gpt-neox-20b"

EMBEDDING_DIMENSIONS = {"thenlper/gte-base": 768}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["query_key_value"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="QA"
)

training_config = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_steps=2,
    max_steps=2,
    learning_rate=2e-4,
    # weight_decay=0.01,
    fp16=True,
    optim="paged_adamw_8bit",
    remove_unused_columns=False,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    logging_steps=1,
    save_steps=1,
    eval_steps=1
)
