from pprint import pprint
import os

import ray
from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig

import torch
from transformers import Trainer, GPTNeoXForQuestionAnswering
from peft import prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset

from src.config import model_id, bnb_config, lora_config, training_config
from src.preprocess import preprocess_function


ray.init()
pprint(ray.cluster_resources())

use_gpu = True  # set this to False to run on CPUs
num_workers = 1  # set this to number of GPUs or CPUs you want to use
trainer_resources = {"CPU": 1}
resources_per_worker = {"CPU": 1, "GPU": 1}

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

data = load_dataset("squad", split="train[:200]")
data = data.train_test_split(0.2)
ray_datasets = {
    "train": ray.data.from_huggingface(data["train"]),
    "validation": ray.data.from_huggingface(data["test"]),
}

def train_func():
    print(f"Is CUDA available: {torch.cuda.is_available()}")

    model = GPTNeoXForQuestionAnswering.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    # print_trainable_parameters(model)

    train_ds = ray.train.get_dataset_shard("train")
    eval_ds = ray.train.get_dataset_shard("eval")

    train_ds_iterable = train_ds.iter_torch_batches(
        batch_size=1, collate_fn=preprocess_function
    )
    eval_ds_iterable = eval_ds.iter_torch_batches(
        batch_size=1, collate_fn=preprocess_function
    )

    args = training_config

    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds_iterable,
        eval_dataset=eval_ds_iterable,
    )

    model.config.use_cache = False

    trainer.add_callback(RayTrainReportCallback())

    trainer = prepare_trainer(trainer)

    print("Starting training")
    trainer.train()

trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(trainer_resources, num_workers, use_gpu, resources_per_worker),
    datasets={
        "train": ray_datasets["train"],
        "eval": ray_datasets["validation"],
    },
    run_config=RunConfig(
        storage_path=os.environ["EXPERIMENT_PATH"],
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="eval_loss",
            checkpoint_score_order="min"
        ),
    ),
)

# Training with Ray Train
result = trainer.fit()
