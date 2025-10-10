import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq,TrainerCallback, TrainerControl, TrainerState
import os
import swanlab


def dataset_jsonl_transfer(origin_path, new_path):
    """
    Convert raw dataset to fine-tuning format for large models
    """
    messages = []

    with open(origin_path, "r") as file:
        for line in file:
            data = json.loads(line)
            context = data["text"]
            category = data["category"]
            label = data["output"]
            message = {
                "instruction": "You are an expert in the medical field and are familiar with other scientific fields, and now you will receive a scientific claim and an abstract of the scientific literature. According to the given claim and abstract, infer whether the viewpoints of the abstract and the claim are consistent. If there is no clear relationship between the content of the claim and the abstract, please return NULL.Please determine whether the content of the abstract supports or contradicts the viewpoint of the claim. If it supports, return SUPPORT; if it contradicts, return CONTRADICT; if there is no clear relationship, return NULL. ",
                "input": f"text:{context},category:{category}",
                "output": label,
            }
            messages.append(message)

    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
            
            
def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 3000 
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\nYou are an expert in the medical field and are familiar with other scientific fields, and now you will receive a scientific claim and an abstract of the scientific literature. According to the given claim and abstract, infer whether the viewpoints of the abstract and the claim are consistent. If there is no clear relationship between the content of the claim and the abstract, please return NULL.Please determine whether the content of the abstract supports or contradicts the viewpoint of the claim. If it supports, return SUPPORT; if it contradicts, return CONTRADICT; if there is no clear relationship, return NULL.<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(response)
     
    return response
    
model_dir = snapshot_download("/path/to/your/model", cache_dir="./", revision="master") # Path to the Qwen-7B

tokenizer = AutoTokenizer.from_pretrained("/path/to/your/model", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/path/to/your/model", device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()

train_dataset_path = "/path/to/your/raw/train/dataset" 
test_dataset_path = "/path/to/your/raw/test/dataset"

train_jsonl_new_path = "/path/to/your/new/train/dataset"
test_jsonl_new_path = "/path/to/your/new/train/dataset"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, config)

class SaveFromEpochCallback(TrainerCallback):
    def __init__(self, start_epoch=5):
        self.start_epoch = start_epoch

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if int(state.epoch) >= self.start_epoch:
            control.should_save = True  
        return control
        
args = TrainingArguments(
    output_dir="/path/to/trained/model", #Path to save the trained model
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=10,
    save_strategy="no",
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

swanlab_callback = SwanLabCallback(project="Qwen-fintune", experiment_name="Qwen1.5-7B-Chat")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[
        swanlab_callback,
        SaveFromEpochCallback(start_epoch=5)  # 
    ],
)

trainer.train()

swanlab.finish()