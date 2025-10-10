import os
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from datasets import Dataset
import logging
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TRAIN_DATASET = "/your/train/dataset"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if torch.cuda.is_available():
    logger.info(f"GPU is available: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    logger.error("GPU is unavailable, program exits")
    sys.exit(1)  

class Args:
    """
    Configuration parameter class, which stores various parameters required during the training process
    """
    def __init__(self):
        self.data_path = os.path.join(SCRIPT_DIR, "data5", TRAIN_DATASET)  # Dataset path
        self.output_dir = os.path.join(SCRIPT_DIR, "scibert_250_full5")   # Model output directory
        self.model_name = "allenai/scibert_scivocab_uncased"  # Pretrained model
        self.max_length = 512 
        self.batch_size = 32  
        self.learning_rate = 3e-5  
        self.epochs = 10  
        self.seed = 42 
        self.class_weights = True  
        self.warmup_ratio = 0.1  

def load_data(data_path, args):
    logger.info(f"from {data_path} Load data")
    df = pd.read_csv(data_path, encoding='utf-8')
    
    assert 'claim' in df.columns, "The dataset must contain a 'claim' column"
    assert 'abstract' in df.columns, "The dataset must contain an 'abstract' column"
    assert 'label' in df.columns, "The dataset must contain a 'label' column"
    

    df['claim'] = df['claim'].astype(str).str.strip()  
    df['abstract'] = df['abstract'].astype(str).str.strip()  
    df['label'] = df['label'].fillna("NULL")  
    
 
    unique_labels = df['label'].unique()
    label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    
    logger.info(f"Automatically generated label mapping: {label_map}")
    
 
    label_map_path = os.path.join(os.path.dirname(data_path), "label_map.txt")
    with open(label_map_path, 'w', encoding='utf-8') as f:
        for label, idx in label_map.items():
            f.write(f"{label}:{idx}\n")
    logger.info(f"The label mapping has been saved to {label_map_path}")
    
 
    if not pd.api.types.is_numeric_dtype(df['label']):
        df['label'] = df['label'].map(label_map)
        if df['label'].isna().any():
            unmapped = df.loc[df['label'].isna(), 'label'].unique()
            raise ValueError(f"There are unmapped labels in the dataset: {unmapped}")
    

    train_dataset = Dataset.from_pandas(df)
    logger.info(f"Training set size: {len(train_dataset)}")
    
    return train_dataset, None, len(label_map)

def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(
        examples["claim"],
        examples["abstract"],
        padding="max_length",  
        truncation=True,       
        max_length=max_length, 
    )

def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    

    os.makedirs(args.output_dir, exist_ok=True)
    

    train_dataset, eval_dataset, num_labels = load_data(args.data_path, args)
    

    logger.info(f"Load model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=num_labels  
    )
    
  
    logger.info("Perform word segmentation on the dataset")
    tokenized_train = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_length),
        batched=True  
    )
    

    class_weights = None
    if args.class_weights:
        train_labels = train_dataset['label']
        class_counts = np.bincount(train_labels)
        class_weights = torch.FloatTensor([len(train_labels)/count for count in class_counts])
        logger.info(f"Use class weights: {class_weights}")
        
        if class_weights is not None:
            model.config.class_weights = class_weights.tolist()
    

    training_args = TrainingArguments(
        output_dir=args.output_dir,  
        num_train_epochs=args.epochs, 
        per_device_train_batch_size=args.batch_size,  
        per_device_eval_batch_size=args.batch_size,   
        warmup_ratio=args.warmup_ratio, 
        weight_decay=0.01, 
        logging_dir=os.path.join(args.output_dir, "logs"),  
        logging_steps=10,  
        eval_strategy="no", 
        save_strategy="epoch",  
        load_best_model_at_end=False,  
        learning_rate=args.learning_rate,  
        fp16=torch.cuda.is_available(), 
        report_to="none",  
        lr_scheduler_type="cosine",  
        save_total_limit=6,  
        gradient_accumulation_steps=1,  
        dataloader_num_workers=4,  
        dataloader_pin_memory=True,  
    )
    

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            if hasattr(model.config, "class_weights") and model.config.class_weights:
                loss_fct = torch.nn.CrossEntropyLoss(
                    weight=torch.tensor(model.config.class_weights).to(model.device)
                )
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=None, 
    )

    logger.info("Start training")
    trainer.train()
    

    logger.info(f"Save the model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    return None  

if __name__ == "__main__":
    args = Args()
    main(args)