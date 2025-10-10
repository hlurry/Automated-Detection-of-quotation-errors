import os
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
import logging
import sys


os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHECKPOINT = "checkpoint" #checkpoint

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
    def __init__(self):
        self.model_dir = os.path.join(SCRIPT_DIR, "scibert_250_full5")  # path/to/checkpoint
        self.checkpoint = CHECKPOINT  
        self.test_data_path = os.path.join(SCRIPT_DIR, "data5", "test_fold5.csv")  # path/to/test.csv
        self.output_path = os.path.join(SCRIPT_DIR, "data5", "output_250_full.csv")  # path/to/output
        self.max_length = 512  
        self.batch_size = 32 

def load_model_and_tokenizer(model_dir, checkpoint=None):
    logger.info(f"Load the tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    if checkpoint and os.path.exists(os.path.join(model_dir, checkpoint)):
        model_path = os.path.join(model_dir, checkpoint)
        logger.info(f"from checkpoint {model_path} load model")
    else:
        model_path = model_dir
        logger.info(f"from {model_path} load model")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    logger.info(f"Move the model to the device: {device}")
    model = model.to(device)
    
    model.eval()
    
    return model, tokenizer

def load_test_data(test_data_path):
    logger.info(f"from {test_data_path} load test dataset")
    df = pd.read_csv(test_data_path, encoding='utf-8')
    
    assert 'claim' in df.columns, "The dataset must contain a 'claim' column"
    assert 'abstract' in df.columns, "The dataset must contain an 'abstract' column"
    
    df['claim'] = df['claim'].astype(str).str.strip()  
    df['abstract'] = df['abstract'].astype(str).str.strip()  
    
    return df

def load_label_map(data_dir):
    label_map_path = os.path.join(data_dir, "label_map.txt")
    logger.info(f"Load the label mapping from {label_map_path}")
    
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f"The label mapping file {label_map_path} does not exist")
    
    label_map_items = []
    with open(label_map_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                label_map_items.append(line)
    
    label_map_str = ",".join(label_map_items)
    logger.info(f"Load the label mapping: {label_map_str}")
    
    return label_map_str

def get_predictions(model, tokenizer, df, args):
    logger.info("Start prediction")
    

    id_to_label = {}
    for item in args.label_map.split(','):
        label, idx = item.split(':')
        id_to_label[int(idx)] = label
    
    predictions = []
    

    for i in range(0, len(df), args.batch_size):
        batch_df = df.iloc[i:i+args.batch_size]  
        
        inputs = tokenizer(
            batch_df['claim'].tolist(),
            batch_df['abstract'].tolist(),
            padding="max_length", 
            truncation=True,     
            max_length=args.max_length, 
            return_tensors="pt"    
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad(): 
            outputs = model(**inputs)
            logits = outputs.logits  
            
            logits_cpu = logits.cpu()
            pred_ids = torch.argmax(logits_cpu, dim=1).numpy() 
            
            probs = torch.nn.functional.softmax(logits_cpu, dim=1).numpy()  
            
            pred_labels = [id_to_label[pred_id] for pred_id in pred_ids]
            
            for j, pred_label in enumerate(pred_labels):
                idx = i + j
                if idx < len(df):
                    row = batch_df.iloc[j]
                    pred_prob = probs[j][pred_ids[j]] 
                    
                    predictions.append({
                        'id': idx,
                        'claim': row['claim'],
                        'abstract': row['abstract'],
                        'label': pred_label,
                        'confidence': pred_prob 
                    })
        
        logger.info(f"Processed  {min(i+args.batch_size, len(df))}/{len(df)} pieces of data")
    
    predictions_df = pd.DataFrame(predictions)
    
    return predictions_df

def main():
    args = Args()
    
    if not os.path.exists(args.model_dir):
        raise ValueError(f"The model directory {args.model_dir} does not exist; please train the model first")
    
    data_dir = os.path.dirname(args.test_data_path)
    args.label_map = load_label_map(data_dir)
    
    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.checkpoint)
    
    test_df = load_test_data(args.test_data_path)
    
    predictions_df = get_predictions(model, tokenizer, test_df, args)
    
    if args.checkpoint:
        output_path = args.output_path.replace('.csv', f'_{args.checkpoint}.csv')
    else:
        output_path = args.output_path
    
    logger.info(f"Save the prediction results to {output_path}")
    predictions_df.to_csv(output_path, index=False)
    
    label_counts = predictions_df['label'].value_counts()  
    logger.info(f"Predicted label distribution:\n{label_counts}")
    
    logger.info("Prediction completed")

if __name__ == "__main__":
    main()