import json
import torch
from modelscope import AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM
import os
import sys

def load_model_and_tokenizer():
    best_model_path = "/path/to/trained/checkpoint" 
    
    model_dir = "/path/to/your/model" 
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True, padding_side='left')
    base_model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)#
    model = PeftModel.from_pretrained(base_model, best_model_path)
    return model, tokenizer

def predict_batch(texts, model, tokenizer, batch_size=8):
    device = next(model.parameters()).device
    all_predictions = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        messages = [
            [
                {"role": "system", "content": "You are an expert in the medical field and are familiar with other scientific fields, and now you will receive a scientific claim and an abstract of the scientific literature. According to the given claim and abstract, infer whether the viewpoints of the abstract and the claim are consistent. If there is no clear relationship between the content of the claim and the abstract, please return NULL.Please determine whether the content of the abstract supports or contradicts the viewpoint of the claim. If it supports, return SUPPORT; if it contradicts, return CONTRADICT; if there is no clear relationship, return NULL."},
                {"role": "user", "content": text}
            ] for text in batch_texts
        ]
        
        input_texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        model_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=model_inputs['input_ids'],
                attention_mask=model_inputs['attention_mask'],
                max_new_tokens=5
            )
        
        responses = tokenizer.batch_decode(generated_ids[:, model_inputs['input_ids'].shape[1]:], skip_special_tokens=True)
#        predictions = [response.strip()[-1] for response in responses]
        predictions = [response.strip() for response in responses]
        all_predictions.extend(predictions)

    return all_predictions


def process_test_data(input_file, output_file, use_batch=True, batch_size=3):
    model, tokenizer = load_model_and_tokenizer()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_outputs = f.readlines()
        num_existing = len(existing_outputs)
    else:
        num_existing = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'a', encoding='utf-8') as outfile:
        data = [json.loads(line) for line in infile]
        data = data[num_existing:]
        
        if use_batch:
            texts = [f"text:{item['text']},category:{item['category']}" for item in data]
            predictions = predict_batch(texts, model, tokenizer, batch_size)
            
            for i, (item, prediction) in enumerate(zip(data, predictions)):
                output_data = {
                    "text": item["text"],
                    "category": item["category"],
                    "output": prediction
                }
                json.dump(output_data, outfile, ensure_ascii=False)
                outfile.write('\n')
                
                progress = f"Processed {i+1+num_existing}/{total_lines} lines"
                sys.stdout.write('\r' + progress)
                sys.stdout.flush()
        else:
            for i, item in enumerate(data):
                text = f"text:{item['text']},category:{item['category']}"
                prediction = predict_batch([text], model, tokenizer)[0]
                
                output_data = {
                    "text": item["text"],
                    "category": item["category"],
                    "output": prediction
                }
                json.dump(output_data, outfile, ensure_ascii=False)
                outfile.write('\n')
                
                progress = f"Processed {i+1+num_existing}/{total_lines} lines"
                sys.stdout.write('\r' + progress)
                sys.stdout.flush()
    
    print()
    print(f"Processing completed. There are a total of {total_lines} data entries, with {total_lines - num_existing} new entries added.")

if __name__ == "__main__":
    input_file = "/path/to/your/raw/test/dataset" 
    output_file = "/path/to/output"
    use_batch = True  
    batch_size = 3
    process_test_data(input_file, output_file, use_batch, batch_size)