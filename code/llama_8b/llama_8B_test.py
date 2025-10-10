from unsloth import FastLanguageModel
from peft import PeftModel
import torch
import json
import os
from tqdm import tqdm

max_seq_length = 2500
dtype = None
load_in_4bit = True
BASE_MODEL = "/path/to/your/model"  # Path to the DeepSeek-R1-Distill-Llama-8B model

CHECKPOINT_NUMBER = None
OUTPUT_DIR = "./outputs_250_cia5"

TEST_FILE = "test_fold5.json"
OUTPUT_FILE = "output.json"

def find_all_checkpoints():
    if not os.path.exists(OUTPUT_DIR):
        raise FileNotFoundError(f"Directory does not exist: {OUTPUT_DIR}")
    
    checkpoint_dirs = [d for d in os.listdir(OUTPUT_DIR) 
                      if d.startswith('checkpoint-') and 
                      os.path.isdir(os.path.join(OUTPUT_DIR, d))]
    
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint folder found in the {OUTPUT_DIR} directory")
    
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
    
    if CHECKPOINT_NUMBER is None:
        checkpoint_paths = [os.path.join(OUTPUT_DIR, d) for d in checkpoint_dirs]
        checkpoint_numbers = [int(d.split('-')[1]) for d in checkpoint_dirs]
        print(f"All checkpoints will be run sequentially: {[d for d in checkpoint_dirs]}")
        return list(zip(checkpoint_paths, checkpoint_numbers))
    else:
        target_checkpoint = f"checkpoint-{CHECKPOINT_NUMBER}"
        if target_checkpoint in checkpoint_dirs:
            checkpoint_path = os.path.join(OUTPUT_DIR, target_checkpoint)
            print(f"Use the specified checkpoint: {checkpoint_path}")
            return [(checkpoint_path, CHECKPOINT_NUMBER)]
        else:
            available_checkpoints = [int(d.split('-')[1]) for d in checkpoint_dirs]
            raise FileNotFoundError(
                f"checkpoint-{CHECKPOINT_NUMBER}"
                f"Available checkpoint: {sorted(available_checkpoints)}"
            )

checkpoint_info_list = find_all_checkpoints()

inference_prompt_style = """
### Question:
You are an expert in the medical field and are familiar with other scientific fields, and now you will receive a scientific claim and an abstract of the scientific literature. According to the given claim and abstract, infer whether the viewpoints of the abstract and the claim are consistent. If there is no clear relationship between the content of the claim and the abstract, please return NULL.Please determine whether the content of the abstract supports or contradicts the viewpoint of the claim. If it supports, return SUPPORT; if it contradicts, return CONTRADICT; if there is no clear relationship, return NULL.{}

### Response:
<think>
"""

def run_inference(test_file_path, output_file_path, model, tokenizer):
    print(f"Loading test data from {test_file_path}...")
    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Total test samples: {len(test_data)}")
    
    results = []
    
    for i, sample in enumerate(tqdm(test_data, desc="Running inference")):
        question = sample["Question"]
        
        prompt = inference_prompt_style.format(question)
        
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1200,
            use_cache=True,
        )
        
        response = tokenizer.batch_decode(outputs)
        generated_text = response[0]
        
        if "### Response:" in generated_text:
            predicted_response = generated_text.split("### Response:")[1].strip()
        else:
            predicted_response = generated_text.strip()
        
        result = {
            "Question": question,
            "Response": predicted_response
        }
        
        results.append(result)
    
    print(f"Saving results to {output_file_path}...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    test_file = TEST_FILE
    
    for checkpoint_path, checkpoint_number in checkpoint_info_list:
        print(f"\nLoading the checkpoint model: {checkpoint_path}")
        
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL, 
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        
        model = PeftModel.from_pretrained(
            base_model,
            checkpoint_path,
            device_map={"": 0},  
        )
        
        FastLanguageModel.for_inference(model)
        
        base_name = os.path.splitext(OUTPUT_FILE)[0]  
        extension = os.path.splitext(OUTPUT_FILE)[1]  
        output_file = f"{base_name}_{checkpoint_number}{extension}"
        
        print(f"Output file: {output_file}")
        

        run_inference(test_file, output_file, model, tokenizer)
        
        del model, base_model, tokenizer
        torch.cuda.empty_cache()
        
        print(f"Checkpoint {checkpoint_number} Processing completed\n")
    
    print("Processing completed")