from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda
import torch
import datetime

device = 'cuda' if cuda.is_available() else 'cpu'
if True:
            model_name = 'sci_ab_output/5/model_large_epoch_ori_notag_5'#Model storage path
            filename = 'sci_ab/5/test_fold5_t5_qs'#test dataset
            fw_file = "sci_ab_out/5/model_epoch_sci_ab_5"#output
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            model = model.to(device)

            def run_model(input_string, **generator_args):

                input_ids = tokenizer.encode(input_string, return_tensors="pt")
                input_ids = input_ids.to(device, dtype=torch.long)
                res = model.generate(input_ids, **generator_args)
                return tokenizer.batch_decode(res, skip_special_tokens=True)

            answers = run_model("How do you descibe the hypernym relation?\\n the hypernym relation")

            with open(filename,'r', encoding='utf-8')as fr:
                with open(fw_file, 'w') as fw:
                    for line in fr:
                        answers = run_model(line.strip())
                        fw.write(answers[0]+"\n")

