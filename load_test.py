import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import numpy as np
import random
import os
import torch.fx as fx
import inspect


from huggingface_hub import login
login()

print('# of gpus: ', torch.cuda.device_count())


seed = 29
    
# set random seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# model_name = "mistralai/Mistral-7B-v0.1"
# model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "meta-llama/Llama-3.2-1B"
model_name = "Qwen/Qwen2.5-0.5B"

print("Loading model:", model_name)

cache_dir="llm_weights"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    cache_dir=cache_dir, 
    low_cpu_mem_usage=True, 
)

model.to(device)

model.seqlen = model.config.max_position_embeddings 

model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("\n===== MODEL ARCHITECTURE =====\n")
print(model)

model_n = model_name.split('/')[1]

with open(model_n + ".txt", "w+") as f:
    # traced = fx.symbolic_trace(model.model.layers[0])
    layer = model.model.layers[0]
    f.write("\n===== execution order =====\n")
    f.write(f'{inspect.getsource(layer.forward)}\n')
    f.write("\n===== attention execution order =====\n")
    f.write(f'{inspect.getsource(layer.self_attn.forward)}\n')
    f.write("\n===== mlp execution order =====\n")
    f.write(f'{inspect.getsource(layer.mlp.forward)}\n')
    f.write(f"\n\n {model}\n\n")
    
    

print("\n===== PARAMETER SUMMARY =====\n")

total_params = 0
layer_params = defaultdict(int)

for name, param in model.named_parameters():
    num = param.numel()
    total_params += num

    # group by top-level module
    layer_name = name.split('.')[0]
    layer_params[layer_name] += num

    print(f"{name:60} {num/1e6:8.2f} M")

print("\n===== PARAMS PER TOP MODULE =====")

for k,v in layer_params.items():
    print(f"{k:20} {v/1e6:.2f} M")

print("\nTOTAL PARAMETERS:", total_params/1e9, "B")


# prompt = "Hello"
# model_inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

# print(type(model_inputs))
# print(model_inputs.shape)

# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=512
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(response)

def main():
    pass

if __name__ == '__main__':
    main()