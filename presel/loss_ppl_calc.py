import os
from dataclasses import dataclass, field
import json
import random
import torch
import tokenizers
import sys
from tqdm import tqdm
import argparse

sys.path.append("./LLaVA")

from llava.model import *
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (tokenizer_image_token, get_model_name_from_path,)
from packaging import version

# Import from dataset_utils instead of loss_calc
from dataset_utils import (_tokenize_fn, _mask_targets, _add_speaker_and_signal, preprocess_multimodal,
                       preprocess_llama_2, preprocess_mpt, preprocess_plain, preprocess, preprocess_v1,
                       LazySupervisedDataset, DataCollatorForSupervisedDataset, make_supervised_data_module, Args)

# -------------------
# rank0_print and seed settings
local_rank = 0  # Set appropriately if using distributed training

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

# Set the seed for reproducibility
seed = 12345
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# -------------------

# Argument parser for user-specified paths
parser = argparse.ArgumentParser(description="Evaluate loss and perplexity for a dataset using a pretrained model.")
parser.add_argument('--data_path', type=str, default='/data/01/sylo/mnt_02/data/annotation_191-task_1k_add_idx.json', help='Path to the dataset JSON file')
parser.add_argument('--model_path', type=str, default='/data/01/sylo/mnt_02/checkpoints/merged_models/llava_v1_5_7b_665k_1per_random_lora_merged', help='Path to the pretrained model checkpoint')
parser.add_argument('--image_folder', type=str, default='/data/01/sylo/mnt_02/data/images_191task_1k', help='Path to the image folder')
parser.add_argument('--output_file', type=str, default='/data/01/sylo/mnt_02/selected/samples_vf_loss_ppl_testtt.json', help='Path to save the output JSON file')
args_cli = parser.parse_args()

# args   
args = Args(
    data_path=args_cli.data_path,
    model_path=args_cli.model_path,
    lazy_preprocess=True,
    is_multimodal=True,
    image_folder=args_cli.image_folder,
    image_aspect_ratio="square"
)

##################################################################################
# Model loading
disable_torch_init()
model_name = get_model_name_from_path(args.model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_name=model_name,
        model_base=None
    )
args.image_processor = image_processor
args.mm_use_im_start_end = ( model.config.mm_use_im_start_end )
###################################################################################
# process dataset
train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_path=args.data_path, data_args=args
    )
###################################################################################
results = []
compute_dtype = (
torch.float16)

model.eval()
model.cuda()

def load_previous_results(output_file):
    if os.path.exists(output_file):
        print(f"Loading previous results from '{output_file}'")
        with open(output_file, "r") as f:
            return json.load(f)
    return []

# Save results to a JSON file
output_file = args_cli.output_file
save_frequency = 500  # Save every N iterations

# Load previously saved results
results = load_previous_results(output_file)
processed_indices = set(result["unique_idx"] for result in results)

with torch.no_grad():
    for i in tqdm(range(len(train_dataset)), desc="Evaluating loss"): 
        unique_idx = train_dataset[i]['unique_idx']
        
        if unique_idx in processed_indices:
            continue   
        
        (
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        inputs_embeds,
        labels,
        ) = model.prepare_inputs_labels_for_multimodal(input_ids=train_dataset[i]['input_ids'].unsqueeze(0).to("cuda:0",dtype=torch.long),
        position_ids=None, 
        attention_mask=None,
        past_key_values=None,
        labels=train_dataset[i]['labels'].unsqueeze(0).to("cuda:0",dtype=torch.long),
        images=train_dataset[i]['image'].unsqueeze(0).to("cuda:0",dtype=compute_dtype)
        )
        
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels
        )
        
        loss = outputs.loss
        perplexity = torch.exp(loss)
        
        # Print or use the loss as needed
        print(f"unique_idx:{train_dataset[i]['unique_idx']}, loss:{loss.item()}, perplexity:{perplexity.item()}")
        
        results.append({"unique_idx": train_dataset[i]['unique_idx'],
                        "loss": loss.item(),
                        "perplexity": perplexity.item()})
        processed_indices.add(unique_idx)
        
        # Save results to a JSON file every 'save_frequency' iterations
        if len(results) % save_frequency == 0:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)

with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

