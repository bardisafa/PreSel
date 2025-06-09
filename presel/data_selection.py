import json
import random
import argparse
import os
import subprocess
import sys
sys.path.append("./presel")
from al_utils import *

def run_deepspeed_command_llava(data_path, output_dir, args):
    """
    DeepSpeed training command for LLAVA.
    
    Args:
        data_path: Path to the training data
        output_dir: Directory to save model checkpoints
        args: Command line arguments containing configuration
    """
    command = [
        "deepspeed", "--master_port=29508", "--include", "localhost:0,1,2,3", "../LLaVA/llava/train/train_mem.py",
        "--lora_enable", "True",
        "--lora_r", "128",
        "--lora_alpha", "256",
        "--mm_projector_lr", "2e-5",
        "--deepspeed", "./LLaVA/scripts/zero3.json",
        "--model_name_or_path", os.path.join(args.base_dir, "models/vicuna-7b-v1.5"),
        "--version", "v1",
        "--data_path", data_path,
        "--image_folder", os.path.join(args.base_dir, "datasets"),
        "--vision_tower", "openai/clip-vit-large-patch14-336",
        "--pretrain_mm_mlp_adapter", os.path.join(args.base_dir, "models/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin"),
        "--mm_projector_type", "mlp2x_gelu",
        "--mm_vision_select_layer", "-2",
        "--mm_use_im_start_end", "False",
        "--mm_use_im_patch_token", "False",
        "--image_aspect_ratio", "pad",
        "--group_by_modality_length", "True",
        "--bf16", "True",
        "--output_dir", output_dir,
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "16",
        "--per_device_eval_batch_size", "4",
        "--gradient_accumulation_steps", "1",
        "--evaluation_strategy", "no",
        "--save_strategy", "steps",
        "--save_steps", "50000",
        "--save_total_limit", "1",
        "--learning_rate", "2e-4",
        "--weight_decay", "0.",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--tf32", "True",
        "--model_max_length", "2048",
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", "4",
        "--lazy_preprocess", "True",
        "--report_to", "tensorboard"
    ]
    subprocess.run(command, check=True)

def run_deepspeed_command_vf(data_path, output_dir, args):
    """
    DeepSpeed training command for Vision-FLAN.
    
    Args:
        data_path: Path to the training data
        output_dir: Directory to save model checkpoints
        args: Command line arguments containing configuration
    """
    command = [
        "deepspeed", "--master_port=29504", "--include", "localhost:0,1,2,3", "../LLaVA/llava/train/train_mem.py",
        "--lora_enable", "True",
        "--lora_r", "128",
        "--lora_alpha", "256",
        "--mm_projector_lr", "2e-5",
        "--deepspeed", "./LLaVA/scripts/zero3.json",
        "--model_name_or_path", os.path.join(args.base_dir, "models/vicuna-7b-v1.5"),
        "--version", "v1",
        "--data_path", data_path,
        "--image_folder", os.path.join(args.base_dir, "datasets/images_191task_1k"),
        "--vision_tower", "openai/clip-vit-large-patch14-336",
        "--pretrain_mm_mlp_adapter", os.path.join(args.base_dir, "models/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin"),
        "--mm_projector_type", "mlp2x_gelu",
        "--mm_vision_select_layer", "-2",
        "--mm_use_im_start_end", "False",
        "--mm_use_im_patch_token", "False",
        "--image_aspect_ratio", "pad",
        "--group_by_modality_length", "True",
        "--bf16", "True",
        "--output_dir", output_dir,
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "16",
        "--per_device_eval_batch_size", "4",
        "--gradient_accumulation_steps", "1",
        "--evaluation_strategy", "no",
        "--save_strategy", "steps",
        "--save_steps", "50000",
        "--save_total_limit", "1",
        "--learning_rate", "2e-4",
        "--weight_decay", "0.",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--tf32", "True",
        "--model_max_length", "2048",
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", "4",
        "--lazy_preprocess", "True",
        "--report_to", "tensorboard"
    ]
    subprocess.run(command, check=True)


def perform_data_selection(dataset, args):
    """
    Perform data selection based on the specified dataset type and method.
    
    Args:
        dataset: The dataset to perform selection on
        args: Command line arguments containing configuration
    """
    # Calculate selection budget
    initial_num_samples = max(1, int(len(dataset) * args.percentage / 100))

    if args.dataset_type == 'llava':
        if args.method == 'presel':
            for round_num in range(1, args.num_rounds + 1):
                
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                print(f"Round {round_num} started.")
                
                # Load the samples selected in the reference set
                data_path = os.path.join(args.base_dir, 'data/round1_665k_notext.json')
                with open(data_path, 'r') as file:
                    selected_samples = json.load(file)
                
                # Filter out already selected samples
                selected_idx_set = {sample['unique_idx'] for sample in selected_samples}
                filtered_data = [entry for entry in dataset if entry['unique_idx'] not in selected_idx_set]
                
                # Combine samples selected from different tasks
                selected_unique_indices = []      
                for task_num in range(1, 11):

                    with open(os.path.join(args.base_dir, f"datasets/clustering_results/dino_5_perc_l2/new_samples_file{task_num}_rnd{round_num}_typical_T_2sqrt_s12345.json"), 'r') as file:
                        print(f"loaded file{task_num}_rnd{round_num}")
                        selected_samples = json.load(file)
                    selected_unique_indices += selected_samples
                
                set_selected_unique_indices = set(selected_unique_indices)
                if len(selected_unique_indices) < (round_num-1) * initial_num_samples:
                    print(f"Randomly selecting {(round_num-1) * initial_num_samples - len(selected_unique_indices)} samples.")
                    remaining_unique_indices = [entry['unique_idx'] for entry in filtered_data if entry['unique_idx'] not in set_selected_unique_indices]
                    selected_unique_indices += random.sample(remaining_unique_indices, (round_num-1) * initial_num_samples - len(selected_unique_indices))
                    print("random samples added")
            
                # Save selected samples for training
                selected_samples = extract_and_add_samples(dataset, data_path, selected_unique_indices)
                data_path = save_samples(selected_samples, args.base_name, round_num, args.method)
                output_dir = os.path.join(args.base_dir, f"checkpoints/llava_{args.method}_llava15_model_round{round_num}_lora")
                
                # Release GPU
                del os.environ["CUDA_VISIBLE_DEVICES"]
                
                # Training
                run_deepspeed_command_llava(data_path, output_dir, args)
        else:
            raise ValueError(f"Invalid method for LLaVA dataset: {args.method}.")
    
    elif args.dataset_type == 'vision_flan':
        
        initial_num_samples = max(1, int(len(dataset) * args.percentage / 100))
        
        if args.method == 'presel':
            for round_num in range(2, args.num_rounds + 1):
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                print(f"Round {round_num} started.")
                data_path = os.path.join(args.base_dir, "data/vf_rnd1_5per_samples.json")
                with open(data_path, 'r') as file:
                    selected_samples = json.load(file)
                selected_idx_set = {sample['unique_idx'] for sample in selected_samples}
                filtered_data = [entry for entry in dataset if entry['unique_idx'] not in selected_idx_set]
                selected_unique_indices = [] 
                 
                with open(os.path.join(args.base_dir, 'data/vf_tasks/task_names.json'), 'r') as file:
                    task_names = json.load(file)    
                for task_num in task_names:
                    with open(os.path.join(args.base_dir, f"datasets/clustering_results/dino_vf_selected_samples/typical_T_sqrt_5per/new_samples_{task_num}_rnd{round_num}_typical_Tsqrt_5per_vf.json"), 'r') as file:
                        print(f"loaded file{task_num}_rnd{round_num}")
                        selected_samples = json.load(file)
                    selected_unique_indices += selected_samples
                set_selected_unique_indices = set(selected_unique_indices)
                if len(selected_unique_indices) < (round_num-1) * initial_num_samples:
                    print(f"Randomly selecting {(round_num-1) * initial_num_samples - len(selected_unique_indices)} samples.")
                    remaining_unique_indices = [entry['unique_idx'] for entry in filtered_data if entry['unique_idx'] not in set_selected_unique_indices]
                    selected_unique_indices += random.sample(remaining_unique_indices, (round_num-1) * initial_num_samples - len(selected_unique_indices))
                    print("random samples added")
            
                selected_samples = extract_and_add_samples(dataset, data_path, selected_unique_indices)
                data_path = save_samples(selected_samples, args.base_name, round_num, args.method)
                output_dir = os.path.join(args.base_dir, f"/checkpoints/llava_{args.method}_vf_model_round{round_num}_lora")
                del os.environ["CUDA_VISIBLE_DEVICES"]
                run_deepspeed_command_vf(data_path, output_dir, args)
        else:
            raise ValueError(f"Invalid method for Vision-FLAN dataset: {args.method}.")
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}. Must be 'llava' or 'vision_flan'.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform multi-round data selection on a dataset")
    parser.add_argument('--file_path', type=str, default="/datasets/llava_v1_5_mix665k_notext_add_idx.json", help="Visual instruction tuning dataset path")
    parser.add_argument('--num_rounds', type=int, default=4, help="Number of rounds of data selection")
    parser.add_argument('--base_dir', type=str, default="/data/01/sylo/", help="Base directory for all data paths")
    parser.add_argument('--percentage', type=float, default=5, help="Percentage of selected samples for each round")
    parser.add_argument('--base_name', type=str, default="/selected/", help="Base name for saving each round's selected samples")
    parser.add_argument('--dataset_type', type=str, default="llava", choices=["llava", "vision_flan"], help="Dataset type to use")
    parser.add_argument('--method', type=str, default="presel", choices=["presel"], help="Method Name.")
    parser.add_argument('--seed', type=int, default=12345, help="Seed")
    parser.add_argument("--lr_mlp", type=float, default=1e-1, help="Learning rate for the loss predictor MLP")
    args = parser.parse_args()
    
    # full paths
    args.file_path = os.path.join(args.base_dir, args.file_path)
    args.base_name = os.path.join(args.base_dir, args.base_name)

    # random seed for reproducibility
    set_seed(args.seed)
    
    # Load dataset and start data selection process
    dataset = load_dataset(args.file_path)
    perform_data_selection(dataset, args)