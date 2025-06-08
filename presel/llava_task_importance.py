import json
import os
import numpy as np
import argparse

def get_ratio(data_w, data_wo):
    data_wo_dict = {item['unique_idx']: item for item in data_wo}
    result = []
    for item1 in data_w:
        unique_idx = item1['unique_idx']
        perplexity1 = item1['loss']
        
        if unique_idx in data_wo_dict:
            perplexity2 = data_wo_dict[unique_idx]['loss']
            
            ratio = perplexity1 / perplexity2
            result.append({
                'unique_idx': unique_idx,
                'perplexity_ratio': ratio
            })
    return result

def calculate_task_difficulties(data_w_path, data_wo_path, reference_data_path, task_files_dir, output_dir):
    # Load data with and without Q tokens
    data_w = json.load(open(data_w_path))
    data_wo = json.load(open(data_wo_path))

    # Calculate perplexity ratios
    data_irs = get_ratio(data_w, data_wo)

    # Load reference data
    with open(reference_data_path, 'r') as file:
        selected_samples = json.load(file)
    selected_idx_set = {sample['unique_idx'] for sample in selected_samples}
    filtered_data = [entry for entry in data_irs if entry['unique_idx'] in selected_idx_set]

    task_difficulties = []

    # Calculate difficulties for each task
    for i in range(1, 11):
        with open(os.path.join(task_files_dir, f"file{i}_665.json"), 'r') as f:
            task_data = json.load(f)
            # Get unique indices for this task
            task_indices = {item['unique_idx'] for item in task_data}
            
            # Get perplexity ratios for these indices from filtered_data
            perplexities = [entry['perplexity_ratio'] for entry in filtered_data if entry['unique_idx'] in task_indices]
            
            # Calculate and store average
            avg_perplexity = np.mean(perplexities)
            print(f"Task {i}:")
            print(f"  Number of samples: {len(perplexities)}")
            print(f"  Average perplexity: {avg_perplexity:.4f}")
            task_difficulties.append(avg_perplexity)

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'task_difficulties.json')
    with open(output_path, 'w') as f:
        json.dump(task_difficulties, f, indent=4)

   
    return task_difficulties

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate task difficulties from perplexity ratios')
    parser.add_argument('--data_w_path', type=str, default='/data/loss_ppl_round1_665k_notext.json',
                      help='Path to data with Q tokens')
    parser.add_argument('--data_wo_path', type=str, default='/data/loss_ppl_round1_665k_notext_img_token.json',
                      help='Path to data without Q tokens')
    parser.add_argument('--reference_data_path', type=str, default='/data/round1_665k_notext.json',
                      help='Path to reference data')
    parser.add_argument('--task_files_dir', type=str, default='/data',
                      help='Directory containing task files (file1_665.json through file10_665.json)')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Directory to save task difficulties')

    args = parser.parse_args()
    calculate_task_difficulties(
        args.data_w_path,
        args.data_wo_path,
        args.reference_data_path,
        args.task_files_dir,
        args.output_dir
    )
