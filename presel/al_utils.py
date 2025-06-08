import json
import random
import os
import torch
import math


def filter_entries(data, indices_to_remove):
    return [item for item in data if item['unique_idx'] not in indices_to_remove]

def select_entries(data, indices_to_keep):
    return [item for item in data if item['unique_idx'] in indices_to_keep]

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_dataset(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_samples(samples, base_name, round_num, method):
    save_path = os.path.join(base_name, f"{method}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print(f"Directory {save_path} already exists. Overwriting files...")
    file_name = os.path.join(save_path, f"round{round_num}.json")
    with open(file_name, 'w') as file:
        json.dump(samples, file, indent=2)
    print(f"Saved {len(samples)} samples to {file_name}")
    return file_name

def save_samples_seed(samples, base_name, round_num, method, seed):
    save_path = os.path.join(base_name, f"{method}_s{seed}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print(f"Directory {save_path} already exists. Overwriting files...")
    file_name = os.path.join(save_path, f"round{round_num}.json")
    with open(file_name, 'w') as file:
        json.dump(samples, file, indent=2)
    print(f"Saved {len(samples)} samples to {file_name}")
    return file_name

def save_remain_samples(samples, base_name, round_num, method):
    save_path = os.path.join(base_name, f"{method}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print(f"Directory {save_path} already exists. Overwriting files...")
    file_name = os.path.join(save_path, f"round{round_num}_remain.json")
    with open(file_name, 'w') as file:
        json.dump(samples, file, indent=2)
    print(f"Saved {len(samples)} samples to {file_name}")
    return file_name

def random_selection(unique_images, num_samples, seed):
    selected_samples = random.sample(unique_images, num_samples)
    return selected_samples

def get_top_n_losses(data, n):
    sorted_data = sorted(data, key=lambda x: x['loss'], reverse=True)    
    top_n_samples = sorted_data[:n]
    unique_indices = [sample['unique_idx'] for sample in top_n_samples]
    return unique_indices

def select_top_k_indices(data, loss_dict, K):
    for item in data:
        item['loss'] = loss_dict.get(item['unique_idx'], float('-inf'))
    sorted_data = sorted(data, key=lambda x: x['loss'], reverse=True)
    
    top_k_data = sorted_data[:K]
    for item in data:
        del item['loss']
        
    top_k_indices = [item['unique_idx'] for item in top_k_data]
    return top_k_indices

def select_top_k_indices_ppl(data, loss_dict, K):
    for item in data:
        item['perplexity_ratio'] = loss_dict.get(item['unique_idx'], float('-inf'))
    
    sorted_data = sorted(data, key=lambda x: x['perplexity_ratio'], reverse=True) 
    top_k_data = sorted_data[:K]
    for item in data:
        del item['perplexity_ratio']
        
    top_k_indices = [item['unique_idx'] for item in top_k_data]
    
    return top_k_indices

def select_top_k_indices_ppl_plain(data, loss_dict, K):
    for item in data:
        item['perplexity'] = loss_dict.get(item['unique_idx'], float('-inf'))
    
    sorted_data = sorted(data, key=lambda x: x['perplexity'], reverse=True)
    top_k_data = sorted_data[:K]
    
    for item in data:
        del item['perplexity']  
    top_k_indices = [item['unique_idx'] for item in top_k_data]
    
    return top_k_indices

def extract_and_add_samples(main_data, selected_samples_path, unique_indices):    
    with open(selected_samples_path, 'r') as selected_file:
        selected_data = json.load(selected_file)
    unique_idx_set = set(unique_indices)
    new_samples = [sample for sample in main_data if sample['unique_idx'] in unique_idx_set]
    selected_data.extend(new_samples)
    
    return selected_data

def get_all_instances(dataset, selected_images):
    return [entry for entry in dataset if entry['image'] in selected_images]


def find_top_n_perplexity_samples(perplexity_dict, n=2000):
    sorted_perplexities = sorted(perplexity_dict.values(), reverse=True)
    return sorted_perplexities[:n]

def calibrate_perplexity(perplexity_dict, top_n_perplexities):
    avg_top_n_perplexity = sum(top_n_perplexities) / len(top_n_perplexities)
    calibrated_perplexity_dict = {k: v / avg_top_n_perplexity for k, v in perplexity_dict.items()}
    return calibrated_perplexity_dict

def calculate_average_perplexity(unique_indices, calibrated_perplexity_dict):
    filtered_data = [calibrated_perplexity_dict[idx] for idx in unique_indices if idx in calibrated_perplexity_dict and not math.isnan(calibrated_perplexity_dict[idx])]
    if not filtered_data:
        return None  
    return sum(filtered_data) / len(filtered_data)

def calc_weights(files, total_samples, perplexity_dict, calibrate=True, n=1000):
    if calibrate: 
        calibrated_perplexity_dict = {}
        for i, file in enumerate(files):
            indices = [item['unique_idx'] for item in file]
            file_perplexity_dict = {idx: perplexity_dict[idx] for idx in indices if idx in perplexity_dict}
            
            top_n_perplexities = find_top_n_perplexity_samples(file_perplexity_dict, n=n)
            
            calibrated_file_perplexity_dict = calibrate_perplexity(file_perplexity_dict, top_n_perplexities)
            calibrated_perplexity_dict.update(calibrated_file_perplexity_dict)
        score_dict = {}
        size_dict = {}  
        for i, file in enumerate(files):
            indices = [item['unique_idx'] for item in file]
            average_perplexity = calculate_average_perplexity(indices, calibrated_perplexity_dict)
            score_dict[f"file{i+1}"] = average_perplexity
            size_dict[f"file{i+1}"] = len(indices)

        weighted_scores = {file: score_dict[file] * size_dict[file] for file in score_dict.keys()}
        total_weighted_score = sum(weighted_scores.values())
        weights = {file: (weighted_scores[file] / total_weighted_score) * total_samples for file in score_dict.keys()}
        weights_list = [int(weight) for weight in weights.values()]
        return weights_list
    else:
        score_dict = {}
        size_dict = {}  
        for i, file in enumerate(files):
            indices = [item['unique_idx'] for item in file]
            average_perplexity = calculate_average_perplexity(indices, perplexity_dict)
            score_dict[f"file{i+1}"] = average_perplexity
            size_dict[f"file{i+1}"] = len(indices)

        weighted_scores = {file: score_dict[file] * size_dict[file] for file in score_dict.keys()}
        total_weighted_score = sum(weighted_scores.values())
        weights = {file: (weighted_scores[file] / total_weighted_score) * total_samples for file in score_dict.keys()}
        weights_list = [int(weight) for weight in weights.values()]
        return weights_list
        