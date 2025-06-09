import os
import torch
import sys
import numpy as np
import argparse
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import json
from scipy.special import softmax
import torch.nn.functional as F
import faiss
sys.path.append("./data_process")
from clust_utils import *

def load_task_features(task_samples, features_dict):
    
    unique_idxs = []
    feature_vectors = []
    for sample in task_samples:
        unique_idx = sample['unique_idx']
        if unique_idx in features_dict:
            unique_idxs.append(unique_idx)
            feature_vectors.append(features_dict[unique_idx])
        else:
            print(f"Warning: Feature for unique_idx {unique_idx} not found.")
    if feature_vectors:
        feature_vectors = np.stack(feature_vectors)
    else:
        feature_vectors = np.array([])
    return unique_idxs, feature_vectors

def typical_sample_selection(unique_idxs, feature_vectors, labels, N, already_selected_idxs):
    
    total_samples = len(unique_idxs)
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}

    # Prepare data structures
    idx_to_unique_idx = {i: uid for i, uid in enumerate(unique_idxs)}
    unique_idx_to_idx = {uid: i for i, uid in enumerate(unique_idxs)}
    # Exclude already selected samples
    available_indices = set(range(total_samples)) - {unique_idx_to_idx[uid] for uid in already_selected_idxs if uid in unique_idx_to_idx}
    
    selected_indices = []

    # Calculate number of samples to select from each cluster
    total_available_samples = len(available_indices)
    cluster_sample_counts = {}
    for label in unique_labels:
        cluster_available_indices = set(cluster_indices[label]) & available_indices
        cluster_size = len(cluster_available_indices)
        proportion = cluster_size / total_available_samples if total_available_samples > 0 else 0
        samples_to_select = int(round(proportion * N))
        cluster_sample_counts[label] = min(samples_to_select, cluster_size)
        
    total_selected = sum(cluster_sample_counts.values())
    if total_selected < N:
        deficit = N - total_selected
        sorted_labels = sorted(unique_labels, key=lambda l: counts[l], reverse=True)
        for label in sorted_labels:
            cluster_available_indices = set(cluster_indices[label]) & available_indices
            cluster_size = len(cluster_available_indices)
            if cluster_sample_counts[label] < cluster_size:
                cluster_sample_counts[label] += 1
                deficit -=1
                if deficit == 0:
                    break
    elif total_selected > N:
        surplus = total_selected - N
        sorted_labels = sorted(unique_labels, key=lambda l: counts[l])
        for label in sorted_labels:
            if cluster_sample_counts[label] > 0:
                cluster_sample_counts[label] -= 1
                surplus -=1
                if surplus == 0:
                    break

    #  select samples from each cluster based on typicality
    for label in unique_labels:
        cluster_available_indices = list(set(cluster_indices[label]) & available_indices)
        num_to_select = cluster_sample_counts[label]
        if num_to_select > 0 and len(cluster_available_indices) > 0:
            cluster_features = feature_vectors[cluster_available_indices]
            typicality = calculate_typicality(cluster_features, min(5, len(cluster_available_indices)-1))
            sorted_indices = np.argsort(-typicality)  
            selected_in_cluster = [cluster_available_indices[i] for i in sorted_indices[:num_to_select]]
            selected_indices.extend(selected_in_cluster)

    selected_unique_idxs = [idx_to_unique_idx[i] for i in selected_indices]

    return selected_unique_idxs



def main(args):
    print(f'Loading features from {args.features_path}')
    features_dict = torch.load(args.features_path)
    features_dict = {int(k.item()): v for k, v in features_dict.items()}

    nan_keys = []
    for key, array in features_dict.items():
        if np.isnan(array).any():
            nan_keys.append(key)
    if nan_keys:
        print(f"Found NaN values in the following keys: {nan_keys}")
    else:
        print("No NaN values found in features_dict.")
    
    with open(args.task_dict_path, 'r') as file:
        task_dict = json.load(file)
    print(f'Total tasks: {len(task_dict)}')

    # Load already selected unique_idxs
    if args.already_selected_file and os.path.exists(args.already_selected_file):
        with open(args.already_selected_file, 'r') as file:
            selected_samples = json.load(file)
        already_selected_idxs = {sample['unique_idx'] for sample in selected_samples}
        print(f'Loaded {len(already_selected_idxs)} already selected unique_idxs from {args.already_selected_file}')
    else:
        already_selected_idxs = set()
        print('No already selected samples found.')

    os.makedirs(args.save_path, exist_ok=True)

    with open('/data/task_diff_vf_rnd1_5per_samples.json', 'r') as f:
        task_difficulties = json.load(f)
    
    with open('/data/vf_tasks/task_names.json', 'r') as file:
        task_names = json.load(file)
    
    adjusted_allocations = allocate_samples(task_difficulties, task_dict, args.N, 1 / np.sqrt(len(task_names)))
    for task_name, task_samples in task_dict.items():
        print(f'\nProcessing task: {task_name}')
        args.task_num = task_name 
        unique_idxs, feature_vectors = load_task_features(task_samples, features_dict)
        if len(unique_idxs) == 0:
            print(f'No features found for task {task_name}')
            continue

        # Normalize features
        feature_vectors_normalized = normalize(feature_vectors, norm='l2')
        clustering_output_path = args.clustering_output_path

        # Determine number of clusters
        n_clusters = max(1, int(0.01 * len(unique_idxs)))
        print(f'Number of clusters: {n_clusters}')

        # Load clustering results or perform clustering
        labels, cluster_centers = load_clustering_results(args, clustering_output_path)
        if labels is None or cluster_centers is None:
            labels, cluster_centers = perform_clustering(args, feature_vectors_normalized, n_clusters, clustering_output_path, 'kmeans')
        else:
            pass 
        print("clustering of this task is done")
        # # Select N samples
        task_budget = adjusted_allocations[task_name]
        
        if args.method == "typical":
            selected_unique_idxs = typical_sample_selection(unique_idxs, feature_vectors_normalized, labels, task_budget, already_selected_idxs)
        else:
            raise ValueError(f"Unknown method: {args.method}")

        # Save selected unique_idxs
        save_path = os.path.join(args.save_path, f'new_samples_{task_name}_rnd{args.round}_{args.method}_Tsqrt_5per_vf.json')
        with open(save_path, 'w') as file:
            json.dump(selected_unique_idxs, file, indent=2)
        print(f"Saved {len(selected_unique_idxs)} samples to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', type=str, default='/data/dino_feats_vf/dino_feats_all_vf.pt',
                        help='Path to the features file (.pt) or directory containing feature files.')
    parser.add_argument('--task_dict_path', type=str, default='/data/vf_tasks/task_dict.json',
                        help='Path to the task dictionary JSON file.')
    parser.add_argument('--clustering_output_path', type=str, default='/datasets/clustering_results/dino_vf/',
                        help='Path to save clustering results.')
    parser.add_argument('--save_path', type=str, default='/datasets/clustering_results/dino_vf_selected_samples/',
                        help='Path to save selected samples.')
    parser.add_argument('--method', type=str, default="typical",
                        choices=["random", "typical"], 
                        help="Method Name")
    parser.add_argument('--round', type=int, default=2,
                        help='Round number.')
    parser.add_argument('--base_dir', type=str, default='/data/01/sylo/',
                        help='Base directory')
    parser.add_argument('--N', type=int, default=5000,
                        help='Total number of samples to select per task.')
    parser.add_argument('--already_selected_file', type=str, default='/data/vf_rnd1_5per_samples.json',
                        help='Path to file containing already selected unique_idxs (JSON list).')
    parser.add_argument('--output_selected_file', type=str, default='selected_unique_idxs.txt',
                        help='Path to save the selected unique_idxs.')
    parser.add_argument('--use_pca', action='store_true',
                        help='Use PCA for dimensionality reduction before clustering.')
    parser.add_argument('--pca_components', type=int, default=50,
                        help='Number of PCA components to keep if using PCA.')
    parser.add_argument('--n_clusters', type=int, default=10,
                        help='Number of clusters for KMeans.')
    args = parser.parse_args()
    # Modify args based on args.base_dir
    args.save_path = os.path.join(args.base_dir, 'datasets', 'clustering_results', 'dino_vf_selected_samples', f'{args.method}_T_sqrt_5per')
    
    data_path = os.path.join(args.base_dir, 'datasets', 'annotation_191-task_1k_add_idx.json')
    with open(data_path, 'r') as file:
        dataset = json.load(file)

    initial_num_samples = max(1, int(len(dataset) * 5 / 100))
    for rnd in [2,3,4]:
        args.round = rnd
        args.N = (rnd - 1) * initial_num_samples
        main(args)
    