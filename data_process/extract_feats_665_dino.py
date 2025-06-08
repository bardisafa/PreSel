import os
import json
import math
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import AutoImageProcessor, AutoModel

def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class ImageDataset(Dataset):
    def __init__(self, data_list, image_folder, image_processor):
        self.data_list = data_list
        self.image_folder = image_folder
        self.image_processor = image_processor

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_entry = self.data_list[idx]
        image_filename = data_entry['image']  # Adjust the key if necessary
        image_path = os.path.join(self.image_folder, image_filename)

        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f'Error loading image {image_path}: {e}')
            return None, None, None

        image = self.image_processor(image, return_tensors='pt')['pixel_values'][0]

        # Get unique_idx
        unique_idx = data_entry['unique_idx']

        return image, unique_idx, idx 

def extract_features(args):
    # Load data
    with open(args.data_path, 'r') as f:
        data = json.load(f)

    # Split data into chunks
    total_chunks = args.num_chunks
    chunk_idx = args.chunk_idx
    data_chunk = get_chunk(data, total_chunks, chunk_idx)

    # Load DINOv2 model and image processor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model_name  
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    dataset = ImageDataset(
        data_list=data_chunk,
        image_folder=args.image_folder,
        image_processor=image_processor
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    features = {}
    num_samples = len(dataset)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, unique_idxs, indices = batch
            valid_indices = [i for i, img in enumerate(images) if img is not None]
            
            if not valid_indices:
                continue  
            images = torch.stack([images[i] for i in valid_indices]).to(device)
            unique_idxs = [unique_idxs[i] for i in valid_indices]

            outputs = model(images)
            # Get the CLS token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  

            cls_embeddings = cls_embeddings.cpu()

            for idx, unique_idx in enumerate(unique_idxs):
                features[unique_idx] = cls_embeddings[idx].numpy()

    # Save features
    output_filename = f'dino_feat_chunk_{chunk_idx}_file_{args.task_num}.pt'
    output_path = os.path.join(args.output_dir, output_filename)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(features, output_path)
    print(f'Features saved to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/datasets/llava_v1_5_mix665k_notext_add_idx.json",
                        help='Path to the JSON file containing image data.')
    parser.add_argument('--task_num', type=str, default="1",
                        help='Path to the JSON file containing image data.')
    parser.add_argument('--image_folder', type=str, default="/datasets",
                        help='Folder containing the images.')
    parser.add_argument('--output_dir', type=str, default='/datasets/features',
                        help='Directory to save the extracted features.')
    parser.add_argument('--model_name', type=str, default='facebook/dinov2-base',
                        help='Name of the DINOv2 model from Hugging Face.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for DataLoader.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for DataLoader.')
    parser.add_argument('--num_chunks', type=int, default=1,
                        help='Total number of chunks to split the data into.')
    parser.add_argument('--chunk_idx', type=int, default=0,
                        help='Index of the current chunk to process.')
    args = parser.parse_args()

    extract_features(args)
