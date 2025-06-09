<div align="center">
  
# PreSel: Pre-Instruction Data Selection <br> for Visual Instruction Tuning
<img src="https://img.shields.io/badge/CVPR-2025-FFA500?style=for-the-badge&logo=google-scholar&logoColor=white">

**🌟 CVPR 2025 Highlight Paper 🌟**

[Bardia Safaei](https://sites.google.com/view/bardiasafaei) &emsp; [Faizan Siddiqui](https://www.linkedin.com/in/faizan-sid/) &emsp; [Jiacong Xu](https://xujiacong.github.io/) &emsp; [Vishal M. Patel](https://engineering.jhu.edu/faculty/vishal-patel/ ) &emsp; [Shao-Yuan Lo](https://shaoyuanlo.github.io/)

Johns Hopkins University, Honda Research Institute USA

<a href='https://bardisafa.github.io/PreSel/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://arxiv.org/abs/2503.07591'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>

</div>
<hr />

## Release Notes 
- [06/08/2025]: 🔥 **PreSel** codebase is released and open to access. The selected 15% data and the finetuned models on these selected data can be downloaded now.

<hr />

## Contents
- [Installation](#installation)
  - [Prepare the Environment](#1-prepare-the-environment)
- [Dataset Preparation](#dataset-preparation)
  - [Download the Datasets](#1-download-the-datasets)
  - [Preprocess the Dataset](#2-preprocess-the-dataset)
  - [Task Splits](#3-task-splits)
  - [Reference Model Training](#4-reference-model-training)
- [Usage](#usage)
  - [Loss/Perplexity Calculations](#1-lossperplexity-calculations)
  - [Task Importance Estimation](#2-task-importance-estimation)
  - [Pre-Instruction Data Selection](#3-pre-instruction-data-selection)
  - [Running on the Vision-FLAN Dataset](#running-on-the-vision-flan-dataset)
- [Finetuned Models & Selected Data](#finetuned-models--selected-data-15)
- [Evaluation](#evaluation)

<hr />

## Installation

### 1. Prepare the Environment
Please first install LLaVA：

```bash
cd PreSel
git clone https://github.com/haotian-liu/LLaVA.git
```

Then prepare the environment for LLaVA [here](https://github.com/haotian-liu/LLaVA).

## Dataset Preparation

### 1. Download the Datasets

#### LLaVA-665K Dataset
For the LLaVA dataset, please download the LLaVA-665K dataset following the instructions from the [LLaVA GitHub repository](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#train). This dataset is used for visual instruction tuning and contains a diverse set of visual-language examples.

#### Vision-FLAN Dataset
For the Vision-FLAN dataset, please download the data from the [Vision-FLAN website](https://vision-flan.github.io/#download). This dataset provides a comprehensive collection of visual-language tasks for instruction tuning.

After downloading the datasets, please place all data files in the `/datasets` directory. 

### 2. Preprocess the Dataset
We first add a unique index for each instruction in the original dataset, to better identify each sample:

```bash
python data_process/preprocess.py \
    --raw_annotation_path datasets/your_dataset.json \
    --new_annotation_save_path datasets/processed_dataset.json
```

This script adds a unique identifier to each sample in your dataset, which is essential for the data selection process. The processed dataset will be saved to the specified path. We will be using the json files with the unique_idx included in the code. 

Please note that as stated in the paper, for the LLaVA-1.5 dataset we remove the text-only instructions from the data, as our method focuses on selecting the images. You can either remove them yourself or use the already processed json file [here](https://drive.google.com/file/d/1j8qBxaHTiLVuBKX04Upsqlh7DdlguAfJ/view?usp=sharing).

### 3. Task Splits
For our method, we need to split the dataset into different tasks. We provide the task splits used in our experiments:

- LLaVA-1.5 task splits: [Download splits](https://drive.google.com/file/d/1g0tns1MOpSgdS_v91T99sB6vDu5R4DKX/view?usp=sharing)
- Vision-FLAN dataset: [Download splits](https://drive.google.com/file/d/19McAjYnghWteV93I-omTnEXDnZUKEceO/view?usp=sharing)

Place the downloaded and unzipped task split files in the `data/` directory.

### 4. Reference Model Training
To estimate task importance values, we need a reference model trained on a small randomly selected reference dataset. You have two options:

#### Option 1: Use Our Pre-selected Reference Datasets 
For LLaVA-1.5 and Vision-FLAN datasets, you can directly use our randomly selected reference datasets (5% of images and their corresponding instructions from each task):

- LLaVA-1.5 reference data (randomly selected 5% images with instructions): [Download JSON](https://drive.google.com/file/d/1zhvJExZNOxumJC9GIHLHHHRlsMaNcs2z/view?usp=sharing)
- Vision-FLAN reference data (randomly selected 5% images with instructions): [Download JSON](https://drive.google.com/file/d/1ZO63umnUZnA0McYdugvCE0TPn-GPaBXS/view?usp=sharing)

Place the downloaded JSON files in the `data/` directory.

#### Option 2: Create Your Own Reference Dataset
For custom datasets, you'll need to create a reference dataset by randomly sampling 5% of images along with their corresponding instructions from each task.

After preparing the reference dataset, fine-tune a LLaVA-7B model on it to obtain the reference model. For this step:

Fine-tune the LLaVA-7B model [huggingface](https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5) using LoRA training following the script provided [here](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_lora.sh)

This reference model will be used in later steps to estimate task-importance values.

## Usage
### 1. Loss/Perplexity Calculations

First, process the reference data to remove the question parts of the instructions:

```bash
python data_process/remove_instruction.py \
    --input_path /data/round1_665k_notext.json \
    --output_path /data/round1_665k_notext_img_token.json
```

This will create a new file (`/data/round1_665k_notext_img_token.json`).

---

Then run the loss/perplexity calculations **twice**:


```bash
python presel/loss_ppl_calc.py \
    --data_path /data/round1_665k_notext.json \
    --model_path /PATH/TO/REFERENCE_MODEL \
    --image_folder /datasets \
    --output_file /data/loss_ppl_round1_665k_notext.json
```

```bash
python presel/loss_ppl_calc.py \
    --data_path /data/round1_665k_notext_img_token.json \
    --model_path /PATH/TO/REFERENCE_MODEL \
    --image_folder /datasets \
    --output_file /data/loss_ppl_round1_665k_notext_img_token.json
```

- Replace `/PATH/TO/REFERENCE_MODEL` with the path to your reference model checkpoint.
- Adjust `--image_folder` and `--output_file` as needed for your setup.

### 2. Task Importance Estimation

Run the following to get the estimated task-importance values required for our data selection approach:

```bash
python presel/llava_task_importance.py \
    --data_w_path /data/loss_ppl_round1_665k_notext.json \
    --data_wo_path /data/loss_ppl_round1_665k_notext_img_token.json \
    --reference_data_path /data/round1_665k_notext.json \
    --task_files_dir /data \
    --output_dir /data
```
### 3. Pre-Instruction Data Selection

First, we extract the visual features using DINOv2 model for each task (1 to 10 for the LLaVA dataset):

```bash
python data_process/extract_feats_665_dino.py --task_num TASK_NUM
```
Then run k-means clustering and sample selection:

```bash
python data_process/kmeans_clust.py --method typical
```

Finally, run the following command to finetune the model on the selected data. Make sure to set the BASE_DIR value appropriately. This code implements multi-round training where each round has a budget of 5% of the total data. Note that the results reported in the main paper correspond to round 3 (15% budget).

```bash
python presel/data_selection.py \
    --base_dir BASE_DIR \
    --method presel \
    --dataset_type llava
```


### Running on the Vision-FLAN Dataset

For the Vision-FLAN dataset, the steps are similar to those for the LLaVA-1.5 dataset mentioned above. For "Loss/Perplexity Calculations", you can follow the same steps, but make sure to adjust the code to match the Vision-FLAN data format (e.g., JSON files, reference set, image folder, etc.).

For "Task Importance Estimation", you can directly download the estimated task importance values [here](https://drive.google.com/file/d/1Qw61VKqE8i-MrUfKWApFdKIXKdbBiuZU/view?usp=sharing) and place it in `/data` directory.

For "Pre-Instruction Data Selection", first use the same script, `data_process/extract_feats_665_dino.py`, to extract VF features. Save the output as `/data/dino_feats_vf/dino_feats_all_vf.pt`. Then, run 
```bash
python data_process/kmeans_clust_vf.py --method typical
```
Finally, run the following command to fine-tune the model on the selected Vision-FLAN data:
```bash
python presel/data_selection.py \
    --base_dir BASE_DIR \
    --method presel \
    --dataset_type vision_flan \
    --file_path /datasets/annotation_191-task_1k_add_idx.json
```

## Finetuned Models & Selected Data (15%)
You can find our selected 15% subset of data via PreSel, as well as the fine-tuned models trained on it here:

| Dataset | 15% Selected Data by PreSel (JSON) | LLaVA-7B Model Finetuned |
|---------|-----------------------------------|--------------------------|
| LLaVA-1.5 | [Download](https://drive.google.com/file/d/1h6ttaIrBKhOR_6_HzY_NxALYID289gFF/view?usp=sharing) | [Download](https://drive.google.com/drive/folders/1M8rWMTrfHFaxD_FTOgBjbw_ABio9G_8i?usp=sharing) |
| Vision-FLAN | [Download](https://drive.google.com/file/d/1HzcV4vfUyPTu8x3CT1xdM5eI96zmm4J_/view?usp=sharing) | [Download](https://drive.google.com/drive/folders/17lbYWWxrsdMVZByK5wLah2iP3YCH_Lig?usp=sharing) |

## Evaluation
Please follow the [original LLaVA page](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#evaluation) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to evaluate models.


---


