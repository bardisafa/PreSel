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

- LLaVA-1.5 dataset: [Download splits](https://drive.google.com/file/d/17dSI8xZMcr4QvRz_qkWKR4RGSzrABHQm/view?usp=sharing)
- Vision-FLAN dataset: [Download splits](https://drive.google.com/file/d/yyy/view?usp=sharing)

Place the downloaded task split files in the `clustering_results/` directory.


