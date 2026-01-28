# Rec-R1: Bridging Generative Large Language Models and User-Centric Recommendation Systems via Reinforcement Learning

<p align="center">
  <a href="https://arxiv.org/pdf/2503.24289">
    <img src="https://img.shields.io/badge/arXiv-2503.00223-b31b1b.svg" alt="arXiv">
  </a>
</p>

REC-R1 is a general framework that bridges generative large language models (LLMs) and recommendation systems via reinforcement learning. Check the paper [here](https://arxiv.org/pdf/2503.24289).

## News
[2026/01/28] We have reorganized the repository structure for better code organization and readability.
[2025/10/11] Rec-R1 is accepted by TMLR!
[2025/03/31] We release the code and paper. We are running more experiments such as more LLM4Rec tasks and more datasets. **More results are coming!**

<p align="center">
  <img  src="resources/llm4rec.png" />
</p>

<p align="center">
  <img  src="resources/rec-r1-compa.png" />
</p>

## Installation

```bash
conda create -n zero python=3.9
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib

# lucene supported by pyserini
pip install pyserini
pip install faiss-gpu

# if you don't have jave in the environment
conda install -c conda-forge openjdk=17
export JAVA_HOME=~/miniconda3/envs/zero
export PATH=$JAVA_HOME/bin:$PATH
```


## Quick Start
This guide demonstrates the complete workflow using **sparse retrieval** (Lucene-based BM25) as an example. The process includes: (1) data preparation, (2) building a search index, (3) training the model with reinforcement learning, and (4) evaluation. The same workflow can be adapted for dense retrieval methods.

**Data Preparation**

```bash
conda activate zero
python src/dataset/amazon_c4/inst/sparse/subset_data.py
```

### Build a Lucene Database
See the `src/Lucene/README.md` file.

**Required Data Downloads**

Before building the Lucene database, you need to download the following data files and place them in the specified locations:

| Dataset | File Path | Google Drive Link |
|---------|-----------|-------------------|
| ESCI | `data/esci/raw/sampled_item_metadata_esci.jsonl` | [Download](https://drive.google.com/file/d/1Nha-36rcY0Y_j7gIFn9fc7KMdwrSsWr0/view?usp=sharing) |
| Amazon C4 | `data/amazon_c4/raw/sampled_item_metadata_1M.jsonl` | [Download](https://drive.google.com/file/d/1duJhGbShZhQehVlDTF1W3lbwqxu8o1J7/view?usp=sharing) |
| Amazon Review (All Beauty) | `data/amazon_review/processed/corpus/All_Beauty_items_filtered.jsonl` | [Download](https://drive.google.com/file/d/10RRHln6zvYe0WFgMLKkaIZguzowSI5AH/view?usp=sharing) |

After downloading, ensure the files are placed in their respective directories as shown above.

**Note:** For dense retrieval methods, please refer to the `src/Dense/` directory for implementation details and usage examples.


### Training
```bash
conda activate zero
```

For the following code, if you see Out-of-vram, try add `critic.model.enable_gradient_checkpointing=True` to the script


**3B+ model**
```bash
export N_GPUS=2
export BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
export DATA_DIR=data/matching/qwen-instruct
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=matching-qwen2.5-3b-inst-ppo
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY="[Your_key]"
export HF_HOME="/srv/local/data/linjc/hub"

export CUDA_VISIBLE_DEVICES=0,1

bash scripts/train/train_rec-amazon_c4_3b.sh
```

### Evaluation

The evaluation process consists of two main steps:

**Step 1: Query Generation**

First, use the trained model to generate search queries based on user preferences and item metadata:

```bash
bash scripts/eval/esci/inst_gen/sparse/rec-r1.sh <DOMAIN_NAME>
```

For example, for the ESCI dataset on the Video Games domain:
```bash
bash scripts/eval/esci/inst_gen/sparse/rec-r1.sh Video_Games
```

This will generate queries and save them to results/esci/rec-r1-esci_<DOMAIN_NAME>.json.

**Step 2: Item Retrieval and Ranking**
After generating the queries, retrieve relevant items using the Lucene-based BM25 search and compute retrieval/ranking metrics (e.g., Recall@K, NDCG@K):

```bash
bash scripts/eval_search/esci/sparse/eval_search.sh
```
This script will:
* Load the generated queries from Step 1
* Perform BM25 search against the Lucene index
* Calculate retrieval metrics (Recall, NDCG, etc.)
* Save results to results/esci/metric_res/




## Citation
```
@article{lin2025rec,
  title={Rec-R1: Bridging Generative Large Language Models and User-Centric Recommendation Systems via Reinforcement Learning},
  author={Lin, Jiacheng and Wang, Tian and Qian, Kun},
  journal={arXiv preprint arXiv:2503.24289},
  year={2025}
  url={https://arxiv.org/pdf/2503.24289}, 
}
```

## Acknowledgements
- [Verl](https://github.com/volcengine/verl) 🔗
- [Pyserini](https://github.com/castorini/pyserini) 🔗
- [Faiss](https://github.com/facebookresearch/faiss) 🔗