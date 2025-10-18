# Quickstart

This guide provides the basic steps to replicate the C-BERT pre-training and evaluation.

## 1. Setup

Install the required Python dependencies.

```bash
# It is recommended to use a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Data Preparation

First, clone the top-100 starred C repositories from GitHub. This will download approximately 5.8 GB of data.

```bash
python scripts/download_corpus.py --output_dir data/raw_c_code
```

Next, pre-process the raw source code to remove comments and prepare it for the tokenizer.

```bash
python src/cbert/data.py --input_dir data/raw_c_code --output_dir data/processed
```

## 3. Pre-training

Train the C-BERT model from scratch using the character-level tokenizer.

```bash
python src/cli/train.py \
    --dataset_dir data/processed \
    --config src/configs/c-bert-base.json \
    --tokenizer char \
    --output_dir models/c-bert-char
```

## 4. Fine-tuning & Evaluation

Fine-tune the pre-trained model on the AST node tagging task and evaluate its performance.

```bash
# (Assuming fine-tuning data is prepared)
python src/cli/evaluate.py \
    --model_dir models/c-bert-char \
    --task ast \
    --dataset_dir data/finetune/ast_tagging
```

