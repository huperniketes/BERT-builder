# BERT-builder
A BERT model laboratory based on C-BERT

BERT-builder is a toolset to recreate and extend the software used to build the C-BERT model presented in the research paper "Exploring Software Naturalness through Neural Language Models" ([2006.12641]), with options to process code written in other languages.

## Overview of C-BERT
The AI paper [Exploring Software Naturalness through Neural Language Models](https://arxiv.org/pdf/2006.12641) by Buratti, et al. introduces C-BERT, a transformer-based language model pre-trained from scratch on a large collection of repositories written specifically in the C programming language.

Key Details of the C-BERT Paper:

Purpose: The primary goal was to investigate the "software naturalness hypothesis," which posits that programming languages can be analyzed using techniques similar to those in natural language processing (NLP).

Model Architecture: C-BERT is based on the original BERT (Bidirectional Encoder Representations from Transformers) architecture, but its training data is source code rather than human language text.

Programming Language Focus: The model was trained exclusively on C language code.

Tasks and Performance: The paper evaluated C-BERT on specific programming language tasks, such as Abstract Syntax Tree (AST) node tagging and vulnerability detection, where it outperformed existing approaches.

Publication: The paper was published on arXiv in June 2020. 

In summary, C-BERT is not a model for processing the natural language surrounding C code, but rather a model designed to understand and process the C source code itself using NLP methods. 

## Model-building Pipeline

Here is the step-by-step pipeline for building a C-BERT model, from raw source code to a trained model, using the scripts in this repository.

  Step 1: Acquire the Dataset

  First, you need a large corpus of source code.

   * Action: Use the scripts/download_corpus.py script to download C repositories from GitHub.
   * Command Example:

   1     python scripts/download_corpus.py --init_bom --output_dir data/corpus --bom_json data/dataset_bom.json
   * Outcome: This will create a data/corpus directory containing the source code of top 100 starred C repositories from GitHub and a BOM (bill of materials) file (data/dataset_bom.json) cataloging them.

  Step 2: Pre-process and Consolidate the Corpus

  The downloaded dataset consists of many individual source files. For training the tokenizer, you need to combine them into a single, large text file.

   * Action: Run the src/cbert/data.py script, providing the input directory of your downloaded corpus and specifying an output file.
   * Command Example:

   1     python src/cbert/data.py --input_dir data/corpus --output_file data/c_corpus_cleaned.txt
   * Outcome: A single, cleaned corpus file named data/c_corpus_cleaned.txt and a corresponding _lineoffsets.json file, which is used for faster processing.

  Step 3: Train a Tokenizer

  A tokenizer is responsible for breaking the raw source code text into tokens that the model can understand.

   * Action: Use the cli/train_tokenizer.py script, pointing it to the consolidated corpus file you just created.
   * Command Example:

   1     python -m cli.train_tokenizer \
   2       --corpus_path data/c_corpus_full.txt \
   3       --model_prefix cbert_tokenizer \
   4       --vocab_size 8000 \
   5       --output_dir models/cbert_tokenizer
   * Outcome: This will generate tokenizer model files (e.g., cbert_tokenizer.model and cbert_tokenizer.vocab) in the models/cbert_tokenizer/ directory.

  Step 4: Create a Model Configuration File

  This configuration file defines the architecture of your BERT model, such as the number of layers and the hidden size.

   * Action: Use the scripts/create_config.py script. The vocab_size must match the one you used to train the tokenizer.
   * Command Example:
   1     python scripts/create_config.py \
   2       --vocab-size 8000 \
   3       --output configs/c-bert-config.json
   * Outcome: A config.json file in the configs/ directory that describes your model's architecture.

  Step 5: Train the C-BERT Model

  This is the main training step where the model learns from the tokenized source code.

   * Action: Use the cli/train.py script. You will need to provide the dataset, the model configuration, and the tokenizer files you created in the previous steps.
   * Command Example:

   1     python -m cli.train \
   2       --dataset-dir data/corpus \
   3       --config configs/c-bert-config.json \
   4       --tokenizer spe \
   5       --spm-model-file models/cbert_tokenizer/cbert_tokenizer.model \
   6       --output-dir models/c-bert-trained
   * Outcome: The trained model and checkpoints will be saved in the models/c-bert-trained/ directory. This process can take a significant amount of time and computational resources.

  Step 6: Evaluate the Model

  After training, you can evaluate your model's performance on various tasks.

   * Action: Use the cli/evaluate.py script, pointing it to your trained model and an evaluation dataset.
   * Command Example:

   1     python -m cli.evaluate \
   2       --model-dir models/c-bert-trained \
   3       --task mlm \
   4       --dataset-dir data/evaluation_data \
   5       --tokenizer-type spe \
   6       --spm-model-file models/cbert_tokenizer/cbert_tokenizer.model
   * Outcome: The script will output evaluation metrics, giving you an indication of how well your model has learned to understand C code.

## Commands

This project aims to provide a full pipeline of tools to build and experiment with models, from creating and managing the datasets to evaluating models.

### Dataset Scripts

*   **`scripts/analyze_data_quality.py`**: Analyzes data quality metrics for C repositories.
    *   `python scripts/analyze_data_quality.py [repo_path]`
    *   **Options:**
        *   `repo_path`: Path to repository (default: current directory).
        *   `--json <file>`: Save detailed results to JSON file.
        *   `--detailed`: Show detailed per-file metrics.

*   **`scripts/create_config.py`**: Creates C-BERT model configuration files.
    *   `python scripts/create_config.py --vocab-size <size> --output <path>`
    *   **Options:**
        *   `--vocab-size <int>`: Vocabulary size (required).
        *   `--hidden-size <int>`: Hidden size (default: 768).
        *   `--num-layers <int>`: Number of layers (default: 12).
        *   `--num-heads <int>`: Number of attention heads (default: 12).
        *   `--max-length <int>`: Max sequence length (default: 512).
        *   `--output <path>`: Output config file path (required).

*   **`scripts/download_corpus.py`**: Downloads C repositories from GitHub.
    *   `python scripts/download_corpus.py`
    *   **Options:**
        *   `--output_dir <dir>`: Directory to clone repositories into (default: `data/`).
        *   `--bom_json <path>`: Path to a dataset BOM file.
        *   `--init_bom`: Initialize a new BOM from GitHub's top repositories.
        *   `--language <lang>`: Language to query from GitHub (default: C).
        *   `--num_repos <int>`: Number of repositories to fetch (default: 100).

*   **`scripts/update_bom.py`**: Extracts additional information from GitHub or local repositories for the BOM.
    *   `python scripts/update_bom.py --output_bom <path>`
    *   **Options:**
        *   `--input_bom <path>`: Path to an input BOM file.
        *   `--output_bom <path>`: Path to an output BOM file (required).
        *   `--github_token <token>`: GitHub personal access token.
        *   `--local_path <path>`: Path to a local directory of Git repositories.
        *   `--update_bom`: Update the BOM with local Git information.

### Train Tokenizer

`python -m cli.train_tokenizer`

Train a SentencePiece tokenizer from a corpus.

| Argument | Description |
|---|---|
| `--corpus_path` | **(Required)** Path to the text corpus file for training (e.g., `c_corpus_cleaned.txt`). |
| `--model_prefix`| **(Required)** Prefix for the output SentencePiece model files (e.g., `cbert_spm`). |
| `--vocab_size` | Vocabulary size for the SentencePiece model. (Default: `8000`) |
| `--output_dir` | Directory to save the trained tokenizer model files. (Default: `.`) |

### Train Model

`python -m cli.train`

Train a C-BERT model from scratch or resume from a checkpoint.

| Argument | Description |
|---|---|
| `--dataset-dir` | **(Required)** Path to the pre-processed training data. |
| `--config` | **(Required)** Path to a model configuration JSON file (e.g., `c-bert-base.json`). |
| `--tokenizer` | **(Required)** Tokenizer to use. Choices: `char`, `keychar`, `spe`. |
| `--vocab-file` | Path to the vocabulary file (required for SentencePiece tokenizer). |
| `--spm-model-file`| Path to the SentencePiece model file (required for SentencePiece tokenizer). |
| `--masking` | Masking strategy. Choices: `mlm`, `wwm`. (Default: `mlm`) |
| `--output-dir` | **(Required)** Directory to save checkpoints and the final model. |
| `--epochs` | Number of training epochs. (Default: `10`) |
| `--batch-size` | Per-GPU batch size. (Default: `8`) |
| `--learning-rate` | Optimizer learning rate. (Default: `2e-5`) |
| `--max-steps` | Total number of training steps. Overrides `epochs`. (Default: `-1`) |
| `--resume-from-checkpoint` | Path to a checkpoint to resume training from. |

### Evaluate Model

`python -m cli.evaluate`

Evaluate a trained C-BERT model on a given task.

| Argument | Description |
|---|---|
| `--model-dir` | **(Required)** Path to the trained model directory. |
| `--task` | The evaluation task. Choices: `mlm`, `ast`, `vi`. (Default: `mlm`) |
| `--dataset-dir` | **(Required)** Path to the pre-processed evaluation data file. |
| `--output-file` | File to save the JSON evaluation results. (Default: prints to stdout) |
| `--batch-size` | Batch size for evaluation. (Default: `8`) |
| `--max-length` | Maximum sequence length for tokenization. (Default: `128`) |
| `--tokenizer-type`| **(Required)** Type of tokenizer used during training. Choices: `char`, `keychar`, `spe`. |
| `--vocab-file` | Path to the vocabulary file. |
| `--spm-model-file`| Path to the SentencePiece model file. |
| `--checkpoint-file`| Path to a checkpoint file for resuming evaluation. |
| `--use-precomputed-offsets` | Use pre-computed line offsets for faster startup. |

### Get Embeddings

`python -m cli.embed`

Get a sentence embedding for a C code snippet.

| Argument | Description |
|---|---|
| `--code` | **(Required)** The C code snippet to embed. |
| `--model-dir` | **(Required)** Path to the trained model directory. |
| `--output-file` | File to save the JSON embeddings. (Default: prints to stdout) |
| `--tokenizer-type`| **(Required)** Type of tokenizer used during training. Choices: `char`, `keychar`, `spe`. |
| `--vocab-file` | Path to the vocabulary file. |
| `--spm-model-file`| Path to the SentencePiece model file. |
| `--max-length` | Maximum sequence length for tokenization. (Default: `128`) |

## Project Development Details

### Project Structure
```
specs/
src/
   + cbert/
   + cli/
   + configs/
tests/
   + contract/
   + integration/
   + unit/
```

### Development Philosophy
This project is focused on research-grade implementations for academic publication. All code should reflect this priority.
- **Strict Adherence:** All implementations must strictly adhere to the methodologies described in the cited research
papers (e.g., the C-BERT paper, 2006.12641v2). The goal is a faithful reproduction of the original work. If the paper is ambiguous on a specific implementation detail, point it out and we will discuss it.
- **Research-Grade Quality:** Code must be suitable for a research publication. Prioritize algorithm
correctness and reproducibility.
- **Configurability:** Avoid hardcoded values. All hyperparameters should be configurable through command-line
arguments or configuration files.
- **Scope:** Focus solely on the core research contributions. Do not add features or optimizations
not explicitly mentioned in the source paper unless directed to do so.
- **Validation:** Include tests to validate *all* input data, parameters, intermediate results and
data shapes. Unexpected values shall be flagged with the rationale, the source, and location for inspection of the values for corrective measures to be undertaken. Errors and exceptions during processing shall likewise be reported.
- **Data Quality:** Implement comprehensive data quality checks throughout the pipeline. This includes validation of data
integrity, consistency, completeness, and adherence to expected formats. All data transformations must preserve semantic meaning while maintaining traceability of quality metrics.
- **Documentation:** Ensure the code is well-documented, especially the parts related to the core
algorithm and data transformations.

### Active Technologies

  Here's a summary of the project's technologies:

  Programming Languages:
   * Python: the programming language used due to the number of ML tools provided by the community.

  Core Machine Learning & NLP:
   * PyTorch (`torch`): The primary deep learning framework.
   * Hugging Face `transformers`: Provides the core BERT architecture and tokenizer infrastructure.
   * `sentencepiece`: Used for subword tokenization.
   * NumPy (`numpy`): For numerical operations.
   * Scikit-learn (`sklearn`): For evaluation metrics.
   * SciPy (`scipy`): For statistical analysis.

  C Code Analysis:
   * `ANTLR`: ANTLR v4 generated classes for parsing/analyzing C code.

  Development & Utility Tools:
   * `argparse`: For creating command-line interfaces.
   * `unittest`: For unit and integration testing.
   * `setuptools`: For packaging the cbert module.
   * `requests`: For making HTTP requests to the GitHub API.
   * `ruff`: For code linting.

### Roadmap
- ~~Initial Release: Complete the core functionality for training and evaluating C-BERT models.~~
- ~~Extended Language Support: Add support for additional programming languages beyond C.~~
- Advanced Tokenization: Implement and test various tokenization strategies.
   * `clang`: Python bindings for parsing/analyzing C code.
- Fine-tuning Capabilities: Enable fine-tuning of pre-trained models on specific tasks.
- Performance Optimization: Optimize training and inference performance.
- Documentation & Tutorials: Provide comprehensive documentation and usage examples.

### Code Style
Python 3.11: Follow standard conventions

### Recent Changes
- 001-c-bert-experiment: Added Python 3.11 + PyTorch, `transformers`, `sentencepiece`, `libclang`
