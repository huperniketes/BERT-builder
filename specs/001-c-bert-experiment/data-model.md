# Data Models

This document defines the key data entities for the C-BERT experiment, based on the source paper.

## 1. C Code Corpus

- **Description**: The collection of C source code used for training, validation, and testing. Sourced from the top-100 starred C repositories on GitHub.
- **Size**: ~5.8 GB.
- **Attributes**:
  - `source_files`: A collection of `.c` and `.h` files.
- **State**: Raw (as cloned) -> Pre-processed (comments removed, de-duplicated).

## 2. C-BERT Model

- **Description**: The trained language model, packaged for use.
- **Format**: Hugging Face Transformers format.
- **Attributes**:
  - `config.json`: Model configuration (12 layers, 768-dim embeddings, 12 heads).
  - `pytorch_model.bin`: The trained model weights.
  - `tokenizer.json` / `vocab.txt`: The vocabulary and tokenizer configuration, specific to the chosen tokenizer (`Char`, `KeyChar`, or `SPE`).
  - `special_tokens_map.json`: Mapping for special tokens like `[MASK]`, `[CLS]`.

## 3. Tokenizer

- **Description**: The component responsible for converting raw C code into a sequence of tokens.
- **Types**:
  - `Char`: Simple ASCII character vocabulary (~103 tokens).
  - `KeyChar`: `Char` vocabulary plus 32 C keywords (~135 tokens).
  - `SPE`: SentencePiece subword tokenizer (5000 token vocabulary).

## 4. Training Configuration

- **Description**: Parameters that define a training run.
- **Attributes**:
  - `dataset_path`: Path to the pre-processed dataset.
  - `tokenizer_type`: One of `char`, `keychar`, or `spe`.
  - `masking_strategy`: One of `mlm` (Masked Language Model) or `wwm` (Whole Word Masking).
  - `learning_rate`: The learning rate for the optimizer (varies by tokenizer and strategy as per paper).
  - `batch_size`: 256 (total across all GPUs).
  - `max_sequence_length`: 512.
  - `output_dir`: Directory to save checkpoints and the final model.
