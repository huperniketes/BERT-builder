# Technical Decisions from Source Paper

This document records the key technical decisions derived from the source paper *Exploring Software Naturalness through Neural Language Models* (arXiv:2006.12641v2).

## 1. Machine Learning Framework

- **Decision**: PyTorch with the Hugging Face `transformers` library.
- **Rationale**: The paper explicitly states in Section 2.4: "All of our models are built using the Huggingface Pytorch Transformers library [41]". This provides a clear path for replication.

## 2. Tokenization Strategy

- **Decision**: The implementation will support all three tokenizers evaluated in the paper to allow for full replication.
  1.  **Character (Char)**: A simple ASCII vocabulary (103 tokens).
  2.  **Character + Keyword (KeyChar)**: The `Char` vocabulary augmented with 32 C-language keywords.
  3.  **SentencePiece (SPE)**: A subword tokenizer with a 5000-token vocabulary.
- **Rationale**: The paper investigates the trade-offs between these different tokenization strategies. To faithfully replicate the experiment, all three must be available. The primary focus for initial implementation will be the `Char` tokenizer, as the CWB (Char-WWM-BiLSTM) model produced the best results on most datasets.

## 3. Pre-training Corpus

- **Decision**: A 5.8 GB dataset of C source code collected from the top-100 starred C repositories on GitHub.
- **Rationale**: Section 2.3 of the paper specifies this dataset. The plan will include a script to automate the cloning of these repositories to assemble the corpus.

## 4. AST & Ground-Truth Data Generation

- **Decision**: Use `Clang` and its Python bindings (`libclang`).
- **Rationale**: The paper states in Section 5: "To generate the gold data for our AST node tagging task we used Clang [48]... We use the libclang python bindings atop Clang 11.0". This is a non-negotiable requirement for replicating the AST-related tasks.
