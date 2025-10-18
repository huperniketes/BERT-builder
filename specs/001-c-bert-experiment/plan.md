# Implementation Plan: Replicate C-BERT Experiment

**Branch**: `001-c-bert-experiment` | **Date**: 2025-10-18 | **Spec**: [spec.md](./spec.md)

**Note**: This document is based on the methodology described in the paper *Exploring Software Naturalness through Neural Language Models* (arXiv:2006.12641v2).

## Summary

This plan outlines the technical approach to replicate the C-BERT experiment. The project will be a Python-based library and CLI to pre-process a 5.8 GB corpus of C code from GitHub, train a BERT-style language model using PyTorch and the Hugging Face `transformers` library, and evaluate its performance. The implementation will support the three tokenizer variations described in the paper (Character, Character+Keyword, SentencePiece) and will package the final models in the standard Hugging Face format.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: PyTorch, `transformers`, `sentencepiece`, `libclang`
**Storage**: Filesystem for C code corpus and model checkpoints.
**Testing**: `pytest`
**Target Platform**: Linux server with NVIDIA GPUs.
**Project Type**: Single project (Library + CLI).
**Performance Goals**: Pre-train the model on the full 5.8 GB corpus in under 24 hours, as per the paper.
**Constraints**: The paper utilized 32 GPUs for training. This plan will support single-GPU training as a baseline, but meeting the performance goal will likely require a multi-GPU setup.
**Scale/Scope**: 5.8 GB C code corpus from the top-100 starred C repositories on GitHub.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. Library-First**: **PASS**. The project will be structured as a Python library with a clear CLI.
- **II. CLI Interface**: **PASS**. The plan defines a CLI for training and evaluation.
- **III. Test-First**: **PASS**. The development workflow will follow TDD principles.
- **Additional Constraints**: **NOTE**. The constitution mentions CMake, which is not suitable for a Python project. This is considered a justified deviation.

## Project Structure

### Documentation (this feature)

```
specs/001-c-bert-experiment/
├── plan.md              # This file
├── research.md          # Technical decisions from source paper
├── data-model.md        # Data models for the project
├── quickstart.md        # Quickstart guide
├── contracts/           # CLI contracts
│   └── cli.md
└── tasks.md             # Phase 2 output (not created by this command)
```

### Source Code (repository root)
```
scripts/
└── download_corpus.py # Script to clone top C repos from GitHub

src/
├── cbert/
│   ├── __init__.py
│   ├── data.py        # Data loading and corpus generation
│   ├── model.py       # C-BERT model architecture
│   ├── tokenizer.py   # Char, KeyChar, and SPE tokenizers
│   └── trainer.py     # PyTorch training loop
├── cli/
│   ├── __init__.py
│   ├── train.py
│   └── evaluate.py
└── configs/
    └── c-bert-base.json

tests/
├── contract/
├── integration/
└── unit/
```

**Structure Decision**: A single project structure is chosen. The core logic will reside in `src/cbert` as a library. A script to download the corpus is added, and the CLI entrypoints are in `src/cli`.

## Complexity Tracking

*No violations requiring justification.*