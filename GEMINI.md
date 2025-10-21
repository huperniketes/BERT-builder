# bert Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-10-18

## Active Technologies
- Python 3.11 + PyTorch, `transformers`, `sentencepiece`, `libclang` (001-c-bert-experiment)

## Project Structure
```
src/
tests/
```

## Commands
cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style
Python 3.11: Follow standard conventions

## Recent Changes
- 001-c-bert-experiment: Added Python 3.11 + PyTorch, `transformers`, `sentencepiece`, `libclang`

## Development Philosophy
This project is focused on research-grade implementations for academic publication. All code should reflect this priority.
- **Strict Adherence:** All implementations must strictly adhere to the methodologies described in the cited research papers (e.g., the C-BERT paper, 2006.12641v2). The goal is a faithful reproduction of the original work. If the paper is ambiguous on a specific implementation detail, point it out and we will discuss it.
- **Research-Grade Quality:** Code must be suitable for a research publication. Prioritize algorithmic correctness and reproducibility.
- **Configurability:** Avoid hardcoded values. All hyperparameters should be configurable through command-line arguments or configuration files.
- **Scope:** Focus solely on the core research contributions. Do not add features or optimizations not explicitly mentioned in the source paper unless directed to do so.
- **Validation:** Include tests to validate *all* input data, parameters, intermediate results and data shapes. Unexpected values shall be flagged with the rationale, the source, and location for inspection of the values for corrective measures to be undertaken. Errors and exceptions during processing shall likewise be reported.
- **Documentation:** Ensure the code is well-documented, especially the parts related to the core algorithm and data transformations.
add data quality to the context


<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
