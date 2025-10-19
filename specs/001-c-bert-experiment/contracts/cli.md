# CLI Contracts

Based on the experiments described in the source paper.

## `train.py`

Initiates a training run for the C-BERT model.

| Argument | Type | Description | Required |
|---|---|---|---|
| `--dataset-dir` | path | Path to the pre-processed training data. | Yes |
| `--config` | path | Path to a model configuration JSON file (e.g., BERT-base). | Yes |
| `--tokenizer` | str | Tokenizer to use. One of: `char`, `keychar`, `spe`. | Yes |
| `--masking` | str | Masking strategy. One of: `mlm`, `wwm`. | No (Default: `mlm`) |
| `--output-dir` | path | Directory to save checkpoints and the final model. | Yes |
| `--epochs` | int | Number of training epochs. | No (Default: 10) |
| `--batch-size` | int | Per-GPU batch size. | No (Default: 8) |
| `--learning-rate` | float | Optimizer learning rate. | No (Default: 2e-5) |
| `--max-steps` | int | Total number of training steps. Overrides epochs. | No (Default: -1) |
| `--resume-from-checkpoint` | path | Path to a checkpoint to resume training from. | No |

## `evaluate.py`

Evaluates a trained C-BERT model on an AST node tagging or vulnerability identification task.

| Argument | Type | Description | Required |
|---|---|---|---|
| `--model-dir` | path | Path to the trained model directory (Hugging Face format). | Yes |
| `--task` | str | The evaluation task. One of: `ast`, `vi`. | Yes |
| `--dataset-dir` | path | Path to the pre-processed and labeled evaluation data. | Yes |
| `--output-file` | path | File to save the JSON evaluation results. | No (Default: stdout) |
