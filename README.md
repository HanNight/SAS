# [Stabilizing Efficient Reasoning with Step-Level Advantage Selection (SAS)](https://arxiv.org/abs/2604.24003)

[Han Wang](https://hannight.github.io/)<sup>1</sup>, [Xiaodong Yu]()<sup>2</sup>, [Jialian Wu]()<sup>2</sup>, [Jiang Liu]()<sup>2</sup>, [Ximeng Sun]()<sup>2</sup>, [Mohit Bansal]()<sup>1</sup>, [Zicheng Liu]()<sup>2</sup>

<sup>1</sup>UNC Chapel Hill, <sup>2</sup>AMD

![image](https://github.com/user-attachments/assets/96ff9959-3992-4203-a8ed-1bbc8fb15a68)

## Installation

1. Create a conda environment

```bash
conda create -n verl_sas python==3.10
conda activate verl_sas
```

2. Clone the repository

```bash
git clone https://github.com/hannight/SAS.git
cd SAS
```

3. Install the dependencies for verl

```bash
cd verl
# Make sure you have activated verl conda env
# If you need to run with megatron
bash scripts/install_vllm_sglang_mcore.sh
# Or if you simply need to run with FSDP
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

```

4. Install verl

```bash
cd verl
pip install --no-deps -e .
```

## Dataset
We provide the training and evaluation data in the `data` folder.

### Training Data
We use [DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) as the training data, which is in the `data/train.parquet` file.

You can also use your own training data, please follow the format of `data/train.parquet`.

### Evaluation Data
We evaluate on five different math reasoning datasets: AIME2024 (`data/aime.parquet`), AIME2025 (`data/aime2025.parquet`), MATH (`data/math.parquet`), AMC (`data/amc.parquet`), and Olympiad-Bench (`data/olympiad_bench.parquet`). In addition, we also include GPQA-Diamond (`data/gpqa.parquet`), LSAT (`data/lsat.parquet`), and MMLU (500 instances subset, `data/mmlu_500.parquet`), three general reasoning benchmarks to test the ability to generalize tot out-of-domain data.

You can also use your own evaluation data, please follow the format of the evaluation data files.

## Training
Train `DeepScaleR-1.5B-Preview` with SAS:
```bash
bash deepscaler_grpo_sas.sh
```
You can also train with other models via modifying the `actor_rollout_ref.model.path` in the `deepscaler_grpo_sas.sh` script.

We explain the important arguments in the `deepscaler_grpo_sas.sh` as follows:

- `data.train_files`: The path to the training data.
- `data.val_files`: The path to the validation data.
- `trainer.sas`: Whether to use SAS.
- `trainer.sas_strategy`: The strategy to use SAS. Available options: `correct_only` (only apply SAS to correct rollouts), `wrong_only` (only apply SAS to wrong rollouts), `both` (apply SAS to both correct and wrong rollouts). Default to `both`.
- `trainer.mask_steps_ratio`: The ratio of steps to set their advantages to 0 (range from 0 to 1, default to 0.3).
- `trainer.random_mask`: Whether to use random selection (for ablation study).

You can train `DeepScaleR-1.5B-Preview` with  standard GRPO post-training under a 4K training context, without any additional RL techniques:
```bash
bash deepscaler_grpo_4k.sh
```

**Note**: Before evaluation, you need to merge the checkpoints from FSDP and Megatron backends. Please refer to the [verl documentation](https://verl.readthedocs.io/en/v0.4.0/advance/checkpoint.html#convert-fsdp-and-megatron-checkpoints-to-huggingface-format-model) for more details. Example command:
```bash
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /path/to/the/saved/model/checkpoints \
    --target_dir /path/to/the/merged/hf/model
```

## Evaluation
Evaluate the model on all the evaluation datasets:
```bash
MODEL_PATH=/path/to/the/model/checkpoint
OUTPUT_DIR=/path/to/the/output/directory
bash eval_model.sh --model ${MODEL_PATH} --num-tokens 8192 --datasets aime aime2025 amc olympiad_bench gpqa lsat mmlu_500 --output-dir ${OUTPUT_DIR}
```

## Acknowledgement
We sincerely thank the authors of [verl](https://github.com/verl-project/verl) and [DeepScaleR](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) for their public code and data release.

## Citation
```bibtex
@article{wang2026stabilizing,
  title={Stabilizing Efficient Reasoning with Step-Level Advantage Selection},
  author={Han Wang and Xiaodong Yu and Jialian Wu and Jiang Liu and Ximeng Sun and Mohit Bansal and Zicheng Liu},
  year={2026},
  journal={arXiv preprint arXiv:2604.24003}
}