#!/bin/bash

set -euo pipefail

TASK="${1:-}"

if [ -z "$TASK" ]; then
  echo "Usage: bash scripts/grpo.sh <frozenlake|maze|minibehaviour>"
  exit 1
fi

case "$TASK" in
  frozenlake)
    TRAIN_SCRIPT="train_rl_frozen.py"
    DATASET_PTH="dataset/frozenlake/tokenized_dataset/SFT/train_dataset.jsonl"
    MODEL_PATH="./models/frozenlake/SFT_LVM_random_merged_ckpts"
    OUTPUT_DIR="./models/frozenlake/GRPO_LVM_random_PEFT_ckpts"
    RUN_NAME="GRPO_LVM_random"
    ;;
  maze)
    TRAIN_SCRIPT="train_rl_maze.py"
    DATASET_PTH="dataset/maze/tokenized_dataset/SFT/train_dataset.jsonl"
    MODEL_PATH="./models/maze/SFT_LVM_random_merged_ckpts"
    OUTPUT_DIR="./models/maze/GRPO_LVM_random_PEFT_ckpts"
    RUN_NAME="GRPO_LVM_random"
    ;;
  minibehaviour)
    TRAIN_SCRIPT="train_rl_mini.py"
    DATASET_PTH="dataset/minibehaviour/tokenized_dataset/SFT/train_dataset.jsonl"
    MODEL_PATH="./models/minibehaviour/SFT_LVM_random_merged_ckpts"
    OUTPUT_DIR="./models/minibehaviour/GRPO_LVM_random_PEFT_ckpts"
    RUN_NAME="GRPO_LVM_random"
    ;;
  *)
    echo "Unknown task: $TASK"
    echo "Expected one of: frozenlake, maze, minibehaviour"
    exit 1
    ;;
esac

python "$TRAIN_SCRIPT" \
  GRPO.dataset_pth="$DATASET_PTH" \
  GRPO.model_path="$MODEL_PATH" \
  GRPO.output_dir="$OUTPUT_DIR" \
  GRPO.run_name="$RUN_NAME"
