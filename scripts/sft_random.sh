#!/bin/bash

set -euo pipefail

TASK="${1:-}"

if [ -z "$TASK" ]; then
  echo "Usage: bash scripts/sft_random.sh <frozenlake|maze|minibehaviour>"
  exit 1
fi

case "$TASK" in
  frozenlake)
    DATASET_PTH="dataset/frozenlake/tokenized_dataset/SFT_random/train_dataset.jsonl"
    MODEL_PATH="./models/LVM_ckpts"
    OUTPUT_DIR="./models/frozenlake/SFT_LVM_random_PEFT_ckpts"
    RUN_NAME="SFT_LVM_random_frozenlake"
    ;;
  maze)
    DATASET_PTH="dataset/maze/tokenized_dataset/SFT_random/train_dataset.jsonl"
    MODEL_PATH="./models/LVM_ckpts"
    OUTPUT_DIR="./models/maze/SFT_LVM_random_PEFT_ckpts"
    RUN_NAME="SFT_LVM_random_maze"
    ;;
  minibehaviour)
    DATASET_PTH="dataset/minibehaviour/tokenized_dataset/SFT_random/train_dataset.jsonl"
    MODEL_PATH="./models/LVM_ckpts"
    OUTPUT_DIR="./models/minibehaviour/SFT_LVM_random_PEFT_ckpts"
    RUN_NAME="SFT_LVM_random_minibehaviour"
    ;;
  *)
    echo "Unknown task: $TASK"
    echo "Expected one of: frozenlake, maze, minibehaviour"
    exit 1
    ;;
esac

python train_sft.py \
  SFT.dataset_pth="$DATASET_PTH" \
  SFT.model_path="$MODEL_PATH" \
  SFT.output_dir="$OUTPUT_DIR" \
  SFT.run_name="$RUN_NAME"
