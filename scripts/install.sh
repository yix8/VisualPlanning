#!/bin/bash

echo ">>> Installing PyTorch 2.5.1 + CUDA 12.4 (torch, torchvision, torchaudio)"
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

echo ">>> Installing additional dependencies from requirements.txt"
pip install -r scripts/requirements.txt

echo "Setup complete!"

echo "Installing git-lfs."
sudo apt install git-lfs
git lfs install

echo "Cloning Dataset."
git clone https://huggingface.co/datasets/xuyi499307483/SFT_random_frozen dataset/frozenlake/tokenized_dataset/SFT_random
git clone https://huggingface.co/datasets/xuyi499307483/SFT_random_maze dataset/maze/tokenized_dataset/SFT_random
git clone https://huggingface.co/datasets/xuyi499307483/SFT_random_mini dataset/minibehaviour/tokenized_dataset/SFT_random

echo "Cloning Models."
mkdir -p models
# Clone into models/ directory
git clone https://huggingface.co/Emma02/LVM_ckpts models/LVM_ckpts
git clone https://huggingface.co/Emma02/vqvae_ckpts vqlm/vqvae_ckpts

echo "Cloning done."
