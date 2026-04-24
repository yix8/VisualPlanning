from trl import GRPOConfig, GRPOTrainer
import torch
import jsonlines
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, TrainerCallback
import hydra
from omegaconf import DictConfig
from transformers import LlamaForCausalLM, AutoModelForCausalLM
import logging
import random
import wandb
import numpy as np
import sys
import einops
import os
from pathlib import Path
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from vqlm.vqvae_muse import get_tokenizer_muse
from CustomGRPO import VisionGRPOTrainer
from layout_parser import ActionParser
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Type,
    Union,
    cast,
)

log = logging.getLogger(__name__)

dtype_map = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16
}

def seed_everything(seed: Optional[int] = 42):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TokenizedDataset(Dataset):
    def __init__(self, filepath: str):
        self.input_ids_list = []
        self.target_ids_list = []
        self.input_state_list = []
        self.meta_list = []
        with jsonlines.open(filepath) as reader:
            for obj in reader:
                self.input_ids_list.append(torch.tensor(obj['input_tokens'], dtype=torch.long))
                self.target_ids_list.append(torch.tensor(obj['output_tokens'], dtype=torch.long))
                self.input_state_list.append(obj['input_state'])
                self.meta_list.append(obj['meta'])
    
    def __len__(self) -> int:
        return len(self.input_ids_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "input_ids": self.input_ids_list[idx],
            "target_ids": self.target_ids_list[idx],
            "input_state": self.input_state_list[idx],
            "meta": self.meta_list[idx],
        }

def collate_fn(batch):
    return batch

def action_reward_func(prompts, completions, action_parser, display, num_generations, **kwargs) -> list[float]:
    rewards = []
    for i in range(len(prompts)):
        if torch.any(completions[i:i+1] >= 8192):
            rewards.append(-5)
            continue

        meta = kwargs['meta'][i]
        level = meta['level']
        current_distance = meta['distance_map'][str(kwargs['input_state'][i])]
        
        action_info = action_parser.parse_maze_action_in_ids(
            prompts[i:i+1], completions[i:i+1], level, kwargs['input_state'][i], meta['start_pos'], meta['target_pos'], meta['layout'], meta['distance_map'], display
        )
        
        action, next_coord = action_info['action'], action_info['pred_coord']
        state = next_coord[0] * level + next_coord[1]

        if action[1] == 'invalid':
            rewards.append(-5)
            continue

        if state == meta['target_pos']:
            rewards.append(1)
            continue

        next_distance = meta['distance_map'].get(str(state), -1)
        rewards.append(1 if next_distance == current_distance - 1 else -5 if next_distance == -1 else 0)

        # if rewards[-1] != 1:
        #     wandb.log({
        #         "action with pred_coord": wandb.Html(str(action) + ", " + str(next_coord)),
        #         "input_pred_images": wandb.Image(action_info['image'], caption="Input (Left) and Pred (Right) Image"),
        #     })

    def log_rewards(rewards, num_generations):
        n = len(rewards) // num_generations
        for i in range(n):
            group = rewards[i * num_generations : (i + 1) * num_generations]
            log.info(f"Rewards for group {i + 1}: {group}")
    
    log_rewards(rewards, num_generations)
    return rewards

# reward functions
def int_reward_func(completions, **kwargs) -> list[float]:
    return [1.0] * len(completions)

@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.GRPO.seed)
    log.info(f"Training Config: {cfg.GRPO}")
    print("CUDA Available:", torch.cuda.is_available())

    os.environ["WANDB_PROJECT"] = "GRPO_experiments"

    torch_dtype = dtype_map.get(cfg.GRPO.dtype, torch.float32)
    torch_device = cfg.GRPO.torch_device

    tokenizer = get_tokenizer_muse().to(torch_device)
    action_parser = ActionParser(tokenizer)

    model = LlamaForCausalLM.from_pretrained(
        cfg.GRPO.model_path,
        torch_dtype=torch_dtype,
        use_safetensors=True
    )

    if cfg.GRPO.lora_path:
        lora_ckpt_path = cfg.GRPO.lora_path
        log.info(f"Loading LoRA from {lora_ckpt_path}")
        model = PeftModel.from_pretrained(model, lora_ckpt_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=32, 
            lora_alpha=64,
            lora_dropout=0.1,
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "down_proj", "up_proj"
            ]
        )
        log.info(f"Starting with LoRA: {lora_config}")
        model = get_peft_model(model, lora_config)
        
    model.print_trainable_parameters()
    
    dataset = TokenizedDataset(cfg.GRPO.dataset_pth)

    save_path = Path(cfg.GRPO.output_dir).parent / f"{cfg.GRPO.run_name}_merged_ckpts"

    training_args = GRPOConfig(
        output_dir=cfg.GRPO.output_dir,
        run_name=cfg.GRPO.run_name,
        learning_rate=cfg.GRPO.lr,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        beta=cfg.GRPO.start_beta,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=cfg.GRPO.num_batch*cfg.GRPO.num_generations,
        gradient_accumulation_steps=1,
        num_generations=cfg.GRPO.num_generations,
        max_prompt_length=256,
        max_completion_length=256,
        temperature=1.1,
        num_train_epochs=cfg.GRPO.num_train_epochs,
        save_strategy="epoch",
        # save_steps=200,
        max_grad_norm=0.1,
        report_to="wandb",
    )

    trainer = VisionGRPOTrainer(
        model=model,
        action_parser=action_parser,
        reward_funcs=[
            action_reward_func
        ],
        data_collator=collate_fn,
        args=training_args,
        train_dataset=dataset,
        on_policy = True,
        if_mixup = False,
        display_counter = 10,
        start_beta = cfg.GRPO.start_beta,
        end_beta = cfg.GRPO.end_beta,
    )

    trainer.train()

    # Save the model
    log.info("Merging LoRA into base model and saving...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(save_path)

if __name__ == "__main__":
    main()
