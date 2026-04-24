import torch
import jsonlines
import hydra
import logging
import random
import wandb
import numpy as np
import sys
import einops
import os
from pathlib import Path
from peft import PeftModel, PeftConfig
from peft import get_peft_model, LoraConfig, TaskType
from vqlm.vqvae_muse import get_tokenizer_muse
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, TrainerCallback
from omegaconf import DictConfig
from transformers import LlamaForCausalLM

log = logging.getLogger(__name__)

dtype_map = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16
}

def seed_everything(seed = 42):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ImageReconstructionCallback(TrainerCallback):
    def __init__(self, eval_steps, model, tokenizer, test_images, logger):
        self.eval_steps = eval_steps
        self.model = model
        self.tokenizer = tokenizer
        self.test_images = test_images
        self.logger = logger

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0:
            self.run_inference(state.global_step)

    def run_inference(self, step):
        # make sure inputs are on the same device as the model
        test_images = self.test_images.to(self.model.device)
        self.tokenizer.to(self.model.device)
        inputs = test_images[:, :256]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                pad_token_id=8192,
                max_new_tokens=256,
                suppress_tokens=list(range(8192, self.model.vocab_size)),)
            generated_tokens = outputs[:, 256:]

        final_sequence = torch.cat((test_images, generated_tokens), dim=1).view(-1, 256)

        new_images = einops.rearrange(
            torch.clamp(self.tokenizer.decode_code(final_sequence), 0.0, 1.0),
            'b c h w -> b h w c'
        ).detach().cpu().numpy()

        self.logger.log({
            "step": step,
            "input, pred, and target images": wandb.Image(
                np.hstack((
                    new_images[0],
                    new_images[2],
                    new_images[1]
                )), 
                caption="Input (Left), Pred (Middle), Target (Right) Image"
            )
        })


# Define the dataset class
class TokenizedDataset(Dataset):
    def __init__(self, filepath):
        self.tokenized_data = []
        with jsonlines.open(filepath) as reader:
            for obj in reader:
                self.tokenized_data.append(torch.tensor(obj['input_tokens'] + obj['output_tokens'], dtype=torch.long))
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx],

# Define custom collate function for DataLoader
def collate_fn(batch):
    batch_inputs = [item[0] for item in batch]
    batch_inputs_padded = pad_sequence(batch_inputs, batch_first=True, padding_value=-100)

    # Create attention masks
    attention_masks = torch.zeros_like(batch_inputs_padded, dtype=torch.long)
    attention_masks = attention_masks.masked_fill(batch_inputs_padded != -100, 1)

    labels = batch_inputs_padded.clone()
    labels[:, :256] = -100

    # batch_inputs_padded = batch_inputs_padded[:, :-1]
    # labels = labels[:, 1:]
    # attention_masks = attention_masks[:, :-1]

    return {'input_ids': batch_inputs_padded, 'attention_mask': attention_masks, 'labels': labels}


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        print("\nChecking inputs before forward pass:")
        print("Checking input_ids:", inputs["input_ids"].shape)
        print("Checking labels:", inputs["labels"].shape)
        print(f"Input IDs: {inputs['input_ids']}")
        print(f"Labels: {inputs['labels']}")

        outputs = model(**inputs)
        loss = outputs.loss

        print(f"\n Forward Pass Completed. Loss: {loss.item()}")
        sys.exit()
        return (loss, outputs) if return_outputs else loss
    
@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.SFT.seed)
    
    log.info(f"Training Config: {cfg.SFT}")
    os.environ["WANDB_PROJECT"] = "SFT_experiments"

    torch_dtype = dtype_map.get(cfg.SFT.dtype, torch.float32)
    torch_device = cfg.SFT.torch_device

    tokenizer = get_tokenizer_muse()

    save_path = Path(cfg.SFT.output_dir).parent / f"{cfg.SFT.run_name}_merged_ckpts"

    # load the model
    model = LlamaForCausalLM.from_pretrained(
        cfg.SFT.model_path,
        torch_dtype=torch_dtype,
        use_safetensors=True
    )

    if cfg.SFT.use_lora:
        log.info("Using LoRA")
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
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # load the dataset
    dataset = TokenizedDataset(cfg.SFT.dataset_pth)

    # Initialize the TrainingArguments
    training_args = TrainingArguments(
        output_dir=cfg.SFT.output_dir,  
        overwrite_output_dir=False,  
        num_train_epochs=cfg.SFT.num_train_epochs,  
        per_device_train_batch_size=cfg.SFT.per_device_train_batch_size, 
        gradient_accumulation_steps=1, 
        learning_rate=1.5e-4, 
        warmup_ratio=0.1,  
        lr_scheduler_type="cosine", 
        bf16=True,
        save_strategy="epoch",
        save_total_limit=20,
        logging_strategy="steps",
        logging_steps=1,
        report_to="wandb",  
        dataloader_num_workers=8,
        run_name=cfg.SFT.run_name, 
    )

    test_images = dataset[0][0].unsqueeze(0)

    image_callback = ImageReconstructionCallback(
        eval_steps=50, 
        model=model,
        tokenizer=tokenizer,  
        test_images=test_images,
        logger=wandb
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        callbacks=[image_callback]
    )

    # Train the model
    trainer.train()
    
    if cfg.SFT.use_lora:
        # Save the model
        log.info("Merging LoRA into base model and saving...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(save_path)

if __name__ == "__main__":
    main()