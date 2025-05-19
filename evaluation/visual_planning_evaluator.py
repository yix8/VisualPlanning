import hydra
import torch
import logging
import random
import numpy as np
import json
import jsonlines
import einops
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf
from transformers import LlamaForCausalLM
from peft import PeftModel
from vqlm.vqvae_muse import get_tokenizer_muse
from layout_parser import ActionParser
from collections import defaultdict, Counter
from accelerate.utils import is_peft_model
log = logging.getLogger(__name__)

def seed_everything(seed = 42):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TokenizedDataset(Dataset):
    def __init__(self, filepath):
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
    
    def __len__(self):
        return len(self.input_ids_list)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids_list[idx],
            "target_ids": self.target_ids_list[idx],
            "input_state": self.input_state_list[idx],
            "meta": self.meta_list[idx],
        }
    
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    target_ids = torch.stack([item['target_ids'] for item in batch])
    
    input_state = [item['input_state'] for item in batch]
    meta = [item['meta'] for item in batch]
    
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "input_state": input_state,
        "meta": meta,
    }

def show_stats(base_dir):
    # Define the base directory
    # base_dir = "test/sft_evaluation_result"

    # Data structure for statistics
    level_stats = defaultdict(lambda: {
        "true": 0,
        "false": 0,
        "false_dirs": [],
        "true_action_lengths": [],
        'all_actions_valid_false': [],
        "any_actions_invalid_false": [],
        "false_action_lengths": [],
        "valid" : 0,
        "invalid" : 0
    })

    # Iterate through all subdirectories
    for subdir in sorted(os.listdir(base_dir), key=lambda x: int(x) if x.isdigit() else float('inf')):  # Ensure numerical order
        subdir_path = os.path.join(base_dir, subdir)

        # Determine the level
        if subdir.isdigit():
            subdir_index = int(subdir)
            level = (subdir_index // 250) + 3  # Level starts at 3

            # Check if it is a directory
            if os.path.isdir(subdir_path):
                json_file_path = os.path.join(subdir_path, "parsed_actions.json")

                # Check if the JSON file exists
                if os.path.exists(json_file_path):
                    with open(json_file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                        # Check the value of the "complete" key
                        if isinstance(data, dict) and "complete" in data:
                            action_list_length = len(data.get("action_list", []))
                            
                            if data["complete"] is True:
                                level_stats[level]["true"] += 1
                                level_stats[level]["true_action_lengths"].append(action_list_length)
                                # if action_list_length == 20:
                                #     print(subdir)
 
                            elif data["complete"] is False:
                                level_stats[level]["false"] += 1
                                level_stats[level]["false_dirs"].append(subdir)
                                level_stats[level]["false_action_lengths"].append(action_list_length)
                                if all(action[1] != "invalid" for action in data['action_list']):
                                    level_stats[level]["all_actions_valid_false"].append(subdir)
                                    # if action_list_length == 20:
                                    #     print(subdir)
                                else:
                                    level_stats[level]["any_actions_invalid_false"].append(subdir)
                                    if action_list_length == 12:
                                        print(subdir)

                            for action in data['action_list']:
                                if action[1] != "invalid":
                                    level_stats[level]['valid'] += 1
                                else:
                                    level_stats[level]['invalid'] += 1

    false_list = []
    acc = []

    # Print statistics per level
    for level in sorted(level_stats.keys()):
        true_count = level_stats[level]["true"]
        false_count = level_stats[level]["false"]
        total_count = true_count + false_count
        true_percentage = (true_count / total_count * 100) if total_count > 0 else 0
        acc.append(true_percentage)
        print(f"\nLevel {level}:")
        print(f"  - Number of True: {true_count}")
        print(f"  - Number of False: {false_count}")
        print(f"        - All actions valid but not optimal: {len(level_stats[level]['all_actions_valid_false'])}")
        print(level_stats[level]['all_actions_valid_false'])
        print(f"        - Any actions invalid: {len(level_stats[level]['any_actions_invalid_false'])}")
        print(f"  - Accuracy (True Percentage): {true_percentage:.2f}%")
        print(f"  - Number of Valid: {level_stats[level]['valid']}")
        print(f"  - Number of Invalid: {level_stats[level]['invalid']}")

        # Print the list of directories where "complete" is False in this level
        # if level_stats[level]["false_dirs"]:
        #     print(f"  - Directories with incomplete path:")
        #     print(f"    {level_stats[level]['false_dirs']}")
        #     false_list.extend(level_stats[level]["false_dirs"])
        if level_stats[level]["any_actions_invalid_false"]:
            print(f"  - Directories with any invalid path:")
            print(f"    {level_stats[level]['any_actions_invalid_false']}")

        # Print action list length distribution
        true_action_lengths = level_stats[level]["true_action_lengths"]
        false_action_lengths = level_stats[level]["false_action_lengths"]

        if true_action_lengths:
            true_length_counts = Counter(true_action_lengths)
            print(f"  - Action List Length Distribution (Complete=True):")
            print(f"    Min: {min(true_action_lengths)}, Max: {max(true_action_lengths)}, Avg: {sum(true_action_lengths)/len(true_action_lengths):.2f}")
            for length, count in sorted(true_length_counts.items()):
                percentage = (count / len(true_action_lengths)) * 100
                print(f"    Length {length}: {count} ({percentage:.2f}%)")

        if false_action_lengths:
            false_length_counts = Counter(false_action_lengths)
            print(f"  - Action List Length Distribution (Complete=False):")
            print(f"    Min: {min(false_action_lengths)}, Max: {max(false_action_lengths)}, Avg: {sum(false_action_lengths)/len(false_action_lengths):.2f}")
            for length, count in sorted(false_length_counts.items()):
                percentage = (count / len(false_action_lengths)) * 100
                print(f"    Length {length}: {count} ({percentage:.2f}%)")

    # # sotre this list in a file
    # with open("false_list.txt", "w") as f:
    #     for item in false_list:
    #         f.write("%s\n" % item)
    # # read this file as a list
    # with open("false_list.txt", "r", encoding="utf-8") as f:
    #     test_false_list = f.read().splitlines()
    print(f"Average accuracy: {sum(acc)/len(acc):.2f}%")
    return false_list

def show_stats_mini(base_dir):
    # Define the base directory
    # base_dir = "test/sft_evaluation_result"

    # Data structure for statistics
    level_stats = defaultdict(lambda: {
        "true": 0,
        "false": 0,
        "false_dirs": [],
        "true_action_lengths": [],
        'all_actions_valid_false': [],
        "any_actions_invalid_false": [],
        "false_action_lengths": [],
        "valid" : 0,
        "invalid" : 0
    })

    def extract_level_and_index(name):
        try:
            level, index = map(int, name.split('_'))
            return (level, index)
        except (ValueError, IndexError):
            return (float('inf'), float('inf'))

    # Iterate through all subdirectories
    for subdir in sorted(os.listdir(base_dir), key=extract_level_and_index):  # Ensure numerical order
        subdir_path = os.path.join(base_dir, subdir)

        # Check if it is a directory
        if os.path.isdir(subdir_path):
            level, index = map(int, subdir.split('_'))
            json_file_path = os.path.join(subdir_path, "parsed_actions.json")

            # Check if the JSON file exists
            if os.path.exists(json_file_path):
                with open(json_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # Check the value of the "complete" key
                    if isinstance(data, dict) and "complete" in data:
                        action_list_length = len(data.get("action_list", []))
                        
                        if data["complete"] is True:
                            level_stats[level]["true"] += 1
                            level_stats[level]["true_action_lengths"].append(action_list_length)


                        elif data["complete"] is False:
                            level_stats[level]["false"] += 1
                            level_stats[level]["false_dirs"].append(subdir)
                            level_stats[level]["false_action_lengths"].append(action_list_length)
                            if all(action[1] != "invalid" for action in data['action_list']):
                                level_stats[level]["all_actions_valid_false"].append(subdir)
                                if action_list_length == 9:
                                    print(subdir)

                            else:
                                level_stats[level]["any_actions_invalid_false"].append(subdir)


                        for action in data['action_list']:
                            if action[1] != "invalid":
                                level_stats[level]['valid'] += 1
                            else:
                                level_stats[level]['invalid'] += 1

    false_list = []

    # Print statistics per level
    for level in sorted(level_stats.keys()):
        true_count = level_stats[level]["true"]
        false_count = level_stats[level]["false"]
        total_count = true_count + false_count
        true_percentage = (true_count / total_count * 100) if total_count > 0 else 0

        print(f"\nLevel {level}:")
        print(f"  - Number of True: {true_count}")
        print(f"  - Number of False: {false_count}")
        print(f"        - All actions valid but not optimal: {len(level_stats[level]['all_actions_valid_false'])}")
        print(f"        - Any actions invalid: {len(level_stats[level]['any_actions_invalid_false'])}")
        print(f"  - Accuracy (True Percentage): {true_percentage:.2f}%")
        print(f"  - Number of Valid: {level_stats[level]['valid']}")
        print(f"  - Number of Invalid: {level_stats[level]['invalid']}")

        # Print the list of directories where "complete" is False in this level
        # if level_stats[level]["false_dirs"]:
        #     print(f"  - Directories with incomplete path:")
        #     print(f"    {level_stats[level]['false_dirs']}")
        #     false_list.extend(level_stats[level]["false_dirs"])
        if level_stats[level]["any_actions_invalid_false"]:
            print(f"  - Directories with any invalid path:")
            print(f"    {level_stats[level]['any_actions_invalid_false']}")

        # Print action list length distribution
        true_action_lengths = level_stats[level]["true_action_lengths"]
        false_action_lengths = level_stats[level]["false_action_lengths"]

        if true_action_lengths:
            true_length_counts = Counter(true_action_lengths)
            print(f"  - Action List Length Distribution (Complete=True):")
            print(f"    Min: {min(true_action_lengths)}, Max: {max(true_action_lengths)}, Avg: {sum(true_action_lengths)/len(true_action_lengths):.2f}")
            for length, count in sorted(true_length_counts.items()):
                percentage = (count / len(true_action_lengths)) * 100
                print(f"    Length {length}: {count} ({percentage:.2f}%)")

        if false_action_lengths:
            false_length_counts = Counter(false_action_lengths)
            print(f"  - Action List Length Distribution (Complete=False):")
            print(f"    Min: {min(false_action_lengths)}, Max: {max(false_action_lengths)}, Avg: {sum(false_action_lengths)/len(false_action_lengths):.2f}")
            for length, count in sorted(false_length_counts.items()):
                percentage = (count / len(false_action_lengths)) * 100
                print(f"    Length {length}: {count} ({percentage:.2f}%)")

    total_false = sum(level_stats[l]["false"] for l in level_stats)
    total_any_invalid_false = sum(len(level_stats[l]["any_actions_invalid_false"]) for l in level_stats)

    if total_false > 0:
        ratio = total_any_invalid_false / total_false
        print(f"Any-invalid false ratio among all false completions: {total_any_invalid_false}/{total_false} = {ratio:.2%}")
    else:
        print("No false completions found.")
        
    # # sotre this list in a file
    # with open("false_list.txt", "w") as f:
    #     for item in false_list:
    #         f.write("%s\n" % item)
    # # read this file as a list
    # with open("false_list.txt", "r", encoding="utf-8") as f:
    #     test_false_list = f.read().splitlines()

    return false_list

def detect_task(test_dataset_pth: str) -> str:
    if "maze" in test_dataset_pth.lower():
        return "maze"
    elif "frozen_lake" in test_dataset_pth.lower():
        return "frozenlake"
    elif "minibehaviour" in test_dataset_pth.lower():
        return "minibehaviour"
    else:
        return "unknown"
    

def evaluate_frozen_or_maze(cfg: DictConfig):
    # Implement the evaluation logic for FrozenLake or Maze here
    if cfg.show_stats:
        if cfg.is_filter:
            raise NotImplementedError("show_stats is not implemented for filter mode.")
        else:
            show_stats(cfg.evaluation_result_folder_pth)
        return

    task = detect_task(cfg.test_dataset_pth)

    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer_muse().to(device)

    action_parser = ActionParser(tokenizer)

    base_model = LlamaForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True
    )

    model = PeftModel.from_pretrained(base_model, cfg.lora_path, torch_dtype=torch.bfloat16) if cfg.lora_path else base_model
    if is_peft_model(model):
        print(f"PEFT model detected. Loading checkpoint from {cfg.lora_path}")
        model = model.merge_and_unload()
    model.to(device)

    dataset = TokenizedDataset(cfg.test_dataset_pth)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    assert cfg.batch_size == 1, "for now, we only support batch size 1 for evaluation."

    correct_count = 0
    total_count = len(dataset)

    # check if evaluation_result_folder_pth exist, if not, create it
    os.makedirs(cfg.evaluation_result_folder_pth, exist_ok=True)

    for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        result_folder = os.path.join(cfg.evaluation_result_folder_pth, str(idx))
        save_path = os.path.join(result_folder, f"evaluation_{idx}.png")
        json_path = os.path.join(result_folder, "parsed_actions.json")

        if os.path.exists(result_folder) and os.path.exists(save_path) and os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data["complete"]:
                    correct_count += 1

            if data["complete"] is False and cfg.is_filter:
                filter_result_folder = os.path.join(f"{cfg.evaluation_result_folder_pth}_uncomplete", str(idx))
                filter_save_path = os.path.join(filter_result_folder, f"evaluation_{idx}.png")
                filter_json_path = os.path.join(filter_result_folder, "parsed_actions.json")
                if os.path.exists(result_folder) and os.path.exists(save_path) and os.path.exists(json_path):
                    log.info(f"Skipping env {idx} as it has already been evaluated.")
                    continue
            else:
                log.info(f"Skipping env {idx} as it has already been evaluated.")
                continue
        os.makedirs(result_folder, exist_ok=True)


        input_ids = batch["input_ids"].to(device)
        input_state = batch["input_state"][0] if cfg.is_filter else batch["input_state"][0][0]
        meta = batch["meta"][0]
        expected_move = 1 if cfg.is_filter else meta["distance_map"][str(input_state)]

        # layout_str = "\n".join(["".join(row) for row in meta['layout']])
        log.info(f"\nEvaluating env {idx}\nStart pos: {input_state}, Expected move: {expected_move}")
        
        new_tokens = [input_ids]
        num_generation = expected_move

        if cfg.double:
            num_generation = num_generation*2

        with torch.no_grad():
            for i in range(num_generation):
                output_ids = model.generate(
                    input_ids=input_ids[:, -256: ],
                    attention_mask=torch.ones_like(input_ids[:, -256: ]),
                    pad_token_id=8192,
                    max_new_tokens=256,
                    do_sample=False,
                    # temperature=temperature,
                    suppress_tokens=list(range(8192, model.vocab_size)),
                )
                input_ids = output_ids[:, -256: ]
                new_tokens.append(input_ids)
        new_tokens = torch.cat(new_tokens, dim=1).view(-1, 256)
        new_images = einops.rearrange(
            torch.clamp(tokenizer.decode_code(new_tokens), 0.0, 1.0),
            'b c h w -> b h w c'
        ).detach().cpu().numpy()

        data = {
            "start_coords": [ActionParser.get_coordinate_from_state(input_state, meta['level'])],
            "action_list": [],
            "complete" : False,
            "level" : meta['level'],
            "start_pos" : input_state,
            "target_pos" : meta['target_pos'],
            "layout" : meta['layout'],
            "distance_map" : meta['distance_map']
        }

        for i in range(len(new_images)-1):
            input_img = new_images[i]
            pred_img = new_images[i+1]
            if task == "maze":
                dict = action_parser.parse_maze_action_in_imgs(input_img, pred_img, meta['level'], data['start_coords'][-1], meta['start_pos'], meta['target_pos'], meta['layout'], meta['distance_map'])
            elif task == 'frozenlake':
                dict = action_parser.parse_action_in_imgs(input_img, pred_img, meta['level'], data['start_coords'][-1], meta['target_pos'])
            else:
                raise NotImplementedError(f"Task {task} not implemented.")
            action = dict['action']
            next_coord = dict['pred_coord']
            data['start_coords'].append(next_coord)
            data['action_list'].append(action)

        if cfg.is_filter:
            assert len(data['action_list']) == expected_move, f"Expected {expected_move} moves, but got {len(data['action_list'])} moves."
            if all(action[1] != "invalid" for action in data['action_list']):
                current_distance = meta['distance_map'][str(input_state)]
                state = data['start_coords'][-1][0] * meta['level'] + data['start_coords'][-1][1]
                next_distance = meta['distance_map'].get(str(state), -1)
                if next_distance == current_distance - 1 or state == meta['target_pos']:
                    data['complete'] = True
                    correct_count += 1
                    log.info(f"Correctly found the optimal path!")
        else:
            if ActionParser.get_coordinate_from_state(meta['target_pos'], meta['level']) == data['start_coords'][expected_move]:
                if all(action[1] != "invalid" for action in data['action_list'][:expected_move]):
                    data['complete'] = True
                    correct_count += 1
                    log.info(f"Correctly found the optimal path!")

        fig, axes = plt.subplots(1, len(new_images), figsize=(len(new_images) * 3, 3))
        
        for ax, image in zip(axes, new_images):
            ax.imshow(image)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        
        if cfg.is_filter and data['complete'] is False:
            filter_result_folder = os.path.join(f"{cfg.evaluation_result_folder_pth}_uncomplete", str(idx))
            filter_save_path = os.path.join(filter_result_folder, f"evaluation_{idx}.png")
            os.makedirs(filter_result_folder, exist_ok=True)
            plt.savefig(filter_save_path, bbox_inches="tight")

        plt.close(fig)

        log.info(f"Saved evaluation results to {save_path}")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        if cfg.is_filter and data['complete'] is False:
            filter_result_folder = os.path.join(f"{cfg.evaluation_result_folder_pth}_uncomplete", str(idx))
            filter_json_path = os.path.join(filter_result_folder, "parsed_actions.json")
            with open(filter_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

        del input_ids, new_tokens
        torch.cuda.empty_cache()
        
    log.info(f"Evaluation Accuracy: {correct_count/total_count:.2%}")


def evaluate_minibehaviour(cfg: DictConfig):
    # Implement the evaluation logic for FrozenLake or Maze here
    log.info("Evaluating minibehaviour...")

    if cfg.show_stats:
        if cfg.is_filter:
            raise NotImplementedError("show_stats is not implemented for filter mode.")
        else:
            show_stats_mini(cfg.evaluation_result_folder_pth)
        return

    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer_muse().to(device)

    action_parser = ActionParser(tokenizer)

    base_model = LlamaForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True
    )

    model = PeftModel.from_pretrained(base_model, cfg.lora_path, torch_dtype=torch.bfloat16) if cfg.lora_path else base_model
    if is_peft_model(model):
        print(f"PEFT model detected. Loading checkpoint from {cfg.lora_path}")
        model = model.merge_and_unload()
    model.to(device)

    dataset = TokenizedDataset(cfg.test_dataset_pth)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    assert cfg.batch_size == 1, "for now, we only support batch size 1 for evaluation."

    correct_count = 0
    total_count = len(dataset)

    # check if evaluation_result_folder_pth exist, if not, create it
    os.makedirs(cfg.evaluation_result_folder_pth, exist_ok=True)

    for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        meta = batch["meta"][0]
        level = meta['level']
        data_id = meta['idx']

        result_folder = os.path.join(cfg.evaluation_result_folder_pth, f"{level}_{data_id}")
        save_path = os.path.join(result_folder, f"evaluation_{idx}.png")
        json_path = os.path.join(result_folder, "parsed_actions.json")

        if os.path.exists(result_folder) and os.path.exists(save_path) and os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data["complete"]:
                    correct_count += 1

            if data["complete"] is False and cfg.is_filter:
                filter_result_folder = os.path.join(f"{cfg.evaluation_result_folder_pth}_uncomplete", f"{level}_{data_id}")
                filter_save_path = os.path.join(filter_result_folder, f"evaluation_{idx}.png")
                filter_json_path = os.path.join(filter_result_folder, "parsed_actions.json")
                if os.path.exists(result_folder) and os.path.exists(save_path) and os.path.exists(json_path):
                    log.info(f"Skipping env {idx} as it has already been evaluated.")
                    continue
            else:
                log.info(f"Skipping env {idx} as it has already been evaluated.")
                continue

        os.makedirs(result_folder, exist_ok=True)
        input_ids = batch["input_ids"].to(device)
        input_info = batch["input_state"][0] if cfg.is_filter else batch["input_state"][0][0]
        expected_move = 1 if cfg.is_filter else len(batch["input_state"][0])-1

        # layout_str = "\n".join(["".join(row) for row in meta['layout']])
        log.info(f"\nEvaluating env: Level {level}, Data {data_id}\nStart pos: {input_info}, Expected move: {expected_move}")
        
        new_tokens = [input_ids]
        num_generation = expected_move

        if cfg.double:
            num_generation = num_generation*2

        with torch.no_grad():
            for i in range(num_generation):
                output_ids = model.generate(
                    input_ids=input_ids[:, -256: ],
                    attention_mask=torch.ones_like(input_ids[:, -256: ]),
                    pad_token_id=8192,
                    max_new_tokens=256,
                    do_sample=False,
                    # temperature=temperature,
                    suppress_tokens=list(range(8192, model.vocab_size)),
                )
                input_ids = output_ids[:, -256: ]
                new_tokens.append(input_ids)
        new_tokens = torch.cat(new_tokens, dim=1).view(-1, 256)
        new_images = einops.rearrange(
            torch.clamp(tokenizer.decode_code(new_tokens), 0.0, 1.0),
            'b c h w -> b h w c'
        ).detach().cpu().numpy()

        data = {
            "start_coords": [input_info],
            "action_list": [],
            "complete" : False,
            "meta": meta
        }

        invalid_flag = False
        for i in range(len(new_images)-1):
            input_img = new_images[i]
            pred_img = new_images[i+1]

            if not invalid_flag:
                action_dict = action_parser.parse_mini_action_in_imgs(input_img, pred_img, data['start_coords'][-1], meta)
                action = action_dict['action']
                next_coord = action_dict['pred_coord']
                carrying = action_dict['carrying']

                if action[1] == 'invalid':
                    invalid_flag = True

            if invalid_flag:
                next_coord = data['start_coords'][-1][0]
                carrying = data['start_coords'][-1][1]
                action = (-1, 'invalid')

            data['start_coords'].append([list(next_coord), carrying])
            data['action_list'].append(action)


        all_valid = all(action[1] != "invalid" for action in data['action_list'][:expected_move])
        last_is_drop = data['action_list'][expected_move - 1][1] == "drop"
        has_pick = any(action[1] == "pick" for action in data['action_list'][:expected_move])

        if all_valid and last_is_drop and has_pick:
            data['complete'] = True
            correct_count += 1
            log.info(f"Correctly found the optimal path!")

        fig, axes = plt.subplots(1, len(new_images), figsize=(len(new_images) * 3, 3))
        
        for ax, image in zip(axes, new_images):
            ax.imshow(image)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

        log.info(f"Saved evaluation results to {save_path}")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


        del input_ids, new_tokens
        torch.cuda.empty_cache()
        
    log.info(f"Evaluation Accuracy: {correct_count/total_count:.2%}")


@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    if cfg.is_filter:
        cfg = OmegaConf.load("configs/filter.yaml")
    log.info(f"Eval Config: {cfg}")

    task = detect_task(cfg.test_dataset_pth)

    if task == "minibehaviour":
        evaluate_minibehaviour(cfg)
    else:
        evaluate_frozen_or_maze(cfg)

if __name__ == "__main__":
    main()