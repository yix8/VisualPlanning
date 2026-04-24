import torch
import re
import numpy as np
from PIL import Image
import einops
from vqlm.vqvae_muse import get_tokenizer_muse
import matplotlib.pyplot as plt
import json
import cv2
import random
import wandb

def plot_grid_patches(coordinates, level, figsize=(8, 8), cmap='gray'):
    fig, axes = plt.subplots(level, level, figsize=figsize)
    
    for i in range(level):
        for j in range(level):
            ax = axes[i, j]
            ax.imshow(coordinates[(i, j)], cmap='gray', vmin=0, vmax=255)
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

class ActionParser():
    def __init__(self, tokenizer):
        assert tokenizer != None
        self.tokenizer = tokenizer
    
    def decode_tokens(self, ids):
        # Check if ids are tensors and have shape (batch_size, 256)
        if not isinstance(ids, torch.Tensor):
            raise TypeError("ids must be a torch.Tensor")
        if ids.dim() != 2 or ids.size(1) != 256:
            raise ValueError("ids must have shape (batch size, 256)")
        
        # make sure ids are on the same device as the tokenizer
        ids_copy = ids.clone()
        ids_copy = ids_copy.to(self.tokenizer.device)
    
        imgs = einops.rearrange(
            torch.clamp(self.tokenizer.decode_code(ids_copy), 0.0, 1.0),
            'b c h w -> b h w c'
        ).detach().cpu().numpy()

        return imgs

    @classmethod
    def get_pixel_location(cls, img, level, crop_ratio=0.0):
        height, width = img.shape[0:2]
        if height != width or height != 256:
            raise ValueError("Resolution of the imgae should be 256 times 256!")
        
        new_height = level * (height // level)
        new_width = level * (width // level)
        if new_height != height or new_width != width:
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        grid_size = new_height // level
        coordinates = {}
        for i in range(level):
            for j in range(level):
                top_left_x = i * grid_size
                top_left_y = j * grid_size
                bottom_right_x = (i + 1) * grid_size
                bottom_right_y = (j + 1) * grid_size
                
                crop_px = int(grid_size * crop_ratio)
                cropped_img = img[
                    top_left_x + crop_px:bottom_right_x - crop_px,
                    top_left_y + crop_px:bottom_right_y - crop_px
                ]
                coordinates[(i, j)] = cropped_img
   
        return coordinates
    
    @classmethod
    def drop_outer_ring(cls, coord_dict):
        # find the grid size (7 here)
        n = max(i for (i, _) in coord_dict) + 1

        inner_coords = {}
        for (i, j), patch in coord_dict.items():
            # skip the first and last row or column
            if 0 < i < n - 1 and 0 < j < n - 1:
                inner_coords[(i - 1, j - 1)] = patch  # re-index to 0...4

        return inner_coords

    @classmethod
    def get_action(cls, start_coord, next_coord):
        ACTIONS = {
            (-1,  0): (0, 'up'),
            ( 1,  0): (1, 'down'),
            ( 0, -1): (2, 'left'),
            ( 0,  1): (3, 'right')
        }
        delta = (next_coord[0] - start_coord[0], next_coord[1] - start_coord[1])
        return ACTIONS.get(delta, (-1, 'invalid'))

    @classmethod
    def get_maze_action(cls, start_coord, next_coord, layout):
        ACTIONS = {
            (-1,  0): (0, 'up'),
            ( 1,  0): (1, 'down'),
            ( 0, -1): (2, 'left'),
            ( 0,  1): (3, 'right')
        }
        delta = (next_coord[0] - start_coord[0], next_coord[1] - start_coord[1])
        action = ACTIONS.get(delta, (-1, 'invalid'))
        
        if action[0] == -1:
            return action
        walls = layout[start_coord[0]][start_coord[1]]
        wall_block = {
            'up': not walls['north'],
            'down': not walls['south'],
            'left': not walls['west'],
            'right': not walls['east']
        }
        print(wall_block)
        print(start_coord)

        if not wall_block[action[1]]:
            return (-1, 'invalid')
    
        return action

    @classmethod
    def get_mini_action(cls, start_coord, next_coord, next_picking, next_dropping):

        if start_coord != next_coord:
            if next_picking or next_dropping:
                action = (-1, 'invalid')
            else:
                ACTIONS = {
                    (-1,  0): (0, 'up'),
                    ( 1,  0): (1, 'down'),
                    ( 0, -1): (2, 'left'),
                    ( 0,  1): (3, 'right')
                }
                delta = (next_coord[0] - start_coord[0], next_coord[1] - start_coord[1])
                action = ACTIONS.get(delta, (-1, 'invalid'))
        else:
            if next_picking or next_dropping:
                if next_picking:
                    action = (4, 'pick')
                else:
                    action = (5, 'drop')
            else:
                action = (-1, 'invalid')
            
        return action
    
    @classmethod
    def get_coordinate_from_state(cls, state, level):
        assert state < level*level, "state must be less than level*level"
        row = state // level
        col = state % level
        return (row, col)

    @classmethod
    def visualize_imgs(cls, input_img, pred_img):
        def to_uint8(img):
            if img.dtype == bool:
                return img.astype(np.uint8) * 255
            elif img.dtype == np.float32 or img.dtype == np.float64:
                if img.max() <= 1.0:
                    return (img * 255).astype(np.uint8)
                else:
                    return img.astype(np.uint8)
            elif img.dtype == np.uint8:
                return img
            else:
                raise TypeError(f"Unsupported image dtype: {img.dtype}")

        input_img = to_uint8(input_img)
        pred_img = to_uint8(pred_img)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(input_img, cmap='gray', vmin=0, vmax=255)
        axs[0].set_title('Input Image')
        axs[0].axis('off')

        axs[1].imshow(pred_img, cmap='gray', vmin=0, vmax=255)
        axs[1].set_title('Pred Image')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    @classmethod 
    def visualize_ids(cls, input_ids, pred_ids, tokenizer):
        def decode_tokens(ids, tokenizer):
            # Check if ids are tensors and have shape (batch_size, 256)
            if not isinstance(ids, torch.Tensor):
                raise TypeError("ids must be a torch.Tensor")
            if ids.dim() != 2 or ids.size(1) != 256:
                raise ValueError("ids must have shape (batch size, 256)")
            
            # make sure ids are on the same device as the tokenizer
            ids_copy = ids.clone()
            ids_copy = ids_copy.to(tokenizer.device)
        
            imgs = einops.rearrange(
                torch.clamp(tokenizer.decode_code(ids_copy), 0.0, 1.0),
                'b c h w -> b h w c'
            ).detach().cpu().numpy()
            return imgs
        input_img = decode_tokens(input_ids, tokenizer)[0]
        pred_img = decode_tokens(pred_ids, tokenizer)[0]
        ActionParser.visualize_imgs(input_img, pred_img)

    @classmethod
    def covert_img_to_white_else_black(cls, img):
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Image must be a 3-channel RGB image")
        
        if img.dtype == np.float32 or img.dtype == np.float64:
            img_uint8 = (img * 255).astype(np.uint8)
        elif img.dtype == np.uint8:
            img_uint8 = img
        else:
            raise TypeError("Image must be either float32/float64 or uint8")
        
        ref = np.array([
            [255,   0,   0],
            [255, 255, 255],
            [  0,   0,   0],
            [139,  69,  19] 
        ], dtype=np.int16)

        img_int16 = img_uint8.astype(np.int16)      # Prevent subtraction overflow
        diff = img_int16[..., None] - ref.T         # (H, W, 3, 4)
        dist2 = np.sum(diff ** 2, axis=2)           # (H, W, 4)

        # Find the index of the nearest color
        nearest = np.argmin(dist2, axis=2)          # (H, W)

        # Build the output: start with all white, then set pixels nearest to black as black
        out_uint8 = np.full_like(img_uint8, 255)    # all white
        out_uint8[nearest == 2] = 0                 # index 2 corresponds to black

        # Return 0-1 float32 for convenient later conversion back to grayscale via multiplication by 255
        return out_uint8.astype(np.float32) / 255.0

    def parse_action_in_ids(self, input_ids, pred_ids, level, start_coord, target_coord, report_to = False):
        if not isinstance(input_ids, torch.Tensor) or not isinstance(pred_ids, torch.Tensor):
            raise TypeError("input_ids and pred_ids must be torch.Tensor")
        
        start_coord = self.get_coordinate_from_state(start_coord, level) if isinstance(start_coord, int) else start_coord
        target_coord = self.get_coordinate_from_state(target_coord, level) if isinstance(target_coord, int) else target_coord

        input_img = self.decode_tokens(input_ids)[0]
        pred_img = self.decode_tokens(pred_ids)[0]

        input_gray = cv2.cvtColor((input_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        pred_gray = cv2.cvtColor((pred_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        input_coordinates = self.get_pixel_location(input_gray, level)
        pred_coordinates = self.get_pixel_location(pred_gray, level)

        def compute_iou(img1, img2, thresh=200, is_show = False):
            _, bin1 = cv2.threshold(img1, thresh, 255, cv2.THRESH_BINARY)
            _, bin2 = cv2.threshold(img2, thresh, 255, cv2.THRESH_BINARY)
            bin1_bool = bin1 == 0
            bin2_bool = bin2 == 0
            if is_show:
                self.visualize_imgs(bin1_bool, bin2_bool)
            intersection = np.logical_and(bin1_bool, bin2_bool).sum()
            union = np.logical_or(bin1_bool, bin2_bool).sum()
            iou = 0.0
            if union > 0:
                iou = intersection / union
            return iou

        def compute_mse(img1, img2):
            return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        
        mse_values = {}
        iou_values = {}
        for coord in input_coordinates.keys():
            mse_values[coord] = compute_mse(input_coordinates[coord], pred_coordinates[coord])
            iou_values[coord] = compute_iou(input_coordinates[start_coord], pred_coordinates[coord])

        sorted_mse = sorted(mse_values.items(), key=lambda x: x[1], reverse=True)
        sorted_iou = sorted(iou_values.items(), key=lambda x: x[1], reverse=True)

        most_changed_coords = [sorted_mse[i][0] for i in range(min(2, len(sorted_mse)))]
        least_changed_coords = [sorted_iou[i][0] for i in range(min(2, len(sorted_iou)))]

        extracted_coord = target_coord if target_coord in most_changed_coords else least_changed_coords[0]

        # not sure
        if extracted_coord not in most_changed_coords:
            extracted_coord = start_coord

        action = self.get_action(start_coord, extracted_coord)

        if report_to:
            wandb.log({
                "action with pred_coord": wandb.Html(str(action) + ", " + str(extracted_coord)),
                "input_pred_images": wandb.Image(np.hstack(((input_img * 255).astype(np.uint8),(pred_img * 255).astype(np.uint8))), caption="Input (Left) and Pred (Right) Image"),
            })

        return {
            "action": action,
            "pred_coord" : extracted_coord
        }

    def parse_action_in_imgs(self, input_img, pred_img, level, start_coord, target_coord, report_to = False):
        
        start_coord = self.get_coordinate_from_state(start_coord, level) if isinstance(start_coord, int) else start_coord
        target_coord = self.get_coordinate_from_state(target_coord, level) if isinstance(target_coord, int) else target_coord

        input_gray = cv2.cvtColor((input_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        pred_gray = cv2.cvtColor((pred_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        input_coordinates = self.get_pixel_location(input_gray, level)
        pred_coordinates = self.get_pixel_location(pred_gray, level)

        def compute_iou(img1, img2, thresh=200, is_show = False):
            _, bin1 = cv2.threshold(img1, thresh, 255, cv2.THRESH_BINARY)
            _, bin2 = cv2.threshold(img2, thresh, 255, cv2.THRESH_BINARY)
            bin1_bool = bin1 == 0
            bin2_bool = bin2 == 0
            if is_show:
                self.visualize_imgs(bin1_bool, bin2_bool)
            intersection = np.logical_and(bin1_bool, bin2_bool).sum()
            union = np.logical_or(bin1_bool, bin2_bool).sum()
            iou = 0.0
            if union > 0:
                iou = intersection / union
            return iou

        def compute_mse(img1, img2):
            return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        
        mse_values = {}
        iou_values = {}
        for coord in input_coordinates.keys():
            mse_values[coord] = compute_mse(input_coordinates[coord], pred_coordinates[coord])
            iou_values[coord] = compute_iou(input_coordinates[start_coord], pred_coordinates[coord])

        sorted_mse = sorted(mse_values.items(), key=lambda x: x[1], reverse=True)
        sorted_iou = sorted(iou_values.items(), key=lambda x: x[1], reverse=True)

        most_changed_coords = [sorted_mse[i][0] for i in range(min(2, len(sorted_mse)))]
        least_changed_coords = [sorted_iou[i][0] for i in range(min(2, len(sorted_iou)))]

        extracted_coord = target_coord if target_coord in most_changed_coords else least_changed_coords[0]

        # not sure
        if extracted_coord not in most_changed_coords:
            extracted_coord = start_coord

        action = self.get_action(start_coord, extracted_coord)

        if report_to:
            wandb.log({
                "action with pred_coord": wandb.Html(str(action) + ", " + str(extracted_coord)),
                "input_pred_images": wandb.Image(np.hstack(((input_img * 255).astype(np.uint8),(pred_img * 255).astype(np.uint8))), caption="Input (Left) and Pred (Right) Image"),
            })

        return {
            "action": action,
            "pred_coord" : extracted_coord
        }
    
    def parse_maze_action_in_ids(self, input_ids, pred_ids, level, start_coord, initial_coord, target_coord, layout, distance_map, report_to = False):
        if not isinstance(input_ids, torch.Tensor) or not isinstance(pred_ids, torch.Tensor):
            raise TypeError("input_ids and pred_ids must be torch.Tensor")
        
        start_coord = self.get_coordinate_from_state(start_coord, level) if isinstance(start_coord, int) else start_coord
        initial_coord = self.get_coordinate_from_state(initial_coord, level) if isinstance(initial_coord, int) else initial_coord
        target_coord = self.get_coordinate_from_state(target_coord, level) if isinstance(target_coord, int) else target_coord

        input_img = self.decode_tokens(input_ids)[0]
        pred_img = self.decode_tokens(pred_ids)[0]

        input_gray = cv2.cvtColor((input_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        pred_gray = cv2.cvtColor((pred_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        input_coordinates = self.get_pixel_location(input_gray, level, 0.05)
        pred_coordinates = self.get_pixel_location(pred_gray, level, 0.05)

        def compute_iou(img1, img2, thresh=200, is_show = False):
            _, bin1 = cv2.threshold(img1, thresh, 255, cv2.THRESH_BINARY)
            _, bin2 = cv2.threshold(img2, thresh, 255, cv2.THRESH_BINARY)
            bin1_bool = bin1 == 0
            bin2_bool = bin2 == 0
            if is_show:
                self.visualize_imgs(bin1_bool, bin2_bool)
            intersection = np.logical_and(bin1_bool, bin2_bool).sum()
            union = np.logical_or(bin1_bool, bin2_bool).sum()
            iou = 0.0
            if union > 0:
                iou = intersection / union
            return iou

        def compute_mse(img1, img2):
            return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        
        mse_values = {}
        iou_values = {}
        for coord in input_coordinates.keys():
            mse_values[coord] = compute_mse(input_coordinates[coord], pred_coordinates[coord])

            # if coord != target_coord:
            iou_values[coord] = compute_iou(input_coordinates[start_coord], pred_coordinates[coord])

        sorted_mse = sorted(
            [(coord, val) for coord, val in mse_values.items() if val > 500],
            key=lambda x: x[1],
            reverse=True
        )
        # sorted_mse = sorted(mse_values.items(), key=lambda x: x[1], reverse=True)
        sorted_iou = sorted(iou_values.items(), key=lambda x: x[1], reverse=True)

        most_changed_coords = [sorted_mse[i][0] for i in range(min(2, len(sorted_mse)))]
        least_changed_coords = [sorted_iou[i][0] for i in range(min(2, len(sorted_iou)))]

        # what if the player does not move?
        if start_coord == initial_coord:
            if len(sorted_mse) == 0:
                extracted_coord = start_coord
            else:
                extracted_coord = most_changed_coords[0] if sorted_mse[0][1] > 1000 else start_coord
        else:
            extracted_coord = target_coord if target_coord in most_changed_coords else least_changed_coords[0]
        
        action = self.get_maze_action(start_coord, extracted_coord, layout)

        if report_to:
            next_coord = extracted_coord
            start_state = start_coord[0] * level + start_coord[1]
            state = next_coord[0] * level + next_coord[1]

            reward = -2
            if action[1] == 'invalid':
                reward = -5
            elif extracted_coord == target_coord:
                reward = 1
            else:
                current_distance = distance_map[str(start_state)]
                next_distance = distance_map[str(state)]
                reward = 1 if next_distance == current_distance - 1 else -5 if next_distance == -1 else 0

            wandb.log({
                "input_pred_images": wandb.Image(
                    np.hstack(((input_img * 255).astype(np.uint8), (pred_img * 255).astype(np.uint8))),
                    caption=f"{action}, {extracted_coord} Reward: {reward}"
                ),
            })

        return {
            "action": action,
            "pred_coord" : extracted_coord,
            "image": np.hstack(((input_img * 255).astype(np.uint8),(pred_img * 255).astype(np.uint8)))
        }       

    def parse_maze_action_in_imgs(self, input_img, pred_img, level, start_coord, initial_coord, target_coord, layout, distance_map, report_to = False):
        
        start_coord = self.get_coordinate_from_state(start_coord, level) if isinstance(start_coord, int) else start_coord
        initial_coord = self.get_coordinate_from_state(initial_coord, level) if isinstance(initial_coord, int) else initial_coord
        target_coord = self.get_coordinate_from_state(target_coord, level) if isinstance(target_coord, int) else target_coord

        input_gray = cv2.cvtColor((input_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        pred_gray = cv2.cvtColor((pred_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        input_coordinates = self.get_pixel_location(input_gray, level, 0.05)
        pred_coordinates = self.get_pixel_location(pred_gray, level, 0.05)

        def compute_iou(img1, img2, thresh=200, is_show = False):
            _, bin1 = cv2.threshold(img1, thresh, 255, cv2.THRESH_BINARY)
            _, bin2 = cv2.threshold(img2, thresh, 255, cv2.THRESH_BINARY)
            bin1_bool = bin1 == 0
            bin2_bool = bin2 == 0
            if is_show:
                self.visualize_imgs(bin1_bool, bin2_bool)
            intersection = np.logical_and(bin1_bool, bin2_bool).sum()
            union = np.logical_or(bin1_bool, bin2_bool).sum()
            iou = 0.0
            if union > 0:
                iou = intersection / union
            return iou

        def compute_mse(img1, img2):
            return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        
        mse_values = {}
        iou_values = {}
        for coord in input_coordinates.keys():
            mse_values[coord] = compute_mse(input_coordinates[coord], pred_coordinates[coord])

            # if coord != target_coord:
            iou_values[coord] = compute_iou(input_coordinates[start_coord], pred_coordinates[coord])

        sorted_mse = sorted(
            [(coord, val) for coord, val in mse_values.items() if val > 500],
            key=lambda x: x[1],
            reverse=True
        )
        # sorted_mse = sorted(mse_values.items(), key=lambda x: x[1], reverse=True)
        sorted_iou = sorted(iou_values.items(), key=lambda x: x[1], reverse=True)

        most_changed_coords = [sorted_mse[i][0] for i in range(min(2, len(sorted_mse)))]
        least_changed_coords = [sorted_iou[i][0] for i in range(min(2, len(sorted_iou)))]

        # what if the player does not move?
        if start_coord == initial_coord:
            if len(sorted_mse) == 0:
                extracted_coord = start_coord
            else:
                extracted_coord = most_changed_coords[0] if sorted_mse[0][1] > 1000 else start_coord
        else:
            extracted_coord = target_coord if target_coord in most_changed_coords else least_changed_coords[0]
        
        action = self.get_maze_action(start_coord, extracted_coord, layout)

        if report_to:

            next_coord = extracted_coord
            start_state = start_coord[0] * level + start_coord[1]
            state = next_coord[0] * level + next_coord[1]

            reward = -2
            if action[1] == 'invalid':
                reward = -5
            elif extracted_coord == target_coord:
                reward = 1
            else:
                current_distance = distance_map[str(start_state)]
                next_distance = distance_map[str(state)]
                reward = 1 if next_distance == current_distance - 1 else -5 if next_distance == -1 else 0
                
            wandb.log({
                "input_pred_images": wandb.Image(
                    np.hstack(((input_img * 255).astype(np.uint8), (pred_img * 255).astype(np.uint8))),
                    caption=f"{action}, {extracted_coord} Reward: {reward}"
                ),
            })

        return {
            "action": action,
            "pred_coord" : extracted_coord,
            "image": np.hstack(((input_img * 255).astype(np.uint8),(pred_img * 255).astype(np.uint8)))
        }
    
    def parse_mini_action_in_ids(self, input_ids, pred_ids, start_info, meta, report_to = False):
        if not isinstance(input_ids, torch.Tensor) or not isinstance(pred_ids, torch.Tensor):
            raise TypeError("input_ids and pred_ids must be torch.Tensor")
        
        level = meta['level']
        start_coord = tuple(start_info[0])
        carrying = start_info[1]

        printer_pos = tuple(meta['printer_pos'])
        table_pos = [tuple(p) for p in meta['table_pos']]

        def binarise_black_vs_white(img, level, thresh=40):
            if img.dtype == np.float32 or img.dtype == np.float64:
                img_u8 = (img * 255).astype(np.uint8)
            elif img.dtype == np.uint8:
                img_u8 = img
            else:
                raise TypeError("Image must be either float32/float64 or uint8")

            gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)

            if thresh is None:
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

            out_u8 = np.full_like(img_u8, 255)
            out_u8[mask == 255] = 0

            return out_u8
            # return out_u8.astype(np.float32)/255

        input_img = self.decode_tokens(input_ids)[0]
        pred_img = self.decode_tokens(pred_ids)[0]

        input_gray = binarise_black_vs_white(input_img, level+2)
        pred_gray = binarise_black_vs_white(pred_img, level+2)

        input_coordinates = self.get_pixel_location(input_gray, level+2, 0.1)
        pred_coordinates = self.get_pixel_location(pred_gray, level+2, 0.1)

        input_coordinates = self.drop_outer_ring(input_coordinates)
        pred_coordinates = self.drop_outer_ring(pred_coordinates)

        def compute_iou(img1, img2, thresh=128, is_show = False):
            _, bin1 = cv2.threshold(img1, thresh, 255, cv2.THRESH_BINARY)
            _, bin2 = cv2.threshold(img2, thresh, 255, cv2.THRESH_BINARY)

            bin1_bool = bin1 == 255
            bin2_bool = bin2 == 255
            if is_show:
                self.visualize_imgs(bin1_bool, bin2_bool)
            intersection = np.logical_and(bin1_bool, bin2_bool).sum()
            union = np.logical_or(bin1_bool, bin2_bool).sum()
            iou = 0.0
            if union > 0:
                iou = intersection / union
            return iou

        def compute_mse(img1, img2):
            return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        

        if carrying:
            masked_pos = table_pos
        else:
            masked_pos = [printer_pos] + table_pos

        mse_values = {}
        iou_values = {}
        for coord in input_coordinates.keys():
            mse_values[coord] = compute_mse(input_coordinates[coord], pred_coordinates[coord])

            if coord not in masked_pos:
                iou_values[coord] = compute_iou(input_coordinates[start_coord], pred_coordinates[coord])

        sorted_mse = sorted(
            [(coord, val) for coord, val in mse_values.items() if val > 500],
            key=lambda x: x[1],
            reverse=True
        )

        sorted_iou = sorted(
            [(coord, val) for coord, val in iou_values.items() if val >= 0.05],
            key=lambda x: x[1],
            reverse=True
        )

        most_changed_coords = [sorted_mse[i][0] for i in range(min(2, len(sorted_mse)))]
        least_changed_coords = [sorted_iou[i][0] for i in range(min(2, len(sorted_iou)))]
        # print(most_changed_coords)
        # print(least_changed_coords)
        # self.visualize_imgs(input_img, pred_img)
        # self.visualize_imgs(input_gray, pred_gray)
        
        # in principle, the extracted_coord should be the coord with highest iou,
        # when picking/dropping, the coord will be the same, so mse is almost 0
        if least_changed_coords:
            extracted_coord = least_changed_coords[0]
        # what if the player disapper?
        else:
            extracted_coord = (-5, -5)
        
        next_picking = False
        next_dropping = False


        if not carrying:
            printer_iou = compute_iou(input_coordinates[printer_pos], pred_coordinates[printer_pos])
            # print("Printer IOU Values:")
            # print(printer_iou)
            if printer_iou < 0.1:
                next_picking = True
        else:
            input_gray_simple = cv2.cvtColor((input_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            pred_gray_simple = cv2.cvtColor((pred_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            input_coordinates_simple = self.get_pixel_location(input_gray_simple, level+2, 0.1)
            pred_coordinates_simple  = self.get_pixel_location(pred_gray_simple, level+2, 0.1)

            input_coordinates_simple = self.drop_outer_ring(input_coordinates_simple)
            pred_coordinates_simple = self.drop_outer_ring(pred_coordinates_simple)

            table_mse_values = {}
            for coord in table_pos:
                table_mse_values[coord] = compute_mse(input_coordinates_simple[coord], pred_coordinates_simple[coord])
            sorted_table_mse = sorted(
                [(coord, val) for coord, val in table_mse_values.items() if val > 500],
                key=lambda x: x[1],
                reverse=True
            )
            most_changed_table_coords = [sorted_table_mse[i][0] for i in range(min(2, len(sorted_table_mse)))]

            if most_changed_table_coords:
                next_dropping = True

            # print("Table MSE Values:")
            # print(table_mse_values)
            # print(sorted_table_mse)
            # print(most_changed_table_coords)

        action = self.get_mini_action(start_coord, extracted_coord, next_picking, next_dropping)

        if action[1] == 'pick':
            printer_neighbors = meta['printer_neighbors']
            printer_neighbors = [tuple(nb) for nb in printer_neighbors]
            if start_coord not in printer_neighbors:
                action = (-1, 'invalid')

        if action[1] == 'drop':
            table_neighbors = meta['table_neighbors']
            table_neighbors = [tuple(nb) for nb in table_neighbors]
            if start_coord not in table_neighbors:
                action = (-1, 'invalid')

        if action[1] == 'pick':
            carrying = True
        elif action[1] == 'drop':
            carrying = False

        if report_to:
            next_coord = extracted_coord
            input_coord = start_coord

            reward = -2
            if action[1] == 'invalid':
                reward = -5
            elif action[1] == 'pick' or action[1] == 'drop':
                reward = 1
            else:
                doing_optimal = False

                printer_neighbors = meta['printer_neighbors']
                table_neighbors = meta['table_neighbors']

                printer_neighbors = [tuple(nb) for nb in printer_neighbors]
                table_neighbors = [tuple(nb) for nb in table_neighbors]
                if not start_info[1]:
                    distance_map_to_printer = meta['distance_map_to_printer']
                    distance_map_to_table = meta['distance_map_to_table']
                    best_total_distance = float('inf')
                    best_paths = []  # store all (printer_nb, table_nb) pairs with minimal total distance

                    for printer_nb in printer_neighbors:
                        # assert agent_locs[-1] in distance_map_to_printer[printer_nb], f"Start {agent_locs[-1]} not reachable from printer {printer_nb}"
                        if str(input_coord) not in distance_map_to_printer[str(printer_nb)]:
                            continue
                        dist_to_printer = distance_map_to_printer[str(printer_nb)][str(input_coord)]

                        for table_nb in table_neighbors:
                            assert str(printer_nb) in distance_map_to_table[str(table_nb)]
                            dist_to_table = distance_map_to_table[str(table_nb)][str(printer_nb)]
                            
                            total_dist = dist_to_printer + dist_to_table

                            if total_dist < best_total_distance:
                                best_total_distance = total_dist
                                best_paths = [(printer_nb, table_nb)]
                            elif total_dist == best_total_distance:
                                best_paths.append((printer_nb, table_nb))
                    
                    for best_path in best_paths:
                        target, table_nb = best_path
                        current_distance = distance_map_to_printer[str(target)][str(input_coord)]
                        next_distance = distance_map_to_printer[str(target)][str(next_coord)]
                        if next_distance == current_distance - 1:
                            doing_optimal = True
                            break
                else:
                    distance_map_to_table = meta['distance_map_to_table']
                    best_distance = float('inf')
                    best_targets = []
                    for table_nb in table_neighbors:
                        assert str(input_coord) in distance_map_to_table[str(table_nb)]
                        dist_to_table = distance_map_to_table[str(table_nb)][str(input_coord)]

                        if dist_to_table < best_distance:
                            best_distance = dist_to_table
                            best_targets = [table_nb]
                        elif dist_to_table == best_distance:
                            best_targets.append(table_nb)
                    for target in best_targets:
                        current_distance = distance_map_to_table[str(target)][str(input_coord)]
                        next_distance = distance_map_to_table[str(target)][str(next_coord)]
                        if next_distance == current_distance - 1:
                            doing_optimal = True
                            break

                if doing_optimal:
                    reward = 1
                else:
                    reward = 0

            wandb.log({
                "input_pred_images": wandb.Image(
                    np.hstack(((input_img * 255).astype(np.uint8), (pred_img * 255).astype(np.uint8))),
                    caption=f"{action}, {extracted_coord} Reward: {reward}"
                ),
            })

        return {
            "action": action,
            "pred_coord" : extracted_coord,
            "carrying": carrying,
            "image": np.hstack(((input_img * 255).astype(np.uint8),(pred_img * 255).astype(np.uint8)))
        }    
    
    def parse_mini_action_in_imgs(self, input_img, pred_img, start_info, meta, report_to = False):
        level = meta['level']
        start_coord = tuple(start_info[0])
        carrying = start_info[1]

        printer_pos = tuple(meta['printer_pos'])
        table_pos = [tuple(p) for p in meta['table_pos']]

        def binarise_black_vs_white(img, level, thresh=40):
            if img.dtype == np.float32 or img.dtype == np.float64:
                img_u8 = (img * 255).astype(np.uint8)
            elif img.dtype == np.uint8:
                img_u8 = img
            else:
                raise TypeError("Image must be either float32/float64 or uint8")

            gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)

            if thresh is None:
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

            out_u8 = np.full_like(img_u8, 255)
            out_u8[mask == 255] = 0

            return out_u8
            # return out_u8.astype(np.float32)/255

        input_gray = binarise_black_vs_white(input_img, level+2)
        pred_gray = binarise_black_vs_white(pred_img, level+2)

        input_coordinates = self.get_pixel_location(input_gray, level+2, 0.1)
        pred_coordinates = self.get_pixel_location(pred_gray, level+2, 0.1)

        input_coordinates = self.drop_outer_ring(input_coordinates)
        pred_coordinates = self.drop_outer_ring(pred_coordinates)

        def compute_iou(img1, img2, thresh=128, is_show = False):
            _, bin1 = cv2.threshold(img1, thresh, 255, cv2.THRESH_BINARY)
            _, bin2 = cv2.threshold(img2, thresh, 255, cv2.THRESH_BINARY)

            bin1_bool = bin1 == 255
            bin2_bool = bin2 == 255
            if is_show:
                self.visualize_imgs(bin1_bool, bin2_bool)
            intersection = np.logical_and(bin1_bool, bin2_bool).sum()
            union = np.logical_or(bin1_bool, bin2_bool).sum()
            iou = 0.0
            if union > 0:
                iou = intersection / union
            return iou

        def compute_mse(img1, img2):
            return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        

        if carrying:
            masked_pos = table_pos
        else:
            masked_pos = [printer_pos] + table_pos

        mse_values = {}
        iou_values = {}
        for coord in input_coordinates.keys():
            mse_values[coord] = compute_mse(input_coordinates[coord], pred_coordinates[coord])

            if coord not in masked_pos:
                iou_values[coord] = compute_iou(input_coordinates[start_coord], pred_coordinates[coord])

        sorted_mse = sorted(
            [(coord, val) for coord, val in mse_values.items() if val > 500],
            key=lambda x: x[1],
            reverse=True
        )

        sorted_iou = sorted(
            [(coord, val) for coord, val in iou_values.items() if val >= 0.05],
            key=lambda x: x[1],
            reverse=True
        )

        most_changed_coords = [sorted_mse[i][0] for i in range(min(2, len(sorted_mse)))]
        least_changed_coords = [sorted_iou[i][0] for i in range(min(2, len(sorted_iou)))]
        # print(most_changed_coords)
        # print(least_changed_coords)
        # self.visualize_imgs(input_img, pred_img)
        # self.visualize_imgs(input_gray, pred_gray)
        
        # in principle, the extracted_coord should be the coord with highest iou,
        # when picking/dropping, the coord will be the same, so mse is almost 0
        if least_changed_coords:
            extracted_coord = least_changed_coords[0]
        # what if the player disapper?
        else:
            extracted_coord = (-5, -5)
        
        next_picking = False
        next_dropping = False


        if not carrying:
            printer_iou = compute_iou(input_coordinates[printer_pos], pred_coordinates[printer_pos])
            # print("Printer IOU Values:")
            # print(printer_iou)
            if printer_iou < 0.1:
                next_picking = True
        else:
            input_gray_simple = cv2.cvtColor((input_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            pred_gray_simple = cv2.cvtColor((pred_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            input_coordinates_simple = self.get_pixel_location(input_gray_simple, level+2, 0.1)
            pred_coordinates_simple  = self.get_pixel_location(pred_gray_simple, level+2, 0.1)

            input_coordinates_simple = self.drop_outer_ring(input_coordinates_simple)
            pred_coordinates_simple = self.drop_outer_ring(pred_coordinates_simple)

            table_mse_values = {}
            for coord in table_pos:
                table_mse_values[coord] = compute_mse(input_coordinates_simple[coord], pred_coordinates_simple[coord])
            sorted_table_mse = sorted(
                [(coord, val) for coord, val in table_mse_values.items() if val > 500],
                key=lambda x: x[1],
                reverse=True
            )
            most_changed_table_coords = [sorted_table_mse[i][0] for i in range(min(2, len(sorted_table_mse)))]

            if most_changed_table_coords:
                next_dropping = True

            # print("Table MSE Values:")
            # print(table_mse_values)
            # print(sorted_table_mse)
            # print(most_changed_table_coords)

        action = self.get_mini_action(start_coord, extracted_coord, next_picking, next_dropping)

        if action[1] == 'pick':
            printer_neighbors = meta['printer_neighbors']
            printer_neighbors = [tuple(nb) for nb in printer_neighbors]
            if start_coord not in printer_neighbors:
                action = (-1, 'invalid')

        if action[1] == 'drop':
            table_neighbors = meta['table_neighbors']
            table_neighbors = [tuple(nb) for nb in table_neighbors]
            if start_coord not in table_neighbors:
                action = (-1, 'invalid')

        if action[1] == 'pick':
            carrying = True
        elif action[1] == 'drop':
            carrying = False

        if report_to:
            next_coord = extracted_coord
            input_coord = start_coord

            reward = -2
            if action[1] == 'invalid':
                reward = -5
            elif action[1] == 'pick' or action[1] == 'drop':
                reward = 1
            else:
                doing_optimal = False

                printer_neighbors = meta['printer_neighbors']
                table_neighbors = meta['table_neighbors']

                printer_neighbors = [tuple(nb) for nb in printer_neighbors]
                table_neighbors = [tuple(nb) for nb in table_neighbors]
                if not start_info[1]:
                    distance_map_to_printer = meta['distance_map_to_printer']
                    distance_map_to_table = meta['distance_map_to_table']
                    best_total_distance = float('inf')
                    best_paths = []  # store all (printer_nb, table_nb) pairs with minimal total distance

                    for printer_nb in printer_neighbors:
                        # assert agent_locs[-1] in distance_map_to_printer[printer_nb], f"Start {agent_locs[-1]} not reachable from printer {printer_nb}"
                        if str(input_coord) not in distance_map_to_printer[str(printer_nb)]:
                            continue
                        dist_to_printer = distance_map_to_printer[str(printer_nb)][str(input_coord)]

                        for table_nb in table_neighbors:
                            assert str(printer_nb) in distance_map_to_table[str(table_nb)]
                            dist_to_table = distance_map_to_table[str(table_nb)][str(printer_nb)]
                            
                            total_dist = dist_to_printer + dist_to_table

                            if total_dist < best_total_distance:
                                best_total_distance = total_dist
                                best_paths = [(printer_nb, table_nb)]
                            elif total_dist == best_total_distance:
                                best_paths.append((printer_nb, table_nb))
                    
                    for best_path in best_paths:
                        target, table_nb = best_path
                        current_distance = distance_map_to_printer[str(target)][str(input_coord)]
                        next_distance = distance_map_to_printer[str(target)][str(next_coord)]
                        if next_distance == current_distance - 1:
                            doing_optimal = True
                            break
                else:
                    distance_map_to_table = meta['distance_map_to_table']
                    best_distance = float('inf')
                    best_targets = []
                    for table_nb in table_neighbors:
                        assert str(input_coord) in distance_map_to_table[str(table_nb)]
                        dist_to_table = distance_map_to_table[str(table_nb)][str(input_coord)]

                        if dist_to_table < best_distance:
                            best_distance = dist_to_table
                            best_targets = [table_nb]
                        elif dist_to_table == best_distance:
                            best_targets.append(table_nb)
                    for target in best_targets:
                        current_distance = distance_map_to_table[str(target)][str(input_coord)]
                        next_distance = distance_map_to_table[str(target)][str(next_coord)]
                        if next_distance == current_distance - 1:
                            doing_optimal = True
                            break

                if doing_optimal:
                    reward = 1
                else:
                    reward = 0

            wandb.log({
                "input_pred_images": wandb.Image(
                    np.hstack(((input_img * 255).astype(np.uint8), (pred_img * 255).astype(np.uint8))),
                    caption=f"{action}, {extracted_coord} Reward: {reward}"
                ),
            })

        return {
            "action": action,
            "pred_coord" : extracted_coord,
            "carrying": carrying,
            "image": np.hstack(((input_img * 255).astype(np.uint8),(pred_img * 255).astype(np.uint8)))
        } 

if __name__ == "__main__":
    # seed everything
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    torch_device = 'cuda'
    tokenizer = get_tokenizer_muse().to(torch_device)
    # wandb.init(project="action_parser_project", name="test_run2")

    # Read the JSONL file
    # jsonl_file_path = 'dataset/maze/tokenized_dataset/SFT_random/train_dataset.jsonl'

    jsonl_file_path = 'dataset/minibehaviour/tokenized_dataset/SFT_random/train_dataset.jsonl'

    with open(jsonl_file_path, 'r') as file:
        data = [json.loads(line) for line in file]
        # data = random.sample(data, 50)

    # randomly sample 5 line from data
    # data = np.random.choice(data, 100, replace=False).tolist()
    # Extract input_ids and pred_ids
    input_ids_list = [item['input_tokens'] for item in data]
    pred_ids_list = [item['output_tokens'] for item in data]
    meta_list = [item['meta'] for item in data]
    input_state_list = [item['input_state'] for item in data]

    evaluator = ActionParser(tokenizer)
    idx = 0
    for input_ids, pred_ids, meta, input_state in zip(input_ids_list, pred_ids_list, meta_list, input_state_list):
        
        input_coord = input_state[0]
        print(f"Evaluating {idx}th data")
        input_ids_tensor = torch.tensor(input_ids).view(-1, 256).to(torch_device)
        pred_ids_tensor = torch.tensor(pred_ids).view(-1, 256).to(torch_device)
        
        action_dict = evaluator.parse_mini_action_in_ids(input_ids_tensor, pred_ids_tensor, input_state, meta, False)
        pred_coord = action_dict['pred_coord']
        print(f"Action: {action_dict['action']}")
        # ActionParser.visualize_ids(input_ids_tensor, pred_ids_tensor, tokenizer)
        next_coord = data[idx]['output_state']
        print(f"Pred coord: {(pred_coord, action_dict['carrying'])}, Next coord: {(tuple(next_coord[0]), next_coord[1])}")
        if (pred_coord, action_dict['carrying']) != (tuple(next_coord[0]), next_coord[1]):
            print(f"Wrong prediction of {idx}!!!!!!!!!!")
            ActionParser.visualize_ids(input_ids_tensor, pred_ids_tensor, tokenizer)
            break

    # level_list = [item['meta']['level'] for item in data]
    # coords_list = [item['input_state'] for item in data]
    # initial_coords_list = [item['meta']['start_pos'] for item in data]
    # target_coords_list = [item['meta']['target_pos'] for item in data]
    # layout_list = [item['meta']['layout'] for item in data]
    # distance_map_list = [item['meta']['distance_map'] for item in data]

    # Example usage of evaluate method
    # evaluator = ActionParser(tokenizer)
    # idx = 0
    # for input_ids, pred_ids, level, coords, initial_coords, target_coords, layout, distance_map in zip(input_ids_list, pred_ids_list, level_list, coords_list, initial_coords_list, target_coords_list, layout_list, distance_map_list):
    #     print(f"Evaluating {idx}th data")
    #     input_ids_tensor = torch.tensor(input_ids).view(-1, 256).to(torch_device)
    #     pred_ids_tensor = torch.tensor(pred_ids).view(-1, 256).to(torch_device)
    #     pred_coord = evaluator.parse_maze_action_in_ids(input_ids_tensor, pred_ids_tensor, level, coords, initial_coords, target_coords, layout, distance_map, False)['pred_coord']
    #     next_coord = ActionParser.get_coordinate_from_state(data[idx]['output_state'], level)
    #     print(f"Pred coord: {pred_coord}, Next coord: {next_coord}")
    #     if next_coord != pred_coord:
    #         print(f"Wrong prediction of {idx}!!!!!!!!!!")
    #         break

        # if idx < len(data) and data[idx]['meta'] == data[idx+1]['meta']:
        #     next_coord = ActionParser.get_coordinate_from_state(data[idx+1]['input_state'], level)
        #     if next_coord != pred_coord:
        #         print(f"Wrong prediction of {idx}!!!!!!!!!!")
        #         break
        # else:
        #     next_coord = ActionParser.get_coordinate_from_state(target_coords, level)
        #     if next_coord != pred_coord:
        #         print(f"Wrong prediction of {idx}!!!!!!!!!!")
        #         break
        # print("="*50)
        idx += 1
