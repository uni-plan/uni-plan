import os
import re
import json
import argparse
import copy
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from einops import rearrange

import sys
sys.path.append('.')

from eval.bagel_plan.planning_utils import *


def get_action_candidates(current_state, goal_state):
    positions = ['top_left', 'top_center', 'top_right', 'center_left', 'center_right', 'bottom_left', 'bottom_center', 'bottom_right']
    blocks = set()
    for pos in positions:
        if goal_state[pos]["occupied"] is not None:
            blocks.add(goal_state[pos]["occupied"])
        if goal_state[pos]["surrounded"] is not None:
            blocks.add(goal_state[pos]["surrounded"])
    occupying_blocks = set()
    for pos in positions:
        if goal_state[pos]["occupied"] is not None:
            occupying_blocks.add(goal_state[pos]["occupied"])
    
    taget_pos = None 
    for pos in positions:
        if goal_state[pos]["occupied"] is not None and current_state[pos]["occupied"] is None:
            taget_pos = pos
            break
    if taget_pos is not None:
        candidates = [f"move {goal_state[taget_pos]['occupied']} to {taget_pos}"]
        for block in blocks:
            if block != goal_state[taget_pos]["occupied"]:
                candidates.append(f"move {block} to {taget_pos}")
        if len(candidates) > 0:
            return candidates, "block2pos"
    
    target_block = None
    for pos in positions:
        if goal_state[pos]["occupied"] is not None and current_state[pos]["occupied"] is not None and goal_state[pos]["occupied"] != current_state[pos]["occupied"]:
            target_block = current_state[pos]["occupied"]
            break
    if target_block is not None:
        candidates = []
        for pos in positions:
            if current_state[pos]["occupied"] is None:
                candidates.append(f"move {target_block} to {pos}")
        if len(candidates) > 0:
            return candidates, "block2pos"
        for pos in positions:
            if current_state[pos]["surrounded"] is None and current_state[pos]["occupied"] != target_block:
                candidates.append(f"move {target_block} to {current_state[pos]['occupied']}")
        if len(candidates) > 0:
            return candidates, "block2block"
    
    target_block = None
    for pos in positions:
        if goal_state[pos]["surrounded"] is not None and current_state[pos]["surrounded"] is None:
            target_block = goal_state[pos]["occupied"]
            break
    if target_block is not None:
        candidates = [f"move {goal_state[pos]['surrounded']} to {target_block}"]
        for block in blocks:
            if block not in occupying_blocks and block != goal_state[pos]["surrounded"]:
                candidates.append(f"move {block} to {target_block}")
        if len(candidates) > 0:
            return candidates, "block2block"
    
    target_block = None
    for pos in positions:
        if goal_state[pos]["surrounded"] is not None and current_state[pos]["surrounded"] is not None and goal_state[pos]["surrounded"] != current_state[pos]["surrounded"]:
            target_block = current_state[pos]["surrounded"]
            break
    if target_block is not None:
        candidates = []
        for pos in positions:
            if current_state[pos]["occupied"] is None:
                candidates.append(f"move {target_block} to {pos}")
        if len(candidates) > 0:
            return candidates, "block2pos"
        for pos in positions:
            if current_state[pos]["surrounded"] is None:
                candidates.append(f"move {target_block} to {current_state[pos]['occupied']}")
        if len(candidates) > 0:
            return candidates, "block2block"
    
    return [], "no_action"


def beam_search(
    model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform,
    image, goal_image,
    dynamics_prompt,
    policy_prompt,
    reward_prompt,
    inverse_dynamics_prompt=None,
    count_blocks_prompt=None,
    num_beams=2,
    plan_branches=2,
    plan_horizon=2,
    num_image_samples=4,
    success_threshold=0,
    num_init_objs=8,
    cfg=False,
    device='cuda:0',
    verbose_image_path=None
):
    if cfg:
        dynamics_func = batch_pred_next_imgs_cfg
    else:
        dynamics_func = batch_pred_next_imgs

    # Initialize: current image, action sequence, and image sequence for each beam
    beam_states = [(image, [], [image])] * num_beams  # (image, action_sequence, image_sequence)
    is_success = False
    
    for step in range(plan_horizon):
        # Expand all current beams
        candidates = []
        batch_images = []
        batch_actions = []
        batch_beam_idxs = []
        num_actions_per_beam = []
        for beam_idx, (current_img, _, _) in enumerate(beam_states):
            # Generate multiple candidate actions for current beam
            outputs = vlm_pred(
                model, tokenizer, new_token_ids, vit_transform,
                reward_prompt, [current_img, goal_image],
                original_image_size=(640, 360),
                do_sample=False,
                num_samples=1,
                device=device
            )
            parsed = json.loads(outputs[0])
            current_state, goal_state = parsed.get('current_state'), parsed.get('goal_state')
            actions, action_type = get_action_candidates(current_state, goal_state)
            actions = actions[:plan_branches]
            num_actions_per_beam.append(len(actions))

            print(actions)

            if len(actions) == 0:
                raise ValueError(f"Cannot find a valid action for step {step+1}")
            
            for action in actions:
                batch_actions.append(action)
                batch_images.append(current_img)
                batch_beam_idxs.append(beam_idx)
            
        if batch_actions:
            # Batch predict next images for all valid actions
            batch_images_repeat = [x for x in batch_images for _ in range(num_image_samples)]
            batch_actions_repeat = [x for x in batch_actions for _ in range(num_image_samples)]
            batch_next_images_repeat = dynamics_func(
                model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform, dynamics_prompt,
                batch_images_repeat, batch_actions_repeat,
                max_image_size=vae_transform.resize_transform.max_size, 
                min_image_size=vae_transform.resize_transform.min_size,
                stride=vae_transform.stride,
                device=device
            )

            if verbose_image_path is not None:
                for b_idx in range(num_beams):
                    for a_idx in range(num_actions_per_beam[b_idx]):
                        for j in range(num_image_samples):
                            cum_sum = int(np.sum(num_actions_per_beam[:b_idx]))
                            obs = batch_images[cum_sum+a_idx]
                            action = batch_actions[cum_sum+a_idx]
                            next_obs = Image.fromarray(batch_next_images_repeat[(cum_sum+a_idx)*num_image_samples+j])
                            vis_transition(obs, next_obs, action, f'{verbose_image_path}/step_{step}_beam_{b_idx}_action_{a_idx}_sample_{j}.jpg')

            # filter out invalid next images
            batch_next_images = []
            for i in range(len(batch_next_images_repeat)//num_image_samples):
                images_ = batch_images_repeat[i*num_image_samples:(i+1)*num_image_samples]
                next_images_ = batch_next_images_repeat[i*num_image_samples:(i+1)*num_image_samples]
                actions_ = batch_actions_repeat[i*num_image_samples:(i+1)*num_image_samples]
                has_valid_next_image = False
                for j in range(len(images_)):
                    image_ = images_[j]
                    next_image_ = Image.fromarray(next_images_[j])
                    action_ = actions_[j]
                    if inverse_dynamics_prompt is not None and count_blocks_prompt is not None:
                        num_objs = vlm_pred(
                            model, tokenizer, new_token_ids, vit_transform,
                            count_blocks_prompt, [next_image_],
                            original_image_size=(640, 360),
                            num_samples=1,
                            do_sample=False
                        )
                        parsed = json.loads(num_objs[0])
                        num_objs = int(parsed.get('num_blocks'))
                        if num_objs != num_init_objs:
                            continue
                        pred_action_ = vlm_pred(
                            model, tokenizer, new_token_ids, vit_transform,
                            inverse_dynamics_prompt, [image_, next_image_],
                            original_image_size=(640, 360),
                            num_samples=1,
                            do_sample=False
                        )
                        objs_in_real_action = re.findall(r'\b\w+_\w+\b', action_)
                        objs_in_pred_action = re.findall(r'\b\w+_\w+\b', pred_action_[0])
                        print(f"objs_in_real_action: {objs_in_real_action}, objs_in_pred_action: {objs_in_pred_action}, num_objs: {num_objs}")
                        position_elements = ['top_left', 'top_center', 'top_right', 'center_left', 'center_right', 'bottom_left', 'bottom_center', 'bottom_right']
                        if any(pos in objs_in_real_action for pos in position_elements):
                            is_action_match = True
                        else:
                            is_action_match = (objs_in_real_action == objs_in_pred_action)
                        if is_action_match:
                            batch_next_images.append(next_images_[j])
                            has_valid_next_image = True
                            break
                    else:
                        batch_next_images.append(next_images_[j])
                        has_valid_next_image = True
                        break
                if not has_valid_next_image:
                    print(f"Cannot find a valid next image for action: {actions_[0]}")
                    raise ValueError(f"Cannot find a valid next image for action: {actions_[0]}")
            
            # Convert numpy arrays to PIL images and create candidates
            for i, (next_img_array, action, b_idx) in enumerate(zip(batch_next_images, batch_actions, batch_beam_idxs)):
                next_img = Image.fromarray(next_img_array)
                new_action_seq = beam_states[b_idx][1] + [action]
                new_image_seq = beam_states[b_idx][2] + [next_img]
                candidates.append((next_img, new_action_seq, new_image_seq))
        
        # Evaluate all candidates
        scores = []
        for next_img, action_seq, _ in candidates:
            outputs = vlm_pred(
                model, tokenizer, new_token_ids, vit_transform,
                reward_prompt, [next_img, goal_image],
                original_image_size=(640, 360),
                num_samples=1,
                do_sample=False,
                device=device
            )
            try:
                parsed = json.loads(outputs[0])
                score = -int(parsed.get('num_steps_left'))
                scores.append(score)
                print(f"plan step: {step}, action seq: {action_seq}, num steps left: {int(parsed.get('num_steps_left'))}")
            except:
                print(f"Invalid output: {outputs[0]}")
                scores.append(float('-inf'))
        
        # Check if there are valid candidates
        if not candidates:
            print(f"Warning: No valid candidates generated at step {step+1}, ending planning early")
            break
        
        # Select top num_beams candidates with highest scores
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted_indices[:num_beams]
        
        # Check if the highest score is 0 (reached goal), terminate planning early
        if -scores[top_indices[0]] <= success_threshold:
            print(f"Goal reached at step {step+1}, terminating planning early")
            beam_states = [candidates[i] for i in top_indices]
            is_success = True
            break
        
        # Update beam_states
        beam_states = [candidates[i] for i in top_indices]
    
    # Return optimal action sequence (highest scoring)
    if not beam_states:
        print("Warning: No valid beam states, returning empty action sequence")
        return []
    
    # Re-evaluate final states to select the best one
    final_scores = []
    for next_img, _, _ in beam_states:
        outputs = vlm_pred(
            model, tokenizer, new_token_ids, vit_transform,
            reward_prompt, [next_img, goal_image],
            original_image_size=(640, 360),
            num_samples=1,
            do_sample=False,
            device=device
        )
        try:
            parsed = json.loads(outputs[0])
            score = -int(parsed.get('num_steps_left'))
            final_scores.append(score)
        except:
            print(f"Invalid output: {outputs[0]}")
            final_scores.append(float('-inf'))
    
    # Select action sequence with highest score
    best_idx = max(range(len(final_scores)), key=lambda i: final_scores[i])
    best_plan = beam_states[best_idx]
    
    return best_plan, is_success


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Image editing with text instructions")
    parser.add_argument("--config_path", type=str, default="/scratch/s/sunyh/Bagel-HF", 
                        help="Path to config")
    parser.add_argument("--ckpt_path", type=str, default="/scratch/s/sunyh/bagel/bagel_plan_language_table_500_trajs-run0/ckpt/0003000", 
                        help="Path to bagel model")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--image_dir", type=str, default="eval/bagel_plan/language_table/planning_eval_samples", 
                        help="Directory containing the images referenced in jsonl")
    parser.add_argument("--output_dir", type=str, default="eval/bagel_plan/language_table/planning_outputs", 
                        help="Directory to save output images")
    parser.add_argument("--dynamics_prompt_path", type=str, default="eval/bagel_plan/language_table/prompts/dynamics_prompt.txt", 
                        help="")
    parser.add_argument("--policy_prompt_path", type=str, default="eval/bagel_plan/language_table/prompts/policy_prompt.txt", 
                        help="")
    parser.add_argument("--reward_prompt_path", type=str, default="eval/bagel_plan/language_table/prompts/reward_prompt.txt", 
                        help="")
    parser.add_argument("--inverse_dynamics_prompt_path", type=str, default="eval/bagel_plan/language_table/prompts/inverse_dynamics_prompt.txt", 
                        help="")
    parser.add_argument("--count_blocks_prompt_path", type=str, default="eval/bagel_plan/language_table/prompts/count_blocks_prompt.txt", 
                        help="")
    parser.add_argument("--cfg", type=bool, default=True, help="")
    parser.add_argument("--sdf", type=int, default=1, help="self-discriminated filtering")
    
    args = parser.parse_args()
    
    # Setup models
    model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform = setup_models(
        args.config_path, args.ckpt_path, device=args.device
    )

    # Set random seeds for reproducibility
    set_seeds(42)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.dynamics_prompt_path, 'r') as f:
        dynamics_prompt = f.read().strip()
    with open(args.policy_prompt_path, 'r') as f:
        policy_prompt = f.read().strip()
    with open(args.reward_prompt_path, 'r') as f:
        reward_prompt = f.read().strip()
    with open(args.inverse_dynamics_prompt_path, 'r') as f:
        if args.sdf == 1:
            inverse_dynamics_prompt = f.read().strip()
        else:
            inverse_dynamics_prompt = None
    with open(args.count_blocks_prompt_path, 'r') as f:
        if args.sdf == 1:
            count_objects_prompt = f.read().strip()
        else:
            count_objects_prompt = None

    num_test_samples = len(os.listdir(args.image_dir))//2
    num_init_objs_list = [9, 9, 9, 9, 10, 10, 9, 10, 9, 9, 12, 10, 12, 9, 9, 11, 11, 10, 10, 11]
    num_image_samples = 4 if args.sdf == 1 else 1
    for i in range(10, 20):
        img_file = os.path.join(args.image_dir, f"{i}_start.jpg")
        goal_img_file = os.path.join(args.image_dir, f"{i}_goal.jpg")
        image = Image.open(img_file).convert('RGB').resize(
            (912, 512)
        )
        goal_image = Image.open(goal_img_file).convert('RGB').resize(
            (912, 512)
        )
        print('Performing Beam Search...')
        try:
            verbose_image_path = os.path.join(args.output_dir, "verbose", f"sample_{i}")
            os.makedirs(verbose_image_path, exist_ok=True)
            plan, is_success = beam_search(
                model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform,
                image, goal_image,
                dynamics_prompt,
                policy_prompt,
                reward_prompt,
                inverse_dynamics_prompt,
                count_objects_prompt,
                num_beams=2,
                plan_branches=4,
                plan_horizon=16,
                num_image_samples=num_image_samples,
                num_init_objs=num_init_objs_list[i],
                success_threshold=0.,
                cfg=args.cfg,
                device=args.device,
                verbose_image_path=verbose_image_path
            )
        except:
            print(f"Failed to find a valid plan for sample {i}")
            continue
        if is_success:
            print(f"Find a valid plan for sample {i}, action seq: {plan[1]}")
        else:
            print(f"Failed to find a valid plan for sample {i}")
        plan_images = plan[2]
        plan_images = rearrange(np.stack(plan_images, 0), "n h w c->h (n w) c")
        output_filename = f"sample_{i}_plan.jpg"
        output_path = os.path.join(args.output_dir, output_filename)
        Image.fromarray(plan_images).save(output_path)


if __name__ == "__main__":
    main()