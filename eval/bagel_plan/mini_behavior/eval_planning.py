import os
import json
import argparse
import torch
from PIL import Image
import numpy as np
from einops import rearrange

import sys
sys.path.append('.')

from eval.bagel_plan.planning_utils import *
from eval.bagel_plan.mini_behavior.utils import can_interact, can_move_forward


ACTION_MAP = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pickup",
    4: "drop"
}


def beam_search(
    model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform,
    image,
    dynamics_prompt,
    policy_prompt,
    reward_prompt,
    num_beams=2,
    plan_branches=4,
    plan_horizon=2,
    device='cuda:0',
    verbose_image_path=None
):
    # Initialize: current image and action sequence for each beam
    beam_states = [(image, [], [image])] * num_beams  # (image, action_sequence, image_sequence)
    is_success = False
    
    for step in range(plan_horizon):
        # Expand all current beams
        candidates = []
        batch_images = []
        batch_actions = []
        batch_beam_idxs = []
        batch_captions = []
        num_actions_per_beam = []
        for beam_idx, (current_img, _, _) in enumerate(beam_states):

            outputs = vlm_pred(
                model, tokenizer, new_token_ids, vit_transform,
                reward_prompt, [current_img],
                original_image_size=(256, 256),
                num_samples=1,
                do_sample=False,
                device=device
            )
            try:
                parsed = json.loads(outputs[0])
                agent_position = tuple(parsed.get('agent_position'))
                object_position = tuple(parsed.get('object_position'))
                table_position = tuple(parsed.get('table_position'))
                is_carrying = bool(parsed.get('is_carrying'))
                num_steps_left = int(parsed.get('total_num_steps'))
            except:
                print(f"Invalid output: {outputs[0]}")
                continue

            # Parse actions and prepare for batch prediction
            action_cnt = 0
            for action_id in range(plan_branches):
                action = ACTION_MAP[action_id]
                if action == 'move forward':
                    if not can_move_forward(4, 4, agent_position, object_position, table_position):
                        continue
                if action == 'pickup':
                    if is_carrying or not can_interact(agent_position, object_position):
                        continue
                if action == 'drop':
                    if not is_carrying or not can_interact(agent_position, table_position):
                        continue
                action_cnt += 1
                batch_actions.append(action)
                batch_images.append(current_img)
                batch_beam_idxs.append(beam_idx)
                batch_captions.append(f"action: {action}, agent_position: {agent_position}, object_position: {object_position}, table_position: {table_position}, is_carrying: {is_carrying}, num_steps_left: {num_steps_left}")
            
            num_actions_per_beam.append(action_cnt)
            
        if batch_actions:
            # Batch predict next images for all valid actions
            batch_next_images = batch_pred_next_imgs(
                model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform, dynamics_prompt,
                batch_images, batch_actions,
                max_image_size=vae_transform.resize_transform.max_size, 
                min_image_size=vae_transform.resize_transform.min_size,
                original_image_size=(256, 256),
                stride=vae_transform.stride,
                device=device
            )
            
            # Convert numpy arrays to PIL images and create candidates
            for img, next_img_array, action, b_idx in zip(batch_images, batch_next_images, batch_actions, batch_beam_idxs):
                next_img = Image.fromarray(next_img_array)
                new_action_seq = beam_states[b_idx][1] + [action]
                new_image_seq = beam_states[b_idx][2] + [next_img]
                candidates.append((next_img, new_action_seq, new_image_seq))
            
            if verbose_image_path is not None:
                for b_idx in range(num_beams):
                    for a_idx in range(num_actions_per_beam[b_idx]):
                        cum_sum = int(np.sum(num_actions_per_beam[:b_idx]))
                        obs = batch_images[cum_sum+a_idx]
                        action = batch_actions[cum_sum+a_idx]
                        next_obs = Image.fromarray(batch_next_images[cum_sum+a_idx])
                        vis_transition(obs, next_obs, action, f'{verbose_image_path}/step_{step}_beam_{b_idx}_action_{a_idx}.jpg')
        
        # Evaluate all candidates
        scores = []
        for next_img, action_seq, _ in candidates:
            outputs = vlm_pred(
                model, tokenizer, new_token_ids, vit_transform,
                reward_prompt, [next_img],
                original_image_size=(256, 256),
                num_samples=1,
                do_sample=False,
                device=device
            )
            try:
                parsed = json.loads(outputs[0])
                agent_position = tuple(parsed.get('agent_position'))
                object_position = tuple(parsed.get('object_position'))
                table_position = tuple(parsed.get('table_position'))
                is_carrying = bool(parsed.get('is_carrying'))
                num_steps_left = int(parsed.get('total_num_steps'))
                scores.append(-num_steps_left)
                print(f"plan step: {step}, action seq: {action_seq}, num steps left: {num_steps_left}")
            except:
                print(f"Invalid output- action seq: {action_seq}, output: {outputs[0]}")
                scores.append(float('-inf'))
        
        # Check if there are valid candidates
        if not candidates:
            print(f"Warning: No valid candidates generated at step {step+1}, ending planning early")
            break
        
        # Select top num_beams candidates with highest scores
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted_indices[:num_beams]
        
        # Check if the highest score is 0 (reached goal), terminate planning early
        if scores[top_indices[0]] == 0:
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
            reward_prompt, [next_img],
            original_image_size=(256, 256),
            num_samples=1,
            do_sample=False,
            device=device
        )
        try:
            parsed = json.loads(outputs[0])
            agent_position = tuple(parsed.get('agent_position'))
            object_position = tuple(parsed.get('object_position'))
            table_position = tuple(parsed.get('table_position'))
            is_carrying = bool(parsed.get('is_carrying'))
            num_steps_left = int(parsed.get('total_num_steps'))
            final_scores.append(-num_steps_left)
        except:
            final_scores.append(float('-inf'))
    
    # Select action sequence with highest score
    best_idx = max(range(len(final_scores)), key=lambda i: final_scores[i])
    best_plan = beam_states[best_idx]
    
    return best_plan, is_success


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Planning for Mini-Behavior")
    parser.add_argument("--config_path", type=str, default="/scratch/s/sunyh/Bagel-HF", 
                        help="Path to config")
    parser.add_argument("--ckpt_path", type=str, default="/scratch/s/sunyh/bagel/bagel_plan_mini_behavior_500_trajs-run0/ckpt/0003000", 
                        help="Path to bagel model")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--image_dir", type=str, default="eval/bagel_plan/mini_behavior/planning_eval_samples", 
                        help="Directory containing the images referenced in jsonl")
    parser.add_argument("--output_dir", type=str, default="eval/bagel_plan/mini_behavior/planning_outputs", 
                        help="Directory to save output images")
    parser.add_argument("--dynamics_prompt_path", type=str, default="eval/bagel_plan/mini_behavior/prompts/dynamics_prompt.txt", 
                        help="")
    parser.add_argument("--reward_prompt_path", type=str, default="eval/bagel_plan/mini_behavior/prompts/reward_prompt.txt", 
                        help="")
    
    args = parser.parse_args()
    
    # Setup models
    model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform = setup_models(
        args.config_path, args.ckpt_path, device=args.device
    )
    set_seeds(42)

    with open(args.dynamics_prompt_path, 'r') as f:
        dynamics_prompt = f.read().strip()
    with open(args.reward_prompt_path, 'r') as f:
        reward_prompt = f.read().strip()

    os.makedirs(args.output_dir, exist_ok=True)

    for idx in range(10):
        image = Image.open(os.path.join(args.image_dir, f"{idx}.jpg")).convert('RGB').resize(
            (512, 512)
        )
        print('Performing Beam Search for sample %d...' % idx)
        verbose_image_path = os.path.join(args.output_dir, "verbose", f"sample_{idx}")
        os.makedirs(verbose_image_path, exist_ok=True)
        plan, is_success = beam_search(
            model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform,
            image,
            dynamics_prompt,
            None,
            reward_prompt,
            num_beams=2,
            plan_branches=5,
            plan_horizon=14,
            device=args.device,
            verbose_image_path=verbose_image_path
        )
        if is_success:
            print(f"Find a valid plan for sample {idx}, action seq: {plan[1]}")
        else:
            print(f"Failed to find a valid plan for sample {idx}")
        plan_images = plan[2]
        plan_images = rearrange(np.stack(plan_images, 0), "n h w c->h (n w) c")
        output_filename = f"sample_{idx}_plan.jpg"
        output_path = os.path.join(args.output_dir, output_filename)
        Image.fromarray(plan_images).save(output_path)


if __name__ == "__main__":
    main()