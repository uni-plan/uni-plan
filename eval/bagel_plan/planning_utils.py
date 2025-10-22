import os
import copy
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from safetensors.torch import load_file
from copy import deepcopy
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from collections import Counter

import sys
sys.path.append('.')

# Import model components
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, 
    Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from data.transforms import ImageTransform
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae


def move_generation_input_to_device(generation_input, device):
    # Utility to move all tensors in generation_input to device
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


def setup_models(config_path, ckpt_path, device='cuda:0'):
    """
    Set up and load all required models.
    
    Args:
        llm_path: Path to the language model
        vit_path: Path to the vision model
        vae_path: Path to the VAE model
        device: GPU device ID
        
    Returns:
        tuple: (model, vae_model, tokenizer, new_token_ids, vae_transform)
    """
    
    # Load LLM
    llm_config = Qwen2Config.from_pretrained(os.path.join(config_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    
    # Load Vision model
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(config_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    
    # Load VAE
    print('Preparing vae model...')
    vae_model, vae_config = load_ae(local_path=os.path.join(config_path, "ae.safetensors"))
    vae_model = vae_model.to(device=device).eval()

    # Bagel config preparing
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    # Tokenizer Preparing
    tokenizer = Qwen2Tokenizer.from_pretrained(config_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Create fusion model
    # print('Preparing language model...')
    # language_model = Qwen2ForCausalLM(llm_config)
    # print('Preparing vit model...')
    # vit_model      = SiglipVisionModel(vit_config)
    # model          = Bagel(language_model, vit_model, config)
    # model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    # ema_state_dict_path = os.path.join(config_path, f"ema.safetensors") # may beed to change
    # ema_state_dict = load_file(ema_state_dict_path, device="cpu")
    # msg = model.load_state_dict(ema_state_dict, strict=False)
    # model = model.to(device=device, dtype=torch.bfloat16).eval()

    with init_empty_weights():
        print('Preparing language model...')
        language_model = Qwen2ForCausalLM(llm_config)
        print('Preparing vit model...')
        vit_model      = SiglipVisionModel(vit_config)
        model          = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(ckpt_path, "model.safetensors"),
        device_map={'': f"{device}"},
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload"
    )
    model = model.eval()

    # Set up transforms
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 378, 14)
    
    return model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform


def vis_transition(img_start, img_end, caption, save_path="output.jpg", font_path=None):

    w, h = img_start.size
    canvas_width = w * 2
    caption_height = 60

    canvas = Image.new("RGB", (canvas_width, h + caption_height), (255, 255, 255))
    canvas.paste(img_start, (0, 0))
    canvas.paste(img_end, (w, 0))

    draw = ImageDraw.Draw(canvas)

    if font_path:
        font = ImageFont.truetype(font_path, 24)
    else:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), caption, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (canvas_width - text_width) // 2
    text_y = h + (caption_height - text_height) // 2

    draw.text((text_x, text_y), caption, fill=(0, 0, 0), font=font)

    canvas.save(save_path)


@torch.no_grad()
def batch_pred_next_imgs(
    model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform,
    prompt, images, actions,
    num_timesteps=50,
    timestep_shift=4.0,
    # Image transform params
    max_image_size=512,
    min_image_size=512,
    original_image_size=(640, 360),
    stride=16,
    device='cuda:0'
):
    """
    Roll out on text instructions using NAVIT model.
    
    Args:
        model: The CausalFusion model
        vae_model: The VAE model
        tokenizer: Tokenizer for text processing
        new_token_ids: Special token IDs
        vae_transform: Transform for VAE input
        image: Input PIL image
        prompt: Text prompt for editing
        num_timesteps: Number of diffusion steps
        timestep_shift: Timestep shift for diffusion
        max_image_size: Maximum size for image dimension
        min_image_size: Minimum size for image dimension
        stride: Stride for resizing
        
    Returns:
        List of rollout images
    """

    assert len(images) == len(actions)
    
    def _make_divisible(value, stride):
        """Ensure the value is divisible by the stride."""
        return max(stride, int(round(value / stride) * stride))

    def _apply_scale(width, height, scale):
        new_width = round(width * scale)
        new_height = round(height * scale)
        new_width = _make_divisible(new_width, stride)
        new_height = _make_divisible(new_height, stride)
        return new_width, new_height
    
    # Prepare image size
    w, h = images[0].size
    scale = min(max_image_size / max(w, h), 1.0)
    scale = max(scale, min_image_size / min(w, h))
    w, h = _apply_scale(w, h, scale)
    
    if max(w, h) > max_image_size:
        scale = max_image_size / max(w, h)
        w, h = _apply_scale(w, h, scale)

    images = [image.resize((w, h)) for image in images]
    batch_size = len(images)
    
    # print(f"Image size: H-{h} W-{w}")

    past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    newlens = [0] * batch_size
    new_rope = [0] * batch_size
    generation_input, newlens, new_rope = model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt] * batch_size,
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = move_generation_input_to_device(generation_input, device)
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)

    # Prepare & forward VAE images
    generation_input, newlens, new_rope = model.prepare_vae_images(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        images=images,
        transforms=vae_transform, 
        new_token_ids=new_token_ids,
        timestep=0.0,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = move_generation_input_to_device(generation_input, device)
        past_key_values = model.forward_cache_update_vae(vae_model, past_key_values, **generation_input)
    
    # prepare & forward VIT images
    generation_input, newlens, new_rope = model.prepare_vit_images(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        images=[image.resize(original_image_size) for image in images],
        transforms=vit_transform, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = move_generation_input_to_device(generation_input, device)
        past_key_values = model.forward_cache_update_vit(past_key_values, **generation_input)
    
    # Prepare & forward action instruction
    generation_input, newlens, new_rope = model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=actions,
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = move_generation_input_to_device(generation_input, device)
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)
        
    # Prepare VAE latent for main branch
    generation_input = model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,  
        image_sizes=[(h, w)] * batch_size, 
        new_token_ids=new_token_ids,
    )        
    
    # Generate final image with mixed CFG
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = move_generation_input_to_device(generation_input, device)
        unpacked_latent = model.generate_image(
            past_key_values=past_key_values,
            num_timesteps=num_timesteps,
            timestep_shift=timestep_shift,
            **generation_input
        )

    # Process and decode the latent representation
    image_list = []
    for latent in unpacked_latent:
        latent = latent.reshape(1, h // 16, w // 16, 2, 2, 16).to(dtype=torch.float32)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, 16, h // 8, w // 8)
        image = vae_model.decode(latent)
        
        # Convert to image
        image = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        image_list.append(image)

    return image_list


@torch.no_grad()
def batch_pred_next_imgs_cfg(
    model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform,
    prompt, images, actions,
    num_timesteps=50,
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_type="parallel",
    cfg_interval=[0.4, 1.0],
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
    timestep_shift=4.0,
    # Image transform params
    max_image_size=512,
    min_image_size=512,
    original_image_size=(640, 360),
    stride=16,
    device='cuda:0'
):
    """
    Roll out on text instructions using NAVIT model.
    
    Args:
        model: The CausalFusion model
        vae_model: The VAE model
        tokenizer: Tokenizer for text processing
        new_token_ids: Special token IDs
        vae_transform: Transform for VAE input
        image: Input PIL image
        prompt: Text prompt for editing
        num_timesteps: Number of diffusion steps
        timestep_shift: Timestep shift for diffusion
        max_image_size: Maximum size for image dimension
        min_image_size: Minimum size for image dimension
        stride: Stride for resizing
        seed: Random seed
        
    Returns:
        List of rollout images
    """

    assert len(images) == len(actions)
    
    def _make_divisible(value, stride):
        """Ensure the value is divisible by the stride."""
        return max(stride, int(round(value / stride) * stride))

    def _apply_scale(width, height, scale):
        new_width = round(width * scale)
        new_height = round(height * scale)
        new_width = _make_divisible(new_width, stride)
        new_height = _make_divisible(new_height, stride)
        return new_width, new_height
    
    # Prepare image size
    w, h = images[0].size
    scale = min(max_image_size / max(w, h), 1.0)
    scale = max(scale, min_image_size / min(w, h))
    w, h = _apply_scale(w, h, scale)
    
    if max(w, h) > max_image_size:
        scale = max_image_size / max(w, h)
        w, h = _apply_scale(w, h, scale)

    images = [image.resize((w, h)) for image in images]
    batch_size = len(images)
    
    # print(f"Image size: H-{h} W-{w}")

    past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    newlens = [0] * batch_size
    new_rope = [0] * batch_size
    generation_input, newlens, new_rope = model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt] * batch_size,
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = move_generation_input_to_device(generation_input, device)
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)

    # Prepare & forward VAE images
    generation_input, newlens, new_rope = model.prepare_vae_images(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        images=images,
        transforms=vae_transform, 
        new_token_ids=new_token_ids,
        timestep=0.0,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = move_generation_input_to_device(generation_input, device)
        past_key_values = model.forward_cache_update_vae(vae_model, past_key_values, **generation_input)
    
    # prepare & forward VIT images
    generation_input, newlens, new_rope = model.prepare_vit_images(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        images=[image.resize(original_image_size) for image in images],
        transforms=vit_transform, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = move_generation_input_to_device(generation_input, device)
        past_key_values = model.forward_cache_update_vit(past_key_values, **generation_input)
    
    # Prepare & forward action instruction
    generation_input, newlens, new_rope = model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=actions,
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = move_generation_input_to_device(generation_input, device)
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)
        
    # Prepare VAE latent for main branch
    generation_input = model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,  
        image_sizes=[(h, w)] * batch_size, 
        new_token_ids=new_token_ids,
    )

    # Setup for text CFG
    cfg_text_past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    cfg_text_newlens = [0] * batch_size
    cfg_text_new_rope = [0] * batch_size

    generation_input_cfg_text, cfg_text_newlens, cfg_text_new_rope = model.prepare_prompts(
        curr_kvlens=cfg_text_newlens,
        curr_rope=cfg_text_new_rope, 
        prompts=[prompt] * batch_size,
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input_cfg_text = move_generation_input_to_device(generation_input_cfg_text, device)
        cfg_text_past_key_values = model.forward_cache_update_text(cfg_text_past_key_values, **generation_input_cfg_text)

    # Prepare & forward VAE images
    generation_input_cfg_text, cfg_text_newlens, cfg_text_new_rope = model.prepare_vae_images(
        curr_kvlens=cfg_text_newlens,
        curr_rope=cfg_text_new_rope, 
        images=images,
        transforms=vae_transform, 
        new_token_ids=new_token_ids,
        timestep=0.0,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input_cfg_text = move_generation_input_to_device(generation_input_cfg_text, device)
        cfg_text_past_key_values = model.forward_cache_update_vae(vae_model, cfg_text_past_key_values, **generation_input_cfg_text)
    
    # prepare & forward VIT images
    generation_input_cfg_text, cfg_text_newlens, cfg_text_new_rope = model.prepare_vit_images(
        curr_kvlens=cfg_text_newlens,
        curr_rope=cfg_text_new_rope, 
        images=[image.resize(original_image_size) for image in images],
        transforms=vit_transform, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input_cfg_text = move_generation_input_to_device(generation_input_cfg_text, device)
        cfg_text_past_key_values = model.forward_cache_update_vit(cfg_text_past_key_values, **generation_input_cfg_text)
        
    generation_input_cfg_text = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_text_newlens,
        curr_rope=cfg_text_new_rope, 
        image_sizes=[(h, w)] * batch_size, 
    )

    # Setup for image CFG
    cfg_img_past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    cfg_img_newlens = [0] * batch_size
    cfg_img_new_rope = [0] * batch_size
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        prompts=[prompt] * batch_size,
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device)
        cfg_img_past_key_values = model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        prompts=actions,
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input_cfg_img = move_generation_input_to_device(generation_input_cfg_img, device)
        cfg_img_past_key_values = model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)
    
    generation_input_cfg_img = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope, 
        image_sizes=[(h, w)] * batch_size, 
    )

    # Extract packed positions and indexes for CFGs
    cfg_text_args = {
        'cfg_text_packed_position_ids': generation_input_cfg_text['cfg_packed_position_ids'],
        'cfg_text_packed_query_indexes': generation_input_cfg_text['cfg_packed_query_indexes'],
        'cfg_text_key_values_lens': generation_input_cfg_text['cfg_key_values_lens'],
        'cfg_text_packed_key_value_indexes': generation_input_cfg_text['cfg_packed_key_value_indexes'],
    }
    
    cfg_img_args = {
        'cfg_img_packed_position_ids': generation_input_cfg_img['cfg_packed_position_ids'],
        'cfg_img_packed_query_indexes': generation_input_cfg_img['cfg_packed_query_indexes'],
        'cfg_img_key_values_lens': generation_input_cfg_img['cfg_key_values_lens'],
        'cfg_img_packed_key_value_indexes': generation_input_cfg_img['cfg_packed_key_value_indexes'],
    }     
    
    # Generate final image with mixed CFG
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = move_generation_input_to_device(generation_input, device)
        cfg_text_args = move_generation_input_to_device(cfg_text_args, device)
        cfg_img_args = move_generation_input_to_device(cfg_img_args, device)
        unpacked_latent = model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_type=cfg_type,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            **cfg_text_args,
            **cfg_img_args,
        )

    # Process and decode the latent representation
    image_list = []
    for latent in unpacked_latent:
        latent = latent.reshape(1, h // 16, w // 16, 2, 2, 16).to(dtype=torch.float32)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, 16, h // 8, w // 8)
        image = vae_model.decode(latent)
        
        # Convert to image
        image = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        image_list.append(image)

    return image_list


@torch.no_grad()
def vlm_pred(
    model, tokenizer, new_token_ids, vit_transform,
    prompt, images,
    original_image_size=(640, 360),
    num_samples=16,
    do_sample=True,
    temperature=0.3,
    max_length=512,
    device='cuda:0'
):
    # Initialize cache and setup
    past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    newlens = [0]
    new_rope = [0]

    # Prepare & forward prompt for main branch
    generation_input, newlens, new_rope = model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = move_generation_input_to_device(generation_input, device)
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)
        
    # prepare & forward VIT images
    for image in images:
        generation_input, newlens, new_rope = model.prepare_vit_images(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=[image.resize(original_image_size)],
            transforms=vit_transform, 
            new_token_ids=new_token_ids,
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generation_input = move_generation_input_to_device(generation_input, device)
            past_key_values = model.forward_cache_update_vit(past_key_values, **generation_input)

    outputs = []

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = model.prepare_start_tokens(newlens, new_rope, new_token_ids)
        
        generation_input = move_generation_input_to_device(generation_input, device)

        for i in range(num_samples):
            unpacked_latent = model.generate_text(
                past_key_values=copy.deepcopy(past_key_values),
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
                )
            output = tokenizer.decode(unpacked_latent[:,0])
            think_output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
            outputs.append(think_output)
        
    return outputs


def set_seeds(seed):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False