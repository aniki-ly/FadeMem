# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from utils.dataset import TextDataset
from utils.misc import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)


def _require_path(label: str, path: str) -> None:
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _validate_config() -> None:
    if config.model_name != "Wan2.1-T2V-1.3B":
        raise ValueError("The released inference path supports only Wan2.1-T2V-1.3B.")
    if int(config.num_frame_per_block) != 3:
        raise ValueError("FadeMem inference requires num_frame_per_block: 3.")
    if int(config.num_output_frames) <= 0 or int(config.num_output_frames) % 3 != 0:
        raise ValueError("num_output_frames must be a positive multiple of 3.")
    if bool(config.model_kwargs.fademem_enabled):
        if int(config.model_kwargs.local_attn_size) != 3:
            raise ValueError("The released FadeMem checkpoint requires local_attn_size: 3.")
        if int(config.model_kwargs.sink_size) != 0:
            raise ValueError("The released FadeMem checkpoint requires sink_size: 0.")

    _require_path("Prompt file", config.data_path)
    _require_path("Wan2.1-T2V-1.3B model directory", "wan_models/Wan2.1-T2V-1.3B")
    _require_path("Generator checkpoint", config.generator_ckpt)
    if getattr(config, "lora_ckpt", None):
        _require_path("LoRA checkpoint", config.lora_ckpt)


def _resolve_low_memory(free_vram_gb: float) -> bool:
    value = os.environ.get("FADEMEM_LOW_MEMORY", getattr(config, "low_memory", "auto"))
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return free_vram_gb < 40
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("low_memory/FADEMEM_LOW_MEMORY must be auto, true, or false.")


if not torch.cuda.is_available():
    raise RuntimeError("FadeMem inference requires a CUDA-capable GPU.")

_validate_config()

# These modules initialize CUDA state during import, so import them only after
# the environment and release configuration have been validated.
from pipeline import CausalInferencePipeline
from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller

print("\n========== FINAL CONFIG (after load) ==========")
print(OmegaConf.to_yaml(config))
print("==============================================\n")

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    os.environ["NCCL_CROSS_NIC"] = "1"
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout,
        )
    set_seed(config.seed + local_rank)
    config.distributed = True  # Mark as distributed for pipeline
    if rank == 0:
        print(f"[Rank {rank}] Initialized distributed processing on device {device}")
else:
    local_rank = 0
    rank = 0
    world_size = 1
    device = torch.device("cuda")
    set_seed(config.seed)
    config.distributed = False  # Mark as non-distributed
    print(f"Single GPU mode on device {device}")

free_vram_gb = get_cuda_free_memory_gb(device)
low_memory = _resolve_low_memory(free_vram_gb)
print(f"Free VRAM: {free_vram_gb:.2f} GB; low-memory mode: {low_memory}")

extended_prompt_path = getattr(config, "extended_data_path", None)
if extended_prompt_path:
    _require_path("Extended prompt file", extended_prompt_path)
dataset = TextDataset(prompt_path=config.data_path, extended_prompt_path=extended_prompt_path)
num_prompts = len(dataset)
if num_prompts == 0:
    raise ValueError(f"Prompt file is empty: {config.data_path}")
if num_prompts < world_size or num_prompts % world_size != 0:
    raise ValueError(
        f"Prompt count ({num_prompts}) must be at least the GPU count ({world_size}) "
        "and divisible by it."
    )
print(f"Number of prompts: {num_prompts}")

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True


# Initialize pipeline
# Note: checkpoint loading is now handled inside the pipeline __init__ method
pipeline = CausalInferencePipeline(config, device=device)

# Load generator checkpoint
if config.generator_ckpt:
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    if "generator" in state_dict or "generator_ema" in state_dict:
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
    elif "model" in state_dict:
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
    if config.use_ema:
        def _clean_key(name: str) -> str:
            """Remove FSDP / checkpoint wrapper prefixes from parameter names."""
            name = name.replace("_fsdp_wrapped_module.", "")
            return name

        cleaned_state_dict = { _clean_key(k): v for k, v in raw_gen_state_dict.items() }
        missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
        if local_rank == 0:
            if len(missing) > 0:
                print(f"[Warning] {len(missing)} parameters are missing when loading checkpoint: {missing[:8]} ...")
            if len(unexpected) > 0:
                print(f"[Warning] {len(unexpected)} unexpected parameters encountered when loading checkpoint: {unexpected[:8]} ...")
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)

# --------------------------- LoRA support (optional) ---------------------------
from utils.lora_utils import configure_lora_for_model
import peft

pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
        print("Applying LoRA to generator (inference)...")
    # Apply the LoRA wrapper after loading the base generator weights.
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    # Load LoRA weights when a checkpoint is provided.
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        # Accept either a wrapped checkpoint or a direct LoRA state dict.
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])  # type: ignore
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)  # type: ignore
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    else:
        if local_rank == 0:
            print("No LoRA checkpoint specified; using base weights with LoRA adapters initialized")

    pipeline.is_lora_enabled = True

# Move pipeline to appropriate dtype and device
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    # For text-to-video, batch is just the text prompt
    prompt = batch['prompts'][0]
    extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
    if extended_prompt is not None:
        prompts = [extended_prompt] * config.num_samples
    else:
        prompts = [prompt] * config.num_samples

    sampled_noise = torch.randn(
        [config.num_samples, config.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
    )

    print("sampled_noise.device", sampled_noise.device)
    print("prompts", prompts)

    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        low_memory=low_memory,
        profile=False,
    )
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()

    # Final output video
    video = 255.0 * current_video

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        # Determine model type for filename
        if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
            model_type = "lora"
        elif getattr(config, 'use_ema', False):
            model_type = "ema"
        else:
            model_type = "regular"
            
        for seed_idx in range(config.num_samples):
            # All processes save their videos
            if config.save_with_index:
                output_path = os.path.join(config.output_folder, f'rank{rank}-{idx}-{seed_idx}_{model_type}.mp4')
            else:
                output_path = os.path.join(config.output_folder, f'rank{rank}-{prompt[:100]}-{seed_idx}.mp4')
            write_video(output_path, video[seed_idx], fps=16)

    if config.inference_iter != -1 and i >= config.inference_iter:
        break
if dist.is_initialized():
    dist.destroy_process_group()
