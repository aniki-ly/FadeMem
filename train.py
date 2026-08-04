# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf
import torch
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--no-auto-resume", action="store_true", help="Disable auto resume from latest checkpoint in logdir")
    parser.add_argument("--no-one-logger", action="store_true", help="Disable One Logger (enabled by default)")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    config_name = Path(args.config_path).stem
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb
    config.auto_resume = not args.no_auto_resume  # Default to True unless --no-auto-resume is specified
    config.use_one_logger = not args.no_one_logger

    if not torch.cuda.is_available():
        raise RuntimeError("FadeMem training requires CUDA-capable GPUs.")
    for label, path in (
        ("Generator checkpoint", config.generator_ckpt),
        ("LoRA checkpoint", config.lora_ckpt),
        ("Training prompt file", config.data_path),
        ("Switch-prompt file", config.switch_prompt_path),
        ("Wan2.1-T2V-1.3B model directory", "wan_models/Wan2.1-T2V-1.3B"),
        ("Wan2.1-T2V-14B model directory", "wan_models/Wan2.1-T2V-14B"),
    ):
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    from trainer import ScoreDistillationTrainer

    if config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    else:
        raise ValueError(f"Unsupported trainer: {config.trainer}")
    try:
        trainer.train()
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
