"""GRPO training pipeline for Vision Coder.

Launch with:
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
        --config_file configs/accelerate_4gpu.yaml \
        vcoder/pipelines/training.py [OPTIONS]
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

for _v in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(_v, None)

import argparse

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer
from accelerate import Accelerator

from vcoder import (
    clip_visual_reward,
    format_reward,
    html_validity_reward,
    structural_similarity_reward,
)
from vcoder.data.websight import load_websight_dataset

def boosted_clip_reward(completions, image, **kwargs):
    """A simple wrapper around the CLIP reward that boosts it to be more comparable in scale to the other rewards."""
    base_scores = clip_visual_reward(completions, image, **kwargs)
    boosted_scores = [score * 3.0 for score in base_scores]
    return boosted_scores


def parse_args():
    p = argparse.ArgumentParser(description="GRPO training for Vision Coder")

    # Model
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--output_dir", default="./outputs/vcoder-grpo")

    # Data
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--image_width", type=int, default=512)

    # Training
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # GRPO
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--max_completion_length", type=int, default=2048)
    p.add_argument("--beta", type=float, default=0.0)

    # vLLM
    p.add_argument("--vllm_max_model_length", type=int, default=4096)

    # Logging / checkpointing
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    accelerator = Accelerator()

    # --- Dataset ---
    accelerator.print("Loading dataset...")
    train_dataset = load_websight_dataset(
        max_samples=args.max_samples,
        max_width=args.image_width,
    )
    accelerator.print(f"Dataset size: {len(train_dataset)}")

    # --- Model & Processor ---
    accelerator.print(f"Loading model: {args.model_id}")
    # max_pixels sized for image_width (assume up to 4:3 aspect ratio)
    max_pixels = args.image_width * int(args.image_width * 3 / 4)
    max_pixels = max(max_pixels // 784, 1) * 784  # align to 28*28

    accelerator.print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        use_fast=True,
        padding_side="left",
        # min_pixels=3136,
        # max_pixels=max_pixels,
    )
    accelerator.print("Processor loaded. Loading model weights...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )
    accelerator.print("Model loaded.")

    # --- Training config ---
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        remove_unused_columns=False,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        beta=args.beta,
        report_to=["tensorboard"],
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        # vLLM colocate: each GPU runs its own vLLM instance sharing GPU with training
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_max_model_length=args.vllm_max_model_length,

        # Skip the expensive forward pass after vLLM generation.
        # Saves ~5s per step (measured: 38% of step time).
        vllm_importance_sampling_correction=False,
    )

    # --- Reward functions ---
    reward_funcs = [
        format_reward,
        html_validity_reward,
        structural_similarity_reward,
        boosted_clip_reward,
    ]

    # --- Trainer ---
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
    )

    accelerator.print(f"\nbeta={trainer.beta}, num_generations={args.num_generations}")
    accelerator.print(f"dataset_size={len(train_dataset)}")
    accelerator.print(f"batch_size={args.batch_size}, grad_accum={args.gradient_accumulation_steps}")
    accelerator.print(f"max_completion_length={args.max_completion_length}")
    accelerator.print(f"vllm_importance_sampling_correction=False")
    accelerator.print(f"\nStarting training...")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    accelerator.print(f"\nTraining complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
