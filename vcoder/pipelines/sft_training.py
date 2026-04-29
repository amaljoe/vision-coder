"""SFT training pipeline for Vision Coder.

Trains Qwen3-VL on WebSight (screenshot → HTML) with completion-only loss
using TRL's SFTTrainer. Used as the warm-start stage before GRPO/RL fine-tuning.

Launch with:
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --config_file configs/accelerate_2gpu.yaml \
        vcoder/pipelines/sft_training.py [OPTIONS]
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

# Keep HF dataset cache in /dev/shm to avoid home filesystem quota issues
os.environ.setdefault("HF_DATASETS_CACHE", "/dev/shm/hf_datasets_cache")
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

for _v in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(_v, None)

for _v in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(_v, None)

import argparse

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer
from accelerate import Accelerator

from vcoder.data.websight import load_websight_sft_dataset


def parse_args():
    p = argparse.ArgumentParser(description="SFT training for Vision Coder")

    p.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--output_dir", default="./outputs/vcoder-sft")

    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--image_width", type=int, default=512)

    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=2048)

    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()

    accelerator.print("Loading SFT dataset...")
    train_dataset = load_websight_sft_dataset(
        max_samples=args.max_samples,
        max_width=args.image_width,
    )
    accelerator.print(f"Dataset size: {len(train_dataset)}")
    accelerator.print(f"Columns: {train_dataset.column_names}")

    accelerator.print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        use_fast=True,
        padding_side="right",
    )
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    accelerator.print(f"Using attention implementation: {attn_impl}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    )
    accelerator.print("Model loaded.")

    training_args = SFTConfig(
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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        dataset_kwargs={"skip_prepare_dataset": True},
        completion_only_loss=True,
        report_to=["tensorboard"],
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
    )

    accelerator.print(f"\nSFT training")
    accelerator.print(f"  dataset_size={len(train_dataset)}")
    accelerator.print(f"  batch_size={args.batch_size}, grad_accum={args.gradient_accumulation_steps}")
    accelerator.print(f"  epochs={args.num_train_epochs}, max_length={args.max_length}")
    accelerator.print(f"  completion_only_loss=True")
    accelerator.print(f"\nStarting SFT training...")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    accelerator.print(f"\nSFT training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
