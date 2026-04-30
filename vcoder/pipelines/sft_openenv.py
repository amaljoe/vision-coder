"""SFT fine-tuning on vision-coder-openenv custom HTML examples.

Fine-tunes the grpo-clip checkpoint (or any model) on 15 custom HTML/screenshot pairs
rendered from the openenv dataset. Uses completion-only loss.

Launch (1 GPU):
    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --config_file configs/accelerate_1gpu.yaml \
        vcoder/pipelines/sft_openenv.py \
        --model_id outputs/vcoder-grpo-clip/checkpoint-500 \
        --output_dir outputs/vcoder-grpo-clip-sft-openenv \
        --num_train_epochs 3
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("HF_DATASETS_CACHE", "/dev/shm/hf_datasets_cache")
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

for _v in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
    os.environ.pop(_v, None)

import argparse

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer
from accelerate import Accelerator

from vcoder.data.openenv import load_openenv_sft_dataset


def parse_args():
    p = argparse.ArgumentParser(description="SFT on openenv custom examples")
    p.add_argument("--model_id", default="outputs/vcoder-grpo-clip/checkpoint-500")
    p.add_argument("--output_dir", default="./outputs/vcoder-grpo-clip-sft-openenv")
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--image_width", type=int, default=512)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=15)
    p.add_argument("--save_total_limit", type=int, default=4)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()

    accelerator.print("Rendering openenv HTML examples...")
    train_dataset = load_openenv_sft_dataset(max_width=args.image_width)
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
    accelerator.print(f"Attention: {attn_impl}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    )

    total_steps = len(train_dataset) * args.num_train_epochs // (args.batch_size * args.gradient_accumulation_steps)
    accelerator.print(f"Estimated optimizer steps: {total_steps}")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        remove_unused_columns=False,
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
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
    )

    accelerator.print(f"\nSFT openenv — {len(train_dataset)} examples × {args.num_train_epochs} epochs")
    accelerator.print(f"  model_id: {args.model_id}")
    accelerator.print(f"  output_dir: {args.output_dir}")
    accelerator.print(f"  lr={args.learning_rate}, max_length={args.max_length}")
    accelerator.print(f"  completion_only_loss=True")
    accelerator.print("\nStarting training...")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    accelerator.print(f"\nTraining complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
