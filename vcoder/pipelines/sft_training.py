"""Supervised fine-tuning (SFT) on synthetic screenshot->HTML data.

Launch example:
    python -m vcoder.pipelines.sft_training \
        --dataset_dir assets/synth_samples \
        --output_dir outputs/vcoder-sft-synth \
        --max_samples 200 \
        --num_train_epochs 1 \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 8
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass

import torch
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from vcoder.data.synth_html import load_synthetic_sft_dataset


ATTN_BACKEND_CHOICES = ("auto", "flash_attention_2", "sdpa", "eager")


@dataclass
class SFTDataCollator:
    """Build batched tensors for VLM SFT and mask prompt tokens in labels."""

    processor: AutoProcessor
    max_length: int

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        prompt_texts: list[str] = []
        full_texts: list[str] = []
        images = []

        for feature in features:
            prompt_messages = feature["prompt"]
            solution = feature["solution"]

            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_messages = prompt_messages + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": solution}],
                }
            ]
            full_text = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_texts.append(prompt_text)
            full_texts.append(full_text)
            images.append(feature["image"])

        batch_full = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_prompt = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch_full["input_ids"].clone()
        labels[batch_full["attention_mask"] == 0] = -100

        prompt_lengths = batch_prompt["attention_mask"].sum(dim=1).tolist()
        for idx, prompt_len in enumerate(prompt_lengths):
            keep = min(int(prompt_len), labels.shape[1])
            labels[idx, :keep] = -100

        batch_full["labels"] = labels
        return batch_full


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT training for synthetic screenshot->HTML data")

    # Model / data
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--dataset_dir", default="assets/synth_samples")
    p.add_argument("--output_dir", default="outputs/vcoder-sft-synth")
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--max_width", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)

    # Optimization
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--attn_backend", choices=ATTN_BACKEND_CHOICES, default="auto")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--no_lora", action="store_true")

    # Logging / checkpointing
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--eval_ratio", type=float, default=0.05)
    p.add_argument("--eval_steps", type=int, default=100)

    return p.parse_args()


def _has_flash_attn() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _resolve_attn_backend(requested: str, has_cuda: bool) -> str | None:
    if not has_cuda:
        return None
    if requested == "auto":
        return "flash_attention_2" if _has_flash_attn() else "sdpa"
    return requested


def _load_model_with_attn_fallback(
    model_id: str,
    dtype: torch.dtype,
    attn_backend: str | None,
) -> tuple[Qwen3VLForConditionalGeneration, str | None]:
    model_kwargs: dict[str, object] = {"dtype": dtype}
    if attn_backend is not None:
        model_kwargs["attn_implementation"] = attn_backend
        print(f"Loading model with attention backend: {attn_backend}")
    else:
        print("Loading model with default CPU attention backend")

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        return model, attn_backend
    except ImportError as exc:
        msg = str(exc).lower()
        needs_fallback = attn_backend == "flash_attention_2" and (
            "flashattention2" in msg or "flash_attn" in msg
        )
        if not needs_fallback:
            raise

        print(
            "Warning: flash_attention_2 requested but flash_attn is unavailable. "
            "Retrying with sdpa backend. "
            "Install flash-attn or pass --attn_backend sdpa to silence this warning."
        )
        retry_kwargs = {"dtype": dtype, "attn_implementation": "sdpa"}
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **retry_kwargs)
        return model, "sdpa"


def main() -> None:
    args = parse_args()

    if args.max_samples <= 0:
        raise ValueError("--max_samples must be > 0")
    if args.max_length <= 0:
        raise ValueError("--max_length must be > 0")

    dataset = load_synthetic_sft_dataset(
        dataset_dir=args.dataset_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        max_width=args.max_width,
    )

    eval_dataset = None
    train_dataset = dataset
    if 0.0 < args.eval_ratio < 1.0 and len(dataset) > 1:
        split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    print(f"Loaded synthetic samples: {len(dataset)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset) if eval_dataset is not None else 0}")

    has_cuda = torch.cuda.is_available()
    model_dtype = torch.bfloat16 if has_cuda else torch.float32
    resolved_attn_backend = _resolve_attn_backend(args.attn_backend, has_cuda)
    if args.attn_backend == "flash_attention_2" and has_cuda and not _has_flash_attn():
        print(
            "Warning: --attn_backend flash_attention_2 was requested, but flash_attn is not installed. "
            "The loader will retry with sdpa if flash2 initialization fails."
        )
    print(f"Requested attention backend: {args.attn_backend}")
    print(f"Resolved attention backend: {resolved_attn_backend or 'cpu-default'}")

    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "right"

    model, loaded_attn_backend = _load_model_with_attn_fallback(
        model_id=args.model_id,
        dtype=model_dtype,
        attn_backend=resolved_attn_backend,
    )
    print(f"Loaded model attention backend: {loaded_attn_backend or 'cpu-default'}")

    if not args.no_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ModuleNotFoundError as exc:
            raise RuntimeError("peft is required for LoRA SFT. Install it or pass --no_lora.") from exc
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    collator = SFTDataCollator(processor=processor, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"SFT training complete. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
