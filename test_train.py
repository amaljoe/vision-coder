# """Short 10-step training run to verify end-to-end pipeline."""
# import os

# # Must be set BEFORE any HF imports — use cached models only, no internet
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"
# # # Clear SOCKS proxy so localhost vLLM server is reachable
# # for var in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
# #     os.environ.pop(var, None)
# # # GPU 0 for training, GPU 3 visible for NCCL weight sync with vLLM server
# # os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

# import torch
# from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
# from peft import LoraConfig
# from trl import GRPOTrainer, GRPOConfig

# from vcoder import (
#     format_reward,
#     html_validity_reward,
#     structural_similarity_reward,
# )
# from vcoder.data.websight import load_websight_dataset

# # --- Dataset (loads from local cache) ---
# print("Loading dataset...")
# train_dataset = load_websight_dataset(max_samples=50)
# print(f"Dataset: {len(train_dataset)} samples")

# # --- Model (load explicitly on cuda:0 to avoid spilling to cuda:1=vLLM GPU) ---
# model_id = "Qwen/Qwen3-VL-2B-Instruct"
# print(f"Loading model and processor: {model_id}")
# processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
# )

# lora_config = LoraConfig(
#     task_type="CAUSAL_LM",
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
# )

# # --- Training (10 steps, vLLM generation) ---
# training_args = GRPOConfig(
#     output_dir="/tmp/vcoder-test-run",
#     learning_rate=5e-6,
#     remove_unused_columns=False,
#     max_steps=10,
#     bf16=True,
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=1,
#     max_completion_length=512,
#     num_generations=4,

#     use_vllm=True,
#     vllm_server_base_url="http://localhost:8000",

#     report_to=["tensorboard"],
#     logging_steps=1,
#     save_strategy="no",
# )

# trainer = GRPOTrainer(
#     model=model,
#     processing_class=processor,
#     reward_funcs=[
#         format_reward,
#         html_validity_reward,
#         structural_similarity_reward,
#     ],
#     args=training_args,
#     train_dataset=train_dataset,
#     peft_config=lora_config,
# )

# # Workaround: TRL sets tools=[] which VLLMClient.chat() rejects (checks `is not None`)
# trainer.vllm_generation.tools = None

# print("Starting 10-step test training run...")
# trainer.train()
# print("\nTest training complete!")

######
"""Short 10-step training run to verify end-to-end pipeline.

Key fixes:
- HF offline env vars set BEFORE any HF imports.
- All HF/TRL/PEFT model loading happens inside main() so
  fsdp_cpu_ram_efficient_loading works correctly.
- Do NOT manually place model on a device (Accelerate/FSDP will handle it).
"""
import os

# --- MUST be set BEFORE any HF/transformers/trl/peft imports ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# (Do NOT set CUDA_VISIBLE_DEVICES here; provide it at launch time.)
# e.g.
# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes=3 test_train.py
# and start vLLM separately:
# CUDA_VISIBLE_DEVICES=3 vllm serve ...

def main():
    # Import inside main to avoid model-loading at import time on all ranks
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig
    from trl import GRPOTrainer, GRPOConfig

    # Your project-specific reward functions and dataset loader
    from vcoder import (
        format_reward,
        html_validity_reward,
        structural_similarity_reward,
    )
    from vcoder.data.websight import load_websight_dataset

    print("Visible CUDA devices (to this process):", torch.cuda.device_count())
    try:
        print("Current device name (cuda:0):", torch.cuda.get_device_name(0))
    except Exception:
        pass

    # --- Dataset (loads from local cache) ---
    print("Loading dataset...")
    train_dataset = load_websight_dataset(max_samples=50)
    print(f"Dataset: {len(train_dataset)} samples")

    # --- Model & processor (loaded here to support CPU-efficient loading) ---
    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    print(f"Loading model and processor from {model_id} (offline)...")
    processor = AutoProcessor.from_pretrained(
        model_id,
        use_fast=True,
        padding_side="left",
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )

    # --- LoRA / PEFT config ---
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    # --- GRPO training args (short test run) ---
    training_args = GRPOConfig(
        output_dir="/tmp/vcoder-test-run",
        learning_rate=5e-6,
        remove_unused_columns=False,
        max_steps=10,
        bf16=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_completion_length=512,
        num_generations=2,          # lowered for quicker debug runs
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url="http://localhost:8000",
        report_to=["tensorboard"],
        logging_steps=1,
        save_strategy="no",
    )

    # --- Trainer ---
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[
            format_reward,
            html_validity_reward,
            structural_similarity_reward,
        ],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
    )

    # Workaround: TRL sets tools=[] which VLLMClient.chat() rejects (checks `is not None`)
    trainer.vllm_generation.tools = None

    print("Starting test training run (10 steps)...")
    trainer.train()
    print("Test training run complete!")


if __name__ == "__main__":
    main()
