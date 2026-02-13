"""Short 10-step training run with comprehensive phase-level profiling."""
import os

# --- MUST be set BEFORE any HF/transformers/trl/peft imports ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

for _proxy_var in (
    "ALL_PROXY", "all_proxy",
    "HTTP_PROXY", "http_proxy",
    "HTTPS_PROXY", "https_proxy",
):
    os.environ.pop(_proxy_var, None)


import copy
import time
from functools import wraps


def timed_reward(fn):
    """Wrap reward function and print execution time."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"      [REWARD] {fn.__name__}: {elapsed:.2f}s", flush=True)
        return out
    return wrapper


def _log(indent, label, elapsed=None):
    prefix = "  " * indent
    if elapsed is not None:
        print(f"{prefix}[{elapsed:>8.2f}s] {label}", flush=True)
    else:
        print(f"{prefix}>> {label}", flush=True)


def main():
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from trl import GRPOTrainer, GRPOConfig
    from trl.data_utils import is_conversational

    from vcoder import (
        format_reward,
        html_validity_reward,
        structural_similarity_reward,
    )
    from vcoder.data.websight import load_websight_dataset

    print(f"Visible CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  cuda:{i} = {torch.cuda.get_device_name(i)}")

    # --- Dataset ---
    print("Loading dataset...")
    train_dataset = load_websight_dataset(max_samples=50)
    print(f"Dataset size: {len(train_dataset)}")

    # --- Model ---
    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    print("Loading model...")

    # OPTIMIZATION: Limit image resolution. Default max_pixels=None uses full
    # 2560x1440 images -> ~3,600 vision tokens/image.
    # 401408 (512*28*28) still gave 1456 tokens. Use 100352 (128*28*28)
    # for ~350 tokens/image -> much faster generation & forward passes.
    processor = AutoProcessor.from_pretrained(
        model_id,
        use_fast=True,
        padding_side="left",
        min_pixels=3136,        # 4 * 28 * 28
        max_pixels=100352,      # 128 * 28 * 28
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )

    # No LoRA — full fine-tune. Eliminates merge/unmerge overhead in vLLM weight sync.

    # Quick sanity check: how many vision tokens does one image produce?
    sample_img = train_dataset[0]["image"]
    sample_inputs = processor(images=[[sample_img]], text=["test"], return_tensors="pt")
    if "image_grid_thw" in sample_inputs:
        grid = sample_inputs["image_grid_thw"]
        n_patches = grid.prod(dim=-1).sum().item()
        print(f"Vision tokens per image: {n_patches} (from {sample_img.size[0]}x{sample_img.size[1]})")
    del sample_inputs

    # --- Training config ---
    training_args = GRPOConfig(
        output_dir="/tmp/vcoder-test-run",
        learning_rate=5e-6,
        remove_unused_columns=False,
        max_steps=3,
        bf16=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_completion_length=512,
        num_generations=2,
        report_to=["tensorboard"],
        logging_steps=1,
        save_strategy="no",

        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_max_model_length=4096,
        # vllm_server_base_url="http://localhost:8003",

        # OPTIMIZATION: Disable importance sampling correction to skip the
        # expensive forward pass that runs AFTER vLLM generation.
        # Only matters when use_vllm=True.
        # vllm_importance_sampling_correction=False,
    )

    # --- Wrap rewards ---
    reward_funcs = [
        timed_reward(format_reward),
        timed_reward(html_validity_reward),
        timed_reward(structural_similarity_reward),
    ]

    # =========================================================================
    # PROFILED TRAINER — dissects every sub-phase with timing
    # =========================================================================
    class ProfiledGRPOTrainer(GRPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._logps_call_idx = 0
            self._ts = {}
            self._train_step_n = 0

            # Monkey-patch vllm_generation methods if vLLM is enabled
            if self.use_vllm and hasattr(self, 'vllm_generation'):
                self._patch_vllm_generation()

        def _patch_vllm_generation(self):
            """Wrap vllm_generation.sync_weights and generate with timing."""
            vg = self.vllm_generation

            orig_sync = vg.sync_weights
            def timed_sync():
                torch.cuda.synchronize()
                t = time.perf_counter()
                _log(2, "sync_weights START")
                orig_sync()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t
                _log(2, "sync_weights", elapsed)
                self._ts["sync_weights"] = elapsed
            vg.sync_weights = timed_sync

            orig_gen = vg.generate
            def timed_gen(*a, **kw):
                torch.cuda.synchronize()
                t = time.perf_counter()
                _log(2, "vllm_generation.generate START")
                res = orig_gen(*a, **kw)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t
                _log(2, "vllm_generation.generate", elapsed)
                self._ts["vllm_gen"] = elapsed
                return res
            vg.generate = timed_gen

        # ----- _generate_single_turn: times weight sync vs vLLM call -----
        def _generate_single_turn(self, prompts):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _log(1, "_generate_single_turn START")

            res = super()._generate_single_turn(prompts)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            _log(1, "_generate_single_turn TOTAL", elapsed)
            self._ts["single_turn"] = elapsed
            return res

        # ----- _generate: dissect deepcopy, single_turn, decode, metrics -----
        def _generate(self, prompts):
            torch.cuda.synchronize()
            self._ts["gen_start"] = time.perf_counter()
            sep = "-" * 60
            print(f"\n{sep}", flush=True)
            _log(0, "_generate START")

            # Phase 1: copy.deepcopy(prompts)
            t = time.perf_counter()
            prompts = copy.deepcopy(prompts)
            dc_time = time.perf_counter() - t
            _log(1, f"copy.deepcopy(prompts) [{len(prompts)} prompts]", dc_time)
            self._ts["deepcopy"] = dc_time

            # Phase 2: _generate_single_turn (includes sync_weights + vLLM gen)
            prompt_ids, completion_ids, logprobs, extra_fields = self._generate_single_turn(prompts)

            # Phase 3: Decode completions
            t = time.perf_counter()
            if is_conversational({"prompt": prompts[0]}):
                contents = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                completions = [[{"role": "assistant", "content": c}] for c in contents]
            else:
                completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            decode_time = time.perf_counter() - t
            _log(1, f"batch_decode [{len(completion_ids)} seqs]", decode_time)
            self._ts["decode"] = decode_time

            # Phase 4: Tool mask
            t = time.perf_counter()
            if self.tools:
                (tool_mask, completions, completion_ids, logprobs,
                 tool_call_count, tool_failure_count) = self._tool_call_loop(
                    prompts, prompt_ids, completion_ids, completions, logprobs)
            else:
                tool_mask = extra_fields.pop("env_mask", None)
            tool_time = time.perf_counter() - t
            self._ts["tools"] = tool_time

            # Phase 5: Metrics (lengths, gathering, logging)
            t = time.perf_counter()
            device = self.accelerator.device
            mode = "train" if self.model.training else "eval"

            prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
            if tool_mask is not None:
                completion_lengths = torch.tensor([sum(mask) for mask in tool_mask], device=device)
            else:
                completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
            agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
            agg_completion_lengths = self.accelerator.gather(completion_lengths)
            total_prompt_tokens = agg_prompt_lengths.sum()
            total_completion_tokens = agg_completion_lengths.sum()

            if mode == "train":
                self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
            self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
            self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
            self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
            self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
            agg_is_truncated = self.accelerator.gather(is_truncated)
            self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
            term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
            if len(term_completion_lengths) == 0:
                term_completion_lengths = torch.zeros(1, device=device)
            self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
            self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
            self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

            if self.tools:
                agg_tool_call_count = self.accelerator.gather(torch.tensor(tool_call_count, device=device)).sum()
                tool_call_frequency = (agg_tool_call_count / len(agg_prompt_lengths)).item()
                self._metrics[mode]["tools/call_frequency"].append(tool_call_frequency)
                agg_tool_failure_count = self.accelerator.gather(torch.tensor(tool_failure_count, device=device)).sum()
                failure_frequency = (agg_tool_failure_count / agg_tool_call_count).item() if agg_tool_call_count > 0 else 0.0
                self._metrics[mode]["tools/failure_frequency"].append(failure_frequency)

            metrics_time = time.perf_counter() - t
            _log(1, "metrics + gather", metrics_time)
            self._ts["metrics"] = metrics_time

            # Completion length stats
            avg_len = agg_completion_lengths.float().mean().item()
            max_len = agg_completion_lengths.float().max().item()
            _log(1, f"completion lengths: avg={avg_len:.0f}, max={max_len:.0f}")

            torch.cuda.synchronize()
            self._ts["gen_end"] = time.perf_counter()
            gen_total = self._ts["gen_end"] - self._ts["gen_start"]

            # Print _generate summary
            print(f"\n  --- _generate SUMMARY ---", flush=True)
            _log(1, "copy.deepcopy(prompts)", dc_time)
            if "sync_weights" in self._ts:
                _log(1, "sync_weights", self._ts["sync_weights"])
            if "vllm_gen" in self._ts:
                _log(1, "vllm_generation.generate", self._ts["vllm_gen"])
            elif "single_turn" in self._ts and "sync_weights" in self._ts:
                rest = self._ts["single_turn"] - self._ts.get("sync_weights", 0) - self._ts.get("vllm_gen", 0)
                if rest > 0.01:
                    _log(1, "single_turn overhead (prep/other)", rest)
            _log(1, "batch_decode", decode_time)
            _log(1, "metrics + gather", metrics_time)
            _log(1, "_generate TOTAL", gen_total)
            print(f"{sep}", flush=True)

            return (
                prompt_ids, completion_ids, tool_mask, completions,
                total_completion_tokens, logprobs, extra_fields,
            )

        # ----- _generate_and_score_completions: outer orchestrator -----
        def _generate_and_score_completions(self, inputs):
            self._logps_call_idx = 0
            self._ts = {}
            torch.cuda.synchronize()
            self._ts["total_start"] = time.perf_counter()

            res = super()._generate_and_score_completions(inputs)

            torch.cuda.synchronize()
            self._ts["total_end"] = time.perf_counter()
            self._print_profile()
            return res

        # ----- _get_per_token_logps (called 1-2x) -----
        def _get_per_token_logps_and_entropies(self, model, *args, **kwargs):
            torch.cuda.synchronize()
            self._logps_call_idx += 1
            idx = self._logps_call_idx
            self._ts[f"logps{idx}_start"] = time.perf_counter()
            res = super()._get_per_token_logps_and_entropies(model, *args, **kwargs)
            torch.cuda.synchronize()
            self._ts[f"logps{idx}_end"] = time.perf_counter()
            return res

        # ----- _calculate_rewards -----
        def _calculate_rewards(self, *args, **kwargs):
            torch.cuda.synchronize()
            self._ts["rew_start"] = time.perf_counter()
            res = super()._calculate_rewards(*args, **kwargs)
            self._ts["rew_end"] = time.perf_counter()
            return res

        # ----- training_step -----
        def training_step(self, model, inputs, num_items_in_batch):
            self._train_step_n += 1
            print(f"\n{'='*70}", flush=True)
            print(f"  TRAINING STEP {self._train_step_n}", flush=True)
            print(f"{'='*70}", flush=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = super().training_step(model, inputs, num_items_in_batch)
            torch.cuda.synchronize()
            total = time.perf_counter() - t0
            print(f"  [TOTAL] training_step: {total:.2f}s", flush=True)
            return res

        # ----- compute_loss -----
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            torch.cuda.synchronize()
            _log(1, "compute_loss (fwd+logps)", time.perf_counter() - t0)
            return res

        # ----- Summary -----
        def _print_profile(self):
            ts = self._ts
            sep = "=" * 60

            def dt(a, b):
                return ts.get(b, 0) - ts.get(a, 0)

            total = dt("total_start", "total_end")
            gen_time = dt("gen_start", "gen_end")

            if "logps1_start" in ts:
                img_gap = ts["logps1_start"] - ts["gen_end"]
            else:
                img_gap = ts.get("rew_start", ts["total_end"]) - ts.get("gen_end", ts["total_start"])

            logps1 = dt("logps1_start", "logps1_end") if "logps1_start" in ts else 0.0
            logps2 = dt("logps2_start", "logps2_end") if "logps2_start" in ts else 0.0

            last_logps_end = ts.get("logps2_end", ts.get("logps1_end", ts.get("gen_end", 0)))
            decode_gap = ts.get("rew_start", 0) - last_logps_end if "rew_start" in ts else 0.0

            rew_time = dt("rew_start", "rew_end") if "rew_start" in ts else 0.0
            remaining = ts["total_end"] - ts.get("rew_end", ts["total_end"])

            print(f"\n{sep}", flush=True)
            print(f"  FULL STEP PROFILE: _generate_and_score_completions", flush=True)
            print(f"{sep}", flush=True)
            _log(0, "_generate (total)", gen_time)
            if "deepcopy" in ts:
                _log(1, "copy.deepcopy", ts["deepcopy"])
            if "sync_weights" in ts:
                _log(1, "sync_weights", ts["sync_weights"])
            if "vllm_gen" in ts:
                _log(1, "vllm generate", ts["vllm_gen"])
            if "decode" in ts:
                _log(1, "batch_decode", ts["decode"])
            if "metrics" in ts:
                _log(1, "metrics+gather", ts["metrics"])
            _log(0, "tensor pad + image proc (gap)", img_gap)
            if logps1 > 0:
                _log(0, "old_per_token_logps (IS correction)", logps1)
            if logps2 > 0:
                _log(0, "ref_per_token_logps (KL)", logps2)
            if abs(decode_gap) > 0.01:
                _log(0, "decode + merge extra fields (gap)", decode_gap)
            _log(0, "reward computation", rew_time)
            if remaining > 0.01:
                _log(0, "normalization + advantages", remaining)
            print(f"{sep}", flush=True)
            _log(0, "TOTAL", total)
            print(f"{sep}\n", flush=True)

    # =========================================================================

    trainer = ProfiledGRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
    )

    print(f"\nbeta={trainer.beta}, use_vllm={trainer.use_vllm}")
    print(f"num_generations={trainer.num_generations}, batch_size={training_args.per_device_train_batch_size}")
    print(f"max_completion_length={training_args.max_completion_length}")
    print("\nStarting training...")
    trainer.train()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
