import argparse
import functools
import os
from typing import Dict

import torch
import torch.distributed as dist
import yaml
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from data_processor import DPODataCollator, HHRLHFDPODataset
from dpo_logic import dpo_loss

try:
    import wandb
except ImportError:  # wandb is optional at runtime
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-parameter DPO training with raw FSDP.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="SFT checkpoint path or model id.")
    parser.add_argument("--wandb_project", type=str, default="dpo-fsdp", help="Weights & Biases project name.")
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def init_distributed() -> Dict[str, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FSDP training.")

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def init_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def init_models(model_name: str):
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(policy_model, "enable_input_require_grads"):
        policy_model.enable_input_require_grads()
    policy_model.config.use_cache = False

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    ref_model.eval()
    ref_model.requires_grad_(False)
    ref_model.config.use_cache = False

    return policy_model, ref_model


def build_fsdp(
    model,
    device_id: torch.device,
    sync_module_states: bool,
    use_orig_params: bool,
):
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer},
    )
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_id=device_id,
        sync_module_states=sync_module_states,
        use_orig_params=use_orig_params,
    )


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def reduce_metrics(reward_margin: torch.Tensor, loss: torch.Tensor) -> Dict[str, float]:
    reward_margin = reward_margin.float()
    loss = loss.float()
    device = reward_margin.device

    count = torch.tensor(reward_margin.numel(), device=device, dtype=torch.float32)
    sum_margin = reward_margin.sum()
    sum_sq = (reward_margin ** 2).sum()
    pos = (reward_margin > 0).sum()
    min_margin = reward_margin.min()
    max_margin = reward_margin.max()
    loss_sum = loss * count

    if dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_margin, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_sq, op=dist.ReduceOp.SUM)
        dist.all_reduce(pos, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(min_margin, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_margin, op=dist.ReduceOp.MAX)

    mean = sum_margin / count.clamp(min=1)
    var = sum_sq / count.clamp(min=1) - mean**2
    std = torch.sqrt(torch.clamp(var, min=0.0))
    accuracy = pos / count.clamp(min=1)
    loss_mean = loss_sum / count.clamp(min=1)

    return {
        "loss": loss_mean.item(),
        "accuracy": accuracy.item(),
        "reward_margin_mean": mean.item(),
        "reward_margin_max": max_margin.item(),
        "reward_margin_min": min_margin.item(),
        "reward_margin_std": std.item(),
    }


def save_checkpoint(model: FSDP, output_dir: str, step: int, rank: int):
    ckpt_dir = os.path.join(output_dir, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg):
        model_state = model.state_dict()

    torch.save(model_state, os.path.join(ckpt_dir, f"model_rank{rank}.pt"))
    torch.save({"step": step}, os.path.join(ckpt_dir, f"trainer_rank{rank}.pt"))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model_name = args.model_name_or_path or cfg["model_name_or_path"]

    dist_state = init_distributed()
    rank = dist_state["rank"]
    world_size = dist_state["world_size"]
    local_rank = dist_state["local_rank"]

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed + rank)

    tokenizer = init_tokenizer(model_name)
    data_collator = DPODataCollator(tokenizer=tokenizer, pad_to_multiple_of=cfg.get("pad_to_multiple_of", 8))

    train_dataset = HHRLHFDPODataset(
        tokenizer=tokenizer,
        split="train",
        max_length=cfg.get("max_length", 2048),
        eval_ratio=cfg.get("eval_ratio", 0.05),
        seed=seed,
        dataset_split=cfg.get("dataset_split", "train"),
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.get("batch_size", 1),
        sampler=train_sampler,
        collate_fn=data_collator,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=True,
    )

    policy_model, ref_model = init_models(model_name)
    sync_module_states = cfg.get("fsdp_sync_module_states", True)
    use_orig_params = cfg.get("fsdp_use_orig_params", True)
    device_id = torch.device("cuda", local_rank)
    policy_model = build_fsdp(policy_model, device_id, sync_module_states, use_orig_params)
    ref_model = build_fsdp(ref_model, device_id, sync_module_states, use_orig_params)
    ref_model.eval()
    ref_model.requires_grad_(False)

    optimizer = AdamW(policy_model.parameters(), lr=cfg.get("lr", 5e-5))

    use_wandb = rank == 0 and wandb is not None
    if use_wandb:
        wandb.init(project=args.wandb_project, config=cfg)

    if rank == 0:
        print("Starting DPO training with FSDP FULL_SHARD.")

    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    log_every = cfg.get("log_every", 1)
    save_every = cfg.get("save_every", 0)
    output_dir = cfg.get("output_dir", "checkpoints")
    num_epochs = cfg.get("num_epochs", 1)

    policy_model.train()
    global_step = 0

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            batch = move_to_device(batch, device_id)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs_w = policy_model(
                    input_ids=batch["input_ids_w"],
                    attention_mask=batch["attention_mask_w"],
                    use_cache=False,
                )
                outputs_l = policy_model(
                    input_ids=batch["input_ids_l"],
                    attention_mask=batch["attention_mask_l"],
                    use_cache=False,
                )

                with torch.no_grad():
                    ref_outputs_w = ref_model(
                        input_ids=batch["input_ids_w"],
                        attention_mask=batch["attention_mask_w"],
                        use_cache=False,
                    )
                    ref_outputs_l = ref_model(
                        input_ids=batch["input_ids_l"],
                        attention_mask=batch["attention_mask_l"],
                        use_cache=False,
                    )

                raw_loss, reward_margin = dpo_loss(
                    policy_logits_w=outputs_w.logits,
                    policy_logits_l=outputs_l.logits,
                    ref_logits_w=ref_outputs_w.logits,
                    ref_logits_l=ref_outputs_l.logits,
                    labels_w=batch["labels_w"],
                    labels_l=batch["labels_l"],
                    beta=cfg.get("beta", 0.1),
                )

            (raw_loss / grad_accum).backward()

            should_step = (step + 1) % grad_accum == 0 or (step + 1) == len(train_dataloader)
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_every == 0:
                    stats = reduce_metrics(reward_margin.detach(), raw_loss.detach())
                    if rank == 0:
                        print(
                            f"step={global_step} loss={stats['loss']:.4f} "
                            f"acc={stats['accuracy']:.4f} margin={stats['reward_margin_mean']:.4f}"
                        )
                        if use_wandb:
                            wandb.log(stats, step=global_step)

                if save_every and global_step % save_every == 0:
                    save_checkpoint(policy_model, output_dir, global_step, rank)

    if cfg.get("save_final", True):
        save_checkpoint(policy_model, output_dir, global_step, rank)

    if dist.is_initialized():
        dist.barrier()

    if use_wandb:
        wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
