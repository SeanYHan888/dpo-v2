import argparse
import os
from typing import Dict, Tuple

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin, HfTransformerPolicy
from torch.distributed.fsdp import ShardingStrategy
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from data_processor import DPODataCollator, HHRLHFDPODataset
from dpo_logic import dpo_loss

try:
    import wandb
except ImportError:  # wandb is optional at runtime
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-parameter DPO training with FSDP.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="SFT checkpoint path or model id.")
    parser.add_argument("--wandb_project", type=str, default="dpo-fsdp", help="Weights & Biases project name.")
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_accelerator(cfg: Dict) -> Accelerator:
    auto_wrap_policy = HfTransformerPolicy({Qwen2DecoderLayer})
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        sync_module_states=cfg.get("fsdp", {}).get("sync_module_states", True),
    )
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        fsdp_plugin=fsdp_plugin,
        cpu_ram_efficient_loading=True,
    )
    return accelerator


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
        attn_implementation="flash_attention_2",
    )
    policy_model.gradient_checkpointing_enable()
    policy_model.config.use_cache = False
    if hasattr(policy_model.config, "use_flash_attention_2"):
        policy_model.config.use_flash_attention_2 = True
    if hasattr(policy_model.config, "use_flash_attn"):
        policy_model.config.use_flash_attn = True

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    ref_model.eval()
    ref_model.config.use_cache = False
    for param in ref_model.parameters():
        param.requires_grad = False

    return policy_model, ref_model


def log_metrics(accelerator: Accelerator, loss: torch.Tensor, reward_margin: torch.Tensor, step: int, use_wandb: bool):
    loss_val = accelerator.gather(loss.detach()).mean()
    margins = accelerator.gather(reward_margin.detach())

    accuracy = (margins > 0).float().mean()
    stats = {
        "loss": loss_val.item(),
        "accuracy": accuracy.item(),
        "reward_margin_mean": margins.mean().item(),
        "reward_margin_max": margins.max().item(),
        "reward_margin_min": margins.min().item(),
        "reward_margin_std": margins.std(unbiased=False).item(),
    }

    if accelerator.is_main_process:
        accelerator.print(f"step={step} loss={stats['loss']:.4f} acc={stats['accuracy']:.4f} margin={stats['reward_margin_mean']:.4f}")
        if use_wandb and wandb is not None:
            wandb.log(stats, step=step)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model_name = args.model_name_or_path or cfg["model_name_or_path"]

    accelerator = setup_accelerator(cfg)
    tokenizer = init_tokenizer(model_name)

    train_dataset = HHRLHFDPODataset(
        tokenizer=tokenizer,
        split="train",
        max_length=cfg.get("max_length", 2048),
    )
    data_collator = DPODataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.get("batch_size", 1),
        shuffle=True,
        collate_fn=data_collator,
    )

    policy_model, ref_model = init_models(model_name)
    optimizer = AdamW(policy_model.parameters(), lr=cfg.get("lr", 5e-5))

    policy_model, ref_model, optimizer, train_dataloader = accelerator.prepare(
        policy_model,
        ref_model,
        optimizer,
        train_dataloader,
    )

    accelerator.print("Starting DPO training with FSDP FULL_SHARD.")
    use_wandb = accelerator.is_main_process and wandb is not None
    if use_wandb:
        wandb.init(project=args.wandb_project, config=cfg)

    policy_model.train()
    total_steps = 0

    for epoch in range(cfg.get("num_epochs", 1)):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(policy_model):
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

                loss, reward_margin = dpo_loss(
                    policy_logits_w=outputs_w.logits,
                    policy_logits_l=outputs_l.logits,
                    ref_logits_w=ref_outputs_w.logits,
                    ref_logits_l=ref_outputs_l.logits,
                    labels_w=batch["labels_w"],
                    labels_l=batch["labels_l"],
                    beta=cfg.get("beta", 0.1),
                )

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_steps += 1
            log_metrics(accelerator, loss, reward_margin, total_steps, use_wandb)

    if accelerator.is_main_process and use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
