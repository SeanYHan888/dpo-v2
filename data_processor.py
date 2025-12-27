import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


CHAT_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"


def _split_prompt_response(text: str) -> Tuple[str, str]:
    """
    HH-RLHF samples look like `Human: ... Assistant: ...`.
    We treat everything before the final `Assistant:` as the prompt context.
    """
    if "Assistant:" in text:
        head, _, tail = text.rpartition("Assistant:")
        prompt = head
        if "Human:" in prompt:
            prompt = prompt.split("Human:", 1)[-1]
        return prompt.strip(), tail.strip()
    return text.strip(), ""


def _format_chatml(prompt: str, response: str) -> str:
    return CHAT_TEMPLATE.format(prompt=prompt.strip(), response=response.strip())


def _prompt_prefix(prompt: str) -> str:
    return "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n".format(prompt=prompt.strip())


def _normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.strip().split())


def _prompt_id(prompt: str) -> str:
    normalized = _normalize_prompt(prompt)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _build_pairs(raw_dataset) -> List[Dict[str, str]]:
    pairs = []
    for sample in raw_dataset:
        chosen_text = sample.get("chosen", "")
        rejected_text = sample.get("rejected", "")
        prompt_w, response_w = _split_prompt_response(chosen_text)
        prompt_l, response_l = _split_prompt_response(rejected_text)
        prompt = prompt_w or prompt_l
        if not prompt:
            continue
        pairs.append(
            {
                "prompt_id": _prompt_id(prompt),
                "prompt": prompt,
                "chosen": response_w,
                "rejected": response_l,
            }
        )
    return pairs


class HHRLHFDPODataset(Dataset):
    """
    Dataset for Anthropic HH-RLHF preference pairs with group-based splits.
    Each item returns tokenized winner/loser sequences with prompt tokens masked to -100.
    """

    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_length: int = 2048,
        eval_ratio: float = 0.05,
        seed: int = 42,
        dataset_split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        raw_dataset = load_dataset("Anthropic/hh-rlhf", split=dataset_split)
        pairs = _build_pairs(raw_dataset)

        prompt_to_pairs: Dict[str, List[Dict[str, str]]] = {}
        for pair in pairs:
            prompt_to_pairs.setdefault(pair["prompt_id"], []).append(pair)

        prompt_ids = sorted(prompt_to_pairs.keys())
        rng = random.Random(seed)
        rng.shuffle(prompt_ids)

        eval_count = int(len(prompt_ids) * eval_ratio)
        eval_ids = set(prompt_ids[:eval_count])

        if split in ("eval", "validation"):
            selected_ids = eval_ids
        elif split == "train":
            selected_ids = set(prompt_ids[eval_count:])
        else:
            raise ValueError("split must be 'train' or 'eval'")

        # Keep all pairs for any selected prompt id to avoid leakage.
        self.pairs = []
        for prompt_id in prompt_ids:
            if prompt_id in selected_ids:
                self.pairs.extend(prompt_to_pairs[prompt_id])

    def __len__(self) -> int:
        return len(self.pairs)

    def _tokenize_with_mask(self, prompt: str, response: str) -> Tuple[List[int], List[int], List[int]]:
        conversation = _format_chatml(prompt, response)
        prompt_prefix = _prompt_prefix(prompt)

        prompt_tokens = self.tokenizer.encode(
            prompt_prefix,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        encoded = self.tokenizer.encode(
            conversation,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )

        prompt_len = min(len(prompt_tokens), len(encoded))
        labels = encoded.copy()
        labels[:prompt_len] = [-100] * prompt_len

        attention_mask = [1] * len(encoded)
        return encoded, attention_mask, labels

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        pair = self.pairs[idx]
        prompt = pair["prompt"]
        response_w = pair["chosen"]
        response_l = pair["rejected"]

        input_ids_w, attn_w, labels_w = self._tokenize_with_mask(prompt, response_w)
        input_ids_l, attn_l, labels_l = self._tokenize_with_mask(prompt, response_l)

        return {
            "input_ids_w": input_ids_w,
            "attention_mask_w": attn_w,
            "labels_w": labels_w,
            "input_ids_l": input_ids_l,
            "attention_mask_l": attn_l,
            "labels_l": labels_l,
        }


@dataclass
class DPODataCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 1

    def _pad(self, sequences: List[List[int]], pad_value: int) -> torch.Tensor:
        if len(sequences) == 0:
            return torch.zeros(0)

        max_len = max(len(seq) for seq in sequences)
        if self.pad_to_multiple_of and self.pad_to_multiple_of > 1:
            max_len = int(math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)

        batch = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_len)
            if length > 0:
                batch[i, :length] = torch.tensor(seq[:length], dtype=torch.long)
        return batch

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids_w = [f["input_ids_w"] for f in features]
        labels_w = [f["labels_w"] for f in features]
        attn_w = [f["attention_mask_w"] for f in features]

        input_ids_l = [f["input_ids_l"] for f in features]
        labels_l = [f["labels_l"] for f in features]
        attn_l = [f["attention_mask_l"] for f in features]

        return {
            "input_ids_w": self._pad(input_ids_w, self.tokenizer.pad_token_id),
            "attention_mask_w": self._pad(attn_w, 0),
            "labels_w": self._pad(labels_w, -100),
            "input_ids_l": self._pad(input_ids_l, self.tokenizer.pad_token_id),
            "attention_mask_l": self._pad(attn_l, 0),
            "labels_l": self._pad(labels_l, -100),
        }
