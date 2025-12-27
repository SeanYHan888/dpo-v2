from typing import Tuple

import torch
import torch.nn.functional as F


def _sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes average log-probability per sequence, ignoring tokens with label == -100.
    """
    # Shift for next-token prediction.
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()

    log_probs = F.log_softmax(logits, dim=-1)
    mask = labels != -100

    safe_labels = labels.clone()
    safe_labels[~mask] = 0

    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * mask

    lengths = mask.sum(dim=-1).clamp(min=1)
    seq_log_prob = token_log_probs.sum(dim=-1) / lengths
    return seq_log_prob


def dpo_loss(
    policy_logits_w: torch.Tensor,
    policy_logits_l: torch.Tensor,
    ref_logits_w: torch.Tensor,
    ref_logits_l: torch.Tensor,
    labels_w: torch.Tensor,
    labels_l: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the DPO loss and reward margins.
    Returns:
        loss: scalar tensor
        reward_margin: tensor of shape (batch,)
    """
    log_pi_w = _sequence_log_probs(policy_logits_w, labels_w)
    log_pi_l = _sequence_log_probs(policy_logits_l, labels_l)
    log_ref_w = _sequence_log_probs(ref_logits_w, labels_w)
    log_ref_l = _sequence_log_probs(ref_logits_l, labels_l)

    reward_margin = (log_pi_w - log_ref_w) - (log_pi_l - log_ref_l)
    loss = -F.logsigmoid(beta * reward_margin).mean()
    return loss, reward_margin
