"""
Correct Timestep Sampling and CFG Dropout for ACE-Step Training V2

Reimplements ``sample_t_r()`` exactly as defined in the model's own
``forward()`` method (``modeling_acestep_v15_turbo.py`` lines 169-194).

Also provides ``apply_cfg_dropout()`` matching lines 1691-1699 of the
same file.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Continuous logit-normal timestep sampling
# ---------------------------------------------------------------------------

def sample_timesteps(
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
    data_proportion: float = 0.0,
    timestep_mu: float = -0.4,
    timestep_sigma: float = 1.0,
    use_meanflow: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample paired timesteps `t` and `r` for ACE‑Step flow‑matching training.
    
    Parameters:
        batch_size (int): Number of samples in the batch.
        device (torch.device | str): Device for the returned tensors.
        dtype (torch.dtype): Dtype for the returned tensors.
        data_proportion (float): Fraction of the batch treated as "data" (those samples will have `r = t`).
        timestep_mu (float): Mean used for the logit‑normal timestep sampling.
        timestep_sigma (float): Standard deviation used for the logit‑normal timestep sampling.
        use_meanflow (bool): If `False`, `data_proportion` is set to 1.0 (forcing `r = t` for all samples).
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: `(t, r)` tensors each of shape `[batch_size]`, where `t >= r` elementwise and `r` equals `t` for the first `int(batch_size * data_proportion)` samples.
    """
    # Logit-normal sampling via sigmoid(N(mu, sigma))
    t = torch.sigmoid(
        torch.randn((batch_size,), device=device, dtype=dtype) * timestep_sigma + timestep_mu
    )
    r = torch.sigmoid(
        torch.randn((batch_size,), device=device, dtype=dtype) * timestep_sigma + timestep_mu
    )

    # Assign t = max, r = min for each pair
    t, r = torch.maximum(t, r), torch.minimum(t, r)

    # When use_meanflow is False the model forces data_proportion = 1.0,
    # which makes r = t for *every* sample (the zero_mask covers the full
    # batch).
    if not use_meanflow:
        data_proportion = 1.0

    data_size = int(batch_size * data_proportion)
    zero_mask = torch.arange(batch_size, device=device) < data_size
    r = torch.where(zero_mask, t, r)

    return t, r


# ---------------------------------------------------------------------------
# CFG dropout
# ---------------------------------------------------------------------------

def apply_cfg_dropout(
    encoder_hidden_states: torch.Tensor,
    null_condition_emb: torch.Tensor,
    cfg_ratio: float = 0.15,
) -> torch.Tensor:
    """
    Apply per-sample classifier-free guidance dropout by replacing some condition embeddings with a null (unconditional) embedding.
    
    Parameters:
        encoder_hidden_states (torch.Tensor): Condition embeddings of shape [B, L, D].
        null_condition_emb (torch.Tensor): Null (unconditional) embedding; will be expanded to match encoder_hidden_states.
        cfg_ratio (float): Probability in [0, 1] of replacing a sample's condition with the null embedding.
    
    Returns:
        torch.Tensor: Encoder hidden states with the same shape as input where, for each batch item, the embedding is replaced by the expanded null embedding with probability `cfg_ratio`.
    """
    bsz = encoder_hidden_states.shape[0]
    device = encoder_hidden_states.device
    dtype = encoder_hidden_states.dtype

    # Per-sample mask: 0 = drop condition (replace with null), 1 = keep
    full_cfg_condition_mask = torch.where(
        torch.rand(size=(bsz,), device=device, dtype=dtype) < cfg_ratio,
        torch.zeros(size=(bsz,), device=device, dtype=dtype),
        torch.ones(size=(bsz,), device=device, dtype=dtype),
    ).view(-1, 1, 1)

    encoder_hidden_states = torch.where(
        full_cfg_condition_mask > 0,
        encoder_hidden_states,
        null_condition_emb.expand_as(encoder_hidden_states),
    )

    return encoder_hidden_states