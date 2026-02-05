"""
AdamW Optimizer and Learning Rate Schedule for CS336 Assignment 1.
"""
from __future__ import annotations

import math
from typing import Iterable
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    AdamW optimizer with weight decay decoupled from gradient updates.

    This follows the algorithm from "Decoupled Weight Decay Regularization"
    (Loshchilov & Hutter, 2019).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """
        Initialize AdamW optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            betas: Coefficients for computing running averages of gradient and its square.
            eps: Term added to denominator for numerical stability.
            weight_decay: Weight decay coefficient.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Loss value if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Get or initialize state for this parameter
                state = self.state[p]

                if len(state) == 0:
                    # Initialize state
                    state["step"] = 0
                    # First moment estimate (exponential moving average of gradients)
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Second moment estimate (exponential moving average of squared gradients)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                t = state["step"]

                m = state["m"]
                v = state["v"]

                # Update biased first moment estimate
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                # Compute bias-corrected estimates
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                # Compute update (Adam step)
                update = m_hat / (torch.sqrt(v_hat) + eps)

                # Apply weight decay (decoupled)
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                # Apply update
                p.data.add_(update, alpha=-lr)

        return loss


def get_lr_cosine_schedule(
    t: int,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Get learning rate at step t using cosine schedule with linear warmup.

    Args:
        t: Current iteration (0-indexed).
        max_lr: Maximum learning rate (reached after warmup).
        min_lr: Minimum learning rate (reached at end of cosine cycle).
        warmup_iters: Number of warmup iterations.
        cosine_cycle_iters: Total number of iterations for the cosine cycle
                           (including warmup).

    Returns:
        Learning rate at step t.
    """
    # Linear warmup phase: t=0 -> 0, t=warmup_iters -> max_lr
    if t < warmup_iters:
        return max_lr * t / warmup_iters

    # At warmup_iters, return max_lr
    if t == warmup_iters:
        return max_lr

    # Cosine decay phase
    if t < cosine_cycle_iters:
        # Progress through the cosine phase (0 to 1)
        progress = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        # Cosine decay from max_lr to min_lr
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

    # After cosine cycle, return min_lr
    return min_lr


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip gradients by L2 norm.

    If the L2 norm of the gradients exceeds max_l2_norm, scale all gradients
    so their combined L2 norm equals max_l2_norm.

    Args:
        parameters: Iterable of parameters whose gradients to clip.
        max_l2_norm: Maximum allowed L2 norm for gradients.
    """
    # Convert to list so we can iterate multiple times
    params = [p for p in parameters if p.grad is not None]

    if len(params) == 0:
        return

    # Compute total L2 norm of all gradients
    total_norm_sq = 0.0
    for p in params:
        total_norm_sq += p.grad.data.pow(2).sum().item()

    total_norm = math.sqrt(total_norm_sq)

    # Clip if necessary
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / total_norm
        for p in params:
            p.grad.data.mul_(clip_coef)
