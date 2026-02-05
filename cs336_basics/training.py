"""
Training utilities for CS336 Assignment 1.
Includes data loading, checkpointing, and generation.
"""
from __future__ import annotations

import os
import numpy as np
import torch
from pathlib import Path
from typing import Any


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of training examples from the data.

    Args:
        data: 1D numpy array of token IDs.
        batch_size: Number of sequences in the batch.
        context_length: Length of each sequence (number of tokens).
        device: Device to place the tensors on.

    Returns:
        x: Input tensor of shape (batch_size, context_length).
        y: Target tensor of shape (batch_size, context_length).
           y[i, j] = data[start + j + 1] for each sequence.
    """
    # Calculate maximum valid starting index
    # Need context_length + 1 tokens for each sequence (context + 1 for target)
    max_start = len(data) - context_length - 1

    if max_start < 0:
        raise ValueError(
            f"Data length ({len(data)}) is too short for context_length ({context_length})"
        )

    # Randomly sample starting indices
    start_indices = np.random.randint(0, max_start + 1, size=batch_size)

    # Extract sequences
    x = np.zeros((batch_size, context_length), dtype=np.int64)
    y = np.zeros((batch_size, context_length), dtype=np.int64)

    for i, start in enumerate(start_indices):
        x[i] = data[start : start + context_length]
        y[i] = data[start + 1 : start + context_length + 1]

    # Convert to tensors and move to device
    x_tensor = torch.from_numpy(x).to(device)
    y_tensor = torch.from_numpy(y).to(device)

    return x_tensor, y_tensor


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out_path: str | os.PathLike,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """
    Save a training checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        iteration: Current training iteration.
        out_path: Path to save the checkpoint.
        extra_state: Optional additional state to save.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }

    if extra_state is not None:
        checkpoint["extra_state"] = extra_state

    # Create parent directories if they don't exist
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, out_path)


def load_checkpoint(
    checkpoint_path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model to load state into.
        optimizer: Optional optimizer to load state into.

    Returns:
        Dictionary containing:
            - "iteration": The saved iteration number.
            - "extra_state": Any extra state that was saved (or None).
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "iteration": checkpoint.get("iteration", 0),
        "extra_state": checkpoint.get("extra_state", None),
    }


def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    Generate new tokens from a language model.

    Args:
        model: The language model.
        input_ids: Initial token IDs of shape (batch_size, seq_len).
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Temperature for sampling (higher = more random).
        top_k: If set, only sample from top k most likely tokens.
        top_p: If set, sample from smallest set of tokens with cumulative prob >= top_p.
        eos_token_id: If set, stop generation when this token is generated.

    Returns:
        Generated token IDs including the input, shape (batch_size, seq_len + num_generated).
    """
    model.eval()
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get model predictions for the next token
            logits = model(generated)  # (batch_size, seq_len, vocab_size)

            # Only need logits for the last position
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                # Get the top k values and their indices
                top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                # Set all values below the kth largest to -inf
                threshold = top_k_values[:, -1, None]
                next_token_logits = torch.where(
                    next_token_logits < threshold,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits = torch.where(
                    indices_to_remove,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS token
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break

    return generated
