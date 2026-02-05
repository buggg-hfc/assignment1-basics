"""
Transformer Language Model Implementation for CS336 Assignment 1.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, einsum


class Linear(nn.Module):
    """Linear transformation without bias."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight with truncated normal
        # σ² = 2 / (d_in + d_out)
        std = math.sqrt(2.0 / (in_features + out_features))
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: Tensor) -> Tensor:
        """Apply linear transformation: y = xW^T."""
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """Token embedding layer."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize with truncated normal, σ² = 1
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Lookup embeddings for token IDs."""
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Learnable gain parameter, initialized to 1
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMSNorm."""
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # RMS(a) = sqrt(1/d * sum(a_i^2) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and apply gain
        result = (x / rms) * self.weight.to(torch.float32)

        return result.to(in_dtype)


def silu(x: Tensor) -> Tensor:
    """SiLU (Swish) activation function: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax."""
    # Subtract max for numerical stability
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model

        # d_ff ≈ 8/3 * d_model, rounded to multiple of 64
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = ((d_ff + 63) // 64) * 64

        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU: W2(SiLU(W1x) ⊙ W3x)."""
        return self.w2(silu(self.w1(x)) * self.w3(x))


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """
    Scaled dot-product attention.

    Args:
        Q: Query tensor of shape (..., n_queries, d_k)
        K: Key tensor of shape (..., n_keys, d_k)
        V: Value tensor of shape (..., n_keys, d_v)
        mask: Boolean mask of shape (..., n_queries, n_keys).
              True = attend, False = don't attend

    Returns:
        Output tensor of shape (..., n_queries, d_v)
    """
    d_k = Q.shape[-1]

    # Compute attention scores: Q @ K^T / sqrt(d_k)
    # Q: (..., n_queries, d_k), K: (..., n_keys, d_k)
    # scores: (..., n_queries, n_keys)
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    scores = scores / math.sqrt(d_k)

    # Apply mask
    if mask is not None:
        # mask: True = attend, False = don't attend
        scores = scores.masked_fill(~mask, float("-inf"))

    # Softmax over keys dimension
    attn_weights = softmax(scores, dim=-1)

    # Apply attention to values
    # attn_weights: (..., n_queries, n_keys), V: (..., n_keys, d_v)
    # output: (..., n_queries, d_v)
    output = einsum(attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")

    return output


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Precompute cos and sin values
        # θ_{i,k} = i / Θ^((2k-2)/d) for k in {1, ..., d/2}
        # Using 0-indexed: θ_{i,j} = i / Θ^(2j/d) for j in {0, ..., d/2-1}

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        # dim_indices: [0, 1, 2, ..., d_k/2 - 1]
        dim_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        # freqs: Θ^(2j/d) = Θ^(dim_indices / d_k)
        freqs = theta ** (dim_indices / d_k)
        # angles: positions / freqs -> (max_seq_len, d_k/2)
        angles = positions.unsqueeze(1) / freqs.unsqueeze(0)

        # Register as buffers (not parameters)
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        Apply RoPE to input tensor.

        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Position indices of shape (..., seq_len)

        Returns:
            Rotated tensor of same shape as input
        """
        # Get cos and sin for the requested positions
        # cos_cached, sin_cached: (max_seq_len, d_k/2)
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k/2)

        # Split x into pairs for rotation
        # x: (..., seq_len, d_k) -> (..., seq_len, d_k/2, 2)
        x_reshaped = rearrange(x, "... seq_len (d half) -> ... seq_len d half", half=2)

        x1 = x_reshaped[..., 0]  # (..., seq_len, d_k/2)
        x2 = x_reshaped[..., 1]  # (..., seq_len, d_k/2)

        # Apply rotation
        # [cos θ, -sin θ] [x1]   [x1 cos θ - x2 sin θ]
        # [sin θ,  cos θ] [x2] = [x1 sin θ + x2 cos θ]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Combine back
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
        rotated = rearrange(rotated, "... seq_len d half -> ... seq_len (d half)")

        return rotated


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention without RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Projection matrices
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply multi-head self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = rearrange(Q, "batch seq (heads d_k) -> batch heads seq d_k", heads=self.num_heads)
        K = rearrange(K, "batch seq (heads d_k) -> batch heads seq d_k", heads=self.num_heads)
        V = rearrange(V, "batch seq (heads d_v) -> batch heads seq d_v", heads=self.num_heads)

        # Create causal mask
        # mask[i, j] = True if j <= i (can attend)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        # Apply scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask)

        # Reshape back
        # (batch, num_heads, seq_len, d_v) -> (batch, seq_len, d_model)
        attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)")

        # Output projection
        return self.output_proj(attn_output)


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    """Multi-Head Self-Attention with RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Projection matrices
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # RoPE
        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=self.d_k,
            max_seq_len=max_seq_len,
            device=device,
        )

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        """
        Apply multi-head self-attention with RoPE.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            token_positions: Optional position indices of shape (batch, seq_len)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Default positions
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
            token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)

        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = rearrange(Q, "batch seq (heads d_k) -> batch heads seq d_k", heads=self.num_heads)
        K = rearrange(K, "batch seq (heads d_k) -> batch heads seq d_k", heads=self.num_heads)
        V = rearrange(V, "batch seq (heads d_v) -> batch heads seq d_v", heads=self.num_heads)

        # Apply RoPE to Q and K (same rotation for all heads)
        # Expand positions for all heads
        positions_expanded = token_positions.unsqueeze(1).expand(-1, self.num_heads, -1)

        Q = self.rope(Q, positions_expanded)
        K = self.rope(K, positions_expanded)

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        # Apply scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask)

        # Reshape back
        attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)")

        # Output projection
        return self.output_proj(attn_output)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Layer norms
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        # Attention
        self.attn = MultiHeadSelfAttentionWithRoPE(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )

        # Feed-forward network
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            token_positions: Optional position indices

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Pre-norm attention sublayer with residual
        x = x + self.attn(self.ln1(x), token_positions)

        # Pre-norm FFN sublayer with residual
        x = x + self.ffn(self.ln2(x))

        return x


class TransformerLM(nn.Module):
    """Transformer Language Model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Token embeddings
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

        # Output projection (LM head)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        input_ids: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass through the language model.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            token_positions: Optional position indices

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Default positions
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=input_ids.device)
            token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)

        # Token embeddings
        x = self.token_embeddings(input_ids)

        # Transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions)

        # Final norm
        x = self.ln_final(x)

        # Output projection
        logits = self.lm_head(x)

        return logits


def cross_entropy_loss(
    inputs: Tensor,
    targets: Tensor,
) -> Tensor:
    """
    Compute cross-entropy loss.

    Args:
        inputs: Logits of shape (batch_size, vocab_size)
        targets: Target indices of shape (batch_size,)

    Returns:
        Scalar loss value
    """
    # Get logits for correct classes
    batch_size = inputs.shape[0]

    # Numerical stability: subtract max
    max_logits = inputs.max(dim=-1, keepdim=True).values
    shifted_logits = inputs - max_logits

    # Log-sum-exp for normalization
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))

    # Get logits for target classes
    target_logits = shifted_logits[torch.arange(batch_size, device=inputs.device), targets]

    # Cross-entropy: -log(softmax(target))
    # = -target_logit + log_sum_exp
    loss = -target_logits + log_sum_exp

    return loss.mean()


def gradient_clipping(
    parameters,
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    """
    Clip gradients by global L2 norm.

    Args:
        parameters: Iterable of parameters
        max_l2_norm: Maximum allowed L2 norm
        eps: Small value for numerical stability
    """
    # Compute global L2 norm
    total_norm_sq = 0.0
    grads = []

    for param in parameters:
        if param.grad is not None:
            grads.append(param.grad)
            total_norm_sq += param.grad.data.norm(2).item() ** 2

    total_norm = math.sqrt(total_norm_sq)

    # Clip if necessary
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for grad in grads:
            grad.data.mul_(clip_coef)
