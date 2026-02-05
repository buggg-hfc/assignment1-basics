from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

# Import implementations
from cs336_basics.model import (
    Linear,
    Embedding,
    RMSNorm,
    silu,
    softmax,
    SwiGLU,
    scaled_dot_product_attention,
    RotaryPositionalEmbedding,
    MultiHeadSelfAttention,
    MultiHeadSelfAttentionWithRoPE,
    TransformerBlock,
    TransformerLM,
    cross_entropy_loss,
    gradient_clipping,
)
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.training import get_batch, save_checkpoint, load_checkpoint
from cs336_basics.tokenizer import train_bpe, Tokenizer


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.
    """
    linear = Linear(d_in, d_out)
    linear.weight.data = weights
    return linear(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.
    """
    embedding = Embedding(vocab_size, d_model)
    embedding.weight.data = weights
    return embedding(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.
    """
    swiglu = SwiGLU(d_model, d_ff)
    swiglu.w1.weight.data = w1_weight
    swiglu.w2.weight.data = w2_weight
    swiglu.w3.weight.data = w3_weight
    return swiglu(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.
    """
    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This function should not use RoPE.
    """
    mha = MultiHeadSelfAttention(d_model, num_heads)
    mha.q_proj.weight.data = q_proj_weight
    mha.k_proj.weight.data = k_proj_weight
    mha.v_proj.weight.data = v_proj_weight
    mha.output_proj.weight.data = o_proj_weight
    return mha(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This version of MHA should include RoPE.
    """
    mha = MultiHeadSelfAttentionWithRoPE(
        d_model, num_heads, max_seq_len, theta, device=in_features.device
    )
    mha.q_proj.weight.data = q_proj_weight
    mha.k_proj.weight.data = k_proj_weight
    mha.v_proj.weight.data = v_proj_weight
    mha.output_proj.weight.data = o_proj_weight
    return mha(in_features, token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.
    """
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=in_query_or_key.device)
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.
    """
    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
        device=in_features.device,
    )

    # Load weights
    block.attn.q_proj.weight.data = weights["attn.q_proj.weight"]
    block.attn.k_proj.weight.data = weights["attn.k_proj.weight"]
    block.attn.v_proj.weight.data = weights["attn.v_proj.weight"]
    block.attn.output_proj.weight.data = weights["attn.output_proj.weight"]
    block.ln1.weight.data = weights["ln1.weight"]
    block.ffn.w1.weight.data = weights["ffn.w1.weight"]
    block.ffn.w2.weight.data = weights["ffn.w2.weight"]
    block.ffn.w3.weight.data = weights["ffn.w3.weight"]
    block.ln2.weight.data = weights["ln2.weight"]

    return block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.
    """
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=in_indices.device,
    )

    # Load weights
    model.token_embeddings.weight.data = weights["token_embeddings.weight"]
    model.ln_final.weight.data = weights["ln_final.weight"]
    model.lm_head.weight.data = weights["lm_head.weight"]

    for i in range(num_layers):
        layer = model.layers[i]
        layer.attn.q_proj.weight.data = weights[f"layers.{i}.attn.q_proj.weight"]
        layer.attn.k_proj.weight.data = weights[f"layers.{i}.attn.k_proj.weight"]
        layer.attn.v_proj.weight.data = weights[f"layers.{i}.attn.v_proj.weight"]
        layer.attn.output_proj.weight.data = weights[f"layers.{i}.attn.output_proj.weight"]
        layer.ln1.weight.data = weights[f"layers.{i}.ln1.weight"]
        layer.ffn.w1.weight.data = weights[f"layers.{i}.ffn.w1.weight"]
        layer.ffn.w2.weight.data = weights[f"layers.{i}.ffn.w2.weight"]
        layer.ffn.w3.weight.data = weights[f"layers.{i}.ffn.w3.weight"]
        layer.ln2.weight.data = weights[f"layers.{i}.ln2.weight"]

    return model(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.
    """
    rmsnorm = RMSNorm(d_model, eps=eps)
    rmsnorm.weight.data = weights
    return rmsnorm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.
    """
    return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.
    """
    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.
    """
    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.
    """
    return cross_entropy_loss(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.
    """
    gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.
    """
    return get_lr_cosine_schedule(
        t=it,
        max_lr=max_learning_rate,
        min_lr=min_learning_rate,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    )


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.
    """
    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.
    """
    result = load_checkpoint(src, model, optimizer)
    return result["iteration"]


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.
    """
    return train_bpe(input_path, vocab_size, special_tokens, **kwargs)
