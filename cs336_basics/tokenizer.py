"""
BPE Tokenizer Implementation for CS336 Assignment 1.
"""
from __future__ import annotations

import os
import json
import regex as re
from collections import Counter
from typing import Iterator
from collections.abc import Iterable
import multiprocessing
from functools import partial


# GPT-2 style pre-tokenization pattern
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    """
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def pretokenize_chunk(chunk: str, special_tokens: list[str]) -> Counter:
    """Pre-tokenize a chunk of text and return counts of pre-tokens."""
    counts: Counter = Counter()

    # Split on special tokens first
    if special_tokens:
        # Escape special tokens for regex and join with |
        escaped_tokens = [re.escape(t) for t in special_tokens]
        split_pattern = "|".join(escaped_tokens)
        parts = re.split(f"({split_pattern})", chunk)
    else:
        parts = [chunk]

    # Process each part
    for part in parts:
        if not part:
            continue
        if special_tokens and part in special_tokens:
            # Skip special tokens - they don't participate in BPE training
            continue
        # Apply GPT-2 pre-tokenization pattern
        for match in re.finditer(GPT2_PAT, part):
            token = match.group()
            token_bytes = tuple(bytes([b]) for b in token.encode("utf-8"))
            counts[token_bytes] += 1

    return counts


def process_chunk_for_pretokenization(args: tuple, special_tokens: list[str]) -> Counter:
    """Process a single chunk for pre-tokenization (for multiprocessing)."""
    filepath, start, end = args
    with open(filepath, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return pretokenize_chunk(chunk, special_tokens)


def get_pair_counts(pretokens: dict[tuple[bytes, ...], int]) -> Counter:
    """Count all adjacent pairs in pre-tokens."""
    pair_counts: Counter = Counter()
    for token_seq, count in pretokens.items():
        if len(token_seq) < 2:
            continue
        for i in range(len(token_seq) - 1):
            pair = (token_seq[i], token_seq[i + 1])
            pair_counts[pair] += count
    return pair_counts


def merge_pair(
    pretokens: dict[tuple[bytes, ...], int],
    pair: tuple[bytes, bytes],
    new_token: bytes
) -> dict[tuple[bytes, ...], int]:
    """Merge a pair of tokens into a new token in all pre-tokens."""
    new_pretokens: dict[tuple[bytes, ...], int] = {}

    for token_seq, count in pretokens.items():
        new_seq: list[bytes] = []
        i = 0
        while i < len(token_seq):
            if i < len(token_seq) - 1 and token_seq[i] == pair[0] and token_seq[i + 1] == pair[1]:
                new_seq.append(new_token)
                i += 2
            else:
                new_seq.append(token_seq[i])
                i += 1
        new_pretokens[tuple(new_seq)] = new_pretokens.get(tuple(new_seq), 0) + count

    return new_pretokens


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Args:
        input_path: Path to the training corpus.
        vocab_size: Maximum vocabulary size (including special tokens and 256 bytes).
        special_tokens: List of special tokens to add to vocabulary.
        num_processes: Number of processes for parallel pre-tokenization.

    Returns:
        vocab: Mapping from token ID to bytes.
        merges: List of (token1, token2) merge operations in order.
    """
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)

    # Step 1: Initialize vocabulary with special tokens and all bytes
    vocab: dict[int, bytes] = {}
    next_id = 0

    # Add special tokens first
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    # Add all 256 byte values
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1

    # Step 2: Pre-tokenize the corpus with parallel processing
    special_token_bytes = special_tokens[0].encode("utf-8") if special_tokens else b""

    with open(input_path, "rb") as f:
        if num_processes > 1 and special_token_bytes:
            boundaries = find_chunk_boundaries(f, num_processes, special_token_bytes)
        else:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            boundaries = [0, file_size]

    # Create chunks
    chunks = [(str(input_path), start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

    # Process chunks in parallel
    if num_processes > 1 and len(chunks) > 1:
        process_func = partial(process_chunk_for_pretokenization, special_tokens=special_tokens)
        with multiprocessing.Pool(num_processes) as pool:
            chunk_counts = pool.map(process_func, chunks)

        # Merge all counts
        pretokens: dict[tuple[bytes, ...], int] = {}
        for counts in chunk_counts:
            for token_seq, count in counts.items():
                pretokens[token_seq] = pretokens.get(token_seq, 0) + count
    else:
        # Serial processing
        pretokens = {}
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        counts = pretokenize_chunk(text, special_tokens)
        for token_seq, count in counts.items():
            pretokens[token_seq] = pretokens.get(token_seq, 0) + count

    # Step 3: Compute BPE merges
    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(special_tokens) - 256

    # Get initial pair counts
    pair_counts = get_pair_counts(pretokens)

    for _ in range(num_merges):
        if not pair_counts:
            break

        # Find most frequent pair, breaking ties lexicographically
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
        best_count = pair_counts[best_pair]

        if best_count < 1:
            break

        # Create new token
        new_token = best_pair[0] + best_pair[1]

        # Add to vocabulary
        vocab[next_id] = new_token
        next_id += 1

        # Add to merges
        merges.append(best_pair)

        # Update pretokens and pair counts incrementally
        new_pretokens: dict[tuple[bytes, ...], int] = {}
        pair_deltas: Counter = Counter()

        for token_seq, count in pretokens.items():
            if len(token_seq) < 2:
                new_pretokens[token_seq] = new_pretokens.get(token_seq, 0) + count
                continue

            # Check if pair exists in this sequence
            has_pair = False
            for i in range(len(token_seq) - 1):
                if token_seq[i] == best_pair[0] and token_seq[i + 1] == best_pair[1]:
                    has_pair = True
                    break

            if not has_pair:
                new_pretokens[token_seq] = new_pretokens.get(token_seq, 0) + count
                continue

            # Build new sequence and track pair count changes
            new_seq: list[bytes] = []
            i = 0
            prev_token = None

            while i < len(token_seq):
                if i < len(token_seq) - 1 and token_seq[i] == best_pair[0] and token_seq[i + 1] == best_pair[1]:
                    # Decrement counts for pairs that are being broken
                    if prev_token is not None:
                        pair_deltas[(prev_token, best_pair[0])] -= count
                    if i + 2 < len(token_seq):
                        pair_deltas[(best_pair[1], token_seq[i + 2])] -= count

                    # Increment counts for new pairs
                    if prev_token is not None:
                        pair_deltas[(prev_token, new_token)] += count
                    if i + 2 < len(token_seq):
                        pair_deltas[(new_token, token_seq[i + 2])] += count

                    new_seq.append(new_token)
                    prev_token = new_token
                    i += 2
                else:
                    new_seq.append(token_seq[i])
                    prev_token = token_seq[i]
                    i += 1

            new_pretokens[tuple(new_seq)] = new_pretokens.get(tuple(new_seq), 0) + count

        pretokens = new_pretokens

        # Update pair counts
        del pair_counts[best_pair]
        for pair, delta in pair_deltas.items():
            if pair in pair_counts:
                pair_counts[pair] += delta
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
            elif delta > 0:
                pair_counts[pair] = delta

    return vocab, merges


class Tokenizer:
    """BPE Tokenizer for encoding and decoding text."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Initialize the tokenizer.

        Args:
            vocab: Mapping from token ID to bytes.
            merges: List of BPE merge operations.
            special_tokens: Optional list of special tokens.
        """
        self.vocab = dict(vocab)  # id -> bytes
        self.inverse_vocab: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.merges = list(merges)
        self.special_tokens = special_tokens or []

        # Add special tokens to vocabulary if not present
        next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.inverse_vocab:
                self.vocab[next_id] = token_bytes
                self.inverse_vocab[token_bytes] = next_id
                next_id += 1

        # Create merge rank lookup for efficient encoding
        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            merge: i for i, merge in enumerate(self.merges)
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        Create a tokenizer from saved vocab and merges files.
        """
        # Load vocab (JSON format: {id_str: base64/hex encoded bytes or list of ints})
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        vocab: dict[int, bytes] = {}
        for id_str, value in vocab_data.items():
            token_id = int(id_str)
            if isinstance(value, list):
                vocab[token_id] = bytes(value)
            elif isinstance(value, str):
                # Try to decode as UTF-8 or use raw bytes
                vocab[token_id] = value.encode("latin-1")
            else:
                vocab[token_id] = bytes(value)

        # Load merges (text format: "token1 token2" per line)
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(" ")
                if len(parts) >= 2:
                    # Decode merge tokens
                    token1 = parts[0].encode("latin-1")
                    token2 = parts[1].encode("latin-1")
                    merges.append((token1, token2))

        return cls(vocab, merges, special_tokens)

    def _apply_bpe(self, token_bytes: tuple[bytes, ...]) -> list[bytes]:
        """Apply BPE merges to a sequence of bytes."""
        if len(token_bytes) <= 1:
            return list(token_bytes)

        tokens = list(token_bytes)

        while len(tokens) >= 2:
            # Find the pair with lowest merge rank
            best_pair = None
            best_rank = float("inf")
            best_idx = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_rank:
                    rank = self.merge_rank[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair
                        best_idx = i

            if best_pair is None:
                break

            # Merge the pair
            new_token = best_pair[0] + best_pair[1]
            tokens = tokens[:best_idx] + [new_token] + tokens[best_idx + 2:]

        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode text into a sequence of token IDs."""
        token_ids: list[int] = []

        # Handle special tokens
        if self.special_tokens:
            # Sort by length descending so longer tokens are matched first
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(t) for t in sorted_tokens]
            split_pattern = "|".join(escaped_tokens)
            parts = re.split(f"({split_pattern})", text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue

            # Check if part is a special token
            if self.special_tokens and part in self.special_tokens:
                token_bytes = part.encode("utf-8")
                if token_bytes in self.inverse_vocab:
                    token_ids.append(self.inverse_vocab[token_bytes])
                continue

            # Pre-tokenize using GPT-2 pattern
            for match in re.finditer(GPT2_PAT, part):
                pretoken = match.group()
                pretoken_bytes = tuple(bytes([b]) for b in pretoken.encode("utf-8"))

                # Apply BPE merges
                merged_tokens = self._apply_bpe(pretoken_bytes)

                # Convert to IDs
                for token in merged_tokens:
                    if token in self.inverse_vocab:
                        token_ids.append(self.inverse_vocab[token])
                    else:
                        # Token not in vocab - should not happen with byte-level BPE
                        # Fall back to individual bytes
                        for b in token:
                            byte_token = bytes([b])
                            if byte_token in self.inverse_vocab:
                                token_ids.append(self.inverse_vocab[byte_token])

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of strings, yielding token IDs lazily."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        byte_sequence = b""
        for token_id in ids:
            if token_id in self.vocab:
                byte_sequence += self.vocab[token_id]

        # Decode with replacement for invalid UTF-8
        return byte_sequence.decode("utf-8", errors="replace")


def save_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str,
    merges_path: str,
):
    """Save tokenizer vocab and merges to files."""
    # Save vocab as JSON
    vocab_data = {str(k): list(v) for k, v in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=2)

    # Save merges as text
    with open(merges_path, "w", encoding="utf-8") as f:
        for token1, token2 in merges:
            f.write(f"{token1.decode('latin-1')} {token2.decode('latin-1')}\n")
