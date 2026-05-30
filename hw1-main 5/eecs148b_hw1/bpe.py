"""Byte-Pair Encoding (BPE) tokenizer training.

Implements byte-level BPE training as described in Sennrich et al. (2016) /
Radford et al. (2019, GPT-2). We pre-tokenize with the GPT-2 regex pattern
and learn merges over the resulting byte sequences.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Iterable

import regex as re  # third-party `regex` (not stdlib `re`) -- needed for \p{L}, \p{N}

# GPT-2 pre-tokenization pattern (see github.com/openai/tiktoken)
GPT2_PRETOKEN_PAT = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

_PRETOKEN_RE = re.compile(GPT2_PRETOKEN_PAT)


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Split text on special tokens so no BPE merge can cross those boundaries."""
    if not special_tokens:
        return [text]
    # Sort by length descending so that longer specials (e.g. "<|eot|><|eot|>")
    # take precedence over shorter ones that are their prefixes.
    sorted_specials = sorted(special_tokens, key=len, reverse=True)
    pat = "|".join(re.escape(t) for t in sorted_specials)
    return re.split(pat, text)


def _count_pretokens(chunks: Iterable[str]) -> Counter:
    """Pre-tokenize each chunk and count occurrences of each pre-token string."""
    counts: Counter = Counter()
    for chunk in chunks:
        if not chunk:
            continue
        for m in _PRETOKEN_RE.finditer(chunk):
            counts[m.group()] += 1
    return counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Returns:
        vocab: dict[int, bytes] mapping token ID -> token bytes.
        merges: list[tuple[bytes, bytes]] in order of creation.
    """
    # ---- 1. Read corpus and split on special tokens ----
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = _split_on_special_tokens(text, special_tokens or [])

    # ---- 2. Pre-tokenize and tally counts ----
    pretoken_counts = _count_pretokens(chunks)

    # Represent every pre-token as a tuple of single-byte `bytes` objects so that
    # merges concatenate into longer byte strings naturally.
    word_freqs: dict[tuple[bytes, ...], int] = {}
    for word_str, count in pretoken_counts.items():
        word_bytes = tuple(bytes([b]) for b in word_str.encode("utf-8"))
        word_freqs[word_bytes] = word_freqs.get(word_bytes, 0) + count

    # ---- 3. Initialize vocab with all 256 single bytes ----
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # ---- 4. Compute the merge budget ----
    num_specials = len(special_tokens or [])
    num_merges = max(0, vocab_size - 256 - num_specials)

    # ---- 5. Index pair counts incrementally ----
    pair_counts: Counter = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            pair_to_words.setdefault(pair, set()).add(word)

    merges: list[tuple[bytes, bytes]] = []

    for _ in range(num_merges):
        if not pair_counts:
            break

        # Pick the most frequent pair; tie-break by lexicographically greater pair
        # (matches the spec: max([("A","B"),("BA","A"),...]) -> ("BA","A")).
        max_count = max(pair_counts.values())
        if max_count <= 0:
            break
        best_pair = max(p for p, c in pair_counts.items() if c == max_count)

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        # Snapshot the affected words; we'll mutate `word_freqs` as we go.
        affected = list(pair_to_words.get(best_pair, ()))
        a, b = best_pair

        for word in affected:
            if word not in word_freqs:
                continue
            freq = word_freqs[word]

            # Build the post-merge word, scanning left-to-right.
            new_word_parts: list[bytes] = []
            i = 0
            L = len(word)
            while i < L:
                if i < L - 1 and word[i] == a and word[i + 1] == b:
                    new_word_parts.append(new_token)
                    i += 2
                else:
                    new_word_parts.append(word[i])
                    i += 1
            new_word = tuple(new_word_parts)

            # If the merge was a no-op for this word (shouldn't happen since the
            # word was in pair_to_words[best_pair]), skip.
            if new_word == word:
                continue

            # Subtract old pair contributions for this word.
            for i in range(L - 1):
                old_pair = (word[i], word[i + 1])
                pair_counts[old_pair] -= freq
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]
                s = pair_to_words.get(old_pair)
                if s is not None:
                    s.discard(word)
                    if not s:
                        pair_to_words.pop(old_pair, None)

            # Move the freq from `word` to `new_word`.
            del word_freqs[word]
            word_freqs[new_word] = word_freqs.get(new_word, 0) + freq

            # Add new pair contributions for the new word.
            for i in range(len(new_word) - 1):
                np_pair = (new_word[i], new_word[i + 1])
                pair_counts[np_pair] += freq
                pair_to_words.setdefault(np_pair, set()).add(new_word)

    # ---- 6. Append special tokens at the END so they have stable IDs ----
    for tok in special_tokens or []:
        vocab[len(vocab)] = tok.encode("utf-8")

    return vocab, merges
