"""BPE Tokenizer: encode/decode using a vocabulary and ordered merges."""

from __future__ import annotations

import json
import os
from typing import Iterable, Iterator

import regex as re

from .bpe import GPT2_PRETOKEN_PAT

_PRETOKEN_RE = re.compile(GPT2_PRETOKEN_PAT)


class Tokenizer:
    """Byte-level BPE tokenizer.

    Constructed from a vocabulary `dict[int, bytes]`, an ordered list of merges
    `list[tuple[bytes, bytes]]`, and an optional list of special token strings.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        # Make sure all special tokens are in the vocab.
        self.special_tokens: list[str] = list(special_tokens) if special_tokens else []
        existing_values = set(self.vocab.values())
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in existing_values:
                next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
                self.vocab[next_id] = tok_bytes
                existing_values.add(tok_bytes)

        # Reverse vocab (bytes -> id) for fast encoding.
        self.bytes_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # Map (b1, b2) -> rank (lower = applied earlier).
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

        # Map special-token string -> id (resolved through bytes_to_id).
        self.special_to_id: dict[str, int] = {
            tok: self.bytes_to_id[tok.encode("utf-8")] for tok in self.special_tokens
        }

        # Pre-compile the special-tokens splitter regex once. Sort by length
        # descending so overlapping specials (e.g. "<|eot|><|eot|>" vs "<|eot|>")
        # prefer the longer match.
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            self._special_split_re = re.compile(
                "(" + "|".join(re.escape(t) for t in sorted_specials) + ")"
            )
        else:
            self._special_split_re = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Load a serialized vocab and merges produced by `save_to_files` (below)."""
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        # Stored format: {str(int_id): hex_string_of_token_bytes}
        vocab = {int(k): bytes.fromhex(v) for k, v in raw_vocab.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                hex1, hex2 = line.split(" ")
                merges.append((bytes.fromhex(hex1), bytes.fromhex(hex2)))

        return cls(vocab, merges, special_tokens)

    def save_to_files(
        self,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
    ) -> None:
        """Serialize vocab and merges to disk in a hex-based, JSON-friendly format."""
        with open(vocab_filepath, "w", encoding="utf-8") as f:
            json.dump({str(k): v.hex() for k, v in self.vocab.items()}, f)
        with open(merges_filepath, "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a.hex()} {b.hex()}\n")

    # ------------------------------------------------------------------
    # BPE encoding for a single pre-token (sequence of single bytes)
    # ------------------------------------------------------------------
    def _bpe_encode_pretoken(self, pretoken_bytes: bytes) -> list[int]:
        """Apply learned merges to a single pre-token's byte sequence."""
        if not pretoken_bytes:
            return []
        # Start with each byte as its own `bytes`-of-length-1 token.
        parts: list[bytes] = [bytes([b]) for b in pretoken_bytes]

        # Repeatedly find the pair with the lowest merge rank and merge it
        # (leftmost occurrence first when ranks tie).
        while len(parts) >= 2:
            best_rank = None
            best_idx = -1
            for i in range(len(parts) - 1):
                rank = self.merge_ranks.get((parts[i], parts[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_idx == -1:
                break
            merged = parts[best_idx] + parts[best_idx + 1]
            parts = parts[:best_idx] + [merged] + parts[best_idx + 2 :]

        return [self.bytes_to_id[p] for p in parts]

    # ------------------------------------------------------------------
    # Public encode / decode
    # ------------------------------------------------------------------
    def _encode_chunk_no_special(self, text: str) -> list[int]:
        """Encode a chunk known to contain no special tokens."""
        if not text:
            return []
        ids: list[int] = []
        for m in _PRETOKEN_RE.finditer(text):
            pretoken = m.group()
            pretoken_bytes = pretoken.encode("utf-8")
            ids.extend(self._bpe_encode_pretoken(pretoken_bytes))
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode a string to a list of token IDs."""
        if not text:
            return []
        if self._special_split_re is None:
            return self._encode_chunk_no_special(text)

        # re.split with a capturing group keeps the matched delimiters in the
        # resulting list, alternating regular text and special tokens.
        parts = self._special_split_re.split(text)
        ids: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
            else:
                ids.extend(self._encode_chunk_no_special(part))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode a stream of strings (e.g. an open file handle).

        Memory strategy:
          - If we have special tokens, chunk the stream at every special-token
            occurrence (so memory stays bounded for inputs like TinyStories
            which separate documents with <|endoftext|>).
          - Otherwise, fall back to one chunk per yielded string from the
            iterable. The standard iteration over a Python file handle yields
            one line at a time, which keeps peak memory low.
        """
        if self._special_split_re is None:
            for piece in iterable:
                if not piece:
                    continue
                for tok_id in self._encode_chunk_no_special(piece):
                    yield tok_id
            return

        buffer = ""
        for piece in iterable:
            if not piece:
                continue
            buffer += piece

            # Flush every complete (content, special-token) pair currently in buffer.
            last_end = 0
            for m in self._special_split_re.finditer(buffer):
                content = buffer[last_end : m.start()]
                for tok_id in self._encode_chunk_no_special(content):
                    yield tok_id
                yield self.special_to_id[m.group()]
                last_end = m.end()
            buffer = buffer[last_end:]

        # Final tail with no further special tokens.
        if buffer:
            for tok_id in self._encode_chunk_no_special(buffer):
                yield tok_id

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to a string."""
        # Concatenate raw bytes, then UTF-8 decode with replacement of any
        # malformed sequences (U+FFFD).
        byte_pieces = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        return byte_pieces.decode("utf-8", errors="replace")
