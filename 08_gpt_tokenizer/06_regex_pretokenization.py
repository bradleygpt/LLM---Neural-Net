"""
06_regex_pretokenization.py
---------------------------
The basic BPE we built in scripts 03-04 has one problem: it can MERGE
ACROSS WORD BOUNDARIES.

For example, suppose ' the' and ' fox' both appear often. BPE might
notice the pair (' the', ' fox') and merge it into one token ' the fox'
because it appeared together a lot.

That's bad for two reasons:
  1. It bakes in specific phrasing as single tokens, which doesn't
     generalize. ' the fox' is a token but ' the dog' isn't.
  2. The vocabulary fills up with phrase-tokens instead of word-tokens.

GPT-2's fix: BEFORE running BPE, split the text on a REGEX pattern that
isolates words from punctuation and whitespace. Then run BPE on each
chunk independently. Merges can only happen WITHIN a chunk.

The actual GPT-2 regex (from the OpenAI repo):

    's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+

Translated:
  's, 't, 're, 've, 'm, 'll, 'd  -> common contractions
  \\p{L}+                          -> letters
  \\p{N}+                          -> numbers
  [^\\s\\p{L}\\p{N}]+              -> punctuation/symbols
  \\s+(?!\\S)                      -> whitespace at end
  \\s+                             -> any whitespace

The leading " ?" on letters and numbers means: a word can include
ONE leading space. That's why GPT tokens often start with " ".

This script implements this preprocessing step and shows the
difference it makes.
"""

import re


# ---------------------------------------------------------------
# The actual GPT-2 splitting regex (slightly simplified)
# ---------------------------------------------------------------
# Note: \p{L} (any letter) and \p{N} (any number) require the third-
# party `regex` module. Python's built-in `re` only supports ASCII.
# For this lesson we'll use a slightly simpler pattern that uses
# Python's built-in re module. The behavior is almost identical for
# English text.
GPT2_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)


# ---------------------------------------------------------------
# Show how it splits text
# ---------------------------------------------------------------
def split(text):
    return GPT2_PATTERN.findall(text)


# Example text demonstrating various features
samples = [
    "Hello world",
    "Hello, world!",
    "I'm not happy",
    "Don't you think?",
    "It's working!",
    "  multiple   spaces",
    "ROMEO: O Juliet, wherefore art thou?",
    "Number 42, plus 100, equals 142.",
    "End of sentence.   New sentence.",
]

print("=" * 60)
print("REGEX PRE-TOKENIZATION (the GPT-2 split)")
print("=" * 60)
print()
print("Text is split BEFORE BPE runs. Each piece is independently")
print("BPE-tokenized. Merges cannot cross piece boundaries.")
print()

for s in samples:
    pieces = split(s)
    print(f"  Input:  {s!r}")
    print(f"  Pieces: {pieces}")
    print()


# ---------------------------------------------------------------
# Why this is better than naive BPE
# ---------------------------------------------------------------
print("=" * 60)
print("WHY REGEX PRE-TOKENIZATION HELPS")
print("=" * 60)
print()
print("Without it, BPE might create tokens like ' the fox' or 'cat. The'")
print("(spanning words) just because they appeared together often.")
print()
print("With the regex split, words are isolated BEFORE BPE runs.")
print("Subword tokens always stay within one word (or one whitespace,")
print("or one punctuation cluster).")
print()
print("Notice in the splits above:")
print("  - 'Hello world' splits into ['Hello', ' world']")
print("    The space is BUNDLED with 'world', not a separate piece.")
print("  - 'Don't you think?' splits into [\"Don\", \"'t\", ' you', ' think', '?']")
print("    Contractions are split BEFORE BPE, since 't isn't part of 'Don'.")
print("  - 'multiple   spaces' splits into ['multiple', '   spaces']")
print("    All consecutive spaces stay together as one piece.")
print()


# ---------------------------------------------------------------
# Build a regex-pretok BPE tokenizer
# ---------------------------------------------------------------
def get_stats(ids, counts=None):
    counts = counts or {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class RegexTokenizer:
    """BPE tokenizer with regex pre-tokenization (GPT-2 style)."""

    def __init__(self, pattern=None):
        self.pattern = pattern or GPT2_PATTERN
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size):
        num_merges = vocab_size - 256

        # Pre-tokenize
        chunks = self.pattern.findall(text)
        # Convert each chunk to a list of byte IDs
        ids_list = [list(ch.encode("utf-8")) for ch in chunks]

        for i in range(num_merges):
            # Count pairs ACROSS ALL CHUNKS
            stats = {}
            for ids in ids_list:
                get_stats(ids, stats)
            if not stats:
                break
            top_pair = max(stats, key=stats.get)
            new_id = 256 + i
            # Apply the merge to every chunk
            ids_list = [merge(ids, top_pair, new_id) for ids in ids_list]
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

    def encode(self, text):
        chunks = self.pattern.findall(text)
        all_ids = []
        for ch in chunks:
            ids = list(ch.encode("utf-8"))
            while len(ids) >= 2:
                stats = get_stats(ids)
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                ids = merge(ids, pair, self.merges[pair])
            all_ids.extend(ids)
        return all_ids

    def decode(self, ids):
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------
# Train and demo
# ---------------------------------------------------------------
import os
SHAKESPEARE_PATH = "../07_build_gpt/input.txt"
if os.path.exists(SHAKESPEARE_PATH):
    with open(SHAKESPEARE_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Training regex BPE on {len(text):,} chars of Shakespeare...")
else:
    text = "The quick brown fox jumps over the lazy dog. " * 1000
    print("Training regex BPE on fallback text...")

regex_tok = RegexTokenizer()
regex_tok.train(text, vocab_size=512)
print(f"Done. {len(regex_tok.merges)} merges learned.")
print()


# Show some merges
print("First 15 merges of regex BPE:")
for (a, b), new_id in list(regex_tok.merges.items())[:15]:
    bytes_repr = regex_tok.vocab[new_id]
    try:
        s = bytes_repr.decode("utf-8")
        print(f"  id {new_id:>4}: {s!r}")
    except UnicodeDecodeError:
        print(f"  id {new_id:>4}: bytes {list(bytes_repr)}")
print()


# Encode/decode test
test = "ROMEO: O Juliet, wherefore art thou? My love is true."
encoded = regex_tok.encode(test)
decoded = regex_tok.decode(encoded)
print(f"Test encoding: {test!r}")
print(f"  -> {len(encoded)} tokens")
print(f"  Round-trip works: {decoded == test}")
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
print("""
============================================================
REGEX-PRETOK BPE = WHAT REAL TOKENIZERS DO
============================================================

The GPT-2 tokenizer is essentially:
  1. Split text on the regex above
  2. UTF-8 encode each chunk
  3. Run BPE within each chunk

The actual GPT-2 tokenizer uses 49,744 merges and handles minor
edge cases we skipped. But the algorithm above produces tokenizations
that look VERY similar to GPT-2's on English text.

Modern improvements (GPT-4, Claude, etc.):
  - More sophisticated regex (handles more languages)
  - Better Unicode normalization
  - Sometimes a "fast" tokenizer using compiled C++ for speed
  - Special "control tokens" (<|endoftext|>, etc.)

But the core algorithm is unchanged. BPE with regex pre-tokenization,
trained on a big text corpus.

Next (final): use the OFFICIAL GPT-2 tokenizer (via tiktoken) and
compare to ours.
""")
