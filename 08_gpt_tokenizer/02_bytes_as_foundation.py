"""
02_bytes_as_foundation.py
-------------------------
Before we build BPE, we need to choose what to operate ON. Two options:

  Option A: Operate on Unicode CODE POINTS.
    A string in Python is a sequence of Unicode code points. Each
    character is one int from 0 to 0x10FFFF (over 1.1 million).
    Problem: huge vocabulary just to represent the alphabet, and
    unfamiliar characters (Chinese, emoji) all become brand new tokens.

  Option B: Operate on UTF-8 BYTES.
    Encode the string to UTF-8 bytes first. Now every string is a
    sequence of integers in [0, 256). The "alphabet" is fixed at 256.
    Multi-byte Unicode characters (Chinese, emoji) become 2-4 bytes
    each, but the algorithm doesn't care -- it just sees integers.

GPT-2 chose Option B. So do GPT-3, GPT-4, Claude, and most modern
LLMs. We do the same.

The reason: a 256-symbol alphabet means BPE starts simple and grows
as needed. It also means EVERY possible string is representable, even
ones with characters we've never seen before. No out-of-vocabulary
problems, ever.

The cost: characters outside ASCII take multiple tokens until BPE
merges them. "café" is 4 characters but 5 bytes (the é is 2 bytes
in UTF-8). Until BPE learns that those bytes go together, "café" is
5 tokens. After learning, maybe 1-2.
"""

# ---------------------------------------------------------------
# Quick UTF-8 refresher
# ---------------------------------------------------------------
# UTF-8 encodes each Unicode code point into 1 to 4 bytes:
#   - ASCII chars (a-z, 0-9, common punctuation): 1 byte each
#   - Latin extended (é, ñ, ü, accented chars):   2 bytes each
#   - CJK chars (Chinese, Japanese, Korean):      3 bytes each
#   - Emoji and rare chars:                       4 bytes each


def show_encoding(s):
    """Print how a string encodes to UTF-8 bytes."""
    encoded = s.encode("utf-8")
    print(f"{s!r}")
    print(f"  Length in chars: {len(s)}")
    print(f"  Length in bytes: {len(encoded)}")
    print(f"  Byte values:     {list(encoded)}")
    print()


print("=" * 60)
print("UTF-8 ENCODING EXAMPLES")
print("=" * 60)
print()

# Simple ASCII -- 1 byte per char
show_encoding("hello")

# Mixed -- accented chars take 2 bytes
show_encoding("café")

# Chinese -- 3 bytes per char
show_encoding("你好")

# Emoji -- 4 bytes per char
show_encoding("👋")

# Combined
show_encoding("Hello 👋 你好 café")


# ---------------------------------------------------------------
# Roundtrip verification
# ---------------------------------------------------------------
print("=" * 60)
print("ROUND-TRIP: bytes can recover the original string")
print("=" * 60)
print()

original = "Hello, 世界! 👋"
encoded = original.encode("utf-8")
decoded = encoded.decode("utf-8")
print(f"Original: {original!r}")
print(f"Encoded:  {list(encoded)}")
print(f"Decoded:  {decoded!r}")
print(f"Identical: {original == decoded}")
print()


# ---------------------------------------------------------------
# Why bytes-as-input matters for BPE
# ---------------------------------------------------------------
print("=" * 60)
print("STARTING ALPHABET: 256 BYTES")
print("=" * 60)
print()
print("Every BPE tokenizer starts with vocabulary {0, 1, 2, ..., 255}.")
print("Each integer represents one possible byte value.")
print()
print("From there, BPE adds ONE NEW TOKEN AT A TIME by finding the most")
print("common adjacent pair of existing tokens and merging them. Each")
print("merge increases the vocabulary by 1.")
print()
print("Starting vocab:  256")
print("After 1 merge:   257  (e.g., 'th' = bytes (116, 104) merged)")
print("After 2 merges:  258  (e.g., 'he' = bytes (104, 101) merged)")
print("After N merges:  256 + N")
print()
print("To get a 50,000-token vocabulary like GPT-2, you do 49,744 merges.")
print("To get the 100,000+ tokens of GPT-4, more.")
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
print("""
============================================================
THE FOUNDATION
============================================================

Strings -> UTF-8 bytes -> sequence of integers in [0, 256).
That sequence is the input to BPE.

This works for ANY string -- English, Chinese, emoji, code, math
symbols, things that don't yet exist. There is no out-of-vocabulary
problem at the byte level. There can't be. Every string in existence
is a sequence of bytes.

The 256 starting tokens are special: they're the fallback. If BPE
hasn't merged some sequence into a single token, the model still
processes it as raw bytes. No string is unrepresentable.

Next: implement BPE itself.
""")
