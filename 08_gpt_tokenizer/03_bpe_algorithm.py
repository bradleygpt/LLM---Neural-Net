"""
03_bpe_algorithm.py
-------------------
Build Byte-Pair Encoding from scratch.

The algorithm in plain English:
  1. Start with a list of tokens (initially just bytes).
  2. Count all adjacent pairs of tokens in the data.
  3. Find the most common pair.
  4. Add a new token to the vocabulary representing that pair.
  5. Replace all occurrences of that pair in the data with the new token.
  6. Repeat steps 2-5 for as many merges as you want.

That's it. Three short functions:
  - get_stats(ids)             -> count adjacent pairs
  - merge(ids, pair, new_id)   -> apply one merge
  - train(text, num_merges)    -> repeat to build a tokenizer

The merges are the LEARNED part. If you train on Shakespeare, you get
merges like ('t', 'h') and ('e', 'r') because those are common in
English. If you train on code, you get ('=', '=') and ('def', ' ').
The merges encode the corpus.

Once trained, encoding/decoding are mechanical:
  - To encode: apply the merges in order until no more apply.
  - To decode: replace each token with the bytes it represents,
              concatenate, and decode UTF-8.
"""


# ---------------------------------------------------------------
# Core algorithm: 3 small functions
# ---------------------------------------------------------------
def get_stats(ids):
    """Count occurrences of every adjacent pair in `ids`.

    Returns a dict {(a, b): count}.

    >>> get_stats([1, 2, 3, 1, 2])
    {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`.

    >>> merge([1, 2, 3, 1, 2, 4], (1, 2), 99)
    [99, 3, 99, 4]
    """
    out = []
    i = 0
    while i < len(ids):
        # If we're at a non-final position and the next two ids match the pair, merge them
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


# ---------------------------------------------------------------
# Demo on a small string
# ---------------------------------------------------------------
print("=" * 60)
print("BPE STEP-BY-STEP DEMO")
print("=" * 60)
print()

text = ("The quick brown fox jumps over the lazy dog. "
        "The quick brown fox is fast. The lazy dog sleeps.")

# Step 1: Convert to bytes
ids = list(text.encode("utf-8"))
print(f"Original text ({len(text)} chars): {text}")
print(f"Initial bytes ({len(ids)} ids): {ids[:30]}...")
print()


# Step 2-5: Do a few merges and show what's happening
NUM_MERGES = 10
vocab_size = 256

merges = {}    # (a, b) -> new_id
print(f"Doing {NUM_MERGES} merges...\n")
for i in range(NUM_MERGES):
    stats = get_stats(ids)
    if not stats:
        break
    # The most common pair
    top_pair = max(stats, key=stats.get)
    new_id = vocab_size + i
    ids = merge(ids, top_pair, new_id)
    merges[top_pair] = new_id

    # Show what got merged
    a_bytes = bytes([top_pair[0]]) if top_pair[0] < 256 else f"<{top_pair[0]}>".encode()
    b_bytes = bytes([top_pair[1]]) if top_pair[1] < 256 else f"<{top_pair[1]}>".encode()
    try:
        a_str = a_bytes.decode("utf-8")
        b_str = b_bytes.decode("utf-8")
        meaning = f"{a_str!r} + {b_str!r}"
    except UnicodeDecodeError:
        meaning = f"bytes {top_pair[0]} + {top_pair[1]}"
    print(f"  merge {i+1:2d}: pair {top_pair} -> id {new_id}  ({meaning}, count was {stats[top_pair]})")

print()
print(f"After {NUM_MERGES} merges:")
print(f"  vocab size:    {vocab_size + NUM_MERGES}")
print(f"  sequence len:  {len(ids)} (down from {len(text.encode('utf-8'))} bytes)")
print(f"  compression:   {len(text.encode('utf-8')) / len(ids):.2f}x")
print()


# ---------------------------------------------------------------
# Encode using the trained merges
# ---------------------------------------------------------------
def encode(text, merges):
    """Apply the merges to a new piece of text."""
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        # For each adjacent pair, look up its merge index. We want to
        # apply merges in the order they were learned, so we pick the
        # pair with the LOWEST merge index that's actually present.
        stats = get_stats(ids)
        # Find the pair that has the earliest merge index (highest priority)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # no more applicable merges
        ids = merge(ids, pair, merges[pair])
    return ids


# Test on a new string (not in training)
test_string = "The quick fox runs."
encoded = encode(test_string, merges)
print(f"Encoding a NEW string: {test_string!r}")
print(f"  Bytes:    {len(test_string.encode('utf-8'))} -> {list(test_string.encode('utf-8'))}")
print(f"  Encoded:  {len(encoded)} tokens -> {encoded}")
print()


# ---------------------------------------------------------------
# Decode (reverse the merges)
# ---------------------------------------------------------------
# Build the inverse: id -> bytes
def build_vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in merges.items():
        vocab[new_id] = vocab[a] + vocab[b]
    return vocab


def decode(ids, vocab):
    """Reverse the encoding: ids -> bytes -> string."""
    raw = b"".join(vocab[i] for i in ids)
    return raw.decode("utf-8", errors="replace")


vocab = build_vocab(merges)
decoded = decode(encoded, vocab)
print(f"Decoded back: {decoded!r}")
print(f"Round-trip works: {decoded == test_string}")
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
print("""
============================================================
BPE IN 30 LINES OF PYTHON
============================================================

The whole BPE algorithm is three short functions:
  - get_stats(ids):     count adjacent pairs
  - merge(ids, pair, new_id): replace pair with new id
  - encode(text):       apply learned merges to new text

Training is just: do those 3 things in a loop until you have enough
tokens. That's it. That's the entire algorithm behind every modern
LLM tokenizer.

GPT-2's tokenizer was trained the same way, on a massive web corpus,
for 49,744 merges. Every "token" in GPT-2 is either:
  - A raw byte (0-255), or
  - A merge of two earlier tokens

Some of the most common merges in GPT-2:
  - " the" (space + 'the')
  - "ing" (common suffix)
  - "tion" (another common suffix)
  - " of"  (space + 'of')

Notice: many of GPT's tokens INCLUDE the leading space. That's because
the actual training was done on text where word boundaries are spaces,
so the BPE algorithm naturally bound spaces to the words that follow
them. This is why GPT tokenizers behave weirdly with leading whitespace.

Next: train on a real corpus and see what merges actually emerge.
""")
