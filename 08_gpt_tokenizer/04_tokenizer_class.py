"""
04_tokenizer_class.py
---------------------
Wrap the BPE algorithm into a clean Tokenizer class with train, encode,
decode, and save/load methods. Then train on Shakespeare and see what
merges emerge.

This is essentially the BasicTokenizer from Karpathy's minbpe repo.
About 60 lines of code -- a complete, working BPE tokenizer.

What you'll see in the trained tokenizer:
  - Common English bigrams: 'th', 'he', 'in', 'er'
  - Common trigrams from Shakespeare specifically: 'the', 'and', 'you'
  - Character names that appear often: 'ROMEO', 'KING', 'MARCIUS'
  - Whole common phrases as eventual single tokens

The corpus shapes the tokenizer.
"""

import os


# ---------------------------------------------------------------
# The 3 core functions (from script 03)
# ---------------------------------------------------------------
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


# ---------------------------------------------------------------
# The Tokenizer class
# ---------------------------------------------------------------
class Tokenizer:
    """A minimal BPE tokenizer.

    After training:
        - self.merges:  dict mapping (a, b) -> new_id
        - self.vocab:   dict mapping id -> bytes representation
    """

    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size, verbose=False):
        """Run BPE for (vocab_size - 256) merges on the given text."""
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            top_pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = merge(ids, top_pair, new_id)
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

            if verbose and (i % 50 == 0 or i == num_merges - 1):
                merged_repr = self.vocab[new_id]
                try:
                    merged_str = merged_repr.decode("utf-8")
                    print(f"  merge {i+1:>4}/{num_merges}: id {new_id} = {merged_str!r} (count={stats[top_pair]})")
                except UnicodeDecodeError:
                    print(f"  merge {i+1:>4}/{num_merges}: id {new_id} = bytes {list(merged_repr)} (count={stats[top_pair]})")

    def encode(self, text):
        """Encode a string into a list of token ids."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Find the merge with the lowest priority (= earliest learned merge)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        """Decode token ids back to a string."""
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path):
        """Save the merges to a text file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"BPE_TOKENIZER_v1\n")
            f.write(f"{len(self.merges)}\n")
            for (a, b), new_id in self.merges.items():
                f.write(f"{a} {b} {new_id}\n")

    def load(self, path):
        """Load merges from a saved file."""
        with open(path, "r", encoding="utf-8") as f:
            assert f.readline().strip() == "BPE_TOKENIZER_v1"
            n = int(f.readline().strip())
            self.merges = {}
            for _ in range(n):
                a, b, new_id = map(int, f.readline().strip().split())
                self.merges[(a, b)] = new_id
        # Rebuild the vocab
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]


# ---------------------------------------------------------------
# Train on Shakespeare
# ---------------------------------------------------------------
SHAKESPEARE_PATH = "../07_build_gpt/input.txt"

if not os.path.exists(SHAKESPEARE_PATH):
    print(f"[note] Couldn't find {SHAKESPEARE_PATH}.")
    print("[note] Will use a small fallback for the demo.")
    text = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?
""" * 200
else:
    with open(SHAKESPEARE_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters of Shakespeare")

print()


# Train the tokenizer
VOCAB_SIZE = 512   # 256 base bytes + 256 merges
print(f"Training BPE tokenizer to vocab size {VOCAB_SIZE} ({VOCAB_SIZE - 256} merges)...")
print()

tok = Tokenizer()
tok.train(text, vocab_size=VOCAB_SIZE, verbose=True)

print()
print(f"Done! Vocab size: {len(tok.vocab)}")
print()


# ---------------------------------------------------------------
# Inspect the most-common merges
# ---------------------------------------------------------------
print("=" * 60)
print("FIRST 20 MERGES (most common patterns in Shakespeare)")
print("=" * 60)
print()
for (a, b), new_id in list(tok.merges.items())[:20]:
    bytes_repr = tok.vocab[new_id]
    try:
        s = bytes_repr.decode("utf-8")
        print(f"  id {new_id:>4}: {s!r}")
    except UnicodeDecodeError:
        print(f"  id {new_id:>4}: bytes {list(bytes_repr)}")
print()


# ---------------------------------------------------------------
# Test encode/decode round-trip
# ---------------------------------------------------------------
print("=" * 60)
print("ENCODE/DECODE TEST")
print("=" * 60)
print()

test = "ROMEO: O Juliet, wherefore art thou? My love is true."
encoded = tok.encode(test)
decoded = tok.decode(encoded)
print(f"Original: {test!r}")
print(f"  Length in chars: {len(test)}")
print(f"  Length in bytes: {len(test.encode('utf-8'))}")
print(f"Encoded ids ({len(encoded)} tokens): {encoded}")
print(f"  Compression: {len(test.encode('utf-8')) / len(encoded):.2f}x bytes per token")
print()
print(f"Decoded: {decoded!r}")
print(f"Round-trip works: {decoded == test}")
print()


# ---------------------------------------------------------------
# Save the trained tokenizer
# ---------------------------------------------------------------
tok.save("shakespeare_tokenizer.txt")
print(f"Saved trained tokenizer to shakespeare_tokenizer.txt")
print(f"({sum(1 for _ in open('shakespeare_tokenizer.txt'))} lines)")
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
print("""
============================================================
A REAL, WORKING TOKENIZER
============================================================

You now have a Tokenizer class with .train(), .encode(), .decode(),
.save(), .load(). About 60 lines of code total. Trained on
Shakespeare, it captures the rhythms and vocabulary of the corpus.

Notice what showed up in the early merges:
  - Common letter pairs ('th', 'he', 'in', 'er')
  - Common trigrams ('the', 'and', 'you')
  - Spaces bound to common following words (' th', ' an')
  - Eventually, character names ('ROMEO', 'KING')

This is Karpathy's minbpe.BasicTokenizer in essence. It's also,
fundamentally, what every modern LLM tokenizer is: BPE with extra
preprocessing rules.

Next: see what happens when you encode quirky strings -- and why
real LLMs have weird behaviors with certain patterns.
""")
