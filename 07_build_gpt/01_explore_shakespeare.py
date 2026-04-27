"""
01_explore_shakespeare.py
-------------------------
The data: a single text file containing the complete works of Shakespeare,
or a curated subset called "Tiny Shakespeare." About 1.1 MB, 1 million
characters, 65 unique characters (letters, punctuation, whitespace).

This is the canonical dataset for "build a tiny transformer" tutorials.
Karpathy uses it because:
  - Small enough to train on in minutes on a CPU
  - Large enough that the model has to actually learn structure
  - Distinctive enough that you can recognize when the model "gets" it
    (Shakespearean cadence is unmistakable)

We also need a fallback for when the download fails. If you can't fetch
input.txt, the scripts will use an embedded sample of ~12k characters
of Shakespeare for shape correctness. The numbers won't match Karpathy's
but the architecture and training will work end-to-end.
"""

import os


# ---------------------------------------------------------------
# Embedded fallback (a small Shakespeare excerpt)
# ---------------------------------------------------------------
# Just enough characters that the dataset has all 65 chars and the
# training procedure works. Real lesson uses ~1M chars.
_FALLBACK = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.
""" * 50  # repeat to get a longer dataset for shape correctness


def load_text(path="input.txt"):
    """Return the full Shakespeare text. Falls back if file missing."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Loaded {len(text):,} characters from {path}")
        return text
    print(f"[note] {path} not found -- using built-in fallback (~{len(_FALLBACK):,} chars).")
    print("[note] For full Karpathy results, download from:")
    print("[note] https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
    return _FALLBACK


# ---------------------------------------------------------------
# Run as a script: load and report
# ---------------------------------------------------------------
if __name__ == "__main__":
    text = load_text()

    print()
    print(f"Total characters: {len(text):,}")
    print()

    # Unique characters in the dataset
    chars = sorted(set(text))
    vocab_size = len(chars)
    print(f"Unique characters: {vocab_size}")
    print(f"Character set: {''.join(chars)!r}")
    print()

    # Build encoder/decoder (character-level tokenization)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    # Test
    sample = text[:80]
    encoded = encode(sample)
    decoded = decode(encoded)
    print(f"Sample text: {sample!r}")
    print(f"Encoded:     {encoded}")
    print(f"Decoded:     {decoded!r}")
    print(f"Encoding round-trips correctly: {decoded == sample}")
    print()

    # ---------------------------------------------------------------
    # What we just learned
    # ---------------------------------------------------------------
    # - The dataset is ~1M characters of Shakespeare (or fallback).
    # - 65 unique characters: lowercase, uppercase, digits, punctuation,
    #   whitespace, newlines.
    # - Tokenization is just character-level: each character maps to an int.
    # - encode(decode(x)) = x. Round-trip lossless.
    #
    # This is BAD tokenization for real language modeling -- 65 tokens
    # means very long sequences (one token per character). Real LLMs use
    # subword tokenization with vocab ~50,000. We get there in lesson 08.
    # For now, character-level keeps things simple.
    #
    # Next: build the train/val splits and the data loader.
