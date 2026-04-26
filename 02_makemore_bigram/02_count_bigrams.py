"""
02_count_bigrams.py
-------------------
Build the bigram count matrix N. It's a (vocab_size x vocab_size) array
where N[i, j] = how many times character j followed character i in the
training data.

This matrix IS the model. Everything else is just normalizing it,
sampling from it, or scoring it.

We don't even use a neural network here. This is pure counting. And
it works -- the names we generate will look like names. The point of
this script is to internalize that DATA + COUNTING gets you a
language model.
"""

import numpy as np
from data_utils import load_names, build_vocab


names = load_names()
stoi, itos = build_vocab(names)
V = len(stoi)  # vocab size


# ---------------------------------------------------------------
# Build the count matrix
# ---------------------------------------------------------------
N = np.zeros((V, V), dtype=np.int32)

for w in names:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        i = stoi[ch1]
        j = stoi[ch2]
        N[i, j] += 1

print(f"N has shape {N.shape}")
print(f"N.sum() = {N.sum()}  (should equal total_bigrams from script 01)")
print()


# ---------------------------------------------------------------
# Inspect: what follows '.' the most? (i.e. most common starting letters)
# ---------------------------------------------------------------
start_row = N[stoi["."]]
top_starts = np.argsort(start_row)[::-1][:5]
print("Top 5 most common starting characters:")
for j in top_starts:
    print(f"  '.' -> '{itos[j]}': {start_row[j]} times")
print()


# ---------------------------------------------------------------
# What follows 'a' the most?
# ---------------------------------------------------------------
a_row = N[stoi["a"]]
top_after_a = np.argsort(a_row)[::-1][:5]
print("Top 5 characters that follow 'a':")
for j in top_after_a:
    print(f"  'a' -> '{itos[j]}': {a_row[j]} times")
print()


# ---------------------------------------------------------------
# The full matrix (just print a corner of it for sanity)
# ---------------------------------------------------------------
print("Top-left 8x8 corner of N:")
print(f"     {' '.join(itos[i].rjust(4) for i in range(8))}")
for i in range(8):
    row = " ".join(str(N[i, j]).rjust(4) for j in range(8))
    print(f" {itos[i]} : {row}")
print()


# Save N for later scripts
np.save("bigram_counts.npy", N)
print("Saved counts to bigram_counts.npy")


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - N[i, j] is the raw frequency of bigram (i, j).
# - The most common bigrams reflect real linguistic patterns -- common
#   starting letters, common letter pairs ('an', 'on', etc.).
# - This matrix IS our model. With more data, it gets sharper.
#
# Right now the matrix is COUNTS. To use it for prediction or generation,
# we need to convert each row into a PROBABILITY DISTRIBUTION over the
# next character. That's the next step.
