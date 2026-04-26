"""
03_sample_from_counts.py
------------------------
We have a (V x V) matrix of counts. To turn it into a model, normalize
each row so it sums to 1. Now P[i, j] = "probability that character j
follows character i".

Then GENERATE a name by sampling:
    1. Start with the previous-character index = stoi['.'] (the start token).
    2. Look up that row in P. It's a probability distribution over what
       comes next.
    3. Sample one index from that distribution.
    4. If the sampled char is '.', stop. Otherwise, append it and the
       sampled char becomes the new "previous" character. Loop.

That's the whole generative process. No magic. No neural network.

Important detail: we use np.random with a SEEDED Generator so output is
reproducible. Karpathy uses torch.Generator the same way in the video.
"""

import numpy as np
from data_utils import load_names, build_vocab


names = load_names()
stoi, itos = build_vocab(names)
V = len(stoi)

# Load counts from the previous script (or rebuild if missing)
try:
    N = np.load("bigram_counts.npy")
except FileNotFoundError:
    N = np.zeros((V, V), dtype=np.int32)
    for w in names:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            N[stoi[ch1], stoi[ch2]] += 1


# ---------------------------------------------------------------
# Normalize rows -> probability matrix P
# ---------------------------------------------------------------
# Cast to float, divide each row by its sum.
# keepdims=True so broadcasting (V,V) / (V,1) works correctly.
P = N.astype(np.float64)
P = P / P.sum(axis=1, keepdims=True)

print("Each row of P sums to ~1.0:")
print(f"  P[0].sum() = {P[0].sum():.6f}")
print(f"  P[1].sum() = {P[1].sum():.6f}")
print()


# ---------------------------------------------------------------
# Inspect P[0] -- the distribution over starting characters
# ---------------------------------------------------------------
# This is what the model thinks the FIRST character of a name should be.
print("Top 5 most likely starting characters per the model:")
start_dist = P[stoi["."]]
top = np.argsort(start_dist)[::-1][:5]
for j in top:
    print(f"  P('{itos[j]}' | start) = {start_dist[j]:.4f}")
print()


# ---------------------------------------------------------------
# Generate names
# ---------------------------------------------------------------
def sample_name(P, stoi, itos, rng, max_len=50):
    out = []
    ix = stoi["."]  # start token
    while True:
        probs = P[ix]
        ix = rng.choice(len(probs), p=probs)  # sample from the distribution
        if ix == stoi["."]:
            break
        out.append(itos[ix])
        if len(out) >= max_len:
            break
    return "".join(out)


rng = np.random.default_rng(2147483647)  # same seed Karpathy uses
print("10 names sampled from the bigram model:")
for _ in range(10):
    print(f"  {sample_name(P, stoi, itos, rng)}")
print()


# ---------------------------------------------------------------
# Compare: what would TRULY random (uniform) sampling produce?
# ---------------------------------------------------------------
# This is the baseline. If our model is doing anything at all, its names
# should look obviously more name-like than these.
P_uniform = np.ones_like(P) / V
print("10 names from a UNIFORM-random model (baseline -- should look like noise):")
for _ in range(10):
    print(f"  {sample_name(P_uniform, stoi, itos, rng)}")
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - P is just N normalized row-wise.
# - Sampling repeatedly from P[prev_char] generates a name.
# - The model's outputs are clearly more name-like than uniform noise --
#   even with only 50 fallback names, you can SEE it learned something.
# - With the full 32k dataset the names are recognizably name-shaped:
#   "junide", "janasah", "konniva" and similar.
#
# The model "works". But how do we MEASURE how good it is? We need a
# single number. That's the loss function -- next script.
