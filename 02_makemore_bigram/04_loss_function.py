"""
04_loss_function.py
-------------------
We have a model. How good is it? We need a single number.

The standard answer in language modeling: NEGATIVE LOG-LIKELIHOOD (NLL).

Idea:
    For each (ch1, ch2) bigram in the training data, the model assigns
    a probability P[ch1, ch2]. A good model assigns HIGH probability to
    bigrams that actually occurred.

    Take the LOG of each probability (avoids underflow when multiplying
    millions of small numbers, and turns products into sums).

    Take the NEGATIVE so lower = better. Now we have something we can
    minimize. Take the AVERAGE so the loss doesn't depend on dataset size.

Final form:
    loss = -mean( log P[ch1, ch2] for every bigram in data )

Lower loss = model assigns higher probability to real data = better model.

Edge case: if a bigram never appeared in training, P[ch1, ch2] = 0, and
log(0) = -infinity. The fix is to add a tiny constant to every count
before normalizing -- "additive smoothing". This costs us a sliver of
accuracy on common bigrams in exchange for finite loss on rare ones.
"""

import numpy as np
from data_utils import load_names, build_vocab


names = load_names()
stoi, itos = build_vocab(names)
V = len(stoi)


# Build P with smoothing
N = np.zeros((V, V), dtype=np.int32)
for w in names:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1

# Add 1 to every count BEFORE normalizing. This is the smoothing.
SMOOTH = 1
P = (N + SMOOTH).astype(np.float64)
P = P / P.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------
# Compute loss on the training data
# ---------------------------------------------------------------
log_likelihood = 0.0
n = 0

for w in names:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        i, j = stoi[ch1], stoi[ch2]
        prob = P[i, j]
        log_likelihood += np.log(prob)
        n += 1

avg_nll = -log_likelihood / n
print(f"Average negative log-likelihood: {avg_nll:.4f}")
print(f"  (sum log-likelihood = {log_likelihood:.2f}, num bigrams = {n})")
print()


# ---------------------------------------------------------------
# What does this number MEAN?
# ---------------------------------------------------------------
# Compare two reference points:
#
#   - Uniform random model: NLL = log(V), since every char is equally likely.
#   - Perfect model: NLL = 0, since it gives probability 1 to actual bigrams.
#
# Real model should land between these.

print("Reference points:")
print(f"  Uniform model NLL  : {np.log(V):.4f}  (lower bound on baseline)")
print(f"  Our bigram model   : {avg_nll:.4f}")
print(f"  Perfect model NLL  : 0.0000")
print()

# With the full dataset (32k names), Karpathy gets ~2.45.
# With the fallback dataset (50 names), it'll be different but should
# still beat the uniform baseline.


# ---------------------------------------------------------------
# Inspect the WORST bigrams (highest NLL contribution)
# ---------------------------------------------------------------
# These are the rarest bigrams in the dataset -- the ones the model
# is most surprised by. Useful to see what the model struggles with.
print("Worst-scoring bigrams in the first 5 names:")
for w in names[:5]:
    chs = ["."] + list(w) + ["."]
    bigram_losses = []
    for ch1, ch2 in zip(chs, chs[1:]):
        i, j = stoi[ch1], stoi[ch2]
        bigram_losses.append((-np.log(P[i, j]), ch1, ch2))
    bigram_losses.sort(reverse=True)
    nll, ch1, ch2 = bigram_losses[0]
    print(f"  {w!r}: worst bigram is '{ch1}{ch2}' with NLL = {nll:.3f}")
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - NLL is one number that summarizes how well the model fits the data.
# - It's well-defined, comparable across models, and minimization-friendly.
# - Smoothing prevents log(0) catastrophes for unseen bigrams.
#
# So far we've used DIRECT counting to build the model. Same problem
# can be solved with a NEURAL NETWORK that learns the same probability
# table via gradient descent. They give equivalent results -- but the
# neural net version generalizes to richer models in a way counting
# never could. That's the second half of the lesson.
