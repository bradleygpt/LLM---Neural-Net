"""
05_neural_net_setup.py
----------------------
We now build the SAME bigram model, but as a neural network.

This is the most important pivot in the lesson. The counting model and
the neural-net-trained model produce IDENTICAL probability tables. The
neural net is not "better". The point is that gradient descent gives us
a path to MUCH richer models later -- this script is just the warm-up.

Setup steps:
    1. Convert the dataset of names into TWO lists:
        xs[i] = the i-th input character index
        ys[i] = the index of the character that should follow it
       So if "emma" gives bigrams (.,e), (e,m), (m,m), (m,a), (a,.),
       we add 5 entries to (xs, ys).

    2. ONE-HOT ENCODE the inputs. Each character index becomes a
       length-V vector with a single 1.0 in position `i` and 0.0
       elsewhere. Why? Because neural nets multiply inputs by weight
       matrices, and one-hot @ W is just "row i of W". One-hot encoding
       is the bridge between "categorical inputs" and "matrix math".

    3. Build a single weight matrix W of shape (V, V). Row i, column j
       will eventually represent "log-probability of char j given char i".
       Initialize randomly.

    4. Forward pass: x_onehot @ W gives us "logits" -- raw scores, one
       per possible next character. We turn logits into probabilities
       via softmax (exponentiate, then normalize).

That's it -- a one-layer neural net with no hidden layer, no
nonlinearity. Just a learned lookup table.
"""

import numpy as np
from data_utils import load_names, build_vocab


names = load_names()
stoi, itos = build_vocab(names)
V = len(stoi)


# ---------------------------------------------------------------
# Build the training set
# ---------------------------------------------------------------
xs, ys = [], []
for w in names:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

xs = np.array(xs, dtype=np.int64)
ys = np.array(ys, dtype=np.int64)
print(f"Training set: {len(xs)} (input, target) pairs")
print(f"  First 5 xs: {xs[:5]}")
print(f"  First 5 ys: {ys[:5]}")
print(f"  Decoded:    {[itos[x] for x in xs[:5]]} -> {[itos[y] for y in ys[:5]]}")
print()


# ---------------------------------------------------------------
# One-hot encode the inputs
# ---------------------------------------------------------------
# x_onehot has shape (N, V). For each row, exactly one column is 1.0.
# This is what np.eye(V)[xs] does in one shot.
x_onehot = np.eye(V)[xs].astype(np.float64)
print(f"x_onehot shape: {x_onehot.shape}")
print(f"First row (one-hot for char index {xs[0]} = '{itos[xs[0]]}'):")
print(f"  {x_onehot[0]}")
print()


# ---------------------------------------------------------------
# Initialize the weights
# ---------------------------------------------------------------
# W has shape (V, V). Random small values.
rng = np.random.default_rng(2147483647)
W = rng.standard_normal((V, V)) * 0.01
print(f"W shape: {W.shape}")
print()


# ---------------------------------------------------------------
# Forward pass -- one example
# ---------------------------------------------------------------
# x_onehot @ W gives us the LOGITS: raw scores for each possible next char.
logits = x_onehot @ W
print(f"logits shape: {logits.shape}  (one row of {V} scores per training example)")
print(f"logits[0] (first example): {logits[0]}")
print()


# ---------------------------------------------------------------
# Softmax: turn logits into probabilities
# ---------------------------------------------------------------
# softmax(z)_i = exp(z_i) / sum_k exp(z_k)
# - exp() makes everything positive (so it can be a probability)
# - dividing by the sum normalizes the row to sum to 1
counts = np.exp(logits)              # "fake counts"
probs = counts / counts.sum(axis=1, keepdims=True)
print(f"probs shape: {probs.shape}")
print(f"probs[0].sum() = {probs[0].sum():.6f} (should be ~1.0)")
print(f"probs[0]: {probs[0]}")
print()


# ---------------------------------------------------------------
# A subtle but important point
# ---------------------------------------------------------------
# Notice the structure:  counts = exp(logits), then normalize.
# This is EXACTLY what we did in script 03, except there `counts` were
# REAL bigram counts and here they're EXPONENTIATED LOGITS.
#
# In other words: W[i] is essentially log-counts of how the network
# expects char j to follow char i. After exp, it's counts. After
# normalize, it's probabilities.
#
# So a neural network with one linear layer is doing the same thing as
# counting -- but the "counts" come from a learned matrix instead of
# from tallying the dataset. Training will push that matrix toward
# numbers that match real bigram statistics.

# Save for next script
np.save("xs.npy", xs)
np.save("ys.npy", ys)
print("Saved xs, ys for the training script.")
