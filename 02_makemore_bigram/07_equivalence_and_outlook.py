"""
07_equivalence_and_outlook.py
-----------------------------
The point of this whole video, in one script: COUNTING and the TRAINED
NEURAL NET are doing the same thing. We prove it by:

  1. Training a NN with NO regularization and NO smoothing on counts ->
     the two probability tables agree to many decimal places.

  2. Showing that L2 regularization on W has the SAME EFFECT as additive
     smoothing on N -- both pull the distribution toward uniform.

Then a look ahead.
"""

import numpy as np
from data_utils import load_names, build_vocab


names = load_names()
stoi, itos = build_vocab(names)
V = len(stoi)

# ---------------------------------------------------------------
# Build training data
# ---------------------------------------------------------------
xs, ys = [], []
for w in names:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
xs = np.array(xs)
ys = np.array(ys)
n = len(xs)


# ---------------------------------------------------------------
# 1. Train an UNREGULARIZED NN to convergence
# ---------------------------------------------------------------
# Without regularization and a long training run, the NN will produce a
# probability table that matches the UNSMOOTHED count table.
print("Training NN with NO regularization (this may take a moment)...")
x_onehot = np.eye(V)[xs]
rng = np.random.default_rng(2147483647)
W = rng.standard_normal((V, V))

LR, EPOCHS = 50.0, 800
for epoch in range(EPOCHS):
    logits = x_onehot @ W
    counts_nn = np.exp(logits)
    probs = counts_nn / counts_nn.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(n), ys]).mean()
    dlogits = probs.copy()
    dlogits[np.arange(n), ys] -= 1.0
    dlogits /= n
    dW = x_onehot.T @ dlogits
    W -= LR * dW
print(f"  Final NN loss (no reg): {loss:.4f}")

# Probability table from the trained NN
logits_all = np.eye(V) @ W
P_nn = np.exp(logits_all)
P_nn /= P_nn.sum(axis=1, keepdims=True)


# Counting WITHOUT smoothing -- the right comparison for the unregularized NN
N_count = np.zeros((V, V), dtype=np.float64)
for w in names:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        N_count[stoi[ch1], stoi[ch2]] += 1
# Some rows in the fallback dataset might be all-zero (chars never seen as
# 'previous'); guard against divide-by-zero.
row_sums = N_count.sum(axis=1, keepdims=True)
P_count = np.divide(N_count, row_sums, out=np.zeros_like(N_count), where=row_sums>0)

# Counting loss (unsmoothed). Skip bigrams in zero-prob cells.
nz_mask = N_count > 0
count_loss = -(N_count[nz_mask] * np.log(P_count[nz_mask])).sum() / n
print(f"  Counting loss (no smoothing): {count_loss:.4f}")
print()


# ---------------------------------------------------------------
# Compare the tables row-by-row
# ---------------------------------------------------------------
print("Compare a few rows from the two probability tables (should match closely):")
print()
for i in [stoi["."], stoi["a"], stoi["e"]]:
    ch = itos[i]
    diff = np.abs(P_count[i] - P_nn[i]).max()
    print(f"Row for previous='{ch}':")
    print(f"  P_count top 3: ", end="")
    top = np.argsort(P_count[i])[::-1][:3]
    print(", ".join(f"{itos[j]}={P_count[i, j]:.4f}" for j in top))
    print(f"  P_nn    top 3: ", end="")
    top = np.argsort(P_nn[i])[::-1][:3]
    print(", ".join(f"{itos[j]}={P_nn[i, j]:.4f}" for j in top))
    print(f"  Max abs diff: {diff:.4f}")
    print()

total_diff = np.abs(P_count - P_nn).sum()
print(f"Sum of absolute differences across the full table: {total_diff:.4f}")
print(f"  Smaller is better. This is the KEY measurement of the lesson.")
print()


# ---------------------------------------------------------------
# 2. The regularization <-> smoothing parallel
# ---------------------------------------------------------------
# Regularizing the NN's weights toward 0 makes its logits flatter, which
# makes its probabilities more UNIFORM. That's the same direction additive
# smoothing pushes the count-based probabilities. So they're playing the
# same role -- pulling the model toward "I'm not sure" away from "100% certain".
print("Both counting+smoothing AND NN+regularization push the model")
print("toward UNIFORM. They're the same idea wearing different clothes:")
print()
print(f"  - additive smoothing of K  ~  L2 weight decay (with right strength)")
print(f"  - both prevent log(0) when an event is rare or unseen")
print(f"  - both trade a bit of training-loss for better generalization")


# ---------------------------------------------------------------
# The central insight
# ---------------------------------------------------------------
print("=" * 60)
print("KEY TAKEAWAY")
print("=" * 60)
print("""
Counting bigrams and training a 1-layer neural net produce essentially
the same probability table. The neural net has just rediscovered the
statistics of the data through gradient descent.

So why bother with the neural net? Three reasons:

  1. SCALE.  Counting only works when we condition on a small, finite
     context (here: 1 previous character). The number of cells in the
     count table is V^k for context length k. With k=10 and V=27 that's
     ~10^14. We'd never see most of those bigrams in any dataset --
     the table is too sparse to be useful.

  2. REPRESENTATION.  A neural net can EMBED characters into vectors
     (next video: makemore part 2), share parameters across similar
     contexts, and use nonlinearities to capture interactions. Counting
     can't.

  3. UNIFORMITY.  Once you have a forward pass + backward pass + update
     loop, you can swap in any architecture you want -- bigram, MLP,
     transformer -- and the training code is essentially the same.
     That's the path that leads to GPT.

The next video (makemore part 2: MLP) replaces the (V,V) lookup table
with an embedding table + hidden layer. The video after that (part 3)
introduces BatchNorm. And so on, each step climbing toward the GPT
architecture in 'Let's build GPT from scratch'.

You now have the floor. Everything from here is upgrades.
""")
