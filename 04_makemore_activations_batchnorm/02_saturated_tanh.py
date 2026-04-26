"""
02_saturated_tanh.py
--------------------
The output layer init wasn't the only problem. Look at what happens INSIDE
the network: the hidden layer's tanh activations.

tanh(x) is roughly linear near 0 (slope 1) but flattens out for large |x|
(slope -> 0). When |x| > 5, tanh(x) is essentially +/-1 and its derivative
is essentially 0.

If too many hidden activations are "saturated" (deep in the flat region),
the gradient that flows back through them is multiplied by ~0. Those
neurons are DEAD -- they don't update during training. Catastrophe.

This script visualizes the problem:
    1. Initialize an MLP with default (large) weights.
    2. Run a forward pass.
    3. Plot a histogram of the pre-activation values (flat @ W1 + b1).
    4. Plot the actual tanh(...) activations.
    5. Compute what fraction of neurons are saturated.
"""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random

from data_utils import load_names, build_vocab, build_dataset


# Build dataset
names = load_names()
stoi, itos = build_vocab(names)
V = len(stoi)
random.seed(42)
random.shuffle(names)
n1 = int(0.8 * len(names))
Xtr, Ytr = build_dataset(names[:n1], stoi, block_size=3)
BLOCK = 3


# ---------------------------------------------------------------
# Compare two initializations side by side
# ---------------------------------------------------------------
def diagnose(init_scale, label):
    torch.manual_seed(42)
    D, H = 10, 200
    C = torch.randn((V, D))
    W1 = torch.randn((BLOCK * D, H)) * init_scale
    b1 = torch.randn(H) * init_scale

    # Forward through the hidden layer on a batch
    ix = torch.randint(0, Xtr.shape[0], (32,))
    emb = C[Xtr[ix]]
    flat = emb.view(emb.shape[0], -1)
    pre = flat @ W1 + b1                          # pre-activations
    h = torch.tanh(pre)                           # post-tanh activations

    # How many activations are saturated? (|tanh| > 0.99)
    saturated = (h.abs() > 0.99).float().mean().item()
    # How many ENTIRE neurons are dead? (every input drives them to saturation)
    dead = ((h.abs() > 0.99).all(dim=0)).float().mean().item()

    print(f"  init scale {init_scale:>5}: {label}")
    print(f"     pre-activation std: {pre.std().item():.3f}")
    print(f"     tanh saturated (|h|>0.99): {100*saturated:5.1f}%")
    print(f"     fully DEAD neurons:        {100*dead:5.1f}%")
    return pre.detach(), h.detach(), label


print("Diagnosing hidden layer activations under different inits:")
print()
pre1, h1, lbl1 = diagnose(1.0, "PyTorch default")
pre2, h2, lbl2 = diagnose(0.2, "scaled by 0.2")
pre3, h3, lbl3 = diagnose(5.0 / ((BLOCK * 10) ** 0.5), "Kaiming-style: 5/3 / sqrt(fan_in)")
print()


# ---------------------------------------------------------------
# Plot histograms of activations
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for col, (pre, h, lbl) in enumerate([(pre1, h1, lbl1), (pre2, h2, lbl2), (pre3, h3, lbl3)]):
    axes[0, col].hist(pre.flatten().tolist(), bins=50, color="steelblue")
    axes[0, col].set_title(f"Pre-activation (flat@W1+b1)\n{lbl}")
    axes[0, col].axvline(-2, color="red", linestyle="--", alpha=0.5)
    axes[0, col].axvline(2, color="red", linestyle="--", alpha=0.5)
    axes[1, col].hist(h.flatten().tolist(), bins=50, color="seagreen")
    axes[1, col].set_title(f"Post-tanh\n{lbl}")
    axes[1, col].set_xlim(-1.05, 1.05)
plt.tight_layout()
plt.savefig("activation_distributions.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved activation_distributions.png")
print()


# ---------------------------------------------------------------
# Why "fully dead neurons" matter
# ---------------------------------------------------------------
print("""
A 'fully dead' neuron is one where, for EVERY example in the batch,
|tanh(pre)| > 0.99. The local gradient through tanh at saturation is
~0, so this neuron's weights get ~0 gradient and never update. It's
permanently stuck.

The fix is simply to make pre-activations smaller in magnitude so tanh
stays in its linear region. Two ways:

  - Scale W1 down at init so the inputs to tanh are smaller.
  - Use the right SCALING formula: W ~ N(0, gain^2 / fan_in). For tanh,
    the recommended gain is 5/3. This is "Kaiming init for tanh".

Next script: derive that scaling factor from first principles.
""")
