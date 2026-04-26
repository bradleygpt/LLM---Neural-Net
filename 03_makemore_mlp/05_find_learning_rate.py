"""
05_find_learning_rate.py
------------------------
Before training for real, we need to find a good learning rate. Karpathy
demonstrates a clever technique that's worth internalizing.

The idea:
    1. Run a SHORT training (e.g. 1000 steps) where the learning rate
       grows EXPONENTIALLY from very small to very large.
    2. Track the loss at each step.
    3. Plot loss vs. log(lr).
    4. The sweet spot is just before the loss starts climbing -- that's
       the largest LR you can use without divergence.

Why this beats guessing:
    - LR is the single most important hyperparameter and it has no
      formula. It depends on architecture, batch size, data, init.
    - Manually trying 0.001, 0.01, 0.1, 1.0 takes forever and is noisy.
    - The exponential sweep covers many orders of magnitude in one shot.
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Load data
data = torch.load("dataset.pt", weights_only=False)
Xtr, Ytr = data["Xtr"], data["Ytr"]
V = len(data["itos"])
BLOCK = data["block_size"]


# ---------------------------------------------------------------
# Re-init parameters from script 04
# ---------------------------------------------------------------
torch.manual_seed(2147483647)
D, HIDDEN = 10, 200
C = torch.randn((V, D))
W1 = torch.randn((BLOCK * D, HIDDEN)) * 0.1
b1 = torch.randn(HIDDEN) * 0.1
W2 = torch.randn((HIDDEN, V)) * 0.1
b2 = torch.randn(V) * 0.1
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad_()


# ---------------------------------------------------------------
# Build the LR sweep
# ---------------------------------------------------------------
# 1000 candidate LRs spaced exponentially from 1e-3 to 1.0
lre = torch.linspace(-3, 0, 1000)
lrs = 10 ** lre

print(f"Sweeping {len(lrs)} learning rates from {lrs[0].item():.4f} to {lrs[-1].item():.4f}")
print()


# ---------------------------------------------------------------
# Train one minibatch per candidate LR, recording the loss
# ---------------------------------------------------------------
BATCH = 32
losses = []

for step in range(len(lrs)):
    # sample a random minibatch
    ix = torch.randint(0, Xtr.shape[0], (BATCH,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward
    emb = C[Xb]
    flat = emb.view(emb.shape[0], -1)
    h = torch.tanh(flat @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)

    # backward
    for p in parameters:
        p.grad = None
    loss.backward()

    # update with the candidate LR
    lr = lrs[step].item()
    for p in parameters:
        p.data += -lr * p.grad

    losses.append(loss.item())


# ---------------------------------------------------------------
# Plot loss vs log(lr)
# ---------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(lre.tolist(), losses)
plt.xlabel("log10(learning rate)")
plt.ylabel("loss")
plt.title("LR sweep: pick the LR just before loss diverges")
plt.grid(alpha=0.3)
plt.savefig("lr_sweep.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved lr_sweep.png. The sweet spot is typically the LR right BEFORE")
print("the curve starts climbing (loss goes back up).")
print()

# Programmatic guess: lowest loss seen, what LR produced it?
best_idx = int(torch.tensor(losses).argmin())
print(f"Step with lowest loss: {best_idx}")
print(f"  loss = {losses[best_idx]:.4f}")
print(f"  log10(lr) = {lre[best_idx].item():.3f}")
print(f"  lr = {lrs[best_idx].item():.4f}")
print()
print("In Karpathy's video, around log10(lr) = -1 (lr ~ 0.1) works well.")
print("Use this as a STARTING POINT. The next script trains with it.")


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - Sweeping LR exponentially is much better than trying values by hand.
# - The optimal LR is the largest one that doesn't diverge -- "as fast
#   as you can without falling over".
# - Common pattern: train at high LR for most of training, then drop
#   ("LR decay") for the last 10-20% to fine-tune. Done in script 06.
