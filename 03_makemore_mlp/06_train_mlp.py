"""
06_train_mlp.py
---------------
Train the MLP to convergence using everything we've built.

Key ideas in the training loop:

  - MINIBATCH GRADIENT DESCENT.  We don't compute the loss on the full
    training set every step -- it's slow and unnecessary. Each step uses
    a random subset (batch) of e.g. 32 examples. The loss is noisy but
    on average points in the right direction. This is what every modern
    deep learning system does.

  - LR DECAY. Train at a high LR for the bulk of steps, then drop to a
    much smaller LR for the final phase. The high LR moves quickly into
    a good basin; the low LR fine-tunes within it.

  - SEPARATE LOSS TRACKING. We periodically evaluate on the FULL train
    set and the FULL dev set. The training-batch loss is jumpy and
    optimistic; the full-split loss tells us where we really are.
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


data = torch.load("dataset.pt", weights_only=False)
Xtr, Ytr = data["Xtr"], data["Ytr"]
Xdev, Ydev = data["Xdev"], data["Ydev"]
itos = data["itos"]
V = len(itos)
BLOCK = data["block_size"]


# ---------------------------------------------------------------
# Init
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
print(f"Total parameters: {sum(p.numel() for p in parameters):,}")
print()


# ---------------------------------------------------------------
# Full-split loss helper
# ---------------------------------------------------------------
@torch.no_grad()
def split_loss(X, Y):
    emb = C[X]
    flat = emb.view(emb.shape[0], -1)
    h = torch.tanh(flat @ W1 + b1)
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Y).item()


# ---------------------------------------------------------------
# Train
# ---------------------------------------------------------------
BATCH = 32
STEPS = 50_000           # bump to 200_000 for the full lesson run
LR_HIGH = 0.1
LR_LOW = 0.01
DECAY_AT = int(STEPS * 0.7)

losses_log = []
for step in range(STEPS):
    # minibatch
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

    # LR-decayed update
    lr = LR_HIGH if step < DECAY_AT else LR_LOW
    for p in parameters:
        p.data += -lr * p.grad

    losses_log.append(loss.log10().item())  # log scale for nicer plots
    if step % 5000 == 0:
        print(f"step {step:>6}/{STEPS}  loss(batch) = {loss.item():.4f}")

print()
print(f"FINAL train loss: {split_loss(Xtr, Ytr):.4f}")
print(f"FINAL dev   loss: {split_loss(Xdev, Ydev):.4f}")
print()


# ---------------------------------------------------------------
# Plot loss curve
# ---------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(losses_log)
plt.xlabel("step")
plt.ylabel("log10(loss)")
plt.title("Training loss (log scale)")
plt.grid(alpha=0.3)
plt.savefig("loss_curve.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved loss_curve.png")


# ---------------------------------------------------------------
# Visualize learned embeddings
# ---------------------------------------------------------------
# This only makes a nice plot if D=2. With D=10 we'd need t-SNE or PCA.
# For pedagogical reruns: change D=2 in the init above and watch how
# vowels and consonants separate.
if D == 2:
    plt.figure(figsize=(8, 8))
    plt.scatter(C[:, 0].detach(), C[:, 1].detach(), s=200)
    for i in range(V):
        plt.text(C[i, 0].item(), C[i, 1].item(), itos[i],
                 ha="center", va="center", color="white", fontweight="bold")
    plt.title("Trained character embeddings")
    plt.grid(alpha=0.3)
    plt.savefig("embeddings_trained.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("Saved embeddings_trained.png -- vowels should now cluster!")


# ---------------------------------------------------------------
# Save trained parameters for the sampling script
# ---------------------------------------------------------------
torch.save({"C": C, "W1": W1, "b1": b1, "W2": W2, "b2": b2,
            "itos": itos, "stoi": data["stoi"], "block_size": BLOCK,
            }, "trained_mlp.pt")
print("Saved trained_mlp.pt")


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - With block_size=3, D=10, hidden=200, after 50k steps you should see
#   train loss around 2.1, dev loss around 2.2. Karpathy's video gets
#   to ~2.17 with similar settings and 200k steps.
# - Train and dev loss should be CLOSE. If train is much lower than dev,
#   you're overfitting -- shrink the model or add regularization.
# - The biggest lever is BLOCK_SIZE. Going from 3 to 8 helps a lot.
#
# Next: sample names from the trained model.
