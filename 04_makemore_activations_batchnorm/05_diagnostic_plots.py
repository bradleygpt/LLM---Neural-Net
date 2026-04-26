"""
05_diagnostic_plots.py
----------------------
Karpathy's lesson introduces four "health check" plots that every deep
learning training run should generate. They tell you whether the
network is happy, dying, or about to explode.

We'll build a small MLP, train briefly, and capture all four:

  1. ACTIVATION HISTOGRAM per layer.  Should look roughly Gaussian-ish
     with std around 0.6-0.8 if using tanh. If everything is at +/-1,
     you have saturation problems.

  2. GRADIENT HISTOGRAM per layer.  Should be roughly the same scale
     across all layers. If gradients shrink with depth ("vanishing
     gradients") or explode, training won't work.

  3. WEIGHT HISTOGRAM per layer.  Should evolve gradually during
     training. Sudden changes mean LR is too high.

  4. UPDATE-TO-WEIGHT RATIO.  At each step, |update| / |weight|.
     Karpathy's rule of thumb: this should be around 1e-3. Higher
     means the LR is too high; lower means too low or the weight
     isn't being updated meaningfully.

These are the tools you reach for first when training isn't working.
They turn "the loss is weird" into a specific diagnosis.
"""

import torch
import torch.nn.functional as F
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
print(f"Training set: {Xtr.shape[0]:,} examples")


# ---------------------------------------------------------------
# Build a deeper MLP (5 hidden layers) so we can see depth effects
# ---------------------------------------------------------------
torch.manual_seed(42)
D, H = 10, 100
N_LAYERS = 5

C = torch.randn((V, D))
layers = []
in_size = BLOCK * D
for _ in range(N_LAYERS):
    W = torch.randn((in_size, H)) * (5/3) / in_size**0.5  # Kaiming for tanh
    b = torch.zeros(H)
    layers.append((W, b))
    in_size = H
W_out = torch.randn((H, V)) * 0.01   # tiny output init (avoid hockey stick)
b_out = torch.zeros(V)

# Collect all params
params = [C] + [w for layer in layers for w in layer] + [W_out, b_out]
for p in params:
    p.requires_grad_()
print(f"Total params: {sum(p.numel() for p in params):,}")
print()


# ---------------------------------------------------------------
# Forward pass that captures intermediate activations
# ---------------------------------------------------------------
def forward(Xb, Yb, capture=False):
    activations = []
    emb = C[Xb]
    h = emb.view(emb.shape[0], -1)
    for W, b in layers:
        h = h @ W + b
        h = torch.tanh(h)
        if capture:
            activations.append(h)
    logits = h @ W_out + b_out
    loss = F.cross_entropy(logits, Yb)
    return loss, activations


# ---------------------------------------------------------------
# Diagnostic 1: activation distribution at init
# ---------------------------------------------------------------
ix = torch.randint(0, Xtr.shape[0], (256,))
Xb, Yb = Xtr[ix], Ytr[ix]
loss, activations = forward(Xb, Yb, capture=True)
print("Activation stats per layer at INIT (Kaiming-initialized):")
for i, h in enumerate(activations):
    sat = (h.abs() > 0.97).float().mean().item()
    print(f"  layer {i}: mean={h.mean().item():+.3f}, std={h.std().item():.3f}, "
          f"saturation={100*sat:.1f}%")
print()


# ---------------------------------------------------------------
# Diagnostic 2: gradient distribution at init
# ---------------------------------------------------------------
loss.backward()
print("Gradient stats per hidden layer at INIT:")
for i, (W, _) in enumerate(layers):
    g = W.grad
    print(f"  layer {i} W.grad: mean={g.mean().item():+.2e}, std={g.std().item():.2e}")
print()


# ---------------------------------------------------------------
# Train with diagnostics
# ---------------------------------------------------------------
LR = 0.1
STEPS = 1000
update_to_weight_ratios = [[] for _ in range(N_LAYERS)]

for step in range(STEPS):
    ix = torch.randint(0, Xtr.shape[0], (32,))
    Xb, Yb = Xtr[ix], Ytr[ix]
    loss, _ = forward(Xb, Yb)
    for p in params:
        p.grad = None
    loss.backward()

    # measure update/weight before applying
    for i, (W, _) in enumerate(layers):
        update = LR * W.grad
        ratio = update.std() / W.std()
        update_to_weight_ratios[i].append(torch.log10(ratio).item())

    for p in params:
        p.data += -LR * p.grad

    if step % 200 == 0:
        print(f"step {step}: loss = {loss.item():.4f}")
print()


# ---------------------------------------------------------------
# Plot the four diagnostic views
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Final activation distributions
loss, activations = forward(Xb, Yb, capture=True)
for i, h in enumerate(activations):
    axes[0, 0].hist(h.flatten().detach().tolist(), bins=50, alpha=0.5,
                    label=f"layer {i}", density=True)
axes[0, 0].set_title("Activation distributions per layer (after training)")
axes[0, 0].set_xlabel("tanh output")
axes[0, 0].legend()

# Final gradient distributions
loss.backward()
for i, (W, _) in enumerate(layers):
    axes[0, 1].hist(W.grad.flatten().tolist(), bins=50, alpha=0.5,
                    label=f"layer {i}", density=True)
axes[0, 1].set_title("Gradient distributions per layer")
axes[0, 1].set_xlabel("dL/dW")
axes[0, 1].legend()

# Weight distributions
for i, (W, _) in enumerate(layers):
    axes[1, 0].hist(W.flatten().detach().tolist(), bins=50, alpha=0.5,
                    label=f"layer {i}", density=True)
axes[1, 0].set_title("Weight distributions per layer")
axes[1, 0].set_xlabel("W value")
axes[1, 0].legend()

# Update / weight ratio over time
for i, ratios in enumerate(update_to_weight_ratios):
    axes[1, 1].plot(ratios, label=f"layer {i}", alpha=0.7)
axes[1, 1].axhline(-3, color="red", linestyle="--", label="rule of thumb: 1e-3")
axes[1, 1].set_title("log10(update / weight ratio) per step")
axes[1, 1].set_xlabel("step")
axes[1, 1].set_ylabel("log10(ratio)")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("diagnostics.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved diagnostics.png -- four key diagnostic plots in one figure.")
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
print("""
The four plots tell a complete story:

  - If activations are healthy and gradients are healthy, the network
    is doing the right thing locally.
  - If weights aren't moving but the loss isn't dropping, the LR is
    too low (or you have dead neurons).
  - If weights jitter wildly, LR is too high.
  - If update/weight ratio is far from 1e-3 in any layer, that layer
    is either ignored (low ratio) or about to explode (high ratio).

These four plots are what professional researchers look at when something
is wrong. Develop the habit of generating them on every long training run.
""")
