"""
08_pytorch_bonus.py
-------------------
BONUS. This script mirrors Karpathy's video code almost line-for-line,
using PyTorch instead of NumPy. Run this if you have torch installed
locally to see that:

    - the API is dramatically smaller (autograd handles the backward pass)
    - the final loss matches the NumPy version closely
    - the SAMPLES look similar in character (no pun intended) to ours

Requires: torch
    pip install torch

If torch isn't installed this script just prints instructions and exits.
The pedagogical content is in scripts 01-07; this is just to ground the
"this is what real PyTorch code looks like" intuition.
"""

import sys

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("PyTorch not installed. To run this script:")
    print("    pip install torch")
    print("Then re-run:  python3 08_pytorch_bonus.py")
    sys.exit(0)

from data_utils import load_names, build_vocab


names = load_names()
stoi, itos = build_vocab(names)
V = len(stoi)


# ---------------------------------------------------------------
# Build training set
# ---------------------------------------------------------------
xs, ys = [], []
for w in names:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
N = xs.numel()
print(f"{N} bigram pairs, vocab size {V}")


# ---------------------------------------------------------------
# Initialize W with autograd
# ---------------------------------------------------------------
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((V, V), generator=g, requires_grad=True)


# ---------------------------------------------------------------
# Training loop -- the WHOLE thing is ~10 lines once autograd does the work
# ---------------------------------------------------------------
LR = 50.0
EPOCHS = 200

for epoch in range(EPOCHS):
    # forward
    x_onehot = F.one_hot(xs, num_classes=V).float()
    logits = x_onehot @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(N), ys].log().mean() + 0.01 * (W**2).mean()

    # backward
    W.grad = None  # zero grad
    loss.backward()

    # update
    W.data += -LR * W.grad

    if epoch % 20 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch:>4}: loss = {loss.item():.4f}")

print()


# ---------------------------------------------------------------
# Sample
# ---------------------------------------------------------------
g_sample = torch.Generator().manual_seed(2147483647)
print("10 names from the PyTorch-trained NN:")
for _ in range(10):
    out = []
    ix = stoi["."]
    while True:
        x_oh = F.one_hot(torch.tensor([ix]), num_classes=V).float()
        logits = x_oh @ W
        probs = logits.exp() / logits.exp().sum(1, keepdims=True)
        ix = torch.multinomial(probs, num_samples=1, generator=g_sample).item()
        if ix == stoi["."]:
            break
        out.append(itos[ix])
    print(f"  {''.join(out)}")
print()


# ---------------------------------------------------------------
# What you should see
# ---------------------------------------------------------------
# - Final loss within ~0.01 of the NumPy version (script 06).
# - Names that "look" similar in shape -- name-like for the full dataset,
#   nonsense-but-pronounceable for the fallback dataset.
# - Loop body is much shorter: no manual softmax-NLL gradient, no manual
#   matmul backprop. Autograd derives the same dW we wrote out by hand.
#
# That's the value proposition of a real DL framework: same math,
# none of the bookkeeping. Now that you've done the bookkeeping yourself,
# you know what autograd is doing under the hood.
