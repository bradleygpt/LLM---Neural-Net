"""
07_sample_and_outlook.py
------------------------
Generate names from the trained MLP, then look at where this lesson hands
off to the next.

Sampling is the same algorithm as in the bigram lesson, just with a
deeper forward pass:

    1. Start with a context of all '.' tokens.
    2. Run the forward pass to get a distribution over the next character.
    3. Sample from that distribution.
    4. Slide the context window: drop the oldest, add the sampled char.
    5. If we sampled '.', stop. Otherwise, loop.
"""

import torch
import torch.nn.functional as F


# Load the trained model
ckpt = torch.load("trained_mlp.pt", weights_only=False)
C, W1, b1, W2, b2 = ckpt["C"], ckpt["W1"], ckpt["b1"], ckpt["W2"], ckpt["b2"]
itos, stoi = ckpt["itos"], ckpt["stoi"]
BLOCK = ckpt["block_size"]
V = len(itos)


# ---------------------------------------------------------------
# Sampling function
# ---------------------------------------------------------------
@torch.no_grad()
def sample_name(generator):
    out = []
    context = [0] * BLOCK  # all '.' tokens
    while True:
        x = torch.tensor([context])
        emb = C[x]
        flat = emb.view(emb.shape[0], -1)
        h = torch.tanh(flat @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=generator).item()
        if ix == 0:  # the '.' token
            break
        out.append(itos[ix])
        context = context[1:] + [ix]
        if len(out) > 50:  # safety cap
            break
    return "".join(out)


g = torch.Generator().manual_seed(2147483647 + 10)
print("20 names sampled from the trained MLP:")
for _ in range(20):
    print(f"  {sample_name(g)}")
print()


# ---------------------------------------------------------------
# What we just learned (and what's next)
# ---------------------------------------------------------------
print("=" * 60)
print("OUTLOOK")
print("=" * 60)
print("""
Compared to the bigram model from the previous lesson, the MLP samples
should look noticeably more name-like. With the full names.txt dataset
the loss drops from ~2.45 (bigram) to ~2.17 (MLP).

But there's a problem we glossed over. Try changing the parameter inits
in script 06 from `* 0.1` to plain `torch.randn(...)` (the PyTorch
default). Watch what happens:

    - Initial loss is HUGE (~27 instead of ~3.3)
    - The model spends thousands of steps just SHRINKING the logits
      to a reasonable scale before it can start learning anything useful.

This is "the hockey stick" curve in early training -- a giant drop in
the first few hundred steps that's mostly recovery from a bad init.

That's the headline of the next lesson:

    LESSON 04 (makemore part 3): Activations, Gradients, BatchNorm

    - Why initialization matters and how to do it right (Kaiming init)
    - Why tanh saturates and what "dead neurons" are
    - The gradient flow problem in deep networks
    - BatchNorm: the trick that made deep networks trainable
    - Diagnostic tools: weight/grad/update ratio plots

Once you've seen those, you have everything you need to build a
transformer. We get there in lesson 07 (Build GPT).
""")
