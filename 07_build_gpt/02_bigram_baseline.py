"""
02_bigram_baseline.py
---------------------
Before we go anywhere near attention, build the simplest possible
language model on Shakespeare: a bigram model, trained as a neural net.

This is essentially lesson 02's bigram NN, but on text instead of names.
The point: establish a BASELINE so we know what loss number "good" looks
like for this dataset. The transformer should beat the baseline by a lot.

The architecture is just a single embedding lookup:
    For each input character index, look up a row in an embedding table.
    That row IS the logits over the vocabulary for the next character.

No hidden layers. No attention. Just a (V, V) lookup table trained with
gradient descent.

Expected loss: ~2.5 nats. For comparison: the 1M-char dataset has 65
unique characters, so a uniform random model has loss log(65) = 4.17.
The bigram does substantially better than uniform.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module


# Import load_text from script 01 (no easy way without making this messier;
# we copy in the function for self-containment)
import os

_FALLBACK = """First Citizen: Before we proceed any further, hear me speak.""" * 5000


def load_text(path="input.txt"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    print(f"[note] {path} not found -- using fallback.")
    return _FALLBACK


# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------
torch.manual_seed(1337)

text = load_text()
chars = sorted(set(text))
V = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s): return [stoi[c] for c in s]
def decode(ids): return "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

# 90/10 train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens, Vocab: {V}")
print()


# ---------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------
# Sample BATCH_SIZE random starting points, then for each grab BLOCK_SIZE
# consecutive tokens as input + the next BLOCK_SIZE as targets.
# Each (input, target) pair is one training example.
BLOCK = 8
BATCH = 32

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - BLOCK, (BATCH,))
    x = torch.stack([d[i:i + BLOCK] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK + 1] for i in ix])
    return x, y


xb, yb = get_batch("train")
print(f"Batch shapes: x = {tuple(xb.shape)}, y = {tuple(yb.shape)}")
print(f"x[0]: {xb[0].tolist()}")
print(f"y[0]: {yb[0].tolist()}")
print(f"  first input char:  {decode([xb[0, 0].item()])!r}")
print(f"  first target char: {decode([yb[0, 0].item()])!r}")
print(f"  -> for input position i, the model predicts the character at position i+1")
print()


# ---------------------------------------------------------------
# Bigram model
# ---------------------------------------------------------------
class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each integer index in [0, V) maps to a row of length V.
        # That row IS the logits over the next character.
        self.embed = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T)
        # logits: (B, T, V) -- one V-vector per position
        logits = self.embed(idx)
        if targets is None:
            return logits, None
        B, T, V = logits.shape
        # cross_entropy expects (N, V) and (N,), so flatten the (B, T) axes
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new):
        for _ in range(max_new):
            logits, _ = self(idx)        # (B, T, V)
            logits = logits[:, -1, :]    # take last position only
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


m = BigramLM(V)
print(f"Bigram model: {sum(p.numel() for p in m.parameters()):,} parameters")
print()


# ---------------------------------------------------------------
# Sanity check: untrained model loss should be near log(V)
# ---------------------------------------------------------------
import math
xb, yb = get_batch("train")
_, untrained_loss = m(xb, yb)
print(f"Untrained loss: {untrained_loss.item():.4f}")
print(f"Expected (log V): {math.log(V):.4f}")
print()


# ---------------------------------------------------------------
# Train
# ---------------------------------------------------------------
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-2)

STEPS = 5000
print(f"Training for {STEPS:,} steps...")
for step in range(STEPS):
    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 1000 == 0 or step == STEPS - 1:
        with torch.no_grad():
            xv, yv = get_batch("val")
            _, val_loss = m(xv, yv)
        print(f"  step {step:>5}/{STEPS}  train loss = {loss.item():.4f}  val loss = {val_loss.item():.4f}")
print()


# ---------------------------------------------------------------
# Generate some text
# ---------------------------------------------------------------
context = torch.zeros((1, 1), dtype=torch.long)
gen = m.generate(context, max_new=300)[0].tolist()
print("Sample generated text from the bigram baseline:")
print("-" * 60)
print(decode(gen))
print("-" * 60)
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - Bigram on Shakespeare gets to ~2.5 nats. That's our floor.
# - Output looks like keyboard mash with occasional real fragments
#   ("hat", "the", "and") because bigram statistics force common
#   letter pairs.
# - The model has NO context awareness beyond the previous character.
# - Real Shakespeare-like output requires LONGER context. That's why
#   the next thing we add is self-attention -- it lets each position
#   look at MANY previous positions and decide what's relevant.
#
# The training loop, batching, and generation pattern set up here are
# IDENTICAL to what we'll use for the transformer. Only the model
# class changes.
