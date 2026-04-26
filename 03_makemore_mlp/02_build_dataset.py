"""
02_build_dataset.py
-------------------
Set up the training data for the MLP.

Two new ideas vs the bigram lesson:

  1. BLOCK SIZE.  We use the previous 3 characters as context, not just 1.
     For "emma" with block_size=3:
        ... -> e   (start of word)
        ..e -> m
        .em -> m
        emm -> a
        mma -> .   (end of word)
     X has shape (N, 3) instead of (N,). Y is still (N,).

  2. TRAIN / DEV / TEST SPLITS.  We don't measure model quality on the
     same data we trained on -- we'd just be measuring memorization. So
     we split the names randomly:
        - train (80%):  what we update parameters on
        - dev (10%):    what we tune hyperparameters on
        - test (10%):   what we report final numbers on, ONCE, at the end

     Touching the test set repeatedly while iterating is a classic
     mistake that turns it into a second dev set. Discipline matters.
"""

import random
import torch
from data_utils import load_names, build_vocab, build_dataset


names = load_names()
stoi, itos = build_vocab(names)
V = len(stoi)
print(f"vocab size: {V}")
print()


# ---------------------------------------------------------------
# Shuffle + split
# ---------------------------------------------------------------
random.seed(42)
random.shuffle(names)

n1 = int(0.8 * len(names))
n2 = int(0.9 * len(names))

train_names = names[:n1]
dev_names = names[n1:n2]
test_names = names[n2:]

print(f"train: {len(train_names)} names")
print(f"dev:   {len(dev_names)} names")
print(f"test:  {len(test_names)} names")
print()


# ---------------------------------------------------------------
# Build (X, Y) tensors for each split
# ---------------------------------------------------------------
BLOCK_SIZE = 3

Xtr, Ytr = build_dataset(train_names, stoi, BLOCK_SIZE)
Xdev, Ydev = build_dataset(dev_names, stoi, BLOCK_SIZE)
Xte, Yte = build_dataset(test_names, stoi, BLOCK_SIZE)

print(f"Xtr: {tuple(Xtr.shape)}  Ytr: {tuple(Ytr.shape)}")
print(f"Xdev: {tuple(Xdev.shape)}  Ydev: {tuple(Ydev.shape)}")
print(f"Xte: {tuple(Xte.shape)}  Yte: {tuple(Yte.shape)}")
print()


# ---------------------------------------------------------------
# Look at the first 10 training examples
# ---------------------------------------------------------------
print("First 10 training examples:")
for i in range(min(10, len(Xtr))):
    ctx = "".join(itos[c.item()] for c in Xtr[i])
    target = itos[Ytr[i].item()]
    print(f"  '{ctx}' --> '{target}'")
print()


# ---------------------------------------------------------------
# Save tensors so later scripts don't rebuild
# ---------------------------------------------------------------
torch.save({
    "Xtr": Xtr, "Ytr": Ytr,
    "Xdev": Xdev, "Ydev": Ydev,
    "Xte": Xte, "Yte": Yte,
    "stoi": stoi, "itos": itos,
    "block_size": BLOCK_SIZE,
}, "dataset.pt")
print("Saved dataset to dataset.pt")


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - Each training example is (context_window, target_char).
# - The window slides one character at a time across each name.
# - Splits keep our evaluation honest: train != dev != test.
# - This dataset is now ready for any model -- bigram, MLP, transformer.
#   The model sees X[i] as input and is asked to predict Y[i].
