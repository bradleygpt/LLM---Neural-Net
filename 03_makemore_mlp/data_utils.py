"""
data_utils.py
-------------
Shared dataset loading and vocabulary building. Same pattern as the
bigram lesson, but with one new function: build_dataset, which produces
the (X, Y) tensors the MLP trains on.

Drop names.txt from
    https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
into this folder for full results. Otherwise, a small fallback runs.
"""

import os
import torch


_FALLBACK = [
    "emma", "olivia", "ava", "isabella", "sophia", "charlotte", "mia",
    "amelia", "harper", "evelyn", "abigail", "emily", "elizabeth", "mila",
    "ella", "avery", "sofia", "camila", "aria", "scarlett", "victoria",
    "madison", "luna", "grace", "chloe", "penelope", "layla", "riley",
    "zoey", "nora", "lily", "eleanor", "hannah", "lillian", "addison",
    "aubrey", "ellie", "stella", "natalie", "zoe", "leah", "hazel",
    "violet", "aurora", "savannah", "audrey", "brooklyn", "bella",
    "claire", "skylar",
]


def load_names(path="names.txt"):
    if os.path.exists(path):
        with open(path) as f:
            names = [line.strip().lower() for line in f if line.strip()]
        # Sanity check: a real names.txt has alphabetic-only entries
        names = [n for n in names if n.isalpha()]
        print(f"Loaded {len(names)} names from {path}")
        return names
    print(f"[note] {path} not found -- using {len(_FALLBACK)}-name fallback.")
    print("[note] Download from https://raw.githubusercontent.com/karpathy/"
          "makemore/master/names.txt for full results.")
    return list(_FALLBACK)


def build_vocab(names):
    chars = sorted(set("".join(names)))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def build_dataset(names, stoi, block_size=3):
    """
    Build (X, Y) for the MLP.
    block_size = number of previous chars used as context (the "context window").
    For block_size=3 and name "emma":
        ... -> e
        ..e -> m
        .em -> m
        emm -> a
        mma -> .
    Returns X of shape (N, block_size), Y of shape (N,), both LongTensors.
    """
    X, Y = [], []
    for w in names:
        context = [0] * block_size  # start with all '.' (index 0)
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # slide the window: drop the oldest, append the new char
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


if __name__ == "__main__":
    import random
    names = load_names()
    stoi, itos = build_vocab(names)
    random.seed(42)
    random.shuffle(names)

    X, Y = build_dataset(names[:5], stoi, block_size=3)
    print(f"Sanity check on first 5 names with block_size=3:")
    print(f"  X shape: {X.shape}, Y shape: {Y.shape}")
    for i in range(min(10, len(X))):
        ctx = "".join(itos[c.item()] for c in X[i])
        print(f"  {ctx} --> {itos[Y[i].item()]}")
