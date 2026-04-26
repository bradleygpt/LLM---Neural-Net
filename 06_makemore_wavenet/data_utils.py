"""
data_utils.py - dataset helpers, same pattern as previous lessons.
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
        names = [n for n in names if n.isalpha()]
        print(f"Loaded {len(names)} names from {path}")
        return names
    print(f"[note] {path} not found -- using {len(_FALLBACK)}-name fallback.")
    print("[note] Download names.txt from Karpathy's makemore repo for full results.")
    return list(_FALLBACK)


def build_vocab(names):
    chars = sorted(set("".join(names)))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def build_dataset(names, stoi, block_size=8):
    """For WaveNet we'll typically use block_size=8 (not 3 like before)."""
    X, Y = [], []
    for w in names:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)
