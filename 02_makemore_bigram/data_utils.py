"""
data_utils.py
-------------
Shared helpers used by every script. Loads the names dataset.

In Karpathy's video, the dataset is `names.txt` -- 32,033 names, one per
line. Get it from:
    https://raw.githubusercontent.com/karpathy/makemore/master/names.txt

Drop that file in this directory and the scripts will pick it up. If you
don't have it, a tiny built-in fallback (~50 names) lets you still run
everything end-to-end. Probabilities and losses will differ from the
video numbers because the dataset is smaller, but the SHAPES of every
output will match.
"""

import os

# A small fallback so scripts run without external files.
# Real lesson uses 32,033 names.
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
    """Return a list of lowercase names. Falls back if file missing."""
    if os.path.exists(path):
        with open(path) as f:
            names = [line.strip().lower() for line in f if line.strip()]
        print(f"Loaded {len(names)} names from {path}")
        return names

    print(f"[note] {path} not found -- using {len(_FALLBACK)}-name fallback.")
    print("[note] Download from https://raw.githubusercontent.com/karpathy/"
          "makemore/master/names.txt for full results.")
    return list(_FALLBACK)


def build_vocab(names):
    """Build the char-to-int and int-to-char maps. '.' is start/end token."""
    chars = sorted(set("".join(names)))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


if __name__ == "__main__":
    # Sanity check
    names = load_names()
    stoi, itos = build_vocab(names)
    print(f"vocab size: {len(stoi)}  (26 letters + '.')")
    print(f"first 5 names: {names[:5]}")
    print(f"stoi['a'] = {stoi['a']}, itos[1] = {itos[1]}")
