"""
01_explore_dataset.py
---------------------
Before any modeling, look at the data. We have a list of names. Our task:
generate plausible NEW names that look like they could belong on this list.

That's it. That's the whole problem. We're not solving a riddle, we're
learning the statistics of how letters follow other letters in real names.

Key concept introduced here: BIGRAM. We model a name as a sequence of
character pairs:
    "emma" -> (., e), (e, m), (m, m), (m, a), (a, .)

The '.' on each end is a SPECIAL token meaning "start" or "end". Without
it, the model wouldn't know where words begin or end.

A bigram model says: P(next character | previous character).
That's a HUGE simplification -- the next letter probably depends on more
than just the one before it -- but it's enough to generate name-like
strings, and it's the simplest possible language model. Start small.
"""

from data_utils import load_names, build_vocab


names = load_names()
print(f"\nFirst 8 names: {names[:8]}")
print(f"Number of names: {len(names)}")
print(f"Shortest name: {min(len(n) for n in names)} chars")
print(f"Longest name:  {max(len(n) for n in names)} chars")
print()

# Build vocabulary: every distinct character + the '.' boundary token
stoi, itos = build_vocab(names)
print(f"Characters in dataset: {sorted(set(''.join(names)))}")
print(f"Vocab size (with '.'): {len(stoi)}")
print()


# ---------------------------------------------------------------
# Show what bigrams look like for the first few names
# ---------------------------------------------------------------
print("Bigrams for the first 3 names:")
for w in names[:3]:
    print(f"  {w!r}:")
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        print(f"    ({ch1}, {ch2})")
    print()


# ---------------------------------------------------------------
# Total number of bigrams in the whole dataset
# ---------------------------------------------------------------
total_bigrams = 0
for w in names:
    chs = ["."] + list(w) + ["."]
    total_bigrams += len(chs) - 1
print(f"Total bigrams across the dataset: {total_bigrams}")
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - The dataset is a list of words.
# - We add a '.' boundary token so the model knows where names start/end.
# - A bigram is just a pair (previous_char, next_char).
# - Our entire training signal is: WHICH bigrams appeared, and HOW OFTEN.
#
# Next: count them. That's it -- the entire first half of the lesson is
# JUST counting bigrams and turning the counts into probabilities.
