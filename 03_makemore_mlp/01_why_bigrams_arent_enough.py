"""
01_why_bigrams_arent_enough.py
------------------------------
We left the previous lesson with a bigram model. It learned the statistics
of (prev_char, next_char) pairs by counting or via a 1-layer NN. The two
gave essentially the same probability table.

Why move to a neural net with embeddings?

Try to extend the count-table approach to TRIGRAMS -- conditioning on the
previous TWO characters instead of one. Now the table has shape
(V, V, V) = 27^3 = 19,683 cells. Manageable.

Now FOUR previous characters: 27^4 = 531,441 cells. Still manageable.

EIGHT previous characters? 27^8 = ~282 billion cells. With only 32k names,
most cells are empty. The model can't learn from data it doesn't have.

This is the CURSE OF DIMENSIONALITY for n-gram language models. The number
of possible contexts grows exponentially with context length, but the
data we have is fixed. A counting model has no way to share information
between similar but distinct contexts.

The neural net approach (Bengio et al., 2003) fixes this with two ideas:

  1. EMBED each character as a low-dimensional vector. Characters that
     play similar roles end up with similar vectors. Information is
     SHARED across contexts.

  2. CONCATENATE the embeddings of the context characters and feed them
     through an MLP. The MLP learns features of the JOINT context, not
     a separate parameter per (c1, c2, c3, ...) tuple.

This script just does the math to show the explosion. The rest of the
lesson builds the MLP that solves it.
"""

import math


V = 27  # vocab size: 26 letters + '.' boundary

print("Cells in an n-gram count table for vocab size 27:")
print()
print(f"  {'context length':>15}  {'cells':>20}")
print(f"  {'-'*15}  {'-'*20}")
for k in range(1, 11):
    cells = V ** k
    print(f"  {k:>15}  {cells:>20,}")
print()


# ---------------------------------------------------------------
# Compare to dataset size
# ---------------------------------------------------------------
# A "name" is on average ~7 characters, plus 2 boundary tokens. With
# 32,033 names that's roughly:
NAMES = 32033
AVG_LEN = 7
TOTAL_BIGRAMS = NAMES * (AVG_LEN + 1)
print(f"Approx total training examples in names.txt: {TOTAL_BIGRAMS:,}")
print()

print("So with context length 4, we have roughly 1 example per cell on average.")
print("With context length 5, most cells are EMPTY -- the model can't learn them.")
print()


# ---------------------------------------------------------------
# Embedding parameter count for comparison
# ---------------------------------------------------------------
# In the MLP we'll build, each character has a learned vector of dim D.
# For D=2 and block_size=3, the embedding-table parameters are 27*2 = 54.
# The context is the CONCATENATION of D*block_size = 6 numbers.
#
# Total parameters in the Bengio MLP (D=10, hidden=200, block_size=3):
D, HIDDEN, BLOCK = 10, 200, 3
embedding_params = V * D
W1_params = (D * BLOCK) * HIDDEN + HIDDEN
W2_params = HIDDEN * V + V
total = embedding_params + W1_params + W2_params

print(f"For an MLP with D={D}, block_size={BLOCK}, hidden={HIDDEN}:")
print(f"  embedding table : {embedding_params:>8,} params")
print(f"  hidden layer    : {W1_params:>8,} params")
print(f"  output layer    : {W2_params:>8,} params")
print(f"  TOTAL           : {total:>8,} params")
print()
print(f"Compare to the {V**BLOCK:,}-cell trigram table: same context window,")
print(f"way fewer parameters, AND the model generalizes across similar contexts.")
