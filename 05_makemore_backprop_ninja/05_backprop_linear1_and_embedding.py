"""
05_backprop_linear1_and_embedding.py
------------------------------------
Two more layers to go. We have dhprebn from script 04. Continue:

    hprebn = embcat @ W1 + b1     (the first linear layer)
    embcat = emb.view(N, -1)      (the flatten/reshape)
    emb = C[Xb]                   (the embedding lookup)

The Linear 1 backward is identical in form to Linear 2 (matmul rule).

The reshape backward is conceptually trivial (.view doesn't move data,
just reinterprets shape), but you have to remember to ".view back" so
the gradient has the original shape.

The embedding lookup backward is the most counterintuitive piece. C[Xb]
is "for each row of Xb, copy that row from C." The gradient flows back
into the SPECIFIC ROWS of C that were indexed. If multiple Xb entries
indexed the same row of C, the gradients ACCUMULATE there.

We use index_add_ for this -- the safe, correct way to accumulate
gradients into specific rows.
"""

import torch


def cmp(name, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    status = "EXACT " if ex else ("CLOSE " if app else "WRONG ")
    print(f"  {status} {name:>20s} | maxdiff = {maxdiff:.2e}")


# Load
state = torch.load("state.pt", weights_only=False)
N, V, BLOCK, D = state["N"], state["V"], state["BLOCK"], state["D"]
Xb = state["Xb"]
embcat = state["embcat"]
emb = state["emb"]
W1 = state["W1"]
b1 = state["b1"]
C = state["C"]
dhprebn = state["dhprebn"]


print("Backpropping through Linear 1, reshape, and embedding...")
print()


# ---------------------------------------------------------------
# hprebn = embcat @ W1 + b1
# ---------------------------------------------------------------
# Same matmul rule as Linear 2.

dembcat = dhprebn @ W1.T
dW1 = embcat.T @ dhprebn
db1 = dhprebn.sum(0)
cmp("embcat", dembcat, embcat)
cmp("W1", dW1, W1)
cmp("b1", db1, b1)


# ---------------------------------------------------------------
# embcat = emb.view(N, BLOCK*D)
# ---------------------------------------------------------------
# .view() reshapes without moving memory. Backward = reshape back.
# emb shape: (N, BLOCK, D). embcat shape: (N, BLOCK*D).

demb = dembcat.view(emb.shape)
cmp("emb", demb, emb)


# ---------------------------------------------------------------
# emb = C[Xb]
# ---------------------------------------------------------------
# This is the tricky one. C has shape (V, D). Xb has shape (N, BLOCK)
# of integer indices. The forward is "for each (i, j), copy row C[Xb[i,j]]
# into emb[i, j]."
#
# Backward: gradient at emb[i, j, :] should ADD into row C[Xb[i,j], :].
# If two different (i, j) pairs share the same Xb value, both contribute.
#
# We use a loop here for clarity, mirroring Karpathy's approach in the
# video. PyTorch has scatter_add_ for the same effect (more efficient).

dC = torch.zeros_like(C)
for i in range(N):
    for j in range(BLOCK):
        ix = Xb[i, j].item()
        dC[ix] += demb[i, j]
cmp("C", dC, C)


print()
print("Linear 1 + reshape + embedding backward complete.")
print()


# ---------------------------------------------------------------
# Save the parameter gradients we'll use for actual training
# ---------------------------------------------------------------
state["dW1"] = dW1
state["db1"] = db1
state["dC"] = dC
state["dembcat"] = dembcat
state["demb"] = demb
torch.save(state, "state.pt")
print("Saved all parameter gradients to state.pt for the final training script.")


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - Linear backward = the matmul rule (dY @ W.T, X.T @ dY) plus a sum
#   for the bias.
# - .view() backward is .view(original_shape). Reshape with no data move.
# - Embedding lookup backward ACCUMULATES gradients into the specific
#   rows that were indexed. If a row is indexed multiple times, all
#   contributions sum.
#
# At this point you have manually computed:
#   dC, dW1, db1, dW2, db2, dbngain, dbnbias
# the gradients of EVERY parameter in the network.
#
# In the next script we use those gradients to TRAIN -- without ever
# calling loss.backward(). Just to prove we can.
