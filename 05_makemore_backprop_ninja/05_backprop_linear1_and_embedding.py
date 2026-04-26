"""
05_backprop_linear1_and_embedding.py
------------------------------------
Two more layers. We have dhprebn from script 04. Continue:

    hprebn = embcat @ W1 + b1     (the first linear layer)
    embcat = emb.view(N, -1)      (the flatten/reshape)
    emb = C[Xb]                   (the embedding lookup)

Linear 1 backward = same matmul rule as Linear 2.
Reshape backward = .view back to original shape (no data movement).
Embedding lookup backward = the most counterintuitive piece. C[Xb] copies
specific rows of C. Gradients flow back into THOSE specific rows. If
multiple Xb entries indexed the same row, gradients ACCUMULATE.
"""

import torch


def cmp(name, dt, t_ref_grad):
    ex = torch.all(dt == t_ref_grad).item()
    app = torch.allclose(dt, t_ref_grad)
    maxdiff = (dt - t_ref_grad).abs().max().item()
    status = "EXACT " if ex else ("CLOSE " if app else "WRONG ")
    print(f"  {status} {name:>20s} | maxdiff = {maxdiff:.2e}")


# Load
state = torch.load("state.pt", weights_only=False)
N, V, BLOCK, D = state["N"], state["V"], state["BLOCK"], state["D"]
Xb = state["Xb"]
embcat = state["embcat"]
emb = state["emb"]
W1 = state["W1"]
C = state["C"]
dhprebn = state["dhprebn"]


print("Backpropping through Linear 1, reshape, and embedding...")
print()


# ---------------------------------------------------------------
# hprebn = embcat @ W1 + b1
# ---------------------------------------------------------------
dembcat = dhprebn @ W1.T
dW1 = embcat.T @ dhprebn
db1 = dhprebn.sum(0)
cmp("embcat", dembcat, state["embcat_grad"])
cmp("W1", dW1, state["W1_grad"])
cmp("b1", db1, state["b1_grad"])


# ---------------------------------------------------------------
# embcat = emb.view(N, BLOCK*D)
# ---------------------------------------------------------------
# .view() reshapes without moving memory. Backward = reshape back.

demb = dembcat.view(emb.shape)
cmp("emb", demb, state["emb_grad"])


# ---------------------------------------------------------------
# emb = C[Xb]
# ---------------------------------------------------------------
# C: (V, D), Xb: (N, BLOCK), emb: (N, BLOCK, D)
# Forward: for each (i, j), emb[i, j, :] = C[Xb[i, j], :]
# Backward: gradient at emb[i, j, :] ADDS into row C[Xb[i, j], :].
# If two (i, j) pairs share the same Xb value, both contribute.

dC = torch.zeros_like(C)
for i in range(N):
    for j in range(BLOCK):
        ix = Xb[i, j].item()
        dC[ix] += demb[i, j]
cmp("C", dC, state["C_grad"])


print()
print("Linear 1 + reshape + embedding backward complete.")
print()


# Save
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
# - Linear backward = matmul rule (dY @ W.T, X.T @ dY) plus a sum for bias.
# - .view() backward is .view(original_shape).
# - Embedding lookup backward ACCUMULATES gradients into the specific
#   rows that were indexed.
#
# At this point we've manually computed:
#   dC, dW1, db1, dW2, db2, dbngain, dbnbias
# the gradients of EVERY parameter in the network.
#
# Next: use those gradients to TRAIN -- without ever calling
# loss.backward(). Just to prove we can.
