"""
04_backprop_batchnorm.py
------------------------
The hardest derivation in the lesson. BatchNorm has multiple paths and
non-obvious dependencies. We'll do it TWICE:

  Part A: ATOMIC backward -- one tiny step at a time, mirroring the
          atomic forward in script 01. Easier to follow, more code.

  Part B: SIMPLIFIED backward -- collapse the whole thing into one
          algebraic expression. Harder to derive, much faster to compute.
          This is what real BatchNorm implementations use.

Forward, with atomic steps:
    bnmeani  = (1/N) * hprebn.sum(0)                  # (1, H)
    bndiff   = hprebn - bnmeani                       # (N, H)
    bndiff2  = bndiff ** 2                            # (N, H)
    bnvar    = (1/(N-1)) * bndiff2.sum(0)             # (1, H)
    bnvar_inv = (bnvar + eps) ** -0.5                 # (1, H)
    bnraw    = bndiff * bnvar_inv                     # (N, H)
    hpreact  = bngain * bnraw + bnbias                # (N, H)

We have dhpreact from script 03. Backprop everything above.

The single trickiest thing: hprebn shows up in TWO places (in `bnmeani`
AND in `bndiff`). Its gradient gets contributions from both paths.
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
N, HIDDEN = state["N"], state["HIDDEN"]
hprebn = state["hprebn"]
bnmeani = state["bnmeani"]
bndiff = state["bndiff"]
bndiff2 = state["bndiff2"]
bnvar = state["bnvar"]
bnvar_inv = state["bnvar_inv"]
bnraw = state["bnraw"]
bngain = state["bngain"]
bnbias = state["bnbias"]
dhpreact = state["dhpreact"]


# ===============================================================
# Part A: ATOMIC BACKWARD
# ===============================================================
print("Part A: Atomic backward (step-by-step, mirrors atomic forward)")
print()

# hpreact = bngain * bnraw + bnbias
# dhpreact -> dbngain, dbnraw, dbnbias
#
# bngain is (1, H), broadcast across N rows. So sum gradient over batch dim.
# bnraw is (N, H), no broadcast.
# bnbias is (1, H), broadcast across N rows. Sum over batch dim.

dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
dbnraw = bngain * dhpreact
dbnbias = dhpreact.sum(0, keepdim=True)
cmp("bngain", dbngain, bngain)
cmp("bnraw", dbnraw, bnraw)
cmp("bnbias", dbnbias, bnbias)


# bnraw = bndiff * bnvar_inv
# bndiff is (N, H), bnvar_inv is (1, H) broadcast across rows.

dbndiff = bnvar_inv * dbnraw                                  # (N, H)
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)           # sum over batch
cmp("bnvar_inv", dbnvar_inv, bnvar_inv)


# bnvar_inv = (bnvar + eps) ** -0.5
# d/dx (x^-0.5) = -0.5 * x^-1.5

dbnvar = -0.5 * (bnvar + 1e-5) ** -1.5 * dbnvar_inv
cmp("bnvar", dbnvar, bnvar)


# bnvar = (1/(N-1)) * bndiff2.sum(0)
# bndiff2 is (N, H), each row contributes 1/(N-1) to bnvar.

dbndiff2 = (1.0 / (N - 1)) * torch.ones_like(bndiff2) * dbnvar
cmp("bndiff2", dbndiff2, bndiff2)


# bndiff2 = bndiff ** 2
# d/dx (x^2) = 2x. ADD to dbndiff (which already has a piece from bnraw).

dbndiff += 2 * bndiff * dbndiff2
cmp("bndiff", dbndiff, bndiff)


# bndiff = hprebn - bnmeani
# d/d(hprebn) = +1 (broadcasts)
# d/d(bnmeani) = -1, summed over batch (since bnmeani is (1, H) broadcast over N)

dhprebn = dbndiff.clone()
dbnmeani = (-dbndiff).sum(0, keepdim=True)
cmp("bnmeani", dbnmeani, bnmeani)


# bnmeani = (1/N) * hprebn.sum(0)
# Each row of hprebn contributes (1/N) to bnmeani.
# ADD to dhprebn (which already has a piece from bndiff above).

dhprebn += (1.0 / N) * torch.ones_like(hprebn) * dbnmeani
cmp("hprebn", dhprebn, hprebn)


print()
print("Part A complete: atomic BatchNorm backward verified.")
print()


# ===============================================================
# Part B: SIMPLIFIED BACKWARD (the real-world one)
# ===============================================================
# By substituting the chain through bnmeani, bndiff, bndiff2, bnvar,
# bnvar_inv, bnraw all the way down, the whole BatchNorm backward
# collapses to a SINGLE expression for dhprebn:
#
# dhprebn = (bngain * bnvar_inv / N) * (
#     N * dhpreact
#     - dhpreact.sum(0)
#     - (N/(N-1)) * bnraw * (dhpreact * bnraw).sum(0)
# )
#
# This is what production BatchNorm implementations use -- one line
# instead of seven, MUCH faster on a GPU.
#
# Let's verify it gives the same answer.
print("Part B: Single-line simplified backward (production formula)")
print()

dhprebn_simplified = (bngain * bnvar_inv / N) * (
    N * dhpreact
    - dhpreact.sum(0)
    - (N / (N - 1)) * bnraw * (dhpreact * bnraw).sum(0)
)
cmp("hprebn (simplified)", dhprebn_simplified, hprebn)


print()
print("BatchNorm backward complete -- both forms verified equivalent.")
print()


# Save what we need for the next script
state["dhprebn"] = dhprebn
state["dbngain"] = dbngain
state["dbnbias"] = dbnbias
torch.save(state, "state.pt")


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - BatchNorm backward through the atomic forward is ~7 chain-rule steps.
# - Each step uses the same three rules from script 02:
#     broadcast/sum, sum/broadcast, identity for elementwise ops.
# - The simplified single-line form is what real BN implementations use.
#   It's algebraically equivalent but much faster.
# - The reason BatchNorm is hard: hprebn appears in TWO places (mean
#   and diff), so its gradient has contributions from BOTH paths.
# - This is also why LayerNorm is preferred in transformers -- it
#   normalizes across features (independent per example), so its backward
#   is much simpler. Same reason GroupNorm exists.
