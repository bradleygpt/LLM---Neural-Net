"""
03_backprop_linear2_and_tanh.py
-------------------------------
Continue walking backward. We now have dlogits. Backprop through:

    logits = h @ W2 + b2     (the second linear layer)
    h = tanh(hpreact)        (the tanh nonlinearity)

The big one here is matrix multiplication. The matmul backward formula
is the most-used identity in deep learning:

    Forward:  Y = X @ W
    Backward: dX = dY @ W.T
              dW = X.T @ dY

It looks magical at first. The intuition: dimensional analysis. If
Y = X @ W has shapes (N,K) = (N,M) @ (M,K), then dY is (N,K). For dX
to be (N,M) we need to multiply dY (N,K) by something (K,M) -- which is
W.T. For dW to be (M,K) we need (M,N) @ (N,K) -- which is X.T @ dY.

For Y = X @ W + b where b is (K,) broadcast over the batch dim:
    db = dY.sum(0)       (sum out the batch dim, since b was broadcast over it)
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
dlogits = state["dlogits"]
h = state["h"]
W2 = state["W2"]
b2 = state["b2"]
hpreact = state["hpreact"]

print("Backpropping through Linear 2 and tanh...")
print()


# ---------------------------------------------------------------
# logits = h @ W2 + b2
# ---------------------------------------------------------------
# h:      (N, HIDDEN)
# W2:     (HIDDEN, V)
# b2:     (V,)            broadcast over batch
# logits: (N, V)
#
# By the matmul backward formula:
#   dh  = dlogits @ W2.T               (N, HIDDEN)
#   dW2 = h.T @ dlogits                (HIDDEN, V)
#   db2 = dlogits.sum(0)               (V,) -- sum out batch since b2 broadcasts

dh = dlogits @ W2.T
dW2 = h.T @ dlogits
db2 = dlogits.sum(0)
cmp("h", dh, h)
cmp("W2", dW2, W2)
cmp("b2", db2, b2)


# ---------------------------------------------------------------
# h = tanh(hpreact)
# ---------------------------------------------------------------
# Element-wise. d(tanh(x))/dx = 1 - tanh(x)^2 = 1 - h^2
# (we use h directly because it's already tanh(hpreact))

dhpreact = (1.0 - h**2) * dh
cmp("hpreact", dhpreact, hpreact)


print()
print("Linear 2 + tanh backward complete.")
print()


# Save dhpreact for the next script (BatchNorm — the hard one)
state["dh"] = dh
state["dW2"] = dW2
state["db2"] = db2
state["dhpreact"] = dhpreact
torch.save(state, "state.pt")
print("Saved dhpreact (and others) to state.pt for script 04.")


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - The matmul backward formula -- the SINGLE most important identity:
#       Y = X @ W   =>   dX = dY @ W.T,   dW = X.T @ dY
# - For y = x + b with b broadcast:  db = dy.sum(broadcast_dims)
# - For tanh:  dx = (1 - tanh(x)^2) * dy
#
# The matmul formula is everywhere -- every dense layer in every
# transformer uses exactly this. Internalize it.
