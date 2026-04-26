"""
02_backprop_through_loss.py
---------------------------
Start at the END of the forward pass and walk BACKWARD. The first piece
to derive: the cross-entropy loss in terms of logprobs.

The forward pass for the loss section was:
    logit_maxes = logits.max(1, keepdim=True).values
    norm_logits = logits - logit_maxes
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdim=True)
    counts_sum_inv = counts_sum ** -1
    probs = counts * counts_sum_inv
    logprobs = probs.log()
    loss = -logprobs[range(N), Yb].mean()

We derive dL/d(each of these) by hand, in reverse order. The pattern is
always: "what does loss depend on directly? Take the partial. Combine
with the upstream gradient via chain rule."

The cmp() helper compares your hand-derived gradient `dt` against the
reference `t_ref_grad` (which we loaded from state.pt -- autograd
computed it in script 01). EXACT means bit-for-bit identical. CLOSE
means within 1e-7 maxdiff (sometimes float ops differ in the last bit).
"""

import torch


# ---------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------
def cmp(name, dt, t_ref_grad):
    """Compare manual gradient `dt` against autograd reference `t_ref_grad`."""
    ex = torch.all(dt == t_ref_grad).item()
    app = torch.allclose(dt, t_ref_grad)
    maxdiff = (dt - t_ref_grad).abs().max().item()
    status = "EXACT " if ex else ("CLOSE " if app else "WRONG ")
    print(f"  {status} {name:>20s} | maxdiff = {maxdiff:.2e}")


# ---------------------------------------------------------------
# Load state from script 01
# ---------------------------------------------------------------
state = torch.load("state.pt", weights_only=False)

N, V = state["N"], state["V"]
Yb = state["Yb"]

# Forward intermediates we'll use to build manual gradients
logprobs = state["logprobs"]
probs = state["probs"]
counts = state["counts"]
counts_sum = state["counts_sum"]
counts_sum_inv = state["counts_sum_inv"]
norm_logits = state["norm_logits"]
logit_maxes = state["logit_maxes"]
logits = state["logits"]


# ---------------------------------------------------------------
# Backprop, step by step
# ---------------------------------------------------------------
print("Backpropping through the loss layer...")
print()


# loss = -logprobs[range(N), Yb].mean()
#
# loss is a scalar. logprobs is (N, V). The loss only "looks at" the
# entry logprobs[i, Yb[i]] for each i. Everything else has gradient 0.
#
# d(loss)/d(logprobs[i, Yb[i]]) = -1/N
# d(loss)/d(logprobs[i, j])     = 0     for j != Yb[i]

dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(N), Yb] = -1.0 / N
cmp("logprobs", dlogprobs, state["logprobs_grad"])


# logprobs = probs.log()
# d(log(x))/dx = 1/x

dprobs = (1.0 / probs) * dlogprobs
cmp("probs", dprobs, state["probs_grad"])


# probs = counts * counts_sum_inv
#
# Two parents: counts (shape (N, V)) and counts_sum_inv (shape (N, 1)).
# counts_sum_inv broadcasts (N, 1) -> (N, V). Whenever forward broadcasts,
# backward SUMS along the broadcast dim.

dcounts = counts_sum_inv * dprobs
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
cmp("counts_sum_inv", dcounts_sum_inv, state["counts_sum_inv_grad"])


# counts_sum_inv = counts_sum ** -1
# d(x^-1)/dx = -x^-2

dcounts_sum = -counts_sum**-2 * dcounts_sum_inv
cmp("counts_sum", dcounts_sum, state["counts_sum_grad"])


# counts_sum = counts.sum(1, keepdim=True)
# Sum: gradient flows BACK to all elements with weight 1. ADD to dcounts
# (which already has a piece from `probs = counts * counts_sum_inv`).

dcounts += torch.ones_like(counts) * dcounts_sum
cmp("counts", dcounts, state["counts_grad"])


# counts = norm_logits.exp()
# d(exp(x))/dx = exp(x) = counts itself

dnorm_logits = counts * dcounts
cmp("norm_logits", dnorm_logits, state["norm_logits_grad"])


# norm_logits = logits - logit_maxes
# d/d(logits) = +1, d/d(logit_maxes) = -1 summed over j (broadcast)

dlogits = dnorm_logits.clone()
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
cmp("logit_maxes", dlogit_maxes, state["logit_maxes_grad"])


# logits used in TWO places: once directly, once inside max().
# .max() picks one element per row. Gradient flows ONLY to argmax, zero
# elsewhere. ADD to dlogits (which already has the piece from norm_logits).

dlogits += torch.zeros_like(logits).scatter_(
    1, logits.max(1, keepdim=True).indices, dlogit_maxes
)
cmp("logits", dlogits, state["logits_grad"])


print()
print("Loss-layer backprop complete. All gradients verified.")
print()


# ---------------------------------------------------------------
# Save dlogits for the next script
# ---------------------------------------------------------------
state["dlogits"] = dlogits
torch.save(state, "state.pt")
print("Saved dlogits to state.pt for the next script.")


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - Backward starts at the loss and walks UP through the graph.
# - dlogprobs is sparse: only the (i, Yb[i]) entries are nonzero.
# - When forward broadcasts, backward SUMS along the broadcast dim.
# - When forward sums, backward BROADCASTS the gradient back.
# - When forward selects (.max, indexing), backward routes the gradient
#   to ONLY the selected position, zero elsewhere.
#
# These three rules -- broadcast/sum, sum/broadcast, select/route --
# cover most of what you'll need for the rest of the lesson.
