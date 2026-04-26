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

We'll derive dL/d(each of these) by hand, in reverse order. The pattern
is always: "what does loss depend on directly? Take the partial. Combine
with the upstream gradient via chain rule."

The cmp() helper at the top compares your hand-derived gradient to
autograd's reference. exact=True means bit-for-bit identical. approx=True
with tiny maxdiff is also fine (sometimes float ops differ in the last bit).
"""

import torch


# ---------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------
def cmp(name, dt, t):
    """Compare a manually-derived gradient `dt` with autograd's `t.grad`."""
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    status = "EXACT " if ex else ("CLOSE " if app else "WRONG ")
    print(f"  {status} {name:>20s} | maxdiff = {maxdiff:.2e}")


# ---------------------------------------------------------------
# Load state from script 01
# ---------------------------------------------------------------
state = torch.load("state.pt", weights_only=False)

N, V = state["N"], state["V"]
Yb = state["Yb"]
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
cmp("logprobs", dlogprobs, logprobs)


# logprobs = probs.log()
#
# Element-wise log. d(log(x))/dx = 1/x. So:
# dprobs[i, j] = (1/probs[i, j]) * dlogprobs[i, j]

dprobs = (1.0 / probs) * dlogprobs
cmp("probs", dprobs, probs)


# probs = counts * counts_sum_inv
#
# Two parents: counts (shape (N, V)) and counts_sum_inv (shape (N, 1)).
# Note the BROADCAST -- counts_sum_inv is (N, 1) and gets broadcast to (N, V).
# Whenever you broadcast in the forward, you SUM along the broadcast dim
# in the backward.
#
# d(probs[i,j])/d(counts[i,j])         = counts_sum_inv[i, 0]
# d(probs[i,j])/d(counts_sum_inv[i,0]) = counts[i, j]   (then SUM over j)

dcounts = counts_sum_inv * dprobs                        # (N, V) * (N, V) -> (N, V)
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True) # (N, V) -> sum over j -> (N, 1)
cmp("counts_sum_inv", dcounts_sum_inv, counts_sum_inv)


# counts_sum_inv = counts_sum ** -1
# d(x^-1)/dx = -x^-2

dcounts_sum = -counts_sum**-2 * dcounts_sum_inv
cmp("counts_sum", dcounts_sum, counts_sum)


# counts_sum = counts.sum(1, keepdim=True)
#
# Sum -- gradient flows BACK to all elements with weight 1, expanded
# along the summed dim. Add this to dcounts (which already had a piece
# from the `probs = counts * counts_sum_inv` step).

dcounts += torch.ones_like(counts) * dcounts_sum
cmp("counts", dcounts, counts)


# counts = norm_logits.exp()
# d(exp(x))/dx = exp(x) = counts itself

dnorm_logits = counts * dcounts
cmp("norm_logits", dnorm_logits, norm_logits)


# norm_logits = logits - logit_maxes
#
# d/d(logits)      = +1 (broadcasts naturally to same shape)
# d/d(logit_maxes) = -1, but logit_maxes is (N, 1) so SUM over j

dlogits = dnorm_logits.clone()
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
cmp("logit_maxes", dlogit_maxes, logit_maxes)


# logits is used in TWO places in the forward pass: as `logits - logit_maxes`
# AND inside `logit_maxes = logits.max(1, ...)`. So its gradient gets
# contributions from BOTH paths.
#
# The first path already gave us dlogits = dnorm_logits.clone() above.
# Now add the second path: gradient from logit_maxes.
#
# .max() picks one element per row. Gradient flows back to ONLY that
# element via scatter_. Other positions get 0.

dlogits += torch.zeros_like(logits).scatter_(
    1, logits.max(1, keepdim=True).indices, dlogit_maxes
)
cmp("logits", dlogits, logits)


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
