"""
04_batchnorm.py
---------------
BatchNorm is the breakthrough that made it possible to train deep networks
without obsessing over initialization. The idea:

Why hand-tune init scales when we can just FORCE the activations to have
the right distribution at every layer? At the start of each training
step, BEFORE applying the activation function, we:

    1. Compute the mean and std of the pre-activations across the BATCH.
    2. Subtract the mean and divide by the std. Now the values are
       standardized: mean 0, std 1.
    3. Then apply a LEARNED affine transform (gamma * x_norm + beta) so
       the network can recover any distribution it wants if normalization
       was the wrong thing to do.

    BN(x) = gamma * (x - mu_batch) / sqrt(sigma_batch^2 + eps) + beta

This is differentiable end-to-end (mu and sigma depend on the batch, so
gradients flow through them), and it dramatically loosens the
init-sensitivity of the network.

Two subtle points the lesson hammers home:

  - BatchNorm couples examples in a batch. The output for example i now
    depends on examples j, k, etc. through the batch statistics. This is
    weird. It causes problems at INFERENCE TIME when you have only one
    example, or want to test with batch size 1.

  - The fix for inference: keep a RUNNING MEAN and RUNNING STD during
    training, and use those at inference time. PyTorch's BatchNorm
    layers do this automatically.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------
# Manual BatchNorm
# ---------------------------------------------------------------
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # learnable parameters
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # running stats (NOT learned, used at inference)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            # batch stats
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True, unbiased=False)
        else:
            mean = self.running_mean
            var = self.running_var
        # normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta
        # update running stats during training
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze(0)
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze(0)
        return out

    def parameters(self):
        return [self.gamma, self.beta]


# ---------------------------------------------------------------
# Sanity check: BN does what we said
# ---------------------------------------------------------------
torch.manual_seed(0)
bn = BatchNorm1d(50)

# A batch of 1000 examples with funky distribution
x = torch.randn(1000, 50) * 5.0 + 7.0   # mean ~7, std ~5
print("Input batch:")
print(f"  mean per dim (avg): {x.mean(0).mean().item():.3f}")
print(f"  std per dim (avg):  {x.std(0).mean().item():.3f}")

y = bn(x)
print("\nAfter BatchNorm (gamma=1, beta=0):")
print(f"  mean per dim (avg): {y.mean(0).mean().item():.3f}  (should be ~0)")
print(f"  std per dim (avg):  {y.std(0).mean().item():.3f}  (should be ~1)")
print()


# ---------------------------------------------------------------
# What happens with gamma != 1, beta != 0?
# ---------------------------------------------------------------
bn.gamma = torch.full((50,), 3.0)
bn.beta = torch.full((50,), -2.0)
y = bn(x)
print("After BatchNorm with gamma=3, beta=-2:")
print(f"  mean per dim (avg): {y.mean(0).mean().item():.3f}  (should be ~-2)")
print(f"  std per dim (avg):  {y.std(0).mean().item():.3f}  (should be ~3)")
print()


# ---------------------------------------------------------------
# Eval mode: inference uses running stats, not batch stats
# ---------------------------------------------------------------
# Reset
bn = BatchNorm1d(50)
# 'Train' on many batches to populate running stats
for _ in range(100):
    bn(torch.randn(32, 50) * 5.0 + 7.0)
print("After 100 batches of training, running stats:")
print(f"  running_mean: ~{bn.running_mean.mean().item():.3f} (should approach 7)")
print(f"  running_var:  ~{bn.running_var.mean().item():.3f} (should approach 25)")

# Now switch to eval and run a single example
bn.training = False
single = torch.tensor([[7.0] * 50]) * 1.0  # one example, exactly mean
out = bn(single)
print(f"\nSingle example through eval-mode BN: out mean = {out.mean().item():.3f}")
print("(Without running stats, batch-of-1 BN would divide by zero!)")
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - BatchNorm normalizes its input to mean 0, std 1 ACROSS the batch.
# - It then applies a learned affine to recover any distribution
#   the network wants.
# - It maintains running statistics so it can be used in eval mode
#   (single-example inference) without breaking.
# - It loosens init-sensitivity dramatically: the network adjusts its
#   own activation distributions instead of relying on us getting it
#   right at init time.
#
# But it has a DARK SIDE: it couples examples within a batch. We'll
# see how to use BatchNorm correctly -- and what alternatives exist
# (LayerNorm, used in transformers) -- in script 05.
