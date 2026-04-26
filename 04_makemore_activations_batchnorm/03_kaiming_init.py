"""
03_kaiming_init.py
------------------
We've seen two init problems: huge initial loss (output side) and
saturated tanh (hidden side). Both come down to the same thing -- the
SCALE of the random weights.

Kaiming initialization (He et al., 2015) gives a principled formula.
Derivation:

We want the activations to maintain roughly UNIT VARIANCE as they pass
through layers. If they shrink toward 0 we lose signal; if they explode
we get saturated activations or NaN.

For a layer y = x @ W with x ~ N(0, sigma_x^2) and W ~ N(0, sigma_W^2):
    Var(y_i) = sum over fan_in of Var(x_j * W_ij)
             = fan_in * sigma_x^2 * sigma_W^2

For Var(y) = Var(x) we need sigma_W^2 = 1 / fan_in.

That's the basic formula:
    W ~ N(0, 1 / fan_in)
or equivalently
    W = randn(...) / sqrt(fan_in)

For activations like tanh that COMPRESS the variance (because tanh(x)
has smaller magnitude than x in expectation), we apply a "gain" g:
    W ~ N(0, g^2 / fan_in)

PyTorch's recommended gains:
    linear / identity:  1
    tanh:               5/3
    relu:               sqrt(2)
    leaky_relu:         sqrt(2 / (1 + a^2))   where a is the slope

This script demonstrates that with Kaiming init, activations stay
well-behaved at every layer.
"""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# Empirical check: the formula works
# ---------------------------------------------------------------
# What goes WRONG with bad init through a tanh stack? Tanh activations
# are bounded in [-1, 1], so they don't "explode" -- but they DO saturate
# at +/-1 when the pre-activation has large magnitude. We measure both:
#   - std of the activation (should stay near healthy bounded value)
#   - SATURATION FRACTION (|tanh| > 0.97). Saturated -> dead gradient.

torch.manual_seed(42)
fan_in, n_layers, n_examples = 100, 50, 1000

def run_stack(init_scale):
    y = torch.randn(n_examples, fan_in)
    stds, sats = [y.std().item()], [0.0]
    for _ in range(n_layers):
        W = torch.randn(fan_in, fan_in) * init_scale
        y = torch.tanh(y @ W)
        stds.append(y.std().item())
        sats.append((y.abs() > 0.97).float().mean().item())
    return stds, sats

# Default init (no scaling): pre-activations huge, tanh saturates
stds_default, sats_default = run_stack(1.0)
# Kaiming for tanh: gain / sqrt(fan_in)
stds_kaiming, sats_kaiming = run_stack((5/3) / fan_in**0.5)

print("Across 50 stacked tanh layers:")
print(f"  Default init  layer  0: std={stds_default[0]:.3f}, sat={100*sats_default[0]:.1f}%")
print(f"  Default init  layer 50: std={stds_default[-1]:.3f}, sat={100*sats_default[-1]:.1f}%")
print(f"  Kaiming init  layer  0: std={stds_kaiming[0]:.3f}, sat={100*sats_kaiming[0]:.1f}%")
print(f"  Kaiming init  layer 50: std={stds_kaiming[-1]:.3f}, sat={100*sats_kaiming[-1]:.1f}%")
print()
print("Default-init activations stay bounded (tanh is bounded), but they")
print("SATURATE -- nearly every neuron pinned to +/-1, where tanh's gradient")
print("is ~0. That kills training even though the std looks 'fine'.")
print("Kaiming keeps neurons in tanh's linear regime where gradients flow.")
print()


# Plot both views
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(stds_default, label="default init", marker=".")
axes[0].plot(stds_kaiming, label="Kaiming init", marker=".")
axes[0].axhline(1.0, color="red", linestyle="--", alpha=0.5)
axes[0].set_xlabel("layer depth")
axes[0].set_ylabel("std of activations")
axes[0].set_title("Activation std (both stay bounded for tanh)")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[1].plot([100*s for s in sats_default], label="default init", marker=".")
axes[1].plot([100*s for s in sats_kaiming], label="Kaiming init", marker=".")
axes[1].set_xlabel("layer depth")
axes[1].set_ylabel("% saturated (|tanh| > 0.97)")
axes[1].set_title("Saturation -- the real failure mode")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("kaiming_demo.png", dpi=100, bbox_inches="tight")
plt.close()
print("Saved kaiming_demo.png -- left plot looks 'fine' for default init,")
print("right plot reveals the actual problem.")
print()


# ---------------------------------------------------------------
# PyTorch has this built in
# ---------------------------------------------------------------
import torch.nn as nn
print("PyTorch's built-in helpers:")
print("  torch.nn.init.kaiming_normal_(W, nonlinearity='tanh')")
print("  torch.nn.init.kaiming_normal_(W, nonlinearity='relu')")
print()
print("In the lesson we'll write the formula out longhand to keep things")
print("transparent, but in real code the built-in is preferred.")


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - Each layer has a magnitude budget. If we don't manage it, activations
#   either vanish (everything goes to 0) or explode (NaN).
# - Kaiming init manages the budget: scale weights by gain/sqrt(fan_in).
# - The gain depends on the nonlinearity: 1 for linear, 5/3 for tanh,
#   sqrt(2) for relu.
# - With Kaiming, activations across many layers stay near std=1, which
#   keeps tanh in its linear region and gradients flowing.
#
# Next: even with great init, there's still a problem in deeper networks.
# That's what BatchNorm is for.
