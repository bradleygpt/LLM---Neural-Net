"""
06_compare_and_outlook.py
-------------------------
Reflect on what WaveNet bought us, and look ahead to lesson 07 (GPT).

Three things to compare:

  1. Loss numbers across the makemore series so far
  2. WaveNet's architecture vs flat MLP -- what changed and why
  3. The bridge from WaveNet to attention / transformers

This script doesn't train anything new -- it just summarizes and points
forward.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------
# Loss progression across the series
# ---------------------------------------------------------------
print("=" * 60)
print("LOSS PROGRESSION ACROSS MAKEMORE LESSONS")
print("=" * 60)
print()
print("  Lesson         Architecture                     Dev loss")
print("  -----------    ------------------------------    --------")
print("  02 (bigram)    27x27 count table               ~ 2.45")
print("  03 (MLP)       3-char window, flat MLP         ~ 2.14")
print("  04 (deep MLP)  3-char window, BN deep MLP      ~ 2.10")
print("  05 (ninja)     same as 04, manual gradients    ~ 2.10")
print("  06 (WaveNet)   8-char window, hierarchical     ~ 1.99-2.05")
print()
print("Each step's improvement comes from:")
print("  02 -> 03: longer context (1 -> 3 chars) + neural net generalization")
print("  03 -> 04: deeper network with proper init + BatchNorm")
print("  05 -> 06: longer context (3 -> 8 chars) + hierarchical structure")
print()


# ---------------------------------------------------------------
# Why hierarchical beats flat
# ---------------------------------------------------------------
print("=" * 60)
print("FLAT MLP vs WAVENET: WHAT CHANGED")
print("=" * 60)
print()
print("Flat MLP (lesson 04):")
print("  - 3 chars * 10 dims = 30 input features")
print("  - One big Linear (30 -> 200) decides EVERYTHING about context")
print("  - Network has no built-in notion of position or proximity")
print("  - Doubling context = doubling first-layer parameters")
print()
print("WaveNet (lesson 06):")
print("  - 8 chars * 24 dims, but processed in 3 merge levels")
print("  - At each level, only NEIGHBORING pairs are merged")
print("  - The network has STRUCTURAL bias: nearby chars are related")
print("  - Doubling context only adds ONE more merge level (constant cost)")
print()


# ---------------------------------------------------------------
# Connection to convolutions
# ---------------------------------------------------------------
print("=" * 60)
print("WAVENET <-> CONVOLUTIONS")
print("=" * 60)
print()
print("FlattenConsecutive(2) + Linear is mathematically equivalent to a")
print("1D CONVOLUTION with kernel size 2 and stride 2. WaveNet is")
print("literally a stack of dilated 1D convolutions in the original paper.")
print()
print("In our implementation we used reshape + Linear because it's clearer")
print("for teaching, but Conv1d would do the same thing with the same")
print("number of parameters. PyTorch has nn.Conv1d that you'd use in")
print("production code.")
print()


# ---------------------------------------------------------------
# Bridge to transformers
# ---------------------------------------------------------------
print("=" * 60)
print("FROM WAVENET TO TRANSFORMERS (LESSON 07)")
print("=" * 60)
print()
print("WaveNet's hierarchy is FIXED: you always merge adjacent pairs.")
print("This works great when the relevant context is local (nearby chars).")
print()
print("But what if char 0 is more relevant than char 6 for predicting")
print("char 7? WaveNet can't know that. The merge structure is fixed at")
print("design time.")
print()
print("The TRANSFORMER fixes this with ATTENTION. Instead of:")
print()
print("    'merge each pair of neighbors' (fixed)")
print()
print("Attention says:")
print()
print("    'each output position LOOKS AT all input positions and")
print("     LEARNS HOW MUCH to weight each one'")
print()
print("In code: the merge operation becomes a learned weighted sum where")
print("the weights are computed from the inputs themselves. That's it.")
print("That's the entire conceptual leap from CNN/WaveNet to Transformer.")
print()
print("The Transformer is doing the SAME job as WaveNet (combine context")
print("info), with the SAME training recipe (forward/loss/backward), but")
print("with LEARNED merge weights instead of fixed pair-merging.")
print()
print("Lesson 07 builds GPT from scratch. After this lesson, you have")
print("everything you need.")
print()


# ---------------------------------------------------------------
# Optional: load and inspect the trained model
# ---------------------------------------------------------------
try:
    state = torch.load("trained_wavenet.pt", weights_only=False)
    print("=" * 60)
    print("TRAINED MODEL STATS (from trained_wavenet.pt)")
    print("=" * 60)
    print()
    n_params = sum(p.numel() for p in state["params"])
    print(f"Total parameters: {n_params:,}")
    print(f"Block size:       {state['block']}")
    print(f"Embed dim:        {state['D']}")
    print(f"Hidden dim:       {state['H']}")
    print()

    # Plot the loss curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        losses = state["losses"]
        # smooth via moving average
        import numpy as np
        smoothed = np.convolve(losses, np.ones(1000)/1000, mode='valid')
        plt.figure(figsize=(10, 5))
        plt.plot(smoothed)
        plt.xlabel("step (log10 loss, smoothed over 1000)")
        plt.ylabel("log10 loss")
        plt.title("WaveNet training loss")
        plt.grid(alpha=0.3)
        plt.savefig("loss_curve.png", dpi=100, bbox_inches="tight")
        plt.close()
        print("Saved loss_curve.png")
    except ImportError:
        print("(matplotlib not available, skipped loss plot)")

except FileNotFoundError:
    print()
    print("[note] trained_wavenet.pt not found. Run script 05 first to")
    print("       train the model, then re-run this script for the plot.")


# ---------------------------------------------------------------
# What we just learned (and what's next)
# ---------------------------------------------------------------
print()
print("=" * 60)
print("YOU'RE READY FOR THE TRANSFORMER")
print("=" * 60)
print("""
At this point you have:
  - A complete forward + backward pass mental model (lessons 01-05)
  - Initialization, normalization, diagnostics (lesson 04)
  - The hierarchical context-merging pattern (lesson 06, this one)

The transformer in lesson 07 replaces FlattenConsecutive's fixed
merge with LEARNED attention weights. Architecturally that's a bigger
piece, but every other primitive (embedding, linear, layernorm, the
training recipe) is the same.

Lesson 07 is where this becomes ChatGPT-shaped.
""")
