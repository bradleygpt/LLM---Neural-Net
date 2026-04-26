# Building Makemore (Part 3): Activations, Gradients, BatchNorm

A coding-along build-out of Andrej Karpathy's lesson:
*"Building makemore Part 3: Activations & Gradients, BatchNorm"*

## What this lesson is

The previous lesson worked but glossed over crucial details. Initial
loss was huge. Hidden activations could saturate. Deep networks could
have vanishing gradients. We hand-tuned `* 0.1` scaling factors and got
lucky.

This lesson opens the hood and fixes all of it properly:
- **Why initialization matters** (and the Kaiming formula)
- **What "saturated tanh" means** and why it kills training
- **BatchNorm**: the layer that loosened the field's dependence on
  hand-tuned init
- **The four diagnostic plots** every researcher uses to tell whether
  a training run is healthy

## Setup

```
pip install torch matplotlib
```

Drop `names.txt` (from
`https://raw.githubusercontent.com/karpathy/makemore/master/names.txt`)
in this folder. Without it, scripts use a 50-name fallback.

## How to use this

```
python 01_the_hockey_stick.py            # bad init -> huge initial loss
python 02_saturated_tanh.py              # dead neurons in the hidden layer
python 03_kaiming_init.py                # the proper init formula, derived
python 04_batchnorm.py                   # what BatchNorm does, how it works
python 05_diagnostic_plots.py            # the four health-check plots
python 06_final_mlp_with_batchnorm.py    # everything together
```

Each script saves a `.png` plot. Look at them. The plots are the lesson.

## The arc

| Script | Concept | What you should be able to explain after |
|---|---|---|
| 01 | Hockey-stick loss | Why initial loss should be near `log(V)` |
| 02 | Saturated tanh | Why some neurons end up "dead" |
| 03 | Kaiming init | The variance argument: `W ~ N(0, gain²/fan_in)` |
| 04 | BatchNorm | Standardize across the batch, then learnable affine |
| 05 | Four diagnostic plots | What healthy activations/grads/weights look like |
| 06 | Final synthesis | A modular, deep, trainable MLP |

## The single most important moment

Script 05's "update / weight ratio" plot. The rule of thumb: this should
sit around `1e-3` for every layer. Higher means LR is too high; lower
means LR is too low or the weight is being ignored.

When training breaks in any future project, the *first* thing to do is
generate this plot. Decide based on it whether to change LR or
architecture.

## Things to try once you finish

- Remove the `* 0.1` from `layers[-1].W` in script 06. Does the hockey
  stick come back? How much?
- Replace the `BatchNorm1d` layers with **LayerNorm** (normalize across
  the FEATURE dim instead of the BATCH dim). LayerNorm is what
  transformers actually use, and it has the nice property of being
  identical at training and inference.
- Set `gain=1.0` everywhere instead of `5/3`. Generate the activation
  histogram. The activations should now be smaller.
- Use SGD with momentum or Adam instead of plain SGD. The diagnostic
  plots for a well-tuned Adam run are gorgeous.

## Pointer to the next lesson

**makemore part 4: Becoming a Backprop Ninja.** We strip out
`loss.backward()` and derive every gradient by hand through the entire
network — including BatchNorm, embedding lookup, cross-entropy. It's
the most painful and most useful lesson in the series.
