# Building Makemore (Part 4): Becoming a Backprop Ninja

A coding-along build-out of Andrej Karpathy's lesson:
*"Building makemore Part 4: Becoming a Backprop Ninja"*

## What this lesson is

In the previous lessons, autograd did the gradient math for you.
`loss.backward()` filled in every `.grad` attribute automatically.

In this lesson, **autograd is forbidden.** You derive every gradient by
hand — through cross-entropy, softmax, matmul, tanh, BatchNorm, the
embedding lookup, and the sliding-window concatenation. Then you verify
your manual gradient against autograd's reference, op by op, until they
match exactly.

The reward: you understand what autograd is doing. PyTorch stops being
magic. Future you, when faced with a NaN gradient at 3 AM, will know
exactly where to look.

## Setup

```
pip install torch matplotlib   # both required
```

Drop `names.txt` (from
`https://raw.githubusercontent.com/karpathy/makemore/master/names.txt`)
in this folder. Without it, scripts use a 50-name fallback.

## How to use this

Scripts share state via `state.pt`. Run them in order:

```
python 01_setup_network.py                  # forward pass with named intermediates
python 02_backprop_through_loss.py          # cross-entropy backward
python 03_backprop_linear2_and_tanh.py      # matmul rule + tanh
python 04_backprop_batchnorm.py             # BatchNorm backward (the hard one)
python 05_backprop_linear1_and_embedding.py # last layers + embedding lookup
python 06_train_with_manual_grads.py        # train end-to-end without autograd
```

Each of scripts 02-05 prints `EXACT` or `CLOSE` for each manual gradient,
verifying it matches autograd's reference. When you see all `EXACT`, you
got the math right.

## The arc

| Script | Concept | What you should be able to explain after |
|---|---|---|
| 01 | Forward with named intermediates | Why we expose every step |
| 02 | Cross-entropy backward | The `(probs - one_hot) / N` closed form, derived |
| 03 | Matmul + tanh backward | The single most-used identity in deep learning |
| 04 | BatchNorm backward | Why it's hard, and why LayerNorm replaced it |
| 05 | Embedding + reshape backward | Index-accumulation pattern |
| 06 | End-to-end manual training | Autograd, demystified |

## The single most important moment

**Script 04, Part B's simplified BatchNorm backward.** A seven-step
chain rule collapses into:

```python
dhprebn = (bngain * bnvar_inv / N) * (
    N * dhpreact
    - dhpreact.sum(0)
    - (N/(N-1)) * bnraw * (dhpreact * bnraw).sum(0)
)
```

That single expression is what production BatchNorm implementations use.
Knowing how to derive it from the atomic form is the rite of passage.

## Things to try once you finish

- Replace `BatchNorm1d` with **LayerNorm** in script 06 (normalize across
  the FEATURE dim instead of BATCH). Derive its backward — it's
  significantly easier than BatchNorm because the normalization happens
  per-example, not across the batch.
- Add an **L2 regularization** term to the loss. Derive the gradient
  contribution from each parameter (it's just `2 * λ * p`). Add it in.
- Implement **gradient clipping**: scale all gradients down if their
  global L2 norm exceeds a threshold. This is a one-liner once you have
  manual access to all gradients.

## Pointer to the next lesson

**makemore part 5: Building a WaveNet.** We replace the flat MLP with a
hierarchical architecture where context information composes
progressively. WaveNet is real — it powered Google's text-to-speech.
After WaveNet, the next stop is the transformer.
