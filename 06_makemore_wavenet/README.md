# Building Makemore (Part 5): WaveNet

A coding-along build-out of Andrej Karpathy's lesson:
*"Building makemore Part 5: Building a WaveNet"*

## What this lesson is

The flat MLP from lessons 03/04 took a fixed window of 3 characters and
mashed them into one big hidden layer. To use longer context, you'd
have to grow the first layer's weights linearly with context length.

This lesson introduces a **hierarchical** alternative: process the
context in a binary tree. Each level merges adjacent pairs, doubling
the receptive field per layer. With 3 merge levels you cover 8 chars;
with 4 you cover 16. Parameter cost stays bounded.

This is the **WaveNet** architecture (DeepMind, 2016) — same one that
powers Google Assistant's text-to-speech today. For our problem we
build a much smaller version trained on names.

## Setup

```
pip install torch matplotlib
```

Drop `names.txt` (from
`https://raw.githubusercontent.com/karpathy/makemore/master/names.txt`)
in this folder. Without it, scripts use a 50-name fallback.

## How to use this

Run scripts in order:

```
python 01_why_flat_mlp_hits_a_wall.py    # motivation, dataset prep
python 02_flatten_consecutive.py         # the new operator (just .view())
python 03_build_modules.py               # Module classes (Linear, BN, etc.)
python 04_build_wavenet.py               # full architecture, forward pass
python 05_train_wavenet.py               # train for 200k steps (~5-10 min)
python 06_compare_and_outlook.py         # summary + bridge to transformer
```

Each script can be run on its own. Script 05 saves `trained_wavenet.pt`
which script 06 uses for the loss plot.

## The arc

| Script | Concept | What you should be able to explain after |
|---|---|---|
| 01 | Hierarchy motivation | Why flat MLPs scale poorly with context |
| 02 | FlattenConsecutive | How `.view()` becomes "merge consecutive pairs" |
| 03 | Module classes | Clean PyTorch-style API for our building blocks |
| 04 | WaveNet architecture | The 13-layer sequential, 3 merge levels |
| 05 | Training | Same training loop as ever, deeper model |
| 06 | Outlook | The bridge from WaveNet to transformers |

## The single most important moment

**Script 02's FlattenConsecutive operator.** It's literally just
`x.view()` — no learned parameters, no compute. But it's the structural
choice that makes hierarchical context processing possible.

When combined with a Linear layer afterward, it becomes mathematically
equivalent to a 1D convolution with kernel=2, stride=2. Most "WaveNet"
implementations use `nn.Conv1d`. We use reshape + Linear because it's
clearer for teaching; the math is identical.

## Things to try once you finish

- Bump `BLOCK = 8` to `BLOCK = 16` in `data_utils.py` and add ANOTHER
  merge level to the architecture in script 05. Loss should drop a bit
  more (16 chars of context).
- Replace the `FlattenConsecutive(2) + Linear` blocks with `nn.Conv1d`
  layers. Same architecture, idiomatic PyTorch.
- Look at the running BatchNorm stats on each layer after training.
  Are they dramatically different at different merge levels? What does
  that tell you about how the network uses each level?

## Pointer to the next lesson

**Let's build GPT: from scratch, in code, spelled out.**

This is the big one. The transformer architecture replaces WaveNet's
*fixed* pair-merging with **learned attention weights**. Each output
position decides for itself which input positions to attend to, and how
strongly.

The training recipe doesn't change. The architecture does. After lesson
07 you will have built a tiny GPT that generates Shakespeare-flavored
text — the same architecture that powers ChatGPT, just smaller.
