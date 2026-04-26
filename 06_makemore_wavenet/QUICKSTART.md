# Lesson 06 Quickstart: WaveNet

**Prerequisites:** Lessons 01-05 done. DEPLOYMENT.md setup complete.

If you don't see `(.venv)` in your prompt, activate the venv first.

Navigate to the repo:

```powershell
cd $HOME\code\LLM---Neural-Net
```

Activate the venv:

```powershell
.venv\Scripts\Activate.ps1
```

---

## Get the dataset

Same `names.txt`. Copy from any earlier lesson:

Navigate to the lesson folder:

```powershell
cd 06_makemore_wavenet
```

Copy from lesson 05:

```powershell
Copy-Item ..\05_makemore_backprop_ninja\names.txt .
```

Verify:

```powershell
ls names.txt
```

---

## Run the lesson

You should be in `06_makemore_wavenet/`.

Run script 1 (motivation: why flat MLPs hit a wall, dataset prep):

```powershell
python 01_why_flat_mlp_hits_a_wall.py
```

Run script 2 (the FlattenConsecutive operator — just .view()):

```powershell
python 02_flatten_consecutive.py
```

Run script 3 (Module classes: Linear, BatchNorm, Tanh, etc.):

```powershell
python 03_build_modules.py
```

Run script 4 (build the full WaveNet architecture):

```powershell
python 04_build_wavenet.py
```

Run script 5 (train for 200k steps — **takes 5-10 min**):

```powershell
python 05_train_wavenet.py
```

Run script 6 (compare to flat MLP, look ahead to transformers):

```powershell
python 06_compare_and_outlook.py
```

---

## What success looks like

| Script | Expected output |
|---|---|
| 01 | Train: 182,625 examples; 5 sample bigrams shown; param tables |
| 02 | Shape transitions (B,8,3) -> (B,4,6) -> (B,2,12) -> (B,24) |
| 03 | Model with 12k params; layer-by-layer shape dump |
| 04 | Full WaveNet (~76k params); initial loss ~3.30 (≈ log(27)) |
| 05 | Final dev loss around **1.99-2.05** (improvement over MLP's 2.10) |
| 06 | Summary table + bridge to transformers; saves loss_curve.png |

---

## Plots saved to disk

After running script 06:

```powershell
ls *.png
```

You'll see `loss_curve.png` — the WaveNet training loss over 200k steps.
Open by double-clicking in File Explorer.

---

## The single most important moment

**Script 02's FlattenConsecutive operator.** It's literally just
`x.view()` — zero parameters, zero compute. But it transforms how the
network handles context: instead of one giant flat input, the model
processes the sequence hierarchically through stacked merge operations.

When combined with the Linear layer that follows it, this is
mathematically equivalent to a 1D convolution with kernel=2, stride=2.
WaveNet is literally a stack of dilated 1D convolutions in the original
paper. We use the reshape-based version because it's clearer for
learning.

---

## Reading order

The README in this folder walks the concepts. Script comments are the
primary teaching material. Predict what each script needs to add before
opening it.

---

## Time

About 90-120 min of focused reading, plus 5-10 min of training time when
script 05 runs. Lighter than lesson 05 because we're back to using
autograd (no manual gradients in this lesson).

---

## When you finish

Tell me. We'll deploy lesson 07 (Build GPT) — the transformer, where
fixed pair-merging becomes learned attention. The architecture that
powers ChatGPT.
