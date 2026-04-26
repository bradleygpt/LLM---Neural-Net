# Lesson 04 Quickstart: Activations, Gradients, BatchNorm

**Prerequisites:** Lessons 01-03 done. DEPLOYMENT.md setup complete.

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

Same `names.txt` as before. Copy from lesson 02 or 03:

Navigate to the lesson folder:

```powershell
cd 04_makemore_activations_batchnorm
```

Copy from lesson 03:

```powershell
Copy-Item ..\03_makemore_mlp\names.txt .
```

Or from lesson 02:

```powershell
Copy-Item ..\02_makemore_bigram\names.txt .
```

Verify:

```powershell
ls names.txt
```

---

## Run the lesson

You should be in `04_makemore_activations_batchnorm/`.

Run script 1 (the hockey stick problem — bad init explodes initial loss):

```powershell
python 01_the_hockey_stick.py
```

Run script 2 (saturated tanh and dead neurons):

```powershell
python 02_saturated_tanh.py
```

Run script 3 (Kaiming initialization derived from first principles):

```powershell
python 03_kaiming_init.py
```

Run script 4 (BatchNorm: what it does, why it works):

```powershell
python 04_batchnorm.py
```

Run script 5 (the four diagnostic plots every researcher uses):

```powershell
python 05_diagnostic_plots.py
```

Run script 6 (everything together: deep MLP + Kaiming + BatchNorm):

```powershell
python 06_final_mlp_with_batchnorm.py
```

---

## What success looks like

| Script | Expected output |
|---|---|
| 01 | Two loss curves: bad init starts ~27, good init starts ~3.3 |
| 02 | Saturation percentages by init scale; saved activation histograms |
| 03 | Default init: ~80% saturation. Kaiming: ~5% saturation |
| 04 | BatchNorm normalizes mean→0, std→1; running stats accumulate |
| 05 | Four diagnostic plots saved to one figure |
| 06 | Final loss around **2.07** train, **2.10** dev with deep+BN model |

If any script errors out, see DEPLOYMENT.md Part 4.

---

## Plots saved to disk

This lesson generates several `.png` files. After running:

```powershell
ls *.png
```

You'll see (approximately):
- `hockey_stick.png` — the bad-init loss curve from script 01
- `activation_distributions.png` — pre/post-tanh distributions from script 02
- `kaiming_demo.png` — saturation demo from script 03
- `diagnostics.png` — the four-plot diagnostic figure from script 05

Open any in Windows by double-clicking in File Explorer.

---

## The single most important moment

**Script 05's "update / weight ratio" plot.** Karpathy's rule of thumb:
this number should sit around `1e-3` (so log10 ≈ -3) for every layer
during training.

When training breaks in any future project, this is the FIRST plot to
generate. It tells you whether each layer is being updated at a healthy
rate. If a layer's ratio is way below `1e-3`, that layer is being
ignored. If it's way above, the LR is too high and the layer is about
to explode.

---

## Things to try once you finish

- In script 06, remove the `* 0.1` from `layers[-1].W *= 0.1`. Re-run.
  Watch the hockey stick come back. The output layer init matters!
- Replace `BatchNorm1d` with **LayerNorm** (normalize across the FEATURE
  dim instead of the BATCH dim). LayerNorm is what transformers use,
  and it has the nice property of being identical at training and
  inference time.
- Bump `STEPS = 50_000` to `STEPS = 200_000` in script 06. Loss should
  drop further. With the final architecture this is competitive with
  Karpathy's video numbers.

---

## A note on what this enables

After this lesson you have everything needed to train a deep network
without it falling over. The exact init + normalization recipe here is
what powers every modern model:

- **Kaiming-style init** is in every PyTorch model out of the box
- **BatchNorm** is in every CNN (ResNet, EfficientNet, etc.)
- **LayerNorm** (BatchNorm's cousin) is in every transformer

The diagnostic plots from script 05 are the same plots researchers at
OpenAI, Anthropic, and Google use when debugging a misbehaving training
run. You're now in the same toolset.

---

## Reading order

The README in this folder walks the concepts. Script comments are the
primary teaching material. Predict what each script needs to add before
opening it.

---

## Time

About 2-3 hours of focused reading. Each script runs in 10 seconds to
2 minutes (script 05 and 06 are the longer ones).

---

## When you finish

Tell me. We'll deploy lesson 05 (Becoming a Backprop Ninja) — you derive
every gradient by hand through the entire network including BatchNorm.
The most painful and most valuable lesson in the series.
