# Lesson 05 Quickstart: Becoming a Backprop Ninja

**Prerequisites:** Lessons 01-04 done. DEPLOYMENT.md setup complete.

This lesson is the most demanding of the series. Set aside dedicated time
(2-3 sessions across a few days works better than one marathon).

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

Same `names.txt` as before. Copy from any earlier lesson:

Navigate to the lesson folder:

```powershell
cd 05_makemore_backprop_ninja
```

Copy from lesson 04:

```powershell
Copy-Item ..\04_makemore_activations_batchnorm\names.txt .
```

Verify:

```powershell
ls names.txt
```

---

## Run the lesson

You should be in `05_makemore_backprop_ninja/`.

Run script 1 (forward pass with all intermediates exposed):

```powershell
python 01_setup_network.py
```

Run script 2 (cross-entropy backward):

```powershell
python 02_backprop_through_loss.py
```

Run script 3 (Linear 2 + tanh backward):

```powershell
python 03_backprop_linear2_and_tanh.py
```

Run script 4 (BatchNorm backward — the hard one):

```powershell
python 04_backprop_batchnorm.py
```

Run script 5 (Linear 1 + reshape + embedding backward):

```powershell
python 05_backprop_linear1_and_embedding.py
```

Run script 6 (train end-to-end with manual gradients):

```powershell
python 06_train_with_manual_grads.py
```

---

## What success looks like

| Script | Expected output |
|---|---|
| 01 | Forward complete; loss prints; state saved |
| 02 | Several `EXACT` results, ending at `logits` |
| 03 | `EXACT` for h, W2, b2, hpreact |
| 04 | `EXACT` for all 7 BatchNorm intermediates, twice (atomic + simplified) |
| 05 | `EXACT` for embcat, W1, b1, emb, C |
| 06 | Final loss around **2.07** train, **2.10** dev — same as autograd would produce |

If any `cmp` line shows `WRONG`, the manual gradient doesn't match autograd. The
script comments explain every derivation step.

---

## The single most important moment

**Script 04, Part B.** The seven-step BatchNorm backward collapses into
one expression:

```python
dhprebn = (bngain * bnvar_inv / N) * (
    N * dhpreact
    - dhpreact.sum(0)
    - (N/(N-1)) * bnraw * (dhpreact * bnraw).sum(0)
)
```

This is what production BN does. The atomic form (Part A) shows you why.
Derive it once, and you'll never be confused by normalization layers again.

---

## How to read the scripts

The comments are the lesson. Don't skim. Each derivation builds:
1. State the forward operation
2. State the local derivative rule
3. Apply the chain rule with the upstream gradient
4. Handle broadcasting (sum) or summing (broadcast) where needed
5. Verify with `cmp()`

When stuck, re-read the comments above the failing `cmp()` line. The
explanation is right there.

---

## Time

The most demanding lesson in the series. Plan accordingly:
- Scripts 01-03: about 30-45 min
- Script 04 (BatchNorm): 60-90 min on its own
- Scripts 05-06: 30-45 min

Total: 2-3 hours of focused reading + tinkering, ideally split into
sessions so the BN derivation can sink in.

---

## When you finish

You'll have manually computed every gradient in a real neural network.
That's the dragon slain. Tell me, and we deploy lesson 06 (WaveNet).
