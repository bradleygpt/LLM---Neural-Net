# Lesson 02 Quickstart: Makemore (Bigram)

**Prerequisites:** Lesson 01 done, DEPLOYMENT.md setup complete.

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

## Get the dataset (one-time, 5 seconds)

This lesson uses Karpathy's `names.txt` (32,033 names). Without it the
scripts fall back to a 50-name sample, but the loss numbers won't match
the video. To get the real one:

Navigate to the lesson folder:

```powershell
cd 02_makemore_bigram
```

Download the dataset:

```powershell
curl -o names.txt https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
```

Verify it downloaded:

```powershell
ls names.txt
```

You should see a file around 230 KB. The `.gitignore` (which doesn't
list `names.txt`) means it'll show up in `git status`, but it's small so
that's fine — you can either commit it or add `names.txt` to `.gitignore`
later. Up to you.

---

## Run the lesson

You should still be in `02_makemore_bigram/`. If not, `cd` back in.

Run script 1 (explore the dataset, understand bigrams):

```powershell
python 01_explore_dataset.py
```

Run script 2 (build the bigram count matrix):

```powershell
python 02_count_bigrams.py
```

Run script 3 (sample names from the counts):

```powershell
python 03_sample_from_counts.py
```

Run script 4 (negative log-likelihood, smoothing):

```powershell
python 04_loss_function.py
```

Run script 5 (neural net setup, one-hot encoding):

```powershell
python 05_neural_net_setup.py
```

Run script 6 (train the bigram NN with manual gradient descent):

```powershell
python 06_train_neural_net.py
```

Run script 7 (demonstrate equivalence between counting and the NN):

```powershell
python 07_equivalence_and_outlook.py
```

Run script 8 (the PyTorch version — bonus, requires torch):

```powershell
python 08_pytorch_bonus.py
```

---

## What success looks like

| Script | Expected output |
|---|---|
| 01 | Bigrams listed for first few names; total bigram count |
| 02 | Count matrix N built; top-followers of `.` and `a` listed |
| 03 | 10 generated names that look name-shaped (e.g. "junide", "konniva") |
| 04 | Average NLL around **2.45** with full dataset (~2.33 with fallback) |
| 05 | Tensor shapes: x_onehot (~228K, 27); probs sum to 1.0 |
| 06 | Loss drops from ~3.7 to ~2.45 over 200 epochs |
| 07 | Counting and trained NN tables match to ~4 decimal places |
| 08 | Same loss as script 06, ~10 lines of training loop using `loss.backward()` |

If any script errors out, see DEPLOYMENT.md Part 4.

---

## The single most important moment

**Script 07.** It compares the probability table from counting against
the table the NN learned. They agree to ~4 decimal places. Counting and
gradient descent rediscover the same statistics — the model is the same,
just with two different paths to find it.

This is the headline insight of the lesson and the launching pad for
everything that follows. Bigger neural nets aren't doing something
different; they're doing the same thing on richer representations.

---

## Reading order

The README in this folder (`README_makemore_bigram.md`) walks the
concepts. The script comments are the primary teaching material.
Predict what each script needs to add before opening it.

---

## Time

About 90 minutes of focused reading + tinkering.

---

## When you finish

Tell me. We'll deploy lesson 03.
