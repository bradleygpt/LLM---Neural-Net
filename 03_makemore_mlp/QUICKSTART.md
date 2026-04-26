# Lesson 03 Quickstart: Makemore MLP

**Prerequisites:** Lessons 01 and 02 done. DEPLOYMENT.md setup complete.

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

This lesson uses the same `names.txt` (32,033 names). If you already have
it from lesson 02, you can copy it over. Otherwise, download fresh.

Navigate to the lesson folder:

```powershell
cd 03_makemore_mlp
```

Either copy from lesson 02:

```powershell
Copy-Item ..\02_makemore_bigram\names.txt .
```

Or download fresh:

```powershell
curl -o names.txt https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
```

Verify it's there:

```powershell
ls names.txt
```

You should see a file around 228 KB.

---

## Run the lesson

You should still be in `03_makemore_mlp/`. If not, `cd` back in.

Run script 1 (motivation: why bigrams hit a wall):

```powershell
python 01_why_bigrams_arent_enough.py
```

Run script 2 (build train/dev/test splits + sliding window dataset):

```powershell
python 02_build_dataset.py
```

Run script 3 (the embedding lookup — the conceptual leap of the lesson):

```powershell
python 03_embeddings.py
```

Run script 4 (the full Bengio MLP forward pass):

```powershell
python 04_build_mlp.py
```

Run script 5 (find a good learning rate by sweeping):

```powershell
python 05_find_learning_rate.py
```

Run script 6 (full training loop — this one trains for ~50,000 steps, takes 1-3 min):

```powershell
python 06_train_mlp.py
```

Run script 7 (sample names from the trained model):

```powershell
python 07_sample_and_outlook.py
```

---

## What success looks like

| Script | Expected output |
|---|---|
| 01 | Tables showing n-gram count growth (V^k); MLP param breakdown |
| 02 | Train/dev/test sizes; first 10 sliding-window examples |
| 03 | Embedding tensor shapes (N, 3, D); saved untrained scatter plot |
| 04 | MLP forward pass works; sanity-check loss near `log(27)` ≈ 3.30 |
| 05 | LR sweep result; saved `lr_sweep.png` showing the loss curve |
| 06 | Final loss around **2.10** train, **2.20** dev; saved loss curve |
| 07 | 20 sampled names; should look more name-like than lesson 02's |

If any script errors out, see DEPLOYMENT.md Part 4.

---

## Plots saved to disk

This lesson generates `.png` files (since we use a headless matplotlib
backend that won't open windows). After running, look in the folder:

```powershell
ls *.png
```

You'll see (approximately):
- `embeddings_untrained.png` — random scatter from script 03
- `lr_sweep.png` — learning rate vs loss curve from script 05
- `loss_curve.png` — training loss over time from script 06
- `embeddings_trained.png` — only if you set `D=2` in script 06

Open any of these in Windows by double-clicking them in File Explorer.

---

## The single most important moment

**Script 03's embedding lookup.** One line:

```python
emb = C[Xtr]
```

`Xtr` has shape `(N, 3)` — N training examples, each with 3 context indices.
`C` has shape `(V, D)` — vocab size by embedding dimension. The result is
shape `(N, 3, D)` — every input character is now a learned vector.

That single tensor indexing operation is **the input layer of every modern
language model.** GPT does this exact thing with token embeddings. The
"meaning" of words is just rows in a learnable matrix.

---

## Things to try once you finish

- Change `D=10` to `D=2` in script 06, re-run, look at `embeddings_trained.png`.
  Vowels should visibly cluster. Pause and appreciate that — the model
  was never told what a vowel is.
- Increase `BLOCK_SIZE = 3` to `BLOCK_SIZE = 8` in script 02, re-run
  scripts 02-06. Loss should drop. Why? More context = more information.
- Bump `STEPS = 50_000` to `STEPS = 200_000` in script 06. Loss should
  drop further. Karpathy gets ~2.17 in the video with 200k steps.

---

## Reading order

The README in this folder (`README.md` after the rename) walks the
concepts. The script comments are the primary teaching material. Predict
what each script needs to add before opening it.

---

## Time

About 2-3 hours of focused reading + tinkering, plus 1-3 min of training
time when script 06 runs.

---

## When you finish

Tell me. We'll deploy lesson 04 (Activations, Gradients, BatchNorm).
