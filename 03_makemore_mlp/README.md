# Building Makemore (Part 2): The MLP

A coding-along build-out of Andrej Karpathy's lesson:
*"Building makemore Part 2: MLP"* — the Bengio 2003 architecture.

## What this lesson is

The bigram model (lesson 02) hits a wall: to use longer context, the
count table grows exponentially. Real-world n-gram models can barely
go beyond trigrams.

This lesson replaces the count table with a **neural net that LEARNS its
own representation of characters**: the embedding lookup. Each character
becomes a small dense vector. The context window of those vectors is fed
through a hidden layer to predict the next character.

Same task as before, way better model. Loss drops from ~2.45 to ~2.17 on
the full `names.txt`.

## Setup

```
pip install torch matplotlib   # both required this time
```

Drop `names.txt` (from `https://raw.githubusercontent.com/karpathy/makemore/master/names.txt`)
in this folder. Without it, scripts use a 50-name fallback.

## How to use this

Run scripts in order:

```
python 01_why_bigrams_arent_enough.py    # motivation (no torch needed)
python 02_build_dataset.py               # train/dev/test splits, sliding window
python 03_embeddings.py                  # the embedding lookup; saves a plot
python 04_build_mlp.py                   # full forward pass; sanity check loss
python 05_find_learning_rate.py          # LR sweep technique; saves a plot
python 06_train_mlp.py                   # full training loop; saves loss curve + embeddings
python 07_sample_and_outlook.py          # generate names; preview lesson 04
```

Scripts 02-07 share state via `dataset.pt` and `trained_mlp.pt` — run
them in order the first time, then you can re-run any individual script.

## The arc

| Script | Concept | What you should be able to explain after |
|---|---|---|
| 01 | n-gram explosion | Why counting can't scale beyond ~trigrams |
| 02 | Train/dev/test splits, sliding window | Why we keep test data separate |
| 03 | Embedding lookup | Why `C[indices]` is a tensor of vectors |
| 04 | The Bengio architecture | The 5-line forward pass |
| 05 | LR sweep | How to find a learning rate without guessing |
| 06 | Minibatch SGD + LR decay | The standard training loop |
| 07 | Sampling + handoff | The hockey-stick problem (preview of lesson 04) |

## The single most important moment

Script 03's lookup operation. `C[Xtr]` where `Xtr` has shape `(N, 3)`
returns a tensor of shape `(N, 3, D)`. That one line is the entire
representational leap of the lesson. Every modern language model uses
the same operation at its input — GPT's token embeddings work exactly
this way.

## Things to try once you finish

- Change `D=2` in script 06 (and re-run 06+07). Embeddings now plot in 2D.
  Vowels should cluster visibly. This is the "I taught it about vowels
  without telling it about vowels" moment.
- Increase `BLOCK_SIZE` from 3 to 5 or 8 (in script 02 and a re-run).
  Loss should drop. Why? Because longer context = more information.
- Forget to call `optimizer.zero_grad()` (we use `p.grad = None` here).
  Predict what happens, then run.
- Use `torch.randn(...)` without the `* 0.1` init scaling. Watch initial
  loss explode. This is the bridge to lesson 04.

## Pointer to the next lesson

**makemore part 3: Activations, Gradients, BatchNorm.** Everything we
hand-tuned (init scales, tanh saturation, the hockey stick) gets
properly diagnosed and fixed. After that the path is straight to
transformers.
