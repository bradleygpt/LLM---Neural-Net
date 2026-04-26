# Building Makemore (Part 1) from Scratch

A coding-along build-out of Andrej Karpathy's lesson:
*"The spelled-out intro to language modeling: building makemore"*

The lesson trains a **bigram character-level language model** on a list of
names, two ways:

1. **By counting** (a 2D frequency table — no neural net at all)
2. **As a neural net** trained with gradient descent

The whole point is to show those two paths produce the SAME model. Counting
is the floor. Once you've seen the equivalence, every video that follows
in the Zero to Hero series is a way of climbing higher than counting can
reach.

## Setup

Only NumPy is needed for scripts 01–07. PyTorch is optional, used only by
the bonus script 08.

```
pip install numpy            # required
pip install matplotlib       # optional, only if you want to plot
pip install torch            # optional, only for script 08
```

### Get the dataset

Karpathy uses `names.txt` (32,033 names, one per line):

```
https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
```

Download it to this folder. **If you don't, scripts will use a small built-in
fallback (~50 names) so everything still runs.** The shapes and structures
will all be correct; the loss numbers won't match the video.

## How to use this

```
python 01_explore_dataset.py        # see the data, understand bigrams
python 02_count_bigrams.py          # build the count matrix N
python 03_sample_from_counts.py     # generate names from N
python 04_loss_function.py          # measure the model with NLL
python 05_neural_net_setup.py       # one-hot encoding, weights, forward pass
python 06_train_neural_net.py       # gradient descent (manual backprop!)
python 07_equivalence_and_outlook.py  # show counting and NN agree
python 08_pytorch_bonus.py          # same thing in PyTorch (optional)
```

## The arc

| Script | Concept | What you should be able to explain after |
|---|---|---|
| 01 | Bigrams + boundary token | Why we add `.` to start and end of names |
| 02 | Count matrix N | Why this matrix IS already a model |
| 03 | Sampling from rows of P | The whole generative process, no NN |
| 04 | Negative log-likelihood | Why we average log-probs, why we smooth |
| 05 | One-hot + softmax | The bridge from indexes to matrix math |
| 06 | Manual gradient descent | The softmax-NLL gradient: `probs - one_hot` |
| 07 | Equivalence | Counting and the NN give the same table |
| 08 | PyTorch (optional) | What autograd buys you |

## The single most important moment

Script 06 has this comment block:

> The combined gradient of "softmax then negative log-likelihood" with
> respect to the LOGITS is `(probs - one_hot(ys)) / N`.

Pause and let that sink in. It's three lines of code. It's the gradient
that powers the cross-entropy loss in every classification network you've
ever seen, including the one at the top of GPT. Knowing where it comes from
is worth more than the rest of the lesson combined.

## Differences from Karpathy's video

- **NumPy instead of PyTorch** in scripts 01–07. Math is identical;
  syntax differs in cosmetic ways (`.numel()` becomes `len()`,
  `F.one_hot` becomes `np.eye(V)[xs]`, etc.).
- **Manual gradient** in script 06 instead of `loss.backward()`. This is
  educational, not just a workaround — Karpathy spends real time in the
  next videos deriving these by hand anyway.
- **No matplotlib visualization** of the bigram heatmap. Easy to add if
  you want it: `plt.imshow(N)` after script 02 builds N.

## Things to try once you finish

- Drop the smoothing constant (set `SMOOTH = 0` in script 04). Watch what
  happens to the loss. Now set it to 100 — what changes? What does
  smoothing buy you, exactly?
- Build a TRIGRAM model: condition on the previous TWO characters instead
  of one. The count table is now (V, V, V). What's the loss? How does the
  generated text look?
- Replace the `.tanh()` in your micrograd MLP from the previous lesson
  and try training the bigram model with that engine instead of NumPy.
  Verify your engine gets the same loss.
- Plot the loss curve over training in script 06. Is it monotonic? Where
  does it plateau?

## Pointer to the next lesson

The natural next video is **makemore part 2: MLP**. It replaces the (V,V)
lookup table with:
1. An embedding lookup (each character becomes a learned vector)
2. A hidden layer with `tanh`
3. A final linear layer to logits

The training loop is identical. The model is more expressive, so it can
condition on longer contexts (3 characters → next character) without the
count-table size exploding to V³.
