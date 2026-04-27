# Building GPT: From Scratch, in Code, Spelled Out

A coding-along build-out of Andrej Karpathy's lesson:
*"Let's build GPT: from scratch, in code, spelled out."*

## What this lesson is

This is the architectural climax of the foundational series. You build
a real, working **transformer** from scratch and train it on Shakespeare.
Same architecture as ChatGPT, Claude, and Gemini — just smaller.

The key intellectual leap: **attention**. WaveNet (lesson 06) merged
adjacent token pairs with fixed weights. The transformer merges *all
previous tokens* with **learned weights** computed from the input itself.
That single change is the difference between a 2010s-era language model
and ChatGPT.

By the end of script 07, you'll have a 10M-parameter model that
generates pages of Shakespeare-flavored text and a `trained_gpt.pt`
file you can use as a starting point for fine-tuning experiments.

## Setup

```
pip install torch
```

Drop `input.txt` (Tiny Shakespeare) into this folder. Source:
`https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`

Without it, scripts use a tiny embedded fallback. The architecture
will work but the loss numbers and generated text won't match Karpathy's.

## How to use this

Run scripts in order:

```
python 01_explore_shakespeare.py   # load + tokenize the dataset
python 02_bigram_baseline.py       # baseline bigram NN, ~30 sec train
python 03_attention_as_matmul.py   # the math trick: weighted average = matmul
python 04_self_attention.py        # one self-attention head, from scratch
python 05_multi_head_and_block.py  # multi-head + feed-forward = full block
python 06_full_gpt.py              # assemble the GPT, sanity check
python 07_train_gpt.py             # train for 5000 steps (CPU: 15-30 min)
```

Scripts 03-06 are quick (seconds). Scripts 02 and 07 train models.

## The arc

| Script | Concept | What you should be able to explain after |
|---|---|---|
| 01 | Dataset + tokenization | Why character-level is simple but inefficient |
| 02 | Bigram baseline | The training/generation loop pattern |
| 03 | The matmul trick | Weighted average of past tokens = matrix multiplication |
| 04 | Self-attention | Q, K, V; scaled dot-product; causal masking |
| 05 | Block | Multi-head, FFN, residuals, LayerNorm — the whole transformer block |
| 06 | Full GPT | Token + position embeddings, stacked blocks, language head |
| 07 | Training | Real transformer, real Shakespeare, real generated text |

## The single most important moment

**Script 03's three forms of weighted aggregation.**

```python
# v1: for loop, average tokens 0..t (slow)
# v2: matmul with normalized triangular matrix (fast, same answer)
# v3: softmax over a triangular -inf mask (same answer, weights now LEARNABLE)
```

All three produce the same output. v3 is the form attention uses,
because the scores can be replaced with anything (and in real attention,
they're computed from the input). Once you internalize that "attention =
matmul with learned weights," the entire transformer architecture
collapses into a few lines.

## Architecture summary

```
GPT(idx) =
    LayerNorm_final(
        Block_N(
            ...
                Block_1(TokenEmbed(idx) + PositionEmbed(0..T-1))
            ...
        )
    ) -> Linear -> logits

Each Block:
    x = x + MultiHeadAttention(LayerNorm(x))   # communicate
    x = x + FeedForward(LayerNorm(x))          # compute
```

## Things to try once you finish

- **Read the original "Attention Is All You Need" paper.** With this
  lesson behind you, every equation in the paper will make sense.
- **Generate longer Shakespeare** (`max_new=2000` in script 07's generate
  call). The model handles long generation fine.
- **Try different prompts.** Initialize the context with `encode("ROMEO:")`
  instead of zeros and see if the model continues in character.
- **Tweak the architecture.** Halve `n_layers`. Halve `n_embed`. Watch
  loss go up. Doubling them (if your machine can handle it) brings loss
  down at the cost of much longer training.
- **Switch from character-level to subword tokens** — which is exactly
  what lesson 08 builds.

## Pointer to the next lesson

**The GPT Tokenizer.** Real LLMs don't use character-level tokenization
(too inefficient). They use **byte-pair encoding (BPE)** to merge common
character sequences into "subword tokens" — typically 50,000 of them.
Lesson 08 builds a BPE tokenizer from scratch and explains why every
quirk of GPT (handling of leading spaces, weird Unicode behavior, etc.)
traces back to tokenization decisions.
