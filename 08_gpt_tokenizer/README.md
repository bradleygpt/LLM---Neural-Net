# Building the GPT Tokenizer

A coding-along build-out of Andrej Karpathy's lesson:
*"Let's build the GPT Tokenizer"*

## What this lesson is

Real LLMs don't use character-level tokenization. They use **subword
tokenization** — specifically, **byte-pair encoding (BPE)** — which is
why a single token in GPT often represents an entire common word.

In this lesson you build BPE from scratch in ~60 lines of Python. By
the end you'll have:

- A working `Tokenizer` class with `train`, `encode`, `decode`, `save`, `load`
- A `RegexTokenizer` that adds the GPT-2 preprocessing step
- A clear understanding of why LLMs fail at certain things
  (counting letters, reversing strings, handling some Unicode)
- A comparison to the actual GPT-2 and GPT-4 tokenizers

This is a lighter lesson than 05 or 07. **No neural network. No GPU.
No long training runs.** Just clever Python.

## Setup

```
pip install tiktoken    # for the comparison in script 07 (optional)
```

You'll also need `input.txt` from lesson 07 (Tiny Shakespeare). If you
already have it from lesson 07, no download needed. The scripts look
for it at `../07_build_gpt/input.txt`.

If for some reason you don't have it, download:
```
curl -o ../07_build_gpt/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## How to use this

Run scripts in order:

```
python 01_why_tokenization_matters.py    # the problem space
python 02_bytes_as_foundation.py         # UTF-8 bytes as the universal alphabet
python 03_bpe_algorithm.py               # BPE in 3 functions
python 04_tokenizer_class.py             # wrap into a clean class, train on Shakespeare
python 05_tokenizer_quirks.py            # why LLMs fail at certain things
python 06_regex_pretokenization.py       # the GPT-2 preprocessing trick
python 07_compare_to_real_tokenizers.py  # compare to actual GPT-2 / GPT-4
```

Each script is fast (seconds, not minutes). Script 04 trains a small
tokenizer; script 06 trains a slightly larger one; script 07 trains
one with vocab=1024 and compares to real tokenizers via tiktoken.

## The arc

| Script | Concept | What you should be able to explain after |
|---|---|---|
| 01 | The tokenization problem | Why character-level and word-level both fail |
| 02 | UTF-8 bytes | Why every modern tokenizer starts with 256 base tokens |
| 03 | BPE algorithm | get_stats, merge, train: the whole algorithm in 3 functions |
| 04 | Tokenizer class | Real, working BPE with save/load |
| 05 | LLM quirks explained | Letter counting, string reversal, glitch tokens |
| 06 | Regex pre-tokenization | How GPT-2 prevents merges across word boundaries |
| 07 | Real-world comparison | How our tokenizer compares to GPT-2 / GPT-4 |

## The single most important moment

**Script 03's three core functions.** The entire BPE algorithm is:

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

def train(text, num_merges):
    ids = list(text.encode("utf-8"))
    merges = {}
    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats: break
        top_pair = max(stats, key=stats.get)
        new_id = 256 + i
        ids = merge(ids, top_pair, new_id)
        merges[top_pair] = new_id
    return merges
```

That's GPT-2's tokenizer in essence. The official version is a few
hundred lines of optimization but the algorithm is exactly this.

## Things to try once you finish

- **Train on a different corpus.** Pick any text file and train.
  Watch what merges emerge — they encode the patterns of your data.
- **Train a code tokenizer.** Run on a folder of Python source. The
  merges will be totally different (`def `, `==`, `    ` whitespace).
- **Train a SQL tokenizer.** All the keywords become single tokens.
  Tokenizing SQL becomes 4x more efficient than with GPT-2's tokenizer.
- **Compare your tokenizer's output to tiktoken's** for various inputs.
  Where does yours do better? Where worse?
- **Read Karpathy's [minbpe repo](https://github.com/karpathy/minbpe)**.
  It's the exact same algorithm with a few extras (special tokens,
  pickle save format, regex variants).

## Pointer to the next lesson

**Reproduce GPT-2.** This is the lesson where everything comes
together: the transformer architecture from lesson 07 + the tokenizer
algorithm from this lesson + scaling up to a real published model.

You'll train a real GPT-2 124M parameter model. This is the actual
model OpenAI released in 2019 that started the modern LLM era. With
your RTX 5050 it'll take a few hours. With Karpathy's tricks it
could even match the original paper's loss numbers.
