# Reproduce GPT-2: Architecture, Weights, and Fine-tuning

A coding-along build-out inspired by Andrej Karpathy's lesson:
*"Let's reproduce GPT-2 (124M)"*

## What this lesson is

The architectural climax of the foundational series. You take everything
built across lessons 01-08 and refine it to **exactly match the published
GPT-2 architecture**, then load OpenAI's actual published weights into
your code, then fine-tune on Shakespeare to demonstrate end-to-end
training works.

**What this lesson does include:**
- GPT-2 small (124M) architecture, byte-for-byte
- Loading OpenAI's pretrained weights via HuggingFace
- Verifying byte-for-byte logit match against HuggingFace's reference
- Fine-tuning on Tiny Shakespeare for ~30 minutes on your GPU
- Real GPT-2 quality output, in your own code

**What this lesson does NOT include:**
- Training GPT-2 from scratch (would take 1-3 weeks on your hardware,
  or $10-50 in cloud GPU costs)
- The full FineWeb-Edu 10B-token dataset (~30GB)
- Multi-GPU distributed training

The script 05 documents exactly what would be needed for full reproduction
if you ever want to scale up. **All of this lesson is zero-cost.**

## Setup

```
pip install transformers
```

`transformers` is HuggingFace's library that distributes OpenAI's GPT-2
weights. About 200MB to install. You should already have `tiktoken`
from lesson 08.

You also need `input.txt` (Tiny Shakespeare) from lesson 07. The
fine-tuning script reads it from `../07_build_gpt/input.txt`.

## How to use this

Run scripts in order:

```
python 01_gpt2_architecture.py            # build the architecture, verify 124M params
python 02_load_pretrained_weights.py      # download + load OpenAI's weights
python 03_compare_with_huggingface.py     # verify byte-for-byte correctness
python 04_finetune_on_shakespeare.py      # 30-min GPU run, Shakespeare-style output
python 05_what_full_reproduction_requires.py  # documentation only
```

**Time estimates:**
- Scripts 01, 03, 05: seconds
- Script 02: ~2 min (downloads ~500MB from HuggingFace, cached after)
- Script 04: ~30 min on RTX 5050 GPU, hours on CPU

## The arc

| Script | Concept | What you should be able to explain after |
|---|---|---|
| 01 | GPT-2 architecture | Weight tying, GELU, scaled init, 124M param count |
| 02 | Weight loading | HF state dict, Conv1D vs Linear transposition |
| 03 | Architecture verification | Logit match to HuggingFace = correctness proof |
| 04 | Fine-tuning | The dominant workflow in production LLM development |
| 05 | What full reproduction requires | The honest cost of training from scratch |

## The single most important moment

**Script 03's logit comparison.** When you see:

```
Max absolute difference: 1.23e-5
  -> MATCH (within numerical precision)
```

That's the moment your architecture is *proven correct* against an
independent reference implementation. You took weights from a model
trained by OpenAI in 2019 and they slot into YOUR code and produce
identical outputs. There's no clearer evidence that you've built a
real GPT-2.

## What you'll be able to do after this lesson

- Implement and load any pretrained transformer model from HuggingFace
- Diagnose architectural mismatches by comparing intermediate outputs
- Fine-tune large language models on custom data
- Read research papers on new architectures and implement them
- Read GPT-2's, GPT-3's, Llama's, and similar papers and follow the math
- Reason about the cost of training a real LLM (compute, data, time)
- Understand what production LLM teams actually do day-to-day

## Things to try once you finish

- **Fine-tune on a different corpus.** Pick anything: code, Wikipedia,
  research papers. The script 04 pattern works for any text.
- **Generate longer text** (`max_new=500`). The model handles long
  generation fine.
- **Try a bigger GPT-2.** Change `model_type="gpt2"` to `"gpt2-medium"`
  (350M) in script 02. Same code, more capacity. Your 8GB VRAM can
  handle medium for fine-tuning, probably not large.
- **Compute perplexity** on held-out data. That's what the field uses
  to compare LLMs quantitatively.
- **Read Karpathy's `nanoGPT` repo** (https://github.com/karpathy/nanoGPT)
  for the production-quality version of this code with all the
  performance optimizations.

## Pointer to the next lesson

**Lesson 10: Andrej Karpathy's "Deep Dive into LLMs" talk** (3.5 hours).

This is a watch-only lesson — no code. Karpathy walks through the entire
modern LLM stack: how teams go from a base model (which you've now
built) to a production system like ChatGPT. RLHF, instruction tuning,
constitutional AI, all the alignment work, plus inference optimization
and serving infrastructure.

It's the synthesis lesson. Everything you've coded across 01-09 is
the foundation. Lesson 10 puts it in production context.
