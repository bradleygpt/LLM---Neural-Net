# Lesson 09 Quickstart: Reproduce GPT-2

**Prerequisites:** Lessons 01-08 done. CUDA-PyTorch installed (lesson 07).
You'll need `input.txt` (Tiny Shakespeare) in `07_build_gpt/`.

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

## Install transformers (HuggingFace library)

This gives you access to OpenAI's published GPT-2 weights:

```powershell
pip install transformers
```

About 200MB install. Takes 30-60 seconds.

Verify install:

```powershell
python -c "from transformers import GPT2LMHeadModel; print('OK')"
```

Should print `OK` (no errors). If it errors, send me the output.

---

## Navigate to the lesson

```powershell
cd 09_reproduce_gpt2
```

---

## Run the lesson

You should be in `09_reproduce_gpt2/`.

Run script 1 (build the architecture, verify 124M params):

```powershell
python 01_gpt2_architecture.py
```

Run script 2 (download and load OpenAI's weights — **first run downloads ~500MB**):

```powershell
python 02_load_pretrained_weights.py
```

Run script 3 (verify the architecture against HuggingFace's reference):

```powershell
python 03_compare_with_huggingface.py
```

**Plan around script 4 — it's the long one (30 min on GPU).**
This trains a real fine-tune on your GPU. You can leave it running
and come back. It saves the result to `gpt2_shakespeare.pt`.

Run script 4 (fine-tune on Shakespeare):

```powershell
python 04_finetune_on_shakespeare.py
```

Run script 5 (documentation of full reproduction requirements):

```powershell
python 05_what_full_reproduction_requires.py
```

---

## What success looks like

| Script | Expected output |
|---|---|
| 01 | Architecture built; total params **124.44M**; sanity forward pass works |
| 02 | Downloads HF weights (cached after first run); 148 tensors copied |
| 03 | **Logit max abs diff < 1e-3 = MATCH**; greedy generation matches HF |
| 04 | Initial loss ~5; final loss ~3.0-3.5; Shakespeare-flavored generations |
| 05 | Documentation only; no computation |

---

## Heads-up about script 02

The first run downloads OpenAI's GPT-2 weights from HuggingFace's CDN.
About **500 MB**. Takes 1-2 min on a typical connection.

After the first run, the weights are cached in your HuggingFace cache
directory (typically `C:\Users\<you>\.cache\huggingface\hub`). Subsequent
runs of script 02 are instant.

---

## Heads-up about script 04

This is the long script. **30 min on RTX 5050. ~6 hours on CPU.**

Pre-flight checklist before starting:
1. Plug in your laptop (if it's a laptop)
2. Verify sleep settings (Settings -> System -> Power & battery)
   are set to "Never" or 8+ hours when plugged in
3. Pause Windows updates (Settings -> Windows Update -> Pause for 1 week)
4. Close memory-hungry apps (browsers with many tabs, etc.)
5. Open a second PowerShell window for `nvidia-smi` heartbeat checks if curious

Then launch:

```powershell
python 04_finetune_on_shakespeare.py
```

You should see "Using device: cuda" in the first line.
Step prints appear every 200 iterations (~4 min between prints on GPU).

**Loss progression to expect:**
- step init: ~5.0 (untrained on Shakespeare specifically)
- step 200: ~4.2
- step 600: ~3.7
- step 1200: ~3.2
- step 1500: ~3.0-3.2 (final)

After training, you'll see Shakespeare-flavored generations from three
prompts ("ROMEO:", "JULIET:\nO Romeo, Romeo, ", "KING HENRY V:\n").
The model saves to `gpt2_shakespeare.pt`.

---

## Files this lesson creates

- `gpt2_loaded.pt` — copy of OpenAI's GPT-2 weights in your model's format (~500MB)
- `gpt2_shakespeare.pt` — the fine-tuned Shakespeare version (~500MB)

These are large binary files. **They're already gitignored** by the
existing `*.pt` rule. They live on your disk only.

---

## The single most important moment

**Script 03's logit comparison.** When you see:

```
Max absolute difference: 1.23e-5
  -> MATCH (within numerical precision)
```

That's the proof that your architecture is byte-for-byte correct against
an independent reference. Not "close enough to look right" but
"mathematically identical to OpenAI's published model, modulo float32
rounding noise."

---

## Time

About **45-60 min of focused work** plus **~30 min unattended training**
during script 04. The reading and architecture work is fast. The
training run is automatic.

---

## When you finish

Tell me. We'll deploy lesson 10 — Karpathy's 3.5-hour synthesis talk on
the modern LLM stack. That's watch-only, no code. The wrap-up.

You will have completed the entire foundational arc.
