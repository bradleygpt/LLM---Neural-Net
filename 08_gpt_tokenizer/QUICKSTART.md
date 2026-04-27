# Lesson 08 Quickstart: GPT Tokenizer

**Prerequisites:** Lessons 01-07 done. DEPLOYMENT.md setup complete.
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

Navigate to the lesson:

```powershell
cd 08_gpt_tokenizer
```

---

## Optional: install tiktoken

Script 07 compares your tokenizer to GPT-2 and GPT-4 via OpenAI's
official `tiktoken` library. Skip this and script 07 will run with a
graceful fallback.

```powershell
pip install tiktoken
```

If you want the full comparison output, install it. The package is
small and installs in seconds.

---

## Run the lesson

You should be in `08_gpt_tokenizer/`.

Run script 1 (motivation):

```powershell
python 01_why_tokenization_matters.py
```

Run script 2 (UTF-8 bytes):

```powershell
python 02_bytes_as_foundation.py
```

Run script 3 (BPE algorithm):

```powershell
python 03_bpe_algorithm.py
```

Run script 4 (Tokenizer class trained on Shakespeare):

```powershell
python 04_tokenizer_class.py
```

Run script 5 (LLM quirks):

```powershell
python 05_tokenizer_quirks.py
```

Run script 6 (regex pre-tokenization):

```powershell
python 06_regex_pretokenization.py
```

Run script 7 (compare to real tokenizers):

```powershell
python 07_compare_to_real_tokenizers.py
```

---

## What success looks like

| Script | Expected output |
|---|---|
| 01 | Char vs token comparison; tradeoffs explained |
| 02 | UTF-8 encoding examples; round-trip works |
| 03 | 10 merges shown step by step; round-trip works |
| 04 | Tokenizer trained on Shakespeare; first 20 merges shown; saves to disk |
| 05 | Demonstration of letter counting, whitespace, Unicode quirks |
| 06 | Regex splits shown for various inputs; trained tokenizer demo |
| 07 | Comparison table: ours vs GPT-2 vs GPT-4 token counts (if tiktoken installed) |

If `tiktoken` isn't installed, script 07 prints "[note] tiktoken not
installed" and shows only your tokenizer's results.

---

## Files this lesson creates

- `shakespeare_tokenizer.txt` — your saved BPE merges from script 04

These are local artifacts that don't need to be committed to the repo.

---

## The single most important moment

**Script 03's three core functions.** You'll see the entire BPE
algorithm written out in about 30 lines of Python:

```python
get_stats(ids)             # count adjacent pairs
merge(ids, pair, new_id)   # replace pairs with new id
# ...repeat in a loop
```

Read those carefully. That's GPT-2's tokenizer in essence.

---

## Time

About 60-90 min total. Each script runs in seconds; the time is in
reading the comments and absorbing the implications.

---

## When you finish

Tell me. We deploy lesson 09: **reproduce GPT-2**. That's the lesson
where everything comes together — your transformer architecture from
lesson 07, your tokenizer concept from this lesson, scaled up to the
actual GPT-2 124M model OpenAI published in 2019. Real published model,
your code, your GPU.
