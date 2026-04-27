# Lesson 07 Quickstart: Build GPT

**Prerequisites:** Lessons 01-06 done. DEPLOYMENT.md setup complete.

This is the big one. You'll build a real transformer and train it on
Shakespeare. Plan for one focused session for scripts 01-06 (~60 min)
plus a separate session for script 07's training run (15-30 min CPU).

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

## Get the dataset (Tiny Shakespeare)

Different from previous lessons — this lesson uses Shakespeare, not
names. Navigate to the lesson folder first:

```powershell
cd 07_build_gpt
```

Download Tiny Shakespeare:

```powershell
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Verify:

```powershell
ls input.txt
```

Should show ~1.1 MB.

---

## Run the lesson

You should be in `07_build_gpt/`.

Run script 1 (load + tokenize Shakespeare):

```powershell
python 01_explore_shakespeare.py
```

Run script 2 (bigram baseline — ~30 sec to train):

```powershell
python 02_bigram_baseline.py
```

Run script 3 (the math trick — three forms of weighted aggregation):

```powershell
python 03_attention_as_matmul.py
```

Run script 4 (self-attention from scratch — Q, K, V):

```powershell
python 04_self_attention.py
```

Run script 5 (multi-head + feed-forward + Block):

```powershell
python 05_multi_head_and_block.py
```

Run script 6 (assemble the full GPT, sanity check):

```powershell
python 06_full_gpt.py
```

**Take a break before script 7.** It trains for 15-30 min on CPU.
You can leave it running and come back.

Run script 7 (train the GPT — **this is the long one**):

```powershell
python 07_train_gpt.py
```

---

## What success looks like

| Script | Expected output |
|---|---|
| 01 | 1,115,394 chars; 65 unique chars; encode/decode round-trip works |
| 02 | Bigram trains to val loss ~2.5; generates keyboard-mash |
| 03 | v1 = v2 = v3 to float precision (max diff ~1e-16) |
| 04 | Single attention head; weights are softmax probability rows |
| 05 | Block has ~99k params; output shapes check out |
| 06 | Full GPT works; tiny config has ~52k params; loss near log(65)=4.17 |
| 07 | **Final val loss ~1.5;** 500 chars of Shakespeare-flavored text |

---

## What "Shakespeare-flavored" means in script 07

After 5000 training steps, generation will look something like:

```
ROMEO:
Why, brave the way thou dost not been here?

JULIET:
And these of mine eyes them with bring this hand
That can lay her father give breath
```

Words are mostly real English, character names appear in caps before
their lines, punctuation works, line breaks happen at appropriate
places. The actual *content* is gibberish — the model is too small
for real semantic content. But the **form** is unmistakable.

That's the architecture working. Lesson 09 (GPT-2 reproduction) is
where scale + data turns form into meaning.

---

## Time

Script 01-06: about 60-90 min of focused reading and tinkering.
Script 07: 15-30 min on CPU (3-5 min on GPU if available).

If you have an NVIDIA GPU, the script will use it automatically.
Otherwise it falls back to CPU and runs fine, just slower.

---

## When you finish

Tell me. We'll deploy lesson 08: **building the GPT tokenizer**. That's
the piece that turns "real text" into "tokens the model can process,"
and it explains every quirky behavior of real LLMs (why they're bad at
counting letters, why they handle URLs weirdly, why some emoji break
them). Lesson 09 reproduces actual GPT-2.

You're three lessons away from having reproduced a real, published
language model from scratch.
