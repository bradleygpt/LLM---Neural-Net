"""
04_finetune_on_shakespeare.py
-----------------------------
Take the loaded GPT-2 weights and fine-tune them on Tiny Shakespeare.

This is a SHORT training run. We're not training from scratch (that
would take days on your GPU). We're starting from OpenAI's pretrained
weights and adapting them to a specific style. Modern LLM development
calls this "fine-tuning."

What you'll see:
  - Initial loss is around 4-5 (GPT-2 hasn't seen Shakespeare specifically)
  - Loss drops quickly in the first few hundred steps
  - Final loss around 3.0-3.5 after ~30 minutes
  - Generated text in clear Shakespeare style

Why fine-tuning works:
  - GPT-2 already knows English (from web text training)
  - It already knows about Shakespeare in general (web mentions, quotes)
  - It just needs to adapt its OUTPUT distribution toward Shakespeare's style
  - That requires far less data and compute than learning English from zero

Hyperparameters (CPU-friendly defaults; tune for your GPU):
  block_size = 256  (smaller than GPT-2's 1024 to save memory)
  batch_size = 4    (small for memory; gradient accumulation expands effective batch)
  lr = 3e-5         (low because we're starting from a good model)
  max_iters = 1500  (~30 min on RTX 5050)
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import our architecture
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("arch", "01_gpt2_architecture.py")
arch_mod = importlib.util.module_from_spec(spec)
sys.modules["arch"] = arch_mod
spec.loader.exec_module(arch_mod)

GPT = arch_mod.GPT
GPTConfig = arch_mod.GPTConfig


# ---------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------
batch_size = 4               # tokens per gradient step = batch_size * block_size
block_size = 256             # cropped from GPT-2's full 1024 to save memory
max_iters = 1500             # ~30 min on RTX 5050
eval_interval = 200
eval_iters = 20              # evaluation batches
learning_rate = 3e-5         # low for fine-tuning
weight_decay = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ---------------------------------------------------------------
# Load Shakespeare and tokenize with the real GPT-2 tokenizer
# ---------------------------------------------------------------
SHAKESPEARE_PATH = "../07_build_gpt/input.txt"
if not os.path.exists(SHAKESPEARE_PATH):
    raise FileNotFoundError(
        f"Couldn't find {SHAKESPEARE_PATH}. Make sure you have the Tiny "
        "Shakespeare dataset from lesson 07 in 07_build_gpt/input.txt."
    )

with open(SHAKESPEARE_PATH, "r", encoding="utf-8") as f:
    text = f.read()
print(f"Loaded {len(text):,} chars of Shakespeare")

import tiktoken
enc = tiktoken.get_encoding("gpt2")
data = torch.tensor(enc.encode(text), dtype=torch.long)
print(f"Tokenized to {len(data):,} GPT-2 tokens (compression: {len(text) / len(data):.2f}x)")

# 90/10 train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------
# Build model and load pretrained weights
# ---------------------------------------------------------------
print()
print("Building GPT-2 small and loading pretrained weights...")

config = GPTConfig()
config.dropout = 0.0    # no dropout for fine-tuning
model = GPT(config)

if os.path.exists("gpt2_loaded.pt"):
    state = torch.load("gpt2_loaded.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    print("  Loaded weights from script 02's saved checkpoint.")
else:
    # Fall back: load fresh from HuggingFace
    print("  gpt2_loaded.pt not found. Loading fresh from HuggingFace...")
    from transformers import GPT2LMHeadModel
    hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd_hf = hf.state_dict()
    sd_ours = model.state_dict()
    transpose_keys = ["attn.c_attn.weight", "attn.c_proj.weight",
                      "mlp.c_fc.weight", "mlp.c_proj.weight"]
    keys_to_skip = [".attn.bias", ".attn.masked_bias"]
    for k_hf, v_hf in sd_hf.items():
        if any(s in k_hf for s in keys_to_skip): continue
        if k_hf not in sd_ours: continue
        if any(s in k_hf for s in transpose_keys):
            v_hf = v_hf.t()
        if v_hf.shape == sd_ours[k_hf].shape:
            with torch.no_grad():
                sd_ours[k_hf].copy_(v_hf)
    print("  Loaded fresh weights from HuggingFace.")

model = model.to(device)


# ---------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------
# AdamW with weight decay only on 2D weights (matrices), not biases / LN params.
# This is what GPT-2 paper recommends.
decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad]
optimizer = torch.optim.AdamW([
    {"params": decay_params, "weight_decay": weight_decay},
    {"params": nodecay_params, "weight_decay": 0.0},
], lr=learning_rate, betas=(0.9, 0.95))
print(f"  Optimizer: AdamW with weight_decay={weight_decay} on {len(decay_params)} matrices, 0.0 on {len(nodecay_params)} bias/LN params")


# ---------------------------------------------------------------
# Loss estimation helper
# ---------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------
# Train
# ---------------------------------------------------------------
print()
print(f"Fine-tuning for {max_iters} iterations on Shakespeare...")
print(f"  Wall-clock estimate: ~30 min on GPU, much longer on CPU")
print()

# Initial loss before training
losses = estimate_loss()
print(f"  step  init  | train loss = {losses['train']:.4f}  val loss = {losses['val']:.4f}")

start = time.time()
for it in range(max_iters):
    if (it + 1) % eval_interval == 0 or it == max_iters - 1:
        elapsed = time.time() - start
        losses = estimate_loss()
        print(f"  step {it+1:>5}/{max_iters} | train loss = {losses['train']:.4f}  val loss = {losses['val']:.4f}  ({elapsed/60:.1f} min)")

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # Clip gradients (helps with training stability)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

print()
print(f"Done. Total time: {(time.time() - start)/60:.1f} min")


# ---------------------------------------------------------------
# Generate Shakespeare from the fine-tuned model
# ---------------------------------------------------------------
print()
print("=" * 60)
print("GENERATING SHAKESPEARE FROM THE FINE-TUNED MODEL")
print("=" * 60)


@torch.no_grad()
def generate(prompt, max_new=200, temperature=0.8, top_k=40):
    model.eval()
    enc_prompt = torch.tensor([enc.encode(prompt)], device=device, dtype=torch.long)
    ids = enc_prompt
    for _ in range(max_new):
        idx_crop = ids if ids.size(1) <= block_size else ids[:, -block_size:]
        logits, _ = model(idx_crop)
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    model.train()
    return enc.decode(ids[0].tolist())


prompts = [
    "ROMEO:",
    "JULIET:\nO Romeo, Romeo, ",
    "KING HENRY V:\n",
]

for p in prompts:
    print(f"\nPrompt: {p!r}")
    output = generate(p, max_new=150, temperature=0.8)
    print(f"--- generation ---")
    print(output)
    print(f"--- end ---")


# ---------------------------------------------------------------
# Save the fine-tuned model
# ---------------------------------------------------------------
torch.save(model.state_dict(), "gpt2_shakespeare.pt")
print("\nSaved fine-tuned model to gpt2_shakespeare.pt")


# ---------------------------------------------------------------
# What we just accomplished
# ---------------------------------------------------------------
print("""

============================================================
YOU FINE-TUNED GPT-2 ON SHAKESPEARE
============================================================

You took weights from a model OpenAI trained on 40GB of web text in
2019, brought them onto your machine, and adapted them to a specific
literary style with ~30 minutes of GPU time.

This is exactly the workflow that powers production LLM applications:
  - Companies start from a powerful pretrained model
  - Fine-tune on their domain-specific data
  - Get a model that's expert in their use case at fraction of the
    cost of training from scratch

The architecture is identical to ChatGPT's, Claude's, Gemini's. The
techniques are identical to what production teams use. You just used
them at 124M parameters and 1MB of training data instead of 70B+
parameters and trillions of tokens.

Lesson 10 (the watch-only deep dive) walks through how the FRONTIER
labs go from this kind of base model to something like ChatGPT --
RLHF, Constitutional AI, instruction tuning, all the alignment work.
""")
