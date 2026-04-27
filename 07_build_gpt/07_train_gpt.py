"""
07_train_gpt.py
---------------
Train the GPT on Shakespeare.

Hyperparameters (chosen to match Karpathy's video):
    block_size  = 256    (context length)
    n_embed     = 384    (embedding dim per token)
    n_heads     = 6      (each gets 384/6 = 64 dims)
    n_layers    = 6      (six transformer blocks)
    dropout     = 0.2

Total parameters: ~10 million. Trains to dev loss ~1.5 in 5000 steps.

Time on CPU: maybe 15-30 minutes for the full 5000 steps. If you have
a GPU and want to set device='cuda' the same code runs in 5 minutes.

What good output looks like:
    The model will produce text that LOOKS LIKE Shakespeare even though
    it's mostly nonsense. Characters speak in rough iambic patterns.
    Names like ROMEO and KING appear in dialogue. Punctuation works.
    But the actual SEMANTIC content is gibberish -- this is a tiny
    model trained on 1MB of data. What's amazing is how much structure
    it captures from such a small substrate.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
n_embed = 384
n_heads = 6
n_layers = 6
dropout = 0.2

# Use CUDA if you have it, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ---------------------------------------------------------------
# Load text
# ---------------------------------------------------------------
_FALLBACK = """First Citizen: Before we proceed any further, hear me speak.""" * 5000


def load_text(path="input.txt"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    print(f"[note] {path} not found -- using fallback. Loss numbers will not match Karpathy's video.")
    return _FALLBACK


text = load_text()
chars = sorted(set(text))
V = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s): return [stoi[c] for c in s]
def decode(ids): return "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"Vocab size: {V}, Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")


# ---------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------
def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------
# Model classes (same as script 06)
# ---------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, n_embed, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, n_embed, block_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_heads, block_size, dropout=0.0):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, n_embed, block_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embed, n_heads, n_layers, block_size, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads, block_size, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(tok + pos)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new):
        for _ in range(max_new):
            idx_crop = idx[:, -self.block_size:]
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


# ---------------------------------------------------------------
# Build the model
# ---------------------------------------------------------------
torch.manual_seed(1337)
model = GPT(V, n_embed, n_heads, n_layers, block_size, dropout).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"GPT parameters: {n_params/1e6:.2f}M ({n_params:,})")


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
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print()
print(f"Training for {max_iters} iterations...")
print(f"  (CPU will take 15-30 min, GPU 3-5 min)")
print()

for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        print(f"  step {it:>5}/{max_iters}  train loss = {losses['train']:.4f}  val loss = {losses['val']:.4f}")

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print()


# ---------------------------------------------------------------
# Generate and save
# ---------------------------------------------------------------
print("Generating 500 characters of Shakespeare...")
print("=" * 60)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(context, max_new=500)[0].tolist()
print(decode(out))
print("=" * 60)
print()


# Save the model
torch.save({
    "model_state": model.state_dict(),
    "config": {"V": V, "n_embed": n_embed, "n_heads": n_heads,
               "n_layers": n_layers, "block_size": block_size},
    "stoi": stoi, "itos": itos,
}, "trained_gpt.pt")
print("Saved trained_gpt.pt")


# ---------------------------------------------------------------
# What we just accomplished
# ---------------------------------------------------------------
print("""
============================================================
YOU TRAINED A TRANSFORMER. FROM SCRATCH. ON SHAKESPEARE.
============================================================

This is the same architecture as ChatGPT, Claude, Gemini, and every
other modern LLM. It's just smaller (10M params vs hundreds of billions)
and trained on far less data (1MB vs trillions of tokens).

The output looks Shakespeare-flavored: character names in caps before
dialogue, line breaks at appropriate places, punctuation that mostly
makes sense, words that mostly look like English. The content is
nonsense -- the model is too small for that -- but the FORM is
unmistakably Shakespeare.

That's the architecture working. Scale + data is what turns the form
into meaning. We get to that in lesson 09 (GPT-2 reproduction).

Lesson 08 next: tokenization. Currently we use one token per character
(65 vocab). Real LLMs use subword tokenization (50,000+ vocab) so they
can handle real text efficiently. That's the next piece.
""")
