"""
07_compare_to_real_tokenizers.py
--------------------------------
Compare our hand-built tokenizer to the real ones used by GPT-2,
GPT-3.5, and GPT-4. We use OpenAI's `tiktoken` library, which is
the actual fast C++ implementation of the algorithm we just built.

Install:
    pip install tiktoken

This script:
  1. Tokenizes various strings using our tokenizer (Shakespeare-trained)
  2. Tokenizes the same strings using GPT-2 and GPT-4 tokenizers
  3. Shows the differences

You'll see:
  - Our tokenizer (trained on 1MB of Shakespeare) is MUCH worse for
    general English text. That's because Shakespeare uses unusual
    vocabulary -- "thou", "hast", "doth" are common but "computer",
    "internet", "pizza" are not.
  - GPT-4's tokenizer is MUCH more efficient than GPT-2's. Larger
    vocabulary means more common patterns get single tokens.
  - All three handle "nonsense" inputs gracefully (no out-of-vocab).
"""

import os


# ---------------------------------------------------------------
# Try to import tiktoken; if not installed, give a helpful note
# ---------------------------------------------------------------
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("[note] tiktoken not installed. To compare with real tokenizers:")
    print("[note]    pip install tiktoken")
    print("[note] This script will only show our local tokenizer's results.")
    print()


# ---------------------------------------------------------------
# Re-import our regex BPE tokenizer (from script 06)
# ---------------------------------------------------------------
import re

GPT2_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)


def get_stats(ids, counts=None):
    counts = counts or {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class RegexTokenizer:
    def __init__(self, pattern=None):
        self.pattern = pattern or GPT2_PATTERN
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size):
        num_merges = vocab_size - 256
        chunks = self.pattern.findall(text)
        ids_list = [list(ch.encode("utf-8")) for ch in chunks]
        for i in range(num_merges):
            stats = {}
            for ids in ids_list:
                get_stats(ids, stats)
            if not stats: break
            top_pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids_list = [merge(ids, top_pair, new_id) for ids in ids_list]
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

    def encode(self, text):
        chunks = self.pattern.findall(text)
        all_ids = []
        for ch in chunks:
            ids = list(ch.encode("utf-8"))
            while len(ids) >= 2:
                stats = get_stats(ids)
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges: break
                ids = merge(ids, pair, self.merges[pair])
            all_ids.extend(ids)
        return all_ids


# ---------------------------------------------------------------
# Load Shakespeare and train our tokenizer
# ---------------------------------------------------------------
SHAKESPEARE_PATH = "../07_build_gpt/input.txt"
if os.path.exists(SHAKESPEARE_PATH):
    with open(SHAKESPEARE_PATH, "r", encoding="utf-8") as f:
        train_text = f.read()
    print(f"Training our tokenizer on {len(train_text):,} chars of Shakespeare (vocab 1024)...")
else:
    train_text = "The quick brown fox jumps over the lazy dog. " * 5000
    print("Training our tokenizer on fallback text (vocab 1024)...")

ours = RegexTokenizer()
ours.train(train_text, vocab_size=1024)
print(f"Done. {len(ours.merges)} merges.\n")


# ---------------------------------------------------------------
# Set up the real tokenizers (if tiktoken is available)
# ---------------------------------------------------------------
if HAS_TIKTOKEN:
    gpt2 = tiktoken.get_encoding("gpt2")        # GPT-2: 50,257 tokens
    gpt4 = tiktoken.get_encoding("cl100k_base") # GPT-3.5/GPT-4: 100,277 tokens

    print(f"GPT-2 tokenizer:  {gpt2.n_vocab:,} tokens")
    print(f"GPT-4 tokenizer:  {gpt4.n_vocab:,} tokens")
    print(f"Our tokenizer:    {len(ours.vocab):,} tokens")
    print()


# ---------------------------------------------------------------
# Compare on different inputs
# ---------------------------------------------------------------
test_strings = [
    # Shakespeare-flavored (our tokenizer should do okay)
    "Romeo, wherefore art thou Romeo?",
    # Modern English (our tokenizer should be worse)
    "I'll grab a pizza and watch Netflix tonight.",
    # Code (none of our tokenizers were trained on much code, but GPT4 has the most)
    "def hello():\n    print('Hello, world!')",
    # Numbers
    "1 + 2 = 3, but 1234 + 5678 = 6912",
    # Unicode / emoji
    "Hello 世界 🌍",
    # Long technical word
    "antidisestablishmentarianism",
]

print("=" * 80)
print("TOKEN COUNTS PER TOKENIZER")
print("=" * 80)
print()
header = f"{'String':<55} {'Ours':>8} {'GPT-2':>8} {'GPT-4':>8}"
print(header)
print("-" * len(header))

for s in test_strings:
    ours_count = len(ours.encode(s))
    gpt2_count = len(gpt2.encode(s)) if HAS_TIKTOKEN else "n/a"
    gpt4_count = len(gpt4.encode(s)) if HAS_TIKTOKEN else "n/a"
    display = s[:53] + "..." if len(s) > 55 else s
    display = display.replace("\n", "\\n")
    print(f"{display:<55} {ours_count:>8} {gpt2_count:>8} {gpt4_count:>8}")
print()


# ---------------------------------------------------------------
# Detailed view of one example
# ---------------------------------------------------------------
if HAS_TIKTOKEN:
    print("=" * 60)
    print("DETAILED TOKENIZATION OF MODERN ENGLISH")
    print("=" * 60)
    print()
    s = "I'll grab a pizza and watch Netflix tonight."
    print(f"Input: {s!r}")
    print(f"Bytes: {len(s.encode('utf-8'))}")
    print()

    print("GPT-2 tokens (each row is one token):")
    for tok_id in gpt2.encode(s):
        piece = gpt2.decode([tok_id])
        print(f"  {tok_id:>6}  {piece!r}")
    print()

    print("GPT-4 tokens (each row is one token):")
    for tok_id in gpt4.encode(s):
        piece = gpt4.decode([tok_id])
        print(f"  {tok_id:>6}  {piece!r}")
    print()


# ---------------------------------------------------------------
# Token cost reality check
# ---------------------------------------------------------------
print("=" * 60)
print("TOKEN COST REALITY CHECK")
print("=" * 60)
print()
print("Real LLM APIs charge per token. Quick math:")
print()
print("  Article (~1000 words, ~5500 chars):")
print(f"    Our tokenizer:  {len(ours.encode('A short article. ' * 100)):,} tokens")
if HAS_TIKTOKEN:
    test_article = "A short article. " * 100
    print(f"    GPT-2:          {len(gpt2.encode(test_article)):,} tokens")
    print(f"    GPT-4:          {len(gpt4.encode(test_article)):,} tokens")
print()
print("Practical implications:")
print("  - GPT-4's tokenizer is ~25-40% more efficient than GPT-2's")
print("    on English text. Same input, fewer tokens, lower cost.")
print("  - Non-English text uses dramatically more tokens.")
print("    Chinese/Japanese can be 2-3x more expensive per character.")
print("  - Code uses moderately more tokens than prose, especially")
print("    for languages with lots of punctuation.")
print()


# ---------------------------------------------------------------
# What we accomplished
# ---------------------------------------------------------------
print("""
============================================================
LESSON 08 COMPLETE
============================================================

You built a Byte-Pair Encoding tokenizer from scratch. Around 60
lines of Python for the Tokenizer class, plus regex preprocessing.

Same algorithm as GPT-2, GPT-3, GPT-4, Claude, Gemini, every
modern LLM tokenizer. The differences at production scale:
  - Trained on 100s of GB of text, not 1MB of Shakespeare
  - 50k-100k merges, not ~750
  - Faster implementations (tiktoken in C++, ~100x our speed)
  - Special tokens (<|endoftext|>, control tokens for chat formats)
  - Unicode normalization

Now that you understand tokenization:
  - Every LLM quirk has a clear explanation
  - Token costs make sense
  - You could swap in a custom tokenizer for a domain (medical, code,
    a specific language) and dramatically improve efficiency

Lesson 09: REPRODUCE GPT-2.
We take everything you've built (transformer + tokenizer) and
reproduce the actual GPT-2 124M model from OpenAI's 2019 paper.
This is where it stops being a learning exercise and becomes a
real, published model. Will need your GPU.
""")
