"""
05_tokenizer_quirks.py
----------------------
Now the fun part. The tokenizer is the source of MANY famous LLM
quirks. Here are the big ones:

  1. LLMs are bad at counting letters in a word.
       "How many R's in strawberry?" trips up many models. Why?
       Because "strawberry" is ONE TOKEN to the model, not a sequence
       of letters. To count R's, the model has to "remember" what
       letters compose that token -- which it can only do if that
       came up in training.

  2. LLMs are bad at reversing strings.
       Same reason. "Reverse the word 'helicopter'" is hard because
       the model sees "helicopter" as one (or few) tokens, not as
       a sequence of characters.

  3. LLMs handle whitespace weirdly.
       Tokenizers often bundle spaces with following words. So " hello"
       and "hello" are different tokens. Trailing spaces in prompts
       sometimes trigger weird completions.

  4. LLMs are worse at non-English text.
       Their tokenizers were trained mostly on English. Non-English
       text gets tokenized into many more tokens per character, which
       means slower processing AND worse performance.

  5. Numbers and arithmetic are weird.
       "1234" might be one token. "1235" might be three tokens. The
       model has no consistent way to "see" digits.

  6. There are "glitch tokens" that break models.
       Famous ones: "SolidGoldMagikarp", "petertodd", various Reddit
       usernames. These appear in the tokenizer's vocab from training
       data, but the model itself never saw them in actual training,
       so it doesn't know what to do with them.

This script demonstrates each of these.
"""

# Reuse the Tokenizer from script 04
import os

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size, verbose=False):
        num_merges = vocab_size - 256
        ids = list(text.encode("utf-8"))
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats: break
            top_pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = merge(ids, top_pair, new_id)
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges: break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------
# Train a tokenizer on Shakespeare for demonstration
# ---------------------------------------------------------------
SHAKESPEARE_PATH = "../07_build_gpt/input.txt"
if os.path.exists(SHAKESPEARE_PATH):
    with open(SHAKESPEARE_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text):,} chars of Shakespeare for training\n")
else:
    text = "The quick brown fox jumps over the lazy dog. " * 1000
    print("(Using fallback text)\n")

tok = Tokenizer()
tok.train(text, vocab_size=512)


# ---------------------------------------------------------------
# Helper: pretty-print how a string tokenizes
# ---------------------------------------------------------------
def show_tokens(s, tok):
    ids = tok.encode(s)
    print(f"  Input:  {s!r}")
    print(f"  Bytes:  {len(s.encode('utf-8'))}, Tokens: {len(ids)}")
    print(f"  Tokens (decoded individually):")
    for tok_id in ids:
        try:
            piece = tok.vocab[tok_id].decode("utf-8")
            print(f"    {tok_id:>4}  {piece!r}")
        except UnicodeDecodeError:
            print(f"    {tok_id:>4}  bytes {list(tok.vocab[tok_id])}")
    print()


# ---------------------------------------------------------------
# Quirk 1: Letter counting
# ---------------------------------------------------------------
print("=" * 60)
print("QUIRK 1: Letter counting is hard")
print("=" * 60)
print()
print("If a word is one token, the model has to learn what letters")
print("compose it -- which it can only do via training. It can't just")
print("'look at' the characters.")
print()
show_tokens("strawberry", tok)
print("  -> Notice how 'strawberry' splits. The model sees a few chunks,")
print("     not 10 individual letters. Counting R's requires it to")
print("     'know' (from training) how each chunk is spelled.")
print()


# ---------------------------------------------------------------
# Quirk 2: Whitespace bonded with words
# ---------------------------------------------------------------
print("=" * 60)
print("QUIRK 2: Whitespace fuses with following words")
print("=" * 60)
print()
print("' the' and 'the' are usually different tokens.")
print()
show_tokens("the", tok)
show_tokens(" the", tok)
print("  -> Note how the space gets bundled with 'th' or absorbed")
print("     into a multi-char token. This is why prompts that end in")
print("     a trailing space sometimes confuse models -- they end")
print("     mid-token-pattern.")
print()


# ---------------------------------------------------------------
# Quirk 3: Non-English / non-ASCII tokenization
# ---------------------------------------------------------------
print("=" * 60)
print("QUIRK 3: Non-English text tokenizes inefficiently")
print("=" * 60)
print()
print("Our tokenizer was trained on English Shakespeare. Foreign")
print("characters fall back to RAW BYTES.")
print()
show_tokens("Hello", tok)
show_tokens("café", tok)
show_tokens("你好", tok)
show_tokens("👋", tok)
print("  -> 'Hello' is just a few tokens. Chinese is 6 raw bytes.")
print("     Emoji is 4 raw bytes. Non-English speakers paying per token")
print("     pay 3-4x what English speakers pay for equivalent meaning.")
print()


# ---------------------------------------------------------------
# Quirk 4: Numbers tokenize unpredictably
# ---------------------------------------------------------------
print("=" * 60)
print("QUIRK 4: Numbers tokenize inconsistently")
print("=" * 60)
print()
print("Some short numbers are single tokens. Long ones are split")
print("based on what subsequences happened to be common in training.")
print()
show_tokens("1", tok)
show_tokens("12", tok)
show_tokens("123", tok)
show_tokens("1234", tok)
show_tokens("12345", tok)
print("  -> The model sees 'inconsistent chunks' of numbers, which is")
print("     why arithmetic is so unreliable in LLMs without specific")
print("     training. Each new number requires re-learning the digits.")
print()


# ---------------------------------------------------------------
# Quirk 5: String reversal
# ---------------------------------------------------------------
print("=" * 60)
print("QUIRK 5: Reversing a string requires knowing its tokens' bytes")
print("=" * 60)
print()
print("To reverse 'helicopter' the model needs to know exactly which")
print("characters compose each of its tokens. If it doesn't know, it")
print("guesses based on common patterns -- often wrong.")
print()
show_tokens("helicopter", tok)


# ---------------------------------------------------------------
# Putting it all together: why LLMs fail in patterned ways
# ---------------------------------------------------------------
print("=" * 60)
print("THE BIG PICTURE")
print("=" * 60)
print("""
Every "weird" behavior of GPT, Claude, Gemini that you've ever
encountered has a tokenization explanation:

  - Math errors: numbers tokenize inconsistently
  - Letter counting: words are tokens, not letter sequences
  - Reversing: same problem
  - Bad non-English: tokenizer was trained mostly on English
  - Glitch tokens: merged tokens that never appeared in actual
    training data, so the model has no learned behavior for them

The model is doing exactly what it was trained to do -- predict the
next token given previous tokens. The "weirdness" is at the
input/output interface, not in the model's reasoning.

When you see an LLM fail at "How many R's in strawberry?", that's
not a reasoning failure. The model literally cannot SEE individual
letters in 'strawberry' because the entire word (or large chunks
of it) is a single token.

Fix: ask differently. "Spell out s-t-r-a-w-b-e-r-r-y and count the
R's." The dashes break it into smaller tokens that include the
individual characters. Same model, different tokenization, much
better answer.

Next: regex-based pre-tokenization, the trick GPT-2 uses to handle
whitespace and punctuation more sanely.
""")
