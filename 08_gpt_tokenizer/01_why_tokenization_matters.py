"""
01_why_tokenization_matters.py
------------------------------
In lesson 07 we used character-level tokenization. Each character of
Shakespeare became one integer in [0, 65). That's the simplest thing
that could work, and it kept the lesson clean.

But character-level tokenization is BAD for real-world LLMs. Three
specific reasons:

  1. Sequence length explodes. "tokenization" is 12 characters but
     ~3 tokens for GPT-style tokenizers. Real text has 4-5x more
     characters than tokens on average. Since attention scales as
     O(T^2), longer sequences mean WAY more compute per training step.

  2. The model wastes capacity learning what "th" means as a unit, what
     "ing" means at the end of words, what "tion" means as a suffix.
     Subword tokens encode these patterns as single units, freeing
     up parameters for higher-level structure.

  3. Cross-language and Unicode handling becomes a nightmare. "naive"
     vs "naïve" become totally different sequences because the i and
     ï are different code points. Subword tokenization handles this
     more gracefully.

The opposite extreme — word-level tokenization — has its own problems:

  1. Vocabulary explosion. English has ~170,000 words. Add proper
     nouns and you're at half a million. Vs ~50,000 for subword.

  2. Out-of-vocabulary words are invisible. If the training set didn't
     have "Brexit" then a word-level model can never even REPRESENT
     "Brexit" — it doesn't exist in the vocabulary.

  3. No generalization between morphologically related words.
     "running", "runner", "ran" are three completely independent
     tokens. The model has to learn each one separately.

The compromise: SUBWORD TOKENIZATION. Specifically, BYTE-PAIR ENCODING
(BPE), which is what GPT-2, GPT-3, GPT-4, Claude, and most modern LLMs
use. The key idea: greedily merge the most-common adjacent pairs of
tokens, building up subword units.

This script just shows the problem space. The next scripts build BPE.
"""

# ---------------------------------------------------------------
# Cost comparison: characters vs tokens
# ---------------------------------------------------------------

samples = [
    "The quick brown fox jumps over the lazy dog.",
    "Tokenization is a fundamental preprocessing step.",
    "Antidisestablishmentarianism",
    "naive vs naïve",  # unicode example
    "café résumé jalapeño",  # accented chars
    "👋 Hello world! 你好世界",  # emoji + Chinese
]

print("=" * 60)
print("CHARACTER COUNT vs APPROXIMATE TOKEN COUNT")
print("=" * 60)
print()

# Real GPT-style tokenizers have specific numbers; these are approximations
# based on typical subword tokenization (~4 chars/token in English).
# When you actually run BPE later, the numbers will be exact.
print(f"{'String':<50} {'Chars':>8} {'~Tokens':>8}")
print("-" * 70)
for s in samples:
    chars = len(s)
    approx_tokens = max(1, chars // 4)  # English heuristic
    print(f"{s[:50]:<50} {chars:>8} {approx_tokens:>8}")
print()


# ---------------------------------------------------------------
# A concrete example of the savings
# ---------------------------------------------------------------
example = "Tokenization is the process of converting text into numerical tokens."
chars = len(example)
print(f"Example: {example!r}")
print(f"  Length in characters: {chars}")
print(f"  Approximate length in subword tokens: ~{chars // 4}")
print()
print(f"With block_size=256 in lesson 07:")
print(f"  Character-level: model sees ~256 chars of context")
print(f"  Subword-level:   model sees ~256 tokens = ~1024 chars of context")
print(f"  -> 4x effectively longer context, with the SAME compute cost.")
print()


# ---------------------------------------------------------------
# A peek at what real GPT tokenizers do
# ---------------------------------------------------------------
print("=" * 60)
print("WHAT REAL TOKENIZATION LOOKS LIKE (heuristic)")
print("=" * 60)
print()
print("The string 'Tokenization' might split as:")
print("  ['Token', 'ization']    -> 2 tokens")
print()
print("'antidisestablishmentarianism' might split as:")
print("  ['anti', 'dis', 'est', 'abl', 'ish', 'ment', 'arian', 'ism']  -> 8 tokens")
print()
print("Real BPE splits depend on the specific tokenizer. GPT-4 has a")
print("vocabulary of about 100,000 subword tokens. GPT-2 had 50,257.")
print("Lesson 09 will use the actual GPT-2 tokenizer; this lesson")
print("builds one from scratch so you understand the algorithm.")
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
print("""
============================================================
THE TOKENIZATION TRADEOFFS
============================================================

  CHARACTER-LEVEL (lesson 07):
    + Tiny vocabulary
    + No out-of-vocab issues
    - Sequences are 4-5x longer
    - Model wastes capacity on common letter patterns

  WORD-LEVEL:
    + Sequences are short
    + Each token has clear meaning
    - Massive vocabulary (170k+ words)
    - Out-of-vocab problem for new/rare words
    - No generalization between morphologically related words

  SUBWORD (BPE):
    + Reasonable vocabulary (~50k)
    + Handles new words via subword decomposition
    + Captures common patterns as single units
    + Sequences only 1.3x longer than word-level
    - More complex to implement
    - The merges encode biases from the training corpus

BPE wins for real LLMs. The next script builds it from scratch.
""")
