"""
05_what_full_reproduction_requires.py
-------------------------------------
You've built and verified GPT-2's architecture, loaded OpenAI's actual
weights, and fine-tuned the model. What you HAVEN'T done is train
GPT-2 from scratch on the original 40GB WebText corpus.

This script documents what that would entail, so you know what you'd
need to scale up. No code runs in this script -- it's just a careful
accounting of the leap from "fine-tune for 30 min" to "train from
scratch like the paper."
"""

print("""
============================================================
THE LEAP FROM YOUR FINE-TUNE TO FULL GPT-2 REPRODUCTION
============================================================

What you did in script 04:
  - Started from OpenAI's pretrained weights
  - Fine-tuned on 1.1MB of Shakespeare
  - 1500 iterations
  - ~30 minutes on RTX 5050 mobile GPU
  - Cost: $0

What full GPT-2 (124M) reproduction requires:

  1. DATA
     - 10B tokens of curated web text
     - Original GPT-2 used WebText (40GB, 8M Reddit-linked pages)
     - Modern reproductions use FineWeb-Edu (10B tokens, ~30GB compressed)
     - Required disk: ~50 GB working space, 30 GB final

  2. COMPUTE
     - ~1.5e21 FLOPs total training compute
     - 600,000 iterations at batch_size=32 with block_size=1024
     - On 8x A100 GPUs (Karpathy's setup): ~90 minutes
     - On a single RTX 4090: ~4-5 days
     - On your RTX 5050 mobile: ~10-14 days (if your laptop survives the heat)

  3. WALL-CLOCK ON YOUR HARDWARE
     - The mobile GPU's smaller VRAM (8GB) means smaller batches and
       gradient accumulation. Combined with thermal throttling, you'd
       likely need 2-3 weeks running 24/7.
     - Reasonable concern: laptops aren't designed for sustained 100% GPU
       load for weeks. Risk of damage.

  4. COST IF YOU RENT CLOUD GPUs
     - AWS p4d (8x A100): $32/hr * 1.5 hr = ~$50
     - Lambda 8x A100: $11/hr * 1.5 hr = ~$17
     - Vast.ai (consumer GPUs): ~$5-10 for full reproduction
     - All have free tiers; none are free for 24+ hour training runs

  5. EXPECTED OUTCOME
     - Final loss matching the paper: ~3.0 on validation
     - Generation quality matching public GPT-2: yes
     - HellaSwag accuracy: ~30% (random=25%, paper=29.4%)
     - WikiText perplexity: ~30 (paper: 29.4)


============================================================
ALTERNATIVES THAT FIT IN A DAY ON YOUR HARDWARE
============================================================

  A. Train GPT-2 small on a SUBSET of FineWeb-Edu
     - 1B tokens (10% of original)
     - ~24 hours on RTX 5050
     - Loss ~3.5 (worse than paper, but real distribution)
     - Demonstrates full training pipeline

  B. Train a SMALLER model (50M params) on full FineWeb-Edu
     - Half the layers and dimensions of GPT-2 small
     - ~12 hours on RTX 5050
     - Loss ~3.5 with less capacity
     - Same code, smaller model

  C. Fine-tune GPT-2 ON MORE DOMAIN DATA
     - Take the script 04 model
     - Fine-tune on a larger corpus (Wikipedia subset, code dataset, etc.)
     - 1-3 hours
     - Most pragmatic application of what you've built


============================================================
WHAT YOU'VE LEARNED THAT TRANSFERS DIRECTLY
============================================================

The skills from this lesson scale unchanged to production:
  - Architecture design: same code patterns at any scale
  - Weight loading from checkpoints: same dict-mapping logic
  - Tokenization: same BPE algorithm at production vocabulary sizes
  - Fine-tuning workflow: same optimizer, same loss, same eval pattern
  - GPU memory management: same considerations, just bigger numbers

The skills you DON'T have yet (covered in lesson 10):
  - RLHF (training the model to be helpful and harmless)
  - Instruction tuning (turning a base model into a chat assistant)
  - Constitutional AI (Anthropic's approach to alignment)
  - Multi-GPU distributed training (model parallelism, FSDP)
  - Production serving (kv-cache, batched inference, quantization)

These are all built ON TOP of what you've now constructed. The base
LLM is the foundation. Everything else is the alignment and serving
layer.


============================================================
LESSON 09 COMPLETE
============================================================

You've finished what's known as the "GPT-2 reproduction" milestone in
ML practitioner training. Most people who claim to "understand LLMs"
have not actually built and verified one against published weights.
You have.

Lesson 10 next: Karpathy's 3.5-hour talk on the modern LLM stack.
That's a watch-only lesson that ties everything together and shows
where the field has gone since GPT-2.
""")
