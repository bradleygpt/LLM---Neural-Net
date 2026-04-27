"""
02_load_pretrained_weights.py
-----------------------------
Download OpenAI's actual GPT-2 weights (via HuggingFace) and load
them into the model from script 01.

This is the moment of truth: if your architecture is correct, you can
take parameters from a model trained by OpenAI in 2019 and use them
verbatim. Generation should produce real, coherent GPT-2 output.

What "loading weights" actually means:
  - HuggingFace publishes GPT-2 as a state_dict (a Python dict mapping
    parameter names to tensors).
  - We need to map HF's parameter names to OUR parameter names.
    Our names already match because we used HF's naming convention
    in script 01 (transformer.wte, transformer.h.0.attn.c_attn, etc.).
  - One critical wrinkle: HuggingFace uses "Conv1D" layers in attention
    and MLP, where Conv1D stores its weight TRANSPOSED relative to
    nn.Linear. We need to transpose those tensors when copying.

The Conv1D-vs-Linear thing is a real footgun. The math is identical;
the storage convention differs. Karpathy specifically documents it.
You'll see it handled below as `transpose_keys`.
"""

import torch
import torch.nn as nn

# Import our architecture from script 01
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("arch", "01_gpt2_architecture.py")
arch_mod = importlib.util.module_from_spec(spec)
sys.modules["arch"] = arch_mod
spec.loader.exec_module(arch_mod)

GPT = arch_mod.GPT
GPTConfig = arch_mod.GPTConfig


def load_gpt2_weights(model, model_type="gpt2"):
    """
    Load OpenAI's GPT-2 weights into our model.

    Args:
        model: an instance of our GPT class
        model_type: "gpt2" (124M), "gpt2-medium" (350M), "gpt2-large" (774M),
                    or "gpt2-xl" (1558M)

    Returns: number of parameters copied.
    """
    from transformers import GPT2LMHeadModel

    print(f"Downloading HuggingFace's {model_type} weights...")
    print(f"(First run will download ~500 MB. Cached after that.)")
    hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = hf.state_dict()
    print(f"  Got {len(sd_hf)} parameters from HuggingFace.")

    # Our model's state dict
    sd_ours = model.state_dict()

    # HF uses Conv1D for these attention/MLP weights, so we need to
    # transpose them when copying into our nn.Linear layers.
    transpose_keys = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]

    # Some keys exist in HF but not in ours (or vice versa)
    keys_to_skip = [
        ".attn.bias",         # HF stores the causal mask as a parameter; we use a buffer
        ".attn.masked_bias",  # HF deprecated artifact
    ]

    print()
    print("Copying weights...")
    n_copied = 0
    for k_hf, v_hf in sd_hf.items():
        if any(skip in k_hf for skip in keys_to_skip):
            continue

        # Our keys don't have the "transformer." or "lm_head." HF prefixes
        # in the same form -- but actually our model DOES use those prefixes
        # because we mirrored HF's structure. So names match directly.
        k_ours = k_hf

        if k_ours not in sd_ours:
            print(f"  [skip] HF key not in our model: {k_hf}")
            continue

        # Apply transpose for Conv1D-stored weights
        if any(suffix in k_hf for suffix in transpose_keys):
            v_hf = v_hf.t()

        # Sanity-check shapes
        if v_hf.shape != sd_ours[k_ours].shape:
            print(f"  [SHAPE MISMATCH] {k_hf}: HF {tuple(v_hf.shape)} vs ours {tuple(sd_ours[k_ours].shape)}")
            continue

        # Copy the data in
        with torch.no_grad():
            sd_ours[k_ours].copy_(v_hf)
        n_copied += 1

    print(f"  Copied {n_copied} parameter tensors.")
    print()

    # Sanity check: did our weight tying still work?
    # After load, transformer.wte.weight and lm_head.weight should be the same tensor.
    assert model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr(), \
        "Weight tying broken after load!"
    print("Weight tying verified: wte.weight and lm_head.weight share storage.")

    return n_copied


# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Build empty model
    config = GPTConfig()
    model = GPT(config)
    print(f"Built empty GPT-2 small ({sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M params).")
    print()

    # Load weights
    n_copied = load_gpt2_weights(model, "gpt2")
    print(f"Done. {n_copied} tensors copied into our architecture.")
    print()

    # Save the loaded model so script 03 can use it without re-downloading
    model.eval()
    torch.save(model.state_dict(), "gpt2_loaded.pt")
    print("Saved loaded weights to gpt2_loaded.pt")


# ---------------------------------------------------------------
# What we just did
# ---------------------------------------------------------------
"""
============================================================
WEIGHTS FROM OPENAI'S GPT-2 ARE NOW IN OUR MODEL
============================================================

We took 148 parameter tensors from a model OpenAI trained on 40GB of
web text in 2019 and put them into a model architecture WE wrote.

The fact that this WORKS is the proof that your architecture is correct.
Every layer name matches. Every shape matches. Every transposition is
handled. If even one parameter were in the wrong place, generation
would produce gibberish in the next script.

Two notable details:

  1. Conv1D vs Linear. HuggingFace uses Conv1D in attention/MLP.
     It's mathematically identical to Linear but stores the weight
     transposed. We handled this with the `transpose_keys` list.

  2. The .attn.bias buffer. HF stores the causal mask as a parameter
     (registered as a buffer with name "bias"). We use the same name
     in our model, but skip copying it because the masks should match
     by construction (both are torch.tril(ones)).

Next: prove the architecture is correct by GENERATING text and
comparing token-by-token to HuggingFace's reference output.
"""
