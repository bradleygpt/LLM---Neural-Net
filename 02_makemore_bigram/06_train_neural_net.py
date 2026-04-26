"""
06_train_neural_net.py
----------------------
Train the bigram neural net with gradient descent. We have W of shape
(V, V), and the training procedure is:

    forward:  logits = x_onehot @ W
              probs  = softmax(logits)
              loss   = -mean(log probs[arange(N), ys])

    backward: compute dW = dLoss/dW
              update W -= lr * dW
              repeat

THE GRADIENT.
We're not using PyTorch's autograd here, so we have to derive dLoss/dW
ourselves. This is one of the most beautiful results in deep learning,
and it's a single line of code in the end.

The combined gradient of "softmax then negative log-likelihood" with
respect to the LOGITS turns out to be:

    dLoss/dlogits = (probs - one_hot(ys)) / N

That's it. The probability of each class minus 1 if it was the true
class, divided by batch size. Walk through the derivation once on paper
and you'll never forget it -- it falls out of the chain rule because the
exp in softmax cancels the log in NLL.

Then dLoss/dW comes from logits = x_onehot @ W:
    dLoss/dW = x_onehot.T @ dLoss/dlogits

And we're done. Here it is:
"""

import numpy as np
from data_utils import load_names, build_vocab


names = load_names()
stoi, itos = build_vocab(names)
V = len(stoi)

# Load (xs, ys) from previous script -- or rebuild them
try:
    xs = np.load("xs.npy")
    ys = np.load("ys.npy")
except FileNotFoundError:
    xs, ys = [], []
    for w in names:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    xs, ys = np.array(xs), np.array(ys)

N = len(xs)
print(f"Training on {N} bigram pairs, vocab size {V}")
print()


# One-hot encode (we could skip this and just index, but doing it
# explicitly here matches the video's pedagogical flow)
x_onehot = np.eye(V)[xs]


# Initialize weights
rng = np.random.default_rng(2147483647)
W = rng.standard_normal((V, V))


# ---------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------
LR = 50.0   # huge learning rate -- but our gradients are small (1/N), so this is fine
EPOCHS = 200

for epoch in range(EPOCHS):
    # --- forward pass ---
    logits = x_onehot @ W                    # (N, V)
    counts = np.exp(logits)
    probs = counts / counts.sum(axis=1, keepdims=True)

    # negative log-likelihood of the correct next character
    correct_log_probs = np.log(probs[np.arange(N), ys])
    loss = -correct_log_probs.mean()

    # add a tiny regularization term: encourage W to stay small
    # (equivalent to "label smoothing" in spirit; matches Karpathy's video)
    reg = 0.01 * (W ** 2).mean()
    loss = loss + reg

    # --- backward pass: dLoss/dW ---
    # softmax + NLL gradient with respect to logits
    dlogits = probs.copy()
    dlogits[np.arange(N), ys] -= 1.0
    dlogits /= N

    # propagate back through the matmul logits = x_onehot @ W
    dW = x_onehot.T @ dlogits

    # plus the gradient of the regularization term
    dW += 2 * 0.01 * W / W.size

    # --- update ---
    W -= LR * dW

    if epoch % 20 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch:>4}: loss = {loss:.4f}")

print()


# ---------------------------------------------------------------
# Sample from the trained NN
# ---------------------------------------------------------------
def sample_from_nn(W, stoi, itos, rng, max_len=50):
    out = []
    ix = stoi["."]
    while True:
        x_oh = np.eye(W.shape[0])[ix]
        logits = x_oh @ W
        counts = np.exp(logits)
        probs = counts / counts.sum()
        ix = rng.choice(len(probs), p=probs)
        if ix == stoi["."]:
            break
        out.append(itos[ix])
        if len(out) >= max_len:
            break
    return "".join(out)


sample_rng = np.random.default_rng(2147483647)
print("10 names sampled from the TRAINED neural net:")
for _ in range(10):
    print(f"  {sample_from_nn(W, stoi, itos, sample_rng)}")
print()


# ---------------------------------------------------------------
# Compare to the counting model
# ---------------------------------------------------------------
# Do the same dataset's smoothed counts give a similar loss? They should.
N_count = np.zeros((V, V), dtype=np.float64)
for w in names:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        N_count[stoi[ch1], stoi[ch2]] += 1
P_count = (N_count + 1) / (N_count + 1).sum(axis=1, keepdims=True)

count_loss = 0.0
for w in names:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        count_loss += -np.log(P_count[stoi[ch1], stoi[ch2]])
count_loss /= sum(len(w) + 1 for w in names)

print(f"Loss from counting (with smoothing=1):  {count_loss:.4f}")
print(f"Loss from trained NN (final):           {loss:.4f}")
print()
print("These should be in the same neighborhood. They won't match exactly:")
print("  - the counting model uses additive smoothing of 1")
print("  - the NN uses L2 regularization on W with strength 0.01")
print("Different priors -> different equilibria. Script 07 explores this.")


# Save the trained W
np.save("W_trained.npy", W)
