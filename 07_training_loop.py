"""
07_training_loop.py
-------------------
The whole point. We have an MLP. We have backprop. Time to train.

The loop is dead simple and you've seen it implied for the last six scripts:

    repeat:
        forward pass:  predict outputs from inputs using the network
        compute loss: how wrong are we? (one number)
        zero gradients: clear out last step's gradients
        backward pass: fill in .grad for every parameter
        update:       nudge each parameter AGAINST its gradient

That's it. That's deep learning. The complexity in real systems is in the
forward pass (huge architectures), the loss (clever objectives), and the
update rule (Adam, momentum, schedulers). The loop itself never changes.

The "zero gradients" step is critical. Our `_backward` functions ACCUMULATE
into `.grad` (always +=, remember script 04). If we don't reset between
training steps, we'd be summing gradients across iterations. Classic bug.
"""

import math
import random


# =====================================================================
# Engine + nn classes (same as script 06)
# =====================================================================
class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self): return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    def _as_value(self, other): return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        other = self._as_value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other): return self + other

    def __mul__(self, other):
        other = self._as_value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other if isinstance(other, Value) else -other)
    def __rsub__(self, other): return (-self) + other

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f"**{other}")
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward(): self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v in visited: return
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)
        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class Module:
    def zero_grad(self):
        for p in self.parameters(): p.grad = 0.0
    def parameters(self): return []


class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    def __call__(self, x):
        return sum((wi * xi for wi, xi in zip(self.w, x)), self.b).tanh()
    def parameters(self): return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self): return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers: x = layer(x)
        return x
    def parameters(self): return [p for layer in self.layers for p in layer.parameters()]


# =====================================================================
# Tiny dataset: 4 inputs, 4 target labels
# =====================================================================
# This is the same toy problem from the lesson. Each input is a 3-vector,
# each target is +1 or -1, and we want the network to learn the mapping.

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

random.seed(1337)
model = MLP(3, [4, 4, 1])
print(f"MLP with {len(model.parameters())} parameters")
print()


# =====================================================================
# Training loop
# =====================================================================
LEARNING_RATE = 0.05
EPOCHS = 100

for epoch in range(EPOCHS):
    # --- forward pass: predict for all inputs ---
    ypred = [model(x) for x in xs]

    # --- loss: mean squared error ---
    # sum over examples of (prediction - target)^2, then divide by N
    losses = [(yp - yt) ** 2 for yp, yt in zip(ypred, ys)]
    loss = sum(losses) * (1.0 / len(losses))

    # --- zero gradients (CRITICAL -- without this, grads accumulate) ---
    model.zero_grad()

    # --- backward pass ---
    loss.backward()

    # --- gradient descent update ---
    # Move each parameter a small step in the direction that REDUCES loss.
    # Loss decreases in the direction of -gradient, so subtract.
    for p in model.parameters():
        p.data -= LEARNING_RATE * p.grad

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch:>3}: loss = {loss.data:.6f}")

print()


# =====================================================================
# Inspect predictions
# =====================================================================
print("Final predictions vs targets:")
final_preds = [model(x) for x in xs]
for x, yt, yp in zip(xs, ys, final_preds):
    print(f"  input={x}  target={yt:>5.2f}  pred={yp.data:>7.4f}")
print()


# =====================================================================
# What just happened
# =====================================================================
# A 41-parameter network learned to map four 3D inputs to their targets,
# using nothing but the Value class we built across scripts 02-05. No
# PyTorch, no NumPy in the engine, no autograd library. Just chain rule
# and a for-loop.
#
# Suggested experiments now that the lesson is built:
#
#   1. Crank EPOCHS up. Watch loss approach 0.
#   2. Drop LEARNING_RATE to 0.001. Convergence crawls. Now try 0.5 and watch
#      it explode. The learning rate is the most important hyperparameter
#      and there's no formula for it -- you tune it.
#   3. Forget to call model.zero_grad(). Predict what happens, then run.
#   4. Add a `relu` op to Value (forward: max(0, x); backward: 1 if x>0 else 0)
#      and use it instead of tanh in Neuron.
#   5. Compare these gradients to PyTorch's autograd on the same expression.
#      They will match to ~6 decimal places.
#
# That's micrograd. PyTorch is the same idea, on tensors, on a GPU, written
# in C++ for speed. Now you know what's underneath.
