"""
06_neuron_layer_mlp.py
----------------------
The autograd engine doesn't care that we're building a neural net. It just
tracks numbers and gradients. We're now going to build the neural net
ABSTRACTIONS on top -- Neuron, Layer, MLP -- in pure Python, using our
Value class as the underlying number type.

This mirrors PyTorch's structure exactly:
    Neuron  ~  a single nn.Linear "row" + activation
    Layer   ~  nn.Linear with multiple outputs
    MLP     ~  nn.Sequential of Layers

The pattern: every module exposes a `parameters()` method that returns ALL
the learnable Values inside it. The training loop will use that list to
zero gradients and apply updates.
"""

import math
import random


# =====================================================================
# The autograd engine (same as script 05, copied for self-containment)
# =====================================================================
class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def _as_value(self, other):
        return other if isinstance(other, Value) else Value(other)

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

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t * t) * out.grad
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


# =====================================================================
# Neural net abstractions
# =====================================================================

class Module:
    """Base class. Knows how to zero its gradients."""
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    """A single neuron: takes nin inputs, outputs one number."""
    def __init__(self, nin):
        # Initialize weights and bias to small random values in [-1, 1]
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # Weighted sum + bias, then tanh
        # sum() with a starting Value of self.b avoids needing a separate bias add
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    """A layer of `nout` neurons, each receiving `nin` inputs."""
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        # If only one output, return scalar. Otherwise return list.
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """Multi-layer perceptron. `nin` inputs, then a list of layer sizes."""
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# =====================================================================
# Try it
# =====================================================================
random.seed(1337)

# A 3-input MLP with two hidden layers of 4 neurons, and 1 output
n = MLP(3, [4, 4, 1])

x = [2.0, 3.0, -1.0]
out = n(x)
print(f"MLP(3, [4, 4, 1]) on input {x}")
print(f"  output: {out.data:.4f}")
print(f"  total parameters: {len(n.parameters())}")
print()

# We can backprop right through it
out.backward()
print("First few parameter gradients after one backward pass:")
for p in n.parameters()[:5]:
    print(f"  {p}")
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# We built a neural net and got gradients for every weight and bias --
# 41 parameters, all updated in a single backward call. The MLP class
# is ~5 lines because all the heavy lifting is done by Value.
#
# We now have everything we need to TRAIN. That's script 07.
