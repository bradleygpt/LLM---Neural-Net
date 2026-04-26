"""
05_more_ops.py
--------------
Adding new operations. The recipe is always the same:

    1. Compute the forward value.
    2. Wrap it in a Value with the right children and op label.
    3. Define a `_backward` closure using the LOCAL derivative of this op,
       multiplied by `out.grad` (the chain rule).

We'll add: negation, subtraction, division, power, exp, tanh.
We'll also let Values play nicely with plain Python numbers (Value + 2 etc).

The tanh activation is the one we'll use in the neural net. Its derivative
is the cleanest non-trivial example:  d/dx tanh(x) = 1 - tanh(x)^2.
"""

import math


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # ---- helper: coerce ints/floats into Values ----
    def _as_value(self, other):
        return other if isinstance(other, Value) else Value(other)

    # ---- addition ----
    def __add__(self, other):
        other = self._as_value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):  # 2 + value
        return self + other

    # ---- multiplication ----
    def __mul__(self, other):
        other = self._as_value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):  # 2 * value
        return self * other

    # ---- negation and subtraction (built from the above) ----
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else -other)

    def __rsub__(self, other):  # 2 - value
        return (-self) + other

    # ---- power (only for fixed numeric exponent) ----
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only number powers supported"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            # d/dx (x^n) = n * x^(n-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    # ---- division: a/b  ==  a * b**-1 ----
    def __truediv__(self, other):
        return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)

    def __rtruediv__(self, other):  # 2 / value
        return other * self ** -1

    # ---- exp ----
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            # d/dx e^x = e^x = out.data
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    # ---- tanh ----
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d/dx tanh(x) = 1 - tanh(x)^2
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    # ---- backward pass ----
    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v in visited:
                return
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)

        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# ---------------------------------------------------------------
# Build a tiny neuron by hand to test the new ops
# ---------------------------------------------------------------
# A single neuron with two inputs:  o = tanh(w1*x1 + w2*x2 + b)

x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
b = Value(6.8813735870195432, label="b")  # chosen so tanh output is a clean number

x1w1 = x1 * w1; x1w1.label = "x1*w1"
x2w2 = x2 * w2; x2w2.label = "x2*w2"
n = x1w1 + x2w2 + b; n.label = "n"
o = n.tanh(); o.label = "o"

o.backward()

print("Single neuron, forward + backward:")
print(f"  o = {o.data:.6f}")
print(f"  do/dw1 = {w1.grad:.6f}")
print(f"  do/dw2 = {w2.grad:.6f}")
print(f"  do/db  = {b.grad:.6f}")
print(f"  do/dx1 = {x1.grad:.6f}")
print(f"  do/dx2 = {x2.grad:.6f}")
print()


# ---------------------------------------------------------------
# Spot-check tanh by building it from primitives
# ---------------------------------------------------------------
# tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
# If our exp, division, +, - all work, we should get the same answer.

def tanh_from_primitives(x):
    e = (2 * x).exp()
    return (e - 1) / (e + 1)


x_test = Value(0.7, label="x")
out_direct = x_test.tanh()
out_built = tanh_from_primitives(Value(0.7))
print("tanh built two ways (should match):")
print(f"  via .tanh()        : {out_direct.data:.10f}")
print(f"  via exp/+/-/division: {out_built.data:.10f}")
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# Adding ops is mechanical. Forward formula + local derivative + closure.
# We can now build any expression we'd find in a small neural net.
#
# Next: stop building neurons by hand. Wrap them in classes -- Neuron,
# Layer, MLP -- so we can spin up real architectures.
