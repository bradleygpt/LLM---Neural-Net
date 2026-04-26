"""
04_auto_backprop.py
-------------------
Time to automate backprop. Two new ideas:

1) Each op stashes a `_backward` CLOSURE on the resulting Value. This little
   function knows how to take the gradient flowing INTO this node and push
   it back to the parents -- using the local derivative rule for that op.

2) When we want to do a full backward pass, we visit nodes in REVERSE
   TOPOLOGICAL ORDER. That guarantees: by the time we call a node's
   `_backward`, that node's `.grad` is already finalized (all incoming
   contributions accumulated).

Why "+=" and not "="? Because a Value can be USED MORE THAN ONCE.
If a appears in two places in the expression, gradients from both paths
must SUM. This is the multivariate chain rule. Always +=, never =.
"""


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None  # default: leaf nodes have no backward

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # ADD: gradient flows through unchanged to both parents
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # MUL: each parent's grad is the OTHER parent's value, times incoming grad
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # Build topological order: parents come before children in the list
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

        # Seed the output's gradient to 1
        self.grad = 1.0

        # Walk in REVERSE: children's grads finalized before their parents are processed
        for node in reversed(topo):
            node._backward()


# ---------------------------------------------------------------
# Try it on the same graph
# ---------------------------------------------------------------
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e = a * b; e.label = "e"
d = e + c; d.label = "d"
f = Value(-2.0, label="f")
L = d * f; L.label = "L"

L.backward()

print("After automatic backprop:")
for v in [a, b, c, d, e, f, L]:
    print(f"  {v.label:>2}: data={v.data:>6.2f}  grad={v.grad:>6.2f}")
print()


# ---------------------------------------------------------------
# The "used twice" gotcha
# ---------------------------------------------------------------
# Watch what happens if a Value appears in two places in the expression.
# This is where the += in _backward earns its keep.

x = Value(3.0, label="x")
y = x + x  # x used TWICE
y.label = "y"
y.backward()
print("Test: y = x + x, so dy/dx should be 2")
print(f"  x.grad = {x.grad}  (correct! both paths contributed 1 each)")
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# We now have a working autograd engine. It's tiny (~40 lines), but the
# core mechanics are exactly what PyTorch does on tensors.
#
# Right now it only knows + and *. Real networks need tanh, exp, and so on.
# Adding new ops is the topic of script 05.
