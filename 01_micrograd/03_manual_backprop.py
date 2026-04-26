"""
03_manual_backprop.py
---------------------
Same graph as script 02. This time we fill in `.grad` ourselves, by hand,
walking BACKWARD from the output and applying the chain rule.

The chain rule, in plain English:
    "How does L respond to a change in some upstream variable x?"
    = "How does L respond to its direct parent?"  *  "How does that parent respond to x?"

We want dL/dx for every node x in the graph.

By definition, dL/dL = 1.  Start there, propagate backward.

For an ADD node `c = a + b`:
    dc/da = 1, dc/db = 1
    so the gradient just FLOWS through unchanged to both parents.

For a MUL node `c = a * b`:
    dc/da = b, dc/db = a
    so each parent receives the OTHER parent's value times the incoming gradient.
    (This is the "swap" rule -- it's the most useful single fact in backprop.)
"""


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), "*")


# Build the same graph as before
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e = a * b; e.label = "e"
d = e + c; d.label = "d"
f = Value(-2.0, label="f")
L = d * f; L.label = "L"


# ---------------------------------------------------------------
# Now do backprop BY HAND. Top-down.
# ---------------------------------------------------------------

# Start: dL/dL = 1
L.grad = 1.0

# L = d * f  ->  MUL rule: dL/dd = f,  dL/df = d
d.grad = f.data * L.grad  # = -2.0 * 1.0 = -2.0
f.grad = d.data * L.grad  # =  4.0 * 1.0 =  4.0

# d = e + c  ->  ADD rule: gradient flows through unchanged
e.grad = 1.0 * d.grad  # = -2.0
c.grad = 1.0 * d.grad  # = -2.0

# e = a * b  ->  MUL rule: swap
a.grad = b.data * e.grad  # = -3.0 * -2.0 =  6.0
b.grad = a.data * e.grad  # =  2.0 * -2.0 = -4.0

print("After manual backprop:")
for v in [a, b, c, d, e, f, L]:
    print(f"  {v.label:>2}: data={v.data:>6.2f}  grad={v.grad:>6.2f}")
print()


# ---------------------------------------------------------------
# Sanity check the gradients numerically
# ---------------------------------------------------------------
# We claimed dL/da = 6.0. Let's verify by nudging a and seeing L change.

def L_of(a_val, b_val=-3.0, c_val=10.0, f_val=-2.0):
    return ((a_val * b_val) + c_val) * f_val

h = 0.0001
print("Numerical check (should match the .grad values above):")
print(f"  dL/da ~ {(L_of(2.0 + h) - L_of(2.0)) / h:.4f}  (manual said {a.grad})")
print(f"  dL/db ~ {(L_of(2.0, b_val=-3.0 + h) - L_of(2.0)) / h:.4f}  (manual said {b.grad})")
print(f"  dL/dc ~ {(L_of(2.0, c_val=10.0 + h) - L_of(2.0)) / h:.4f}  (manual said {c.grad})")
print(f"  dL/df ~ {(L_of(2.0, f_val=-2.0 + h) - L_of(2.0)) / h:.4f}  (manual said {f.grad})")
print()


# ---------------------------------------------------------------
# What just happened?
# ---------------------------------------------------------------
# We computed gradients for EVERY variable in the graph -- a, b, c, d, e, f --
# in a single backward pass. Compare to the numerical method, where we'd
# need a separate forward pass per variable.
#
# The pattern is also mechanical:
#   - For each node, we know its op and its children.
#   - The op tells us how to push the gradient back to each child.
#   - Repeat until we've visited every node.
#
# "Mechanical" is a hint. We can automate this.
#
# Two pieces remain:
#   1. Each op type needs to know HOW to push gradients back.
#      (We'll attach a `_backward` function to each Value.)
#   2. We need to visit nodes in the right ORDER so that when we process
#      a node, all of its dependents have already been processed.
#      (Topological sort.)
#
# Both arrive in script 04.
