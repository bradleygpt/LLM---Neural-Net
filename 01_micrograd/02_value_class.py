"""
02_value_class.py
-----------------
To do backprop, we need to RECORD what happens during a forward pass --
which values were combined with which operations to produce which results.
That's the computational graph.

We'll wrap every number in a `Value` object. When two Values get added or
multiplied, the result is a new Value that REMEMBERS its parents and the
op that created it.

This is the single most important idea in the whole lesson. Everything
else is plumbing on top of it.
"""


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0  # we'll fill this in later via backprop
        self._prev = set(_children)  # who created me?
        self._op = _op  # what operation created me?
        self.label = label  # for pretty-printing only

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), "*")


# ---------------------------------------------------------------
# Build a small expression
# ---------------------------------------------------------------
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")

e = a * b
e.label = "e"

d = e + c
d.label = "d"

f = Value(-2.0, label="f")
L = d * f
L.label = "L"

print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c}")
print(f"e = a*b = {e}")
print(f"d = e+c = {d}")
print(f"f = {f}")
print(f"L = d*f = {L}")
print()


# ---------------------------------------------------------------
# Walk the graph
# ---------------------------------------------------------------
# Each Value remembers its parents. We can recursively visit them.

def show_graph(v, indent=0):
    pad = "  " * indent
    op = f" via '{v._op}'" if v._op else ""
    print(f"{pad}{v.label or '?'} = {v.data}{op}")
    for child in v._prev:
        show_graph(child, indent + 1)


print("The graph behind L:")
show_graph(L)
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - Every Value knows three things: its number, its parents, its op.
# - Building an expression now also builds a tree we can walk later.
# - We have a `grad` slot but it's just sitting at 0. Filling it in --
#   correctly, automatically -- is what backprop does.
#
# Next: do it MANUALLY for this same little graph, by hand, applying the
# chain rule. Then in script 04 we automate it.
