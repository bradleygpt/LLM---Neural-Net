"""
01_derivatives.py
-----------------
Before we write a single line of neural network code, we need to be solid
on what a derivative actually IS.

A derivative answers one question: "if I nudge this input a tiny bit,
how much does the output change, and in what direction?"

That's it. That's the whole game. Backpropagation is just this question
asked over and over, all the way back through a computation.
"""


# ---------------------------------------------------------------
# Part 1: A simple function and its derivative, by feel
# ---------------------------------------------------------------

def f(x):
    return 3 * x**2 - 4 * x + 5


print("Part 1: a simple function f(x) = 3x^2 - 4x + 5")
print(f"f(3.0) = {f(3.0)}")
print(f"f(-3.0) = {f(-3.0)}")
print()

# What is the derivative at x = 3? Calculus says f'(x) = 6x - 4, so f'(3) = 14.
# But pretend we don't know that. We can ESTIMATE it numerically:
# nudge x by a tiny h, see how much f changes, divide.

h = 0.000001
x = 3.0
slope = (f(x + h) - f(x)) / h
print(f"Numerical derivative at x=3.0:  {slope:.6f}  (should be ~14)")

x = -3.0
slope = (f(x + h) - f(x)) / h
print(f"Numerical derivative at x=-3.0: {slope:.6f}  (should be ~-22)")

x = 2 / 3  # the minimum: derivative should be ~0
slope = (f(x + h) - f(x)) / h
print(f"Numerical derivative at x=2/3:  {slope:.6f}  (should be ~0)")
print()


# ---------------------------------------------------------------
# Part 2: A function of three inputs
# ---------------------------------------------------------------
# This is where it gets interesting. With multiple inputs, each one has
# its own derivative -- the "partial derivative". Each tells us how
# sensitive the output is to nudging that one input alone.

a = 2.0
b = -3.0
c = 10.0

d = a * b + c
print(f"Part 2: d = a*b + c  with a={a}, b={b}, c={c}")
print(f"d = {d}")

h = 0.0001

# Nudge a, see how d changes
d1 = a * b + c
d2 = (a + h) * b + c
print(f"  dd/da ~ {(d2 - d1) / h:.6f}   (analytically: b = {b})")

# Nudge b
d2 = a * (b + h) + c
print(f"  dd/db ~ {(d2 - d1) / h:.6f}   (analytically: a = {a})")

# Nudge c
d2 = a * b + (c + h)
print(f"  dd/dc ~ {(d2 - d1) / h:.6f}   (analytically: 1)")
print()


# ---------------------------------------------------------------
# What we just learned
# ---------------------------------------------------------------
# - A derivative is a local sensitivity: "if I wiggle this input,
#   how does the output respond?"
# - We can ALWAYS estimate it numerically with the (f(x+h) - f(x)) / h trick.
# - With multiple inputs, each input has its own partial derivative.
#
# Numerical estimation works but it's slow and noisy. For a real network with
# millions of parameters, we'd have to do millions of forward passes per step.
#
# Backpropagation is a clever way to compute ALL the partial derivatives
# in basically one backward pass. To get there, we first need a way to
# RECORD the computation as it happens, so we can walk back through it.
#
# That's the Value class. On to script 02.
