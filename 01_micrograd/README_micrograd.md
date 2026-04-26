# Building Micrograd from Scratch

A coding-along build-out of Andrej Karpathy's lesson:
*"The spelled-out intro to neural networks and backpropagation: building micrograd"*

## How to use this

Work through the scripts **in order**. Each one is runnable on its own and
adds exactly one new concept on top of the previous. The goal is for you to
run each, read the output, then try to predict what the next one needs to add
before opening it.

```
python 01_derivatives.py
python 02_value_class.py
python 03_manual_backprop.py
python 04_auto_backprop.py
python 05_more_ops.py
python 06_neuron_layer_mlp.py
python 07_training_loop.py
```

Only Python 3.8+ and `matplotlib` (optional, for one plot) are needed. No
PyTorch, no NumPy in the engine itself — that's the whole point.

## The arc

| Script | Concept | What you should be able to explain after |
|---|---|---|
| 01 | Numerical derivatives | What a gradient *is*, intuitively |
| 02 | The `Value` class | Why we need to track operations as a graph |
| 03 | Manual backprop | The chain rule, applied by hand |
| 04 | Automatic backprop | Topological sort + `_backward` closures |
| 05 | More operations | How any new op slots into the framework |
| 06 | Neuron / Layer / MLP | A neural net is just a math expression |
| 07 | Training loop | Forward → loss → backward → update, repeat |

## Things to try once you finish

- Add a `relu` op to script 05 and use it in the MLP
- Try a different loss (MAE instead of MSE)
- Plot the loss curve over training
- Break a `_backward` function on purpose and watch what happens
- Compare your gradients to PyTorch's `autograd` on the same expression

## A note on the comparison to PyTorch

This engine is `O(slow)` and operates on scalars, not tensors. PyTorch does
the same conceptual thing but on n-dimensional arrays with vectorized ops on
the GPU. The *ideas* are identical. Once you've built micrograd, PyTorch
stops being magic.
