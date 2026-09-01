 Calculus Basics for Autograd

## Overview

Calculus is one of the mathematical foundations behind automatic differentiation, backpropagation, and gradient-based optimization.

Before learning PyTorch Autograd, it is useful to understand a few basic calculus concepts.

The most important concepts are:

- Function
- Variable
- Derivative
- Partial Derivative
- Chain Rule
- Gradient

---

## 1. Function

A function maps an input to an output.

For example:

y = x²

If:

x = 3

then:

y = 3² = 9

### Python

```python
x = 3
y = x ** 2

print(y)
