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
```
2. Rate of Change

Suppose:

y = x²

When x changes, y also changes.

Calculus allows us to measure how quickly y changes with respect to x.

This rate of change is represented by the derivative:

dy/dx

3. Derivative

For:

y = x²

the derivative is:

dy/dx = 2x

If:

x = 3

then:

dy/dx = 6

The derivative tells us how sensitive the output is to a small change in the input.

4. Why Derivatives Matter in Machine Learning

Machine learning models contain parameters such as:

weights
biases

During training, we want to minimize a loss function.

For example:

L = Loss(w)

We need to know:

dL/dw

This tells us how the loss changes when the weight changes.

The gradient is then used by optimization algorithms such as Gradient Descent.

5. Gradient Descent Connection

A basic parameter update is:

w_new = w - η * dL/dw

where:

w = current parameter
η = learning rate
dL/dw = gradient

The derivative tells us the direction in which the parameter should move.

6. Manual Derivative Implementation

For:

y = x²

we know:

dy/dx = 2x

We can implement the derivative manually:
### Python
```python
def derivative(x):
    return 2 * x


x = 3

gradient = derivative(x)

print("Gradient:", gradient)
