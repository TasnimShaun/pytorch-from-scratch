# Tensor Shapes and Data Types

## Tensor Shapes

Shape describes the size of a tensor along each dimension.

Example:

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x.shape)
```

The tensor has shape `(2, 3)`.

Also practiced:

- `empty_like`
- `zeros_like`
- `ones_like`
- `rand_like`

## Data Types

Practiced checking and converting tensor dtypes using:

- `tensor.dtype`
- `dtype=...`
- `.to(...)`

Examples include `torch.float32`, `torch.float64`, `torch.int32`, and `torch.float64`.

## AI/ML connection

Tensor shape and dtype are critical because neural network layers expect inputs with compatible dimensions, while dtype affects memory usage, precision, and computation.
