# Reshaping Tensors

Practiced:

- `reshape`
- `flatten`
- `permute`
- `unsqueeze`
- `squeeze`

## Why shape manipulation matters

Deep learning models frequently require tensors to be arranged in specific dimensions.

For example, an image may be represented as:

```text
Height x Width x Channels
```

Adding a batch dimension changes it to:

```text
Batch x Height x Width x Channels
```

## AI/ML connection

Shape manipulation is common when preparing image, text, and tabular data for neural network layers.
