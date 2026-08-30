# GPU and Performance

## Device Check

The code checks whether CUDA is available:

```python
torch.cuda.is_available()
```

If a GPU exists, a tensor can be created directly on the GPU or moved there using `.to("cuda")`.

## CPU vs GPU Experiment

The original notebook includes a large matrix multiplication benchmark and compares CPU and GPU execution time.

A `torch.cuda.synchronize()` call is used before reading GPU timing so asynchronous CUDA work has completed.

## Important note

The benchmark runs only when CUDA is available. The matrix size is intentionally large (`10000 x 10000`) to make the performance difference measurable, so it requires substantial memory.

## AI/ML connection

GPU acceleration is essential for training and inference of modern deep learning models because neural networks perform large amounts of parallel tensor computation.
