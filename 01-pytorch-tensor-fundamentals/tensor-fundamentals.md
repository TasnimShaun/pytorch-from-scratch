import time
import numpy as np
import torch


# ============================================================
# 1. PyTorch setup
# ============================================================

print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available. Using CPU.")


# ============================================================
# 2. Creating a Tensor
# ============================================================

# Using empty
a = torch.empty(2, 3)
print("\nempty:\n", a)

# Using zeros
print("\nzeros:\n", torch.zeros(2, 3))

# Using ones
print("\nones:\n", torch.ones(2, 3))

# Using rand
print("\nrand:\n", torch.rand(2, 3))

# Reproducibility with manual_seed
torch.manual_seed(100)
print("\nrandom tensor with seed:\n", torch.rand(2, 3))

torch.manual_seed(100)
print("\nsame seed -> same tensor:\n", torch.rand(2, 3))

# Using tensor()
print("\nusing tensor:\n", torch.tensor([[1, 2, 3], [4, 5, 6]]))

# arange
print("\nusing arange ->", torch.arange(0, 10, 2))

# linspace
print("\nusing linspace ->", torch.linspace(0, 10, 10))

# eye
print("\nusing eye ->\n", torch.eye(5))

# full
print("\nusing full ->\n", torch.full((3, 3), 5))


# ============================================================
# 3. Tensor Shapes
# ============================================================

x = torch.tensor([[1, 2, 3], [4, 5, 6]])

print("\nx:\n", x)
print("x.shape:", x.shape)

print("\nempty_like:\n", torch.empty_like(x))
print("\nzeros_like:\n", torch.zeros_like(x))
print("\nones_like:\n", torch.ones_like(x))
print("\nrand_like:\n", torch.rand_like(x, dtype=torch.float32))


# ============================================================
# 4. Tensor Data Types
# ============================================================

print("\nx.dtype:", x.dtype)

print(
    "\nfloat values with int32 dtype:",
    torch.tensor([1.0, 2.0, 3.0], dtype=torch.int32),
)

print(
    "integer values with float64 dtype:",
    torch.tensor([1, 2, 3], dtype=torch.float64),
)

print("x converted to float32:", x.to(torch.float32))


# ============================================================
# 5. Mathematical Operations
# ============================================================

# 5.1 Scalar operations
x = torch.rand(2, 2)

print("\nScalar operations input:\n", x)
print("x + 2:\n", x + 2)
print("x - 2:\n", x - 2)
print("x * 3:\n", x * 3)
print("x / 3:\n", x / 3)
print("(x * 100) // 3:\n", (x * 100) // 3)
print("((x * 100) // 3) % 2:\n", ((x * 100) // 3) % 2)
print("x ** 2:\n", x ** 2)

# 5.2 Element-wise operations
a = torch.rand(2, 3)
b = torch.rand(2, 3)

print("\na:\n", a)
print("b:\n", b)
print("a + b:\n", a + b)
print("a - b:\n", a - b)
print("a * b:\n", a * b)
print("a / b:\n", a / b)
print("a ** b:\n", a ** b)
print("a % b:\n", a % b)

c = torch.tensor([1, -2, 3, -4])
print("\nabs(c):", torch.abs(c))
print("neg(c):", torch.neg(c))

d = torch.tensor([1.9, 2.3, 3.7, 4.4])
print("\nround(d):", torch.round(d))
print("ceil(d):", torch.ceil(d))
print("floor(d):", torch.floor(d))
print("clamp(d, 2, 3):", torch.clamp(d, min=2, max=3))

# 5.3 Reduction operations
e = torch.randint(size=(2, 3), low=0, high=10, dtype=torch.float32)

print("\ne:\n", e)
print("sum:", torch.sum(e))
print("sum dim=0:", torch.sum(e, dim=0))
print("sum dim=1:", torch.sum(e, dim=1))
print("mean:", torch.mean(e))
print("mean dim=0:", torch.mean(e, dim=0))
print("median:", torch.median(e))
print("max:", torch.max(e))
print("min:", torch.min(e))
print("prod:", torch.prod(e))
print("std:", torch.std(e))
print("var:", torch.var(e))
print("argmax:", torch.argmax(e))
print("argmin:", torch.argmin(e))

# 5.4 Matrix operations
f = torch.randint(size=(2, 3), low=0, high=10)
g = torch.randint(size=(3, 2), low=0, high=10)

print("\nf:\n", f)
print("g:\n", g)
print("matrix multiplication:\n", torch.matmul(f, g))

vector1 = torch.tensor([1, 2])
vector2 = torch.tensor([3, 4])
print("dot product:", torch.dot(vector1, vector2))

print("transpose of f:\n", torch.transpose(f, 0, 1))

h = torch.randint(size=(3, 3), low=0, high=10, dtype=torch.float32)
print("\nh:\n", h)
print("determinant:", torch.det(h))

# Inverse only exists for an invertible square matrix.
if torch.det(h) != 0:
    print("inverse:\n", torch.inverse(h))
else:
    print("inverse: h is singular, so no inverse exists.")

# 5.5 Comparison operations
i = torch.randint(size=(2, 3), low=0, high=10)
j = torch.randint(size=(2, 3), low=0, high=10)

print("\ni:\n", i)
print("j:\n", j)
print("i > j:\n", i > j)
print("i < j:\n", i < j)
print("i == j:\n", i == j)
print("i != j:\n", i != j)
print("i >= j:\n", i >= j)
print("i <= j:\n", i <= j)

# 5.6 Special functions
k = torch.randint(size=(2, 3), low=0, high=10, dtype=torch.float32)

print("\nk:\n", k)
print("log(k):\n", torch.log(k))
print("exp(k):\n", torch.exp(k))
print("sqrt(k):\n", torch.sqrt(k))
print("sigmoid(k):\n", torch.sigmoid(k))
print("softmax(k, dim=0):\n", torch.softmax(k, dim=0))
print("relu(k):\n", torch.relu(k))


# ============================================================
# 6. In-place Operations
# ============================================================

m = torch.rand(2, 3)
n = torch.rand(2, 3)

print("\nm before add_:\n", m)
print("n:\n", n)

m.add_(n)
print("m after m.add_(n):\n", m)

print("relu(m) without in-place:\n", torch.relu(m))

m.relu_()
print("m after m.relu_():\n", m)


# ============================================================
# 7. Copying a Tensor
# ============================================================

a = torch.rand(2, 3)
b = a

print("\na:", a)
print("b = a:", b)

a[0][0] = 0
print("\na after modification:", a)
print("b after a modification:", b)
print("id(a):", id(a))
print("id(b):", id(b))

b = a.clone()

a[0][0] = 10

print("\na after clone and modification:", a)
print("b after clone and modification:", b)
print("id(a):", id(a))
print("id(b):", id(b))


# ============================================================
# 8. Tensor Operations on GPU
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")

    gpu_tensor = torch.rand((2, 3), device=device)
    print("\nnew tensor on GPU:\n", gpu_tensor)

    a = torch.rand(3, 3)
    b = a.to(device)

    print("\nCPU tensor a:\n", a)
    print("GPU tensor b:\n", b)
    print("b + 5:\n", b + 5)

    # CPU vs GPU matrix multiplication
    size = 10000

    matrix_cpu1 = torch.randn(size, size)
    matrix_cpu2 = torch.randn(size, size)

    start_time = time.time()
    result_cpu = torch.matmul(matrix_cpu1, matrix_cpu2)
    cpu_time = time.time() - start_time

    print(f"\nTime on CPU: {cpu_time:.4f} seconds")

    matrix_gpu1 = matrix_cpu1.to("cuda")
    matrix_gpu2 = matrix_cpu2.to("cuda")

    start_time = time.time()
    result_gpu = torch.matmul(matrix_gpu1, matrix_gpu2)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time

    print(f"Time on GPU: {gpu_time:.4f} seconds")
    print("Speedup (CPU time / GPU time):", cpu_time / gpu_time)
else:
    print("\nSkipping GPU section because CUDA is not available.")


# ============================================================
# 9. Reshaping Tensors
# ============================================================

a = torch.ones(4, 4)

print("\na:\n", a)
print("reshape to (2, 2, 2, 2):\n", a.reshape(2, 2, 2, 2))
print("flatten:", a.flatten())

b = torch.rand(2, 3, 4)
print("\nb shape:", b.shape)
print("b.permute(2, 1, 0) shape:", b.permute(2, 1, 0).shape)

# Image example: H x W x C
c = torch.rand(226, 226, 3)
print("\nc shape:", c.shape)
print("unsqueeze(0) shape:", c.unsqueeze(0).shape)
print("unsqueeze(1) shape:", c.unsqueeze(1).shape)
print("unsqueeze(2) shape:", c.unsqueeze(2).shape)

d = torch.rand(1, 20)
print("\nd:", d)
print("d.squeeze(0):", d.squeeze(0))
print("d.squeeze(0).shape:", d.squeeze(0).shape)


# ============================================================
# 10. NumPy and PyTorch
# ============================================================

# Tensor -> NumPy
a = torch.tensor([1, 2, 33, 4])
b = a.numpy()

print("\nTensor:", a)
print("NumPy array:", b)
print("NumPy type:", type(b))

# NumPy -> Tensor
c = np.array([1, 2, 3, 4, 5, 6])
d = torch.from_numpy(c)

print("\nNumPy array:", c)
print("Tensor:", d)
print("Tensor type:", type(d))
