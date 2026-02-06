# mizchi/blas

BLAS (Basic Linear Algebra Subprograms) bindings for MoonBit.

- **macOS**: Apple Accelerate.framework
- **Linux**: OpenBLAS

## Installation

```bash
moon add mizchi/blas
```

## Usage

```moonbit
// Matrix multiplication: C = A @ B
let a = [1.0, 2.0, 3.0, 4.0]  // 2x2 matrix
let b = [5.0, 6.0, 7.0, 8.0]  // 2x2 matrix
let c = [0.0, 0.0, 0.0, 0.0]  // result
@blas.sgemm(a, b, c, 2, 2, 2)
// c = [19.0, 22.0, 43.0, 50.0]
```

### High-performance MLP forward pass

```moonbit
// Create C-side buffers (zero-copy after initialization)
let bufs = @blas.mlp_buffers_create(batch_size, input_dim, hidden_dim, output_dim)
@blas.mlp_buffers_init(bufs, input, weight1, bias1, weight2, bias2)

// Forward pass (all computation in C memory)
@blas.mlp_forward_fused(bufs)

// Get output
let output = Array::make(batch_size * output_dim, 0.0)
@blas.mlp_buffers_get_output(bufs, output)

// Cleanup
@blas.mlp_buffers_free(bufs)
```

### Batch training (forward + backward + update)

```moonbit
let bufs = @blas.mlp_train_buffers_create(batch_size, input_dim, hidden_dim, output_dim)
@blas.mlp_train_buffers_init_weights(bufs, weight1, bias1, weight2, bias2)

// Single call does forward, backward, and parameter update
let (loss_sum, correct_count) = @blas.mlp_train_step(bufs, batch_input, batch_labels, learning_rate)

@blas.mlp_train_buffers_free(bufs)
```

## Quick Commands

```bash
just           # check + test
just bench     # run benchmark
just docker-build  # build Linux container
just docker-test   # test on Linux
just docker-bench  # benchmark on Linux
```

## Platform Support

### macOS

Uses Apple Accelerate.framework (no additional setup required).

### Linux

Requires OpenBLAS:

```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# Or use Docker
just docker-build
just docker-test
```

## Benchmark

MNIST 2-layer MLP (784→128→10), batch size 128:

| Backend | Time/epoch | vs Pure MoonBit |
|---------|-----------|-----------------|
| Pure MoonBit CPU | 130s | 1x |
| BLAS batch | 0.7s | **186x** |

## API

### Low-level BLAS operations

- `sgemm(a, b, c, m, n, k)` - Matrix multiply: C = A @ B
- `sgemv(a, x, y, m, n)` - Matrix-vector multiply: y = A @ x
- `saxpy(alpha, x, y)` - Vector add: y = alpha * x + y
- `sdot(x, y)` - Dot product
- `snrm2(x)` - L2 norm

### High-level MLP operations

- `mlp_buffers_create/init/free` - C-side buffer management
- `mlp_forward_fused` - Fused forward pass (layer1+layer2)
- `mlp_train_buffers_create/init_weights/get_weights/free` - Training buffer management
- `mlp_train_step` - Complete training step (forward + backward + update)

## License

MIT
