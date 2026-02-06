///|
struct BenchArgs {
  input_size : Int
  hidden_size : Int
  output_size : Int
  batch_size : Int
  warmup : Int
  iters : Int
} derive(Show, Eq)

///|
fn bench_args_default() -> BenchArgs {
  {
    input_size: 784,
    hidden_size: 128,
    output_size: 10,
    batch_size: 128,
    warmup: 20,
    iters: 200,
  }
}

///|
fn print_usage(program : String) -> Unit {
  println(
    "Usage: " +
    program +
    " [--input N] [--hidden N] [--output N] [--batch N] [--warmup N] [--iters N]",
  )
  println("  --input N    input size (default 784)")
  println("  --hidden N   hidden size (default 128)")
  println("  --output N   output size (default 10)")
  println("  --batch N    batch size (default 128)")
  println("  --warmup N   warmup iterations (default 20)")
  println("  --iters N    measure iterations (default 200)")
}

///|
fn parse_positive_int(name : String, value : String) -> Result[Int, String] {
  let parsed = try? @strconv.parse_int(value)
  match parsed {
    Ok(v) => if v > 0 { Ok(v) } else { Err(name + " must be > 0") }
    Err(err) => Err("invalid " + name + ": " + err.to_string())
  }
}

///|
fn parse_args(args : Array[String]) -> Result[BenchArgs, String] {
  let default = bench_args_default()
  let mut input_size = default.input_size
  let mut hidden_size = default.hidden_size
  let mut output_size = default.output_size
  let mut batch_size = default.batch_size
  let mut warmup = default.warmup
  let mut iters = default.iters
  let mut i = 1
  while i < args.length() {
    let arg = args[i]
    if arg == "--help" || arg == "-h" {
      return Err("help")
    }
    if arg == "--input" && i + 1 < args.length() {
      match parse_positive_int("input", args[i + 1]) {
        Ok(v) => input_size = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg == "--hidden" && i + 1 < args.length() {
      match parse_positive_int("hidden", args[i + 1]) {
        Ok(v) => hidden_size = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg == "--output" && i + 1 < args.length() {
      match parse_positive_int("output", args[i + 1]) {
        Ok(v) => output_size = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg == "--batch" && i + 1 < args.length() {
      match parse_positive_int("batch", args[i + 1]) {
        Ok(v) => batch_size = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg == "--warmup" && i + 1 < args.length() {
      match parse_positive_int("warmup", args[i + 1]) {
        Ok(v) => warmup = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg == "--iters" && i + 1 < args.length() {
      match parse_positive_int("iters", args[i + 1]) {
        Ok(v) => iters = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    return Err("unknown argument: " + arg)
  }
  Ok({
    input_size,
    hidden_size,
    output_size,
    batch_size,
    warmup,
    iters,
  })
}

///|
fn seeded_values(seed : Int, count : Int, offset : Int) -> Array[Float] {
  let arr = Array::make(count, Float::from_int(0))
  let a = 1664525
  let c = 1013904223
  let mut state = seed
  for i = 0; i < count; i = i + 1 {
    state = state * a + c
    let v = (state.land(0x7FFFFFFF) % 1000).to_float() / 1000.0 - 0.5
    arr[i] = v + Float::from_int(offset) * 0.001
  }
  arr
}

///|
fn run_mlp_fused_bench(cfg : BenchArgs) -> Unit {
  // Setup data
  let zero = Float::from_int(0)
  let batch_input = seeded_values(0, cfg.batch_size * cfg.input_size, 0)
  let weight1_data = seeded_values(1, cfg.input_size * cfg.hidden_size, 100)
  let bias1_data = seeded_values(2, cfg.hidden_size, 200)
  let weight2_data = seeded_values(3, cfg.hidden_size * cfg.output_size, 300)
  let bias2_data = seeded_values(4, cfg.output_size, 400)

  // Create C-side buffers (single copy, then operate in-place)
  let bufs = @blas.mlp_buffers_create(
    cfg.batch_size,
    cfg.input_size,
    cfg.hidden_size,
    cfg.output_size,
  )
  @blas.mlp_buffers_init(
    bufs,
    batch_input,
    weight1_data,
    bias1_data,
    weight2_data,
    bias2_data,
  )

  // Warmup
  let mut i = 0
  while i < cfg.warmup {
    @blas.mlp_forward_fused(bufs)
    i = i + 1
  }

  // Benchmark
  let start = @env.now()
  i = 0
  while i < cfg.iters {
    @blas.mlp_forward_fused(bufs)
    i = i + 1
  }
  let end = @env.now()
  let elapsed_ms = end - start

  // Get output for verification
  let output = Array::make(cfg.batch_size * cfg.output_size, zero)
  @blas.mlp_buffers_get_output(bufs, output)

  // Clean up
  @blas.mlp_buffers_free(bufs)

  // Output (use Int64 to avoid overflow)
  let ops_per_iter : Int64 = cfg.batch_size.to_int64() * (
    cfg.input_size.to_int64() * cfg.hidden_size.to_int64() * 2L + // layer1 matmul
    cfg.hidden_size.to_int64() + // layer1 bias
    cfg.hidden_size.to_int64() * cfg.output_size.to_int64() * 2L + // layer2 matmul
    cfg.output_size.to_int64() // layer2 bias
  )
  let total_ops = ops_per_iter * cfg.iters.to_int64()
  let elapsed_ms_f = elapsed_ms.to_double()
  let gflops = if elapsed_ms_f > 0.0 {
    total_ops.to_double() / elapsed_ms_f / 1_000_000.0
  } else {
    0.0
  }

  println(
    "{\"type\":\"mlp_forward_fused\"" +
    ",\"input\":" + cfg.input_size.to_string() +
    ",\"hidden\":" + cfg.hidden_size.to_string() +
    ",\"output\":" + cfg.output_size.to_string() +
    ",\"batch\":" + cfg.batch_size.to_string() +
    ",\"iters\":" + cfg.iters.to_string() +
    ",\"elapsed_ms\":" + elapsed_ms.to_string() +
    ",\"gflops\":" + gflops.to_string() +
    ",\"sample_output\":" + output[0].to_string() +
    "}"
  )
}

///|
fn run_sgemm_bench(cfg : BenchArgs) -> Unit {
  // Simple SGEMM benchmark: C = A @ B
  // A: batch x input, B: input x hidden, C: batch x hidden
  let m = cfg.batch_size
  let k = cfg.input_size
  let n = cfg.hidden_size

  let a = seeded_values(0, m * k, 0)
  let b = seeded_values(1, k * n, 100)
  let c = Array::make(m * n, Float::from_int(0))

  // Warmup
  let mut i = 0
  while i < cfg.warmup {
    @blas.sgemm(a, b, c, m, n, k)
    i = i + 1
  }

  // Benchmark
  let start = @env.now()
  i = 0
  while i < cfg.iters {
    @blas.sgemm(a, b, c, m, n, k)
    i = i + 1
  }
  let end = @env.now()
  let elapsed_ms = end - start

  // GFLOPS calculation: 2*m*n*k ops per matmul (use Int64 to avoid overflow)
  let ops_per_iter : Int64 = 2L * m.to_int64() * n.to_int64() * k.to_int64()
  let total_ops = ops_per_iter * cfg.iters.to_int64()
  let elapsed_ms_f = elapsed_ms.to_double()
  let gflops = if elapsed_ms_f > 0.0 {
    total_ops.to_double() / elapsed_ms_f / 1_000_000.0
  } else {
    0.0
  }

  println(
    "{\"type\":\"sgemm\"" +
    ",\"m\":" + m.to_string() +
    ",\"n\":" + n.to_string() +
    ",\"k\":" + k.to_string() +
    ",\"iters\":" + cfg.iters.to_string() +
    ",\"elapsed_ms\":" + elapsed_ms.to_string() +
    ",\"gflops\":" + gflops.to_string() +
    ",\"sample_output\":" + c[0].to_string() +
    "}"
  )
}

///|
fn main {
  let args = @env.args()
  let program = if args.length() > 0 { args[0] } else { "bench" }

  match parse_args(args) {
    Err("help") => {
      print_usage(program)
      return
    }
    Err(msg) => {
      println("Error: " + msg)
      print_usage(program)
      return
    }
    Ok(cfg) => {
      println("# BLAS Benchmark")
      println("# config: " + cfg.to_string())
      run_sgemm_bench(cfg)
      run_mlp_fused_bench(cfg)
    }
  }
}
