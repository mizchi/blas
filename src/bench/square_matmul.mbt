///|
fn square_matmul_iterations(size : Int) -> Int {
  match size {
    128 => 100000
    256 => 32000
    512 => 8000
    1024 => 1024
    _ => 1024
  }
}

///|
fn square_legacy_iterations(size : Int) -> Int {
  match size {
    128 => 2000
    256 => 800
    512 => 200
    1024 => 50
    _ => 50
  }
}

///|
fn measure_single_sgemm_us(
  a : @blas.FloatBuffer,
  b : @blas.FloatBuffer,
  c : @blas.FloatBuffer,
  size : Int,
  iterations : Int,
) -> Double {
  let start = @env.now()
  for _ in 0..<iterations {
    @blas.sgemm_direct(a, b, c, size, size, size)
  }
  (@env.now() - start).to_double() * 1000.0 / iterations.to_int64().to_double()
}

///|
fn measure_batch_sgemm_us(
  a : @blas.FloatBuffer,
  b : @blas.FloatBuffer,
  c : @blas.FloatBuffer,
  size : Int,
  batch_calls : Int,
  batch_width : Int,
) -> Double {
  let start = @env.now()
  for _ in 0..<batch_calls {
    @blas.sgemm_direct_batch(a, b, c, size, size, size, iterations=batch_width)
  }
  let operations = batch_calls * batch_width
  (@env.now() - start).to_double() * 1000.0 / operations.to_int64().to_double()
}

///|
fn median_three(a : Double, b : Double, c : Double) -> Double {
  if a < b {
    if b < c {
      b
    } else if a < c {
      c
    } else {
      a
    }
  } else if a < c {
    a
  } else if b < c {
    c
  } else {
    b
  }
}

///|
fn run_square_matmul_case(size : Int) -> Unit {
  let iterations = square_matmul_iterations(size)
  let legacy_iterations = square_legacy_iterations(size)
  let batch_width = 16
  let batch_calls = iterations / batch_width
  let data = seeded_values(size, size * size, 0)
  let a = @blas.alloc_floats(size * size)
  let b = @blas.alloc_floats(size * size)
  let c = @blas.alloc_floats(size * size)
  @blas.copy_to_buffer(a, data)
  @blas.copy_to_buffer(b, data)

  for _ in 0..<20 {
    @blas.sgemm_direct(a, b, c, size, size, size)
  }
  for _ in 0..<5 {
    @blas.sgemm_direct_batch(a, b, c, size, size, size, iterations=batch_width)
  }
  let single_samples : FixedArray[Double] = FixedArray::make(3, 0.0)
  let batch_samples : FixedArray[Double] = FixedArray::make(3, 0.0)
  for sample in 0..<3 {
    if sample % 2 == 0 {
      single_samples[sample] = measure_single_sgemm_us(
        a, b, c, size, iterations,
      )
      batch_samples[sample] = measure_batch_sgemm_us(
        a, b, c, size, batch_calls, batch_width,
      )
    } else {
      batch_samples[sample] = measure_batch_sgemm_us(
        a, b, c, size, batch_calls, batch_width,
      )
      single_samples[sample] = measure_single_sgemm_us(
        a, b, c, size, iterations,
      )
    }
  }

  let legacy_output = Array::make(size * size, Float::from_int(0))
  for _ in 0..<5 {
    @blas.sgemm(data, data, legacy_output, size, size, size)
  }
  let legacy_start = @env.now()
  for _ in 0..<legacy_iterations {
    @blas.sgemm(data, data, legacy_output, size, size, size)
  }
  let legacy_elapsed_ms = @env.now() - legacy_start

  let single_us = median_three(
    single_samples[0],
    single_samples[1],
    single_samples[2],
  )
  let batch_us = median_three(
    batch_samples[0],
    batch_samples[1],
    batch_samples[2],
  )
  let legacy_us = legacy_elapsed_ms.to_double() *
    1000.0 /
    legacy_iterations.to_int64().to_double()
  let sample = @blas.get_float(c, 0)
  @blas.free_floats(a)
  @blas.free_floats(b)
  @blas.free_floats(c)

  println(
    "{\"type\":\"square_sgemm\"" +
    ",\"size\":" +
    size.to_string() +
    ",\"iterations\":" +
    iterations.to_string() +
    ",\"legacy_array_us\":" +
    legacy_us.to_string() +
    ",\"single_us\":" +
    single_us.to_string() +
    ",\"batch16_us_per_op\":" +
    batch_us.to_string() +
    ",\"sample_output\":" +
    sample.to_string() +
    "}",
  )
}

///|
fn run_square_matmul_bench() -> Unit {
  println("# Square SGEMM benchmark (persistent C buffers)")
  for size in [128, 256, 512, 1024] {
    run_square_matmul_case(size)
  }
}
