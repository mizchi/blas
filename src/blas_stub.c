// BLAS bindings for MoonBit
// Supports: Apple Accelerate (macOS), OpenBLAS (Linux/others)

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
  #define ACCELERATE_NEW_LAPACK
  #include <Accelerate/Accelerate.h>
#else
  #include <cblas.h>
#endif

// MoonBit FixedArray[Byte] is passed as pointer to data
typedef uint8_t *moonbit_bytes_t;

// ============================================================================
// Float buffer operations
// ============================================================================

float* blas_alloc_floats(int count) {
  return (float*)calloc(count, sizeof(float));
}

void blas_free_floats(float* buf) {
  if (buf) free(buf);
}

void blas_copy_from_bytes(float* dst, moonbit_bytes_t src, int count) {
  memcpy(dst, src, count * sizeof(float));
}

void blas_copy_to_bytes(moonbit_bytes_t dst, float* src, int count) {
  memcpy(dst, src, count * sizeof(float));
}

float blas_get_float(float* buf, int idx) {
  return buf[idx];
}

void blas_set_float(float* buf, int idx, float val) {
  buf[idx] = val;
}

void blas_zero(float* buf, int n) {
  memset(buf, 0, n * sizeof(float));
}

// ============================================================================
// Int buffer operations (for labels)
// ============================================================================

int* blas_alloc_ints(int count) {
  return (int*)calloc(count, sizeof(int));
}

void blas_free_ints(int* buf) {
  if (buf) free(buf);
}

void blas_set_int(int* buf, int idx, int val) {
  buf[idx] = val;
}

int blas_get_int(int* buf, int idx) {
  return buf[idx];
}

// ============================================================================
// Direct BLAS operations on C buffers
// ============================================================================

void blas_sgemm_direct(
  int trans_a, int trans_b,
  int m, int n, int k,
  float alpha,
  float* a, int lda,
  float* b, int ldb,
  float beta,
  float* c, int ldc
) {
  cblas_sgemm(
    CblasRowMajor,
    trans_a ? CblasTrans : CblasNoTrans,
    trans_b ? CblasTrans : CblasNoTrans,
    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc
  );
}

void blas_sgemv_direct(
  int trans, int m, int n,
  float alpha,
  float* a, int lda,
  float* x, int incx,
  float beta,
  float* y, int incy
) {
  cblas_sgemv(
    CblasRowMajor,
    trans ? CblasTrans : CblasNoTrans,
    m, n, alpha, a, lda, x, incx, beta, y, incy
  );
}

void blas_saxpy_direct(int n, float alpha, float* x, int incx, float* y, int incy) {
  cblas_saxpy(n, alpha, x, incx, y, incy);
}

// ============================================================================
// Fused layer operations
// ============================================================================

void blas_add_bias(float* out, float* bias, int batch, int dim) {
  for (int b = 0; b < batch; b++) {
    for (int j = 0; j < dim; j++) {
      out[b * dim + j] += bias[j];
    }
  }
}

void blas_relu_inplace(float* x, int n) {
  for (int i = 0; i < n; i++) {
    if (x[i] < 0.0f) x[i] = 0.0f;
  }
}

void blas_layer1_fused(
  float* input, float* weight, float* bias, float* output,
  int batch, int in_dim, int out_dim
) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    batch, out_dim, in_dim,
    1.0f, input, in_dim, weight, out_dim, 0.0f, output, out_dim);
  for (int b = 0; b < batch; b++) {
    for (int j = 0; j < out_dim; j++) {
      float v = output[b * out_dim + j] + bias[j];
      output[b * out_dim + j] = v > 0.0f ? v : 0.0f;
    }
  }
}

void blas_layer2_fused(
  float* input, float* weight, float* bias, float* output,
  int batch, int in_dim, int out_dim
) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    batch, out_dim, in_dim,
    1.0f, input, in_dim, weight, out_dim, 0.0f, output, out_dim);
  for (int b = 0; b < batch; b++) {
    for (int j = 0; j < out_dim; j++) {
      output[b * out_dim + j] += bias[j];
    }
  }
}

// ============================================================================
// Batch training operations
// ============================================================================

void blas_softmax_batch(float* logits, float* probs, int batch, int dim) {
  for (int b = 0; b < batch; b++) {
    float* logit_row = logits + b * dim;
    float* prob_row = probs + b * dim;
    float max_val = logit_row[0];
    for (int k = 1; k < dim; k++) {
      if (logit_row[k] > max_val) max_val = logit_row[k];
    }
    float sum = 0.0f;
    for (int k = 0; k < dim; k++) {
      prob_row[k] = expf(logit_row[k] - max_val);
      sum += prob_row[k];
    }
    for (int k = 0; k < dim; k++) {
      prob_row[k] /= sum;
    }
  }
}

void blas_compute_loss_acc(
  float* probs, int* labels, float* result,
  int batch, int output_dim
) {
  float loss_sum = 0.0f;
  int correct = 0;
  float eps = 1e-7f;
  for (int b = 0; b < batch; b++) {
    int label = labels[b];
    float* prob_row = probs + b * output_dim;
    loss_sum += -logf(prob_row[label] + eps);
    int pred = 0;
    float pred_val = prob_row[0];
    for (int k = 1; k < output_dim; k++) {
      if (prob_row[k] > pred_val) {
        pred = k;
        pred_val = prob_row[k];
      }
    }
    if (pred == label) correct++;
  }
  result[0] = loss_sum;
  result[1] = (float)correct;
}

void blas_train_step(
  float* input, int* labels,
  float* weight1, float* bias1,
  float* weight2, float* bias2,
  float* hidden, float* output, float* probs,
  float* grad_w1, float* grad_b1,
  float* grad_w2, float* grad_b2,
  float* delta2, float* delta1,
  float* result,
  float lr,
  int batch, int input_dim, int hidden_dim, int output_dim
) {
  // Forward: hidden = ReLU(input @ weight1 + bias1)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    batch, hidden_dim, input_dim,
    1.0f, input, input_dim, weight1, hidden_dim, 0.0f, hidden, hidden_dim);
  for (int b = 0; b < batch; b++) {
    for (int j = 0; j < hidden_dim; j++) {
      float v = hidden[b * hidden_dim + j] + bias1[j];
      hidden[b * hidden_dim + j] = v > 0.0f ? v : 0.0f;
    }
  }

  // Forward: output = hidden @ weight2 + bias2
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    batch, output_dim, hidden_dim,
    1.0f, hidden, hidden_dim, weight2, output_dim, 0.0f, output, output_dim);
  for (int b = 0; b < batch; b++) {
    for (int j = 0; j < output_dim; j++) {
      output[b * output_dim + j] += bias2[j];
    }
  }

  // Softmax and loss
  blas_softmax_batch(output, probs, batch, output_dim);
  blas_compute_loss_acc(probs, labels, result, batch, output_dim);

  // Backward: delta2 = probs - one_hot(labels)
  for (int b = 0; b < batch; b++) {
    for (int k = 0; k < output_dim; k++) {
      delta2[b * output_dim + k] = probs[b * output_dim + k];
    }
    delta2[b * output_dim + labels[b]] -= 1.0f;
  }

  // grad_b2 = sum(delta2)
  for (int b = 0; b < batch; b++) {
    for (int k = 0; k < output_dim; k++) {
      grad_b2[k] += delta2[b * output_dim + k];
    }
  }

  // grad_w2 = hidden^T @ delta2
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
    hidden_dim, output_dim, batch,
    1.0f, hidden, hidden_dim, delta2, output_dim, 0.0f, grad_w2, output_dim);

  // delta1 = delta2 @ weight2^T * relu_grad(hidden)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    batch, hidden_dim, output_dim,
    1.0f, delta2, output_dim, weight2, output_dim, 0.0f, delta1, hidden_dim);
  for (int b = 0; b < batch; b++) {
    for (int j = 0; j < hidden_dim; j++) {
      int idx = b * hidden_dim + j;
      if (hidden[idx] <= 0.0f) delta1[idx] = 0.0f;
    }
  }

  // grad_b1 = sum(delta1)
  for (int b = 0; b < batch; b++) {
    for (int j = 0; j < hidden_dim; j++) {
      grad_b1[j] += delta1[b * hidden_dim + j];
    }
  }

  // grad_w1 = input^T @ delta1
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
    input_dim, hidden_dim, batch,
    1.0f, input, input_dim, delta1, hidden_dim, 0.0f, grad_w1, hidden_dim);

  // Update: param -= lr/batch * grad
  float scale = lr / batch;
  cblas_saxpy(input_dim * hidden_dim, -scale, grad_w1, 1, weight1, 1);
  cblas_saxpy(hidden_dim, -scale, grad_b1, 1, bias1, 1);
  cblas_saxpy(hidden_dim * output_dim, -scale, grad_w2, 1, weight2, 1);
  cblas_saxpy(output_dim, -scale, grad_b2, 1, bias2, 1);
}

// ============================================================================
// Legacy API (with byte conversion)
// ============================================================================

void blas_sgemm(
  int trans_a, int trans_b,
  int m, int n, int k,
  float alpha,
  moonbit_bytes_t a_bytes, int lda,
  moonbit_bytes_t b_bytes, int ldb,
  float beta,
  moonbit_bytes_t c_bytes, int ldc
) {
  cblas_sgemm(CblasRowMajor,
    trans_a ? CblasTrans : CblasNoTrans,
    trans_b ? CblasTrans : CblasNoTrans,
    m, n, k, alpha, (float*)a_bytes, lda, (float*)b_bytes, ldb, beta, (float*)c_bytes, ldc);
}

void blas_sgemv(
  int trans, int m, int n,
  float alpha,
  moonbit_bytes_t a_bytes, int lda,
  moonbit_bytes_t x_bytes, int incx,
  float beta,
  moonbit_bytes_t y_bytes, int incy
) {
  cblas_sgemv(CblasRowMajor,
    trans ? CblasTrans : CblasNoTrans,
    m, n, alpha, (float*)a_bytes, lda, (float*)x_bytes, incx, beta, (float*)y_bytes, incy);
}

void blas_saxpy(int n, float alpha, moonbit_bytes_t x_bytes, int incx, moonbit_bytes_t y_bytes, int incy) {
  cblas_saxpy(n, alpha, (float*)x_bytes, incx, (float*)y_bytes, incy);
}

void blas_sscal(int n, float alpha, moonbit_bytes_t x_bytes, int incx) {
  cblas_sscal(n, alpha, (float*)x_bytes, incx);
}

float blas_sdot(int n, moonbit_bytes_t x_bytes, int incx, moonbit_bytes_t y_bytes, int incy) {
  return cblas_sdot(n, (float*)x_bytes, incx, (float*)y_bytes, incy);
}

float blas_snrm2(int n, moonbit_bytes_t x_bytes, int incx) {
  return cblas_snrm2(n, (float*)x_bytes, incx);
}

void blas_scopy(int n, moonbit_bytes_t x_bytes, int incx, moonbit_bytes_t y_bytes, int incy) {
  cblas_scopy(n, (float*)x_bytes, incx, (float*)y_bytes, incy);
}
