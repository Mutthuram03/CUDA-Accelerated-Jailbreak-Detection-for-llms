/*
 * Custom Fused Linear Classification and Sigmoid Kernel
 * 
 * Computes:
 *   logit_i = sum_j (X_ij * W_j) + bias
 *   prob_i  = 1.0 / (1.0 + exp(-logit_i))
 * 
 * Optimized for high-throughput batch inference in LLM jailbreak gatekeepers.
 */

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void fused_dense_linear_sigmoid_kernel(
    const float* __restrict__ X,         // Shape: [batch_size, n_features]
    const float* __restrict__ W,         // Shape: [n_features]
    const float bias,                    // Scalar bias
    float* __restrict__ probs,           // Output shape: [batch_size]
    const int batch_size,
    const int n_features
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch_size) {
        float sum = 0.0f;
        const float* row_ptr = X + (row * n_features);

        #pragma unroll 4
        for (int j = 0; j < n_features; ++j) {
            sum += row_ptr[j] * W[j];
        }
        sum += bias;

        // Sigmoid activation
        probs[row] = 1.0f / (1.0f + __expf(-sum));
    }
}
