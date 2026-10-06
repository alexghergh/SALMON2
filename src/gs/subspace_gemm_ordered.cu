// CUDA versions of the two ZGEMMs of ssdg_zwf_cblas for 80 orbitals on a 4096-point grid:
// overlap CN 80x80x4096 and rotation NN 4096x80x80. They repeat the operation order of the
// ZGEMM of the OpenBLAS-based BLAS bundled with NVHPC, as dispatched to its SSE3 kernels on the
// Xeon test host. Written from that operation order, not from the BLAS source.
#include "subspace_gemm_ordered.h"

namespace {
constexpr int points = 4096, states = 80;

// The CPU kernel forms all four alpha products even when alpha is real. Separately rounded
// operations keep that expression tree and the update of C after each panel.
__device__ __forceinline__ double2 update(double r, double i, double ar, double ai,
                                          double2 old) {
    const double real = __dsub_rn(__dmul_rn(r, ar), __dmul_rn(i, ai));
    const double imag = __dadd_rn(__dmul_rn(i, ar), __dmul_rn(r, ai));
    return make_double2(__dadd_rn(real, old.x), __dadd_rn(imag, old.y));
}

// One thread owns one output. Shared tiles move inputs only: no parallel reduction.
// The CN call runs over 32 ascending K panels of 128 points; C is rounded after each panel.
__global__ void cn80(const double2 *a, const double2 *b, double2 *c,
                     double ar, double ai) {
    constexpr int tile = 8, chunk = 32, panel = 128;
    __shared__ double2 sa[tile][chunk], sb[tile][chunk];
    const int row = int(blockIdx.x) * tile + int(threadIdx.x);
    const int col = int(blockIdx.y) * tile + int(threadIdx.y);
    const int tid = int(threadIdx.x) + tile * int(threadIdx.y);
    double2 value = make_double2(0.0, 0.0); // original beta=0 stores positive zero
    for (int start = 0; start < points; start += panel) {
        double negative_rr = 0.0, negative_ir = 0.0, ri = 0.0, ii = 0.0;
        for (int base = start; base < start + panel; base += chunk) {
            for (int t = tid; t < tile * chunk; t += tile * tile) {
                const int v = t / chunk, q = t % chunk;
                sa[v][q] = a[base + q + points * (int(blockIdx.x) * tile + v)];
                sb[v][q] = b[base + q + points * (int(blockIdx.y) * tile + v)];
            }
            __syncthreads();
            #pragma unroll 1
            for (int q = 0; q < chunk; ++q) {
                const double2 av = sa[threadIdx.x][q], bv = sb[threadIdx.y][q];
                // As in the CPU kernel, these two channels are subtracted from +0; negating a sum would
                // give different signed zeros.
                negative_rr = __dsub_rn(negative_rr, __dmul_rn(av.x, bv.x));
                negative_ir = __dsub_rn(negative_ir, __dmul_rn(av.y, bv.x));
                ri = __dadd_rn(ri, __dmul_rn(av.x, bv.y));
                ii = __dadd_rn(ii, __dmul_rn(av.y, bv.y));
            }
            __syncthreads();
        }
        value = update(__dsub_rn(ii, negative_rr), __dadd_rn(ri, negative_ir),
                       ar, ai, value);
    }
    c[row + states * col] = value;
}

// NN has one K=80 panel. The CPU's 224/144-row and 6/2-column packing partitions
// independent outputs, so they need not be duplicated as GPU launch partitions.
__global__ void nn80(const double2 *a, const double2 *e, double2 *c,
                     double ar, double ai) {
    constexpr int rows = 16, cols = 8, chunk = 16;
    __shared__ double2 sa[chunk][rows], se[cols][chunk];
    const int row = int(blockIdx.x) * rows + int(threadIdx.x);
    const int col = int(blockIdx.y) * cols + int(threadIdx.y);
    const int tid = int(threadIdx.x) + rows * int(threadIdx.y);
    double rr = 0.0, ir = 0.0, ri = 0.0, ii = 0.0;
    for (int base = 0; base < states; base += chunk) {
        for (int t = tid; t < rows * chunk; t += rows * cols) {
            const int q = t / rows, r = t % rows;
            sa[q][r] = a[int(blockIdx.x) * rows + r + points * (base + q)];
        }
        for (int t = tid; t < cols * chunk; t += rows * cols) {
            const int j = t / chunk, q = t % chunk;
            se[j][q] = e[base + q + states * (int(blockIdx.y) * cols + j)];
        }
        __syncthreads();
        #pragma unroll 1
        for (int q = 0; q < chunk; ++q) {
            const double2 av = sa[q][threadIdx.x], ev = se[threadIdx.y][q];
            rr = __dadd_rn(rr, __dmul_rn(av.x, ev.x));
            ir = __dadd_rn(ir, __dmul_rn(av.y, ev.x));
            ri = __dadd_rn(ri, __dmul_rn(av.x, ev.y));
            ii = __dadd_rn(ii, __dmul_rn(av.y, ev.y));
        }
        __syncthreads();
    }
    const int offset = row + points * col;
    c[offset] = update(__dsub_rn(rr, ii), __dadd_rn(ir, ri), ar, ai, c[offset]);
}
} // namespace

extern "C" int salmon_subspace_cn80_ordered(const double2 *a, const double2 *b,
                                           double2 *c, double ar, double ai) {
    if (!a || !b || !c || a == c || b == c) return int(cudaErrorInvalidValue);
    if (ar == 0.0 && ai == 0.0)
        return int(cudaMemsetAsync(c, 0, states * states * sizeof(double2)));
    cn80<<<dim3(states / 8, states / 8), dim3(8, 8)>>>(a, b, c, ar, ai);
    return int(cudaGetLastError());
}

extern "C" int salmon_subspace_nn80_ordered(const double2 *a, const double2 *e,
                                           double2 *c, double ar, double ai) {
    if (!a || !e || !c || a == c || e == c) return int(cudaErrorInvalidValue);
    if (ar == 0.0 && ai == 0.0) return int(cudaSuccess); // beta=1; original early return
    nn80<<<dim3(points / 16, states / 8), dim3(16, 8)>>>(a, e, c, ar, ai);
    return int(cudaGetLastError());
}
