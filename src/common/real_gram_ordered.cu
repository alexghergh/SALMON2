// CUDA versions of the two DGEMMs of the largest block pair in gram_schmidt_col_rblas:
// overlap TN 1027x1027x21296 and update NN 21296x1027x1027. They repeat the scalar
// multiply/add order of the SSE3 DGEMM kernels that the OpenBLAS-based BLAS bundled with
// NVHPC selects on the Xeon test host, including the partial micro-tiles at the edges.
// Another BLAS or kernel set may round differently. Written from that operation order,
// not from the BLAS source.
#include "real_gram_ordered.h"

namespace {
constexpr int points = 21296, states = 1027, chunk = 32;

// CPU K panels: overlap 165*128+88+88; update 7*128+68+63.
// These are arithmetic boundaries: alpha and C addition occur after EVERY panel.
template<bool overlap> __device__ __forceinline__ int panel_size(int start) {
    if (overlap) return start < 21120 ? 128 : 88;
    return start < 896 ? 128 : (start == 896 ? 68 : 63);
}

// CPU row panels are 448+292+287 (TN), 46*448+344+344 (NN).
// Every panel starts at a multiple of four, so only the TN final 3 rows have tails.
__device__ __forceinline__ int micro_extent(int index) {
    return index < 1024 ? 4 : (index < 1026 ? 2 : 1);
}

struct Sum {
    double s0, s1, s2, s3;
    int chains;
    bool corner;

    __device__ Sum(int rows, int cols)
        : s0(0.0), s1(0.0), s2(0.0), s3(0.0),
          chains(cols == 4 || (rows == 4 && cols == 2) ? 1 :
                 ((rows == 2 && cols == 2) || (rows == 4 && cols == 1) ? 2 : 4)),
          corner(rows == 1 && cols == 1) {}

    __device__ __forceinline__ void add(double a, double b, int k, int size) {
        // Main CPU loop handles floor(K/8)*8 terms. All remainder terms use s0;
        // they are NOT distributed to the main-loop chains by k%chains.
        const int lane = k < (size & ~7) ? k % chains : 0;
        const double p = __dmul_rn(a, b);
        if (lane == 0) s0 = __dadd_rn(s0, p);
        else if (lane == 1) s1 = __dadd_rn(s1, p);
        else if (lane == 2) s2 = __dadd_rn(s2, p);
        else s3 = __dadd_rn(s3, p);
    }
    __device__ __forceinline__ double finish() const {
        if (chains == 1) return s0;
        if (chains == 2) return __dadd_rn(s0, s1);
        // The 1x1 path packs adjacent K values into SSE lanes and ends in haddpd.
        if (corner) return __dadd_rn(__dadd_rn(s0, s2), __dadd_rn(s1, s3));
        return __dadd_rn(__dadd_rn(s0, s1), __dadd_rn(s2, s3));
    }
};

// Shared tiles move inputs only. One thread owns an output through every K panel;
// no thread reduction, atomic accumulation, FMA or changed scalar expression.
template<bool overlap, int rows, int cols>
__global__ void ordered(const double *a, const double *b, double *c, double alpha) {
    __shared__ double sa[overlap ? rows : chunk][overlap ? chunk + 1 : rows + 1];
    __shared__ double sb[cols][chunk + 1];
    const int row0 = int(blockIdx.x) * rows, col0 = int(blockIdx.y) * cols;
    const int row = row0 + int(threadIdx.x), col = col0 + int(threadIdx.y);
    const int tid = int(threadIdx.x) + rows * int(threadIdx.y);
    constexpr int m = overlap ? states : points, depth = overlap ? points : states;
    const bool active = row < m && col < states;
    const int offset = row + m * col;
    double value = overlap ? 0.0 : (active ? c[offset] : 0.0);
    for (int start = 0; start < depth;) {
        const int size = panel_size<overlap>(start);
        Sum sum(overlap ? micro_extent(row) : 4, micro_extent(col));
        for (int base = 0; base < size; base += chunk) {
            const int count = size - base < chunk ? size - base : chunk;
            for (int t = tid; t < rows * chunk; t += rows * cols) {
                if (overlap) {
                    const int r = t / chunk, k = t % chunk;
                    sa[r][k] = row0 + r < m && k < count
                        ? a[start + base + k + points * (row0 + r)] : 0.0;
                } else {
                    const int k = t / rows, r = t % rows;
                    sa[k][r] = row0 + r < m && k < count
                        ? a[row0 + r + points * (start + base + k)] : 0.0;
                }
            }
            for (int t = tid; t < cols * chunk; t += rows * cols) {
                const int j = t / chunk, k = t % chunk;
                sb[j][k] = col0 + j < states && k < count
                    ? b[start + base + k + depth * (col0 + j)] : 0.0;
            }
            __syncthreads();
            #pragma unroll 1
            for (int k = 0; k < count; ++k) {
                const double av = overlap ? sa[threadIdx.x][k] : sa[k][threadIdx.x];
                sum.add(av, sb[threadIdx.y][k], base + k, size);
            }
            __syncthreads();
        }
        value = __dadd_rn(__dmul_rn(sum.finish(), alpha), value);
        start += size;
    }
    if (active) c[offset] = value;
}
} // namespace

extern "C" int salmon_real_gram_overlap_ordered(const double *a, const double *b,
                                               double *s, double hvol) {
    if (!a || !b || !s || a == s || b == s || !(hvol > 0.0))
        return int(cudaErrorInvalidValue);
    ordered<true,16,16><<<dim3((states+15)/16, (states+15)/16), dim3(16,16)>>>(a,b,s,hvol);
    return int(cudaGetLastError());
}

extern "C" int salmon_real_gram_update_ordered(const double *a, const double *u,
                                              double *b) {
    if (!a || !u || !b || a == b || u == b) return int(cudaErrorInvalidValue);
    ordered<false,32,8><<<dim3((points+31)/32, (states+7)/8), dim3(32,8)>>>(a,u,b,-1.0);
    return int(cudaGetLastError());
}
