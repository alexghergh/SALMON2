// SPDX-License-Identifier: Apache-2.0
// Device workspace for one ssdg_zwf_cblas call; the ZGEMM kernels themselves own no memory.
#include "subspace_gemm_ordered.h"
#include <cstdio>
#include <initializer_list>
#include <new>

extern "C" int salmon_subspace_overlap_device(const double2 *, const double2 *, double2 *, double);
extern "C" int salmon_subspace_rotation_device(const double2 *, const double2 *, double2 *);

namespace {
constexpr size_t field_bytes = 4096 * 80 * sizeof(double2);
constexpr size_t matrix_bytes = 80 * 80 * sizeof(double2);
struct Workspace {
    double2 *a = nullptr, *b = nullptr, *matrix = nullptr;
    int phase = 0; // 0: next overlap; 1: its rotation; -1: failed, destroy only.
};

cudaError_t release(Workspace *w) {
    cudaError_t first = cudaSuccess;
    for (auto p : {w->a, w->b, w->matrix}) if (p) {
        const auto s = cudaFree(p);
        if (first == cudaSuccess) first = s;
    }
    delete w;
    return first;
}
} // namespace

extern "C" int salmon_subspace_context_create(void **handle) {
    if (!handle || *handle) return int(cudaErrorInvalidValue);
    auto *w = new (std::nothrow) Workspace;
    if (!w) return int(cudaErrorMemoryAllocation);
    cudaError_t status = cudaSuccess;
    int count = 0;
    #define CHECK(call) do { status = (call); if (status != cudaSuccess) goto failed; } while (0)
    CHECK(cudaGetDeviceCount(&count));
    if (count != 1) { status = cudaErrorInvalidDevice; goto failed; }
    CHECK(cudaSetDevice(0));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&w->a), field_bytes));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&w->b), field_bytes));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&w->matrix), matrix_bytes));
    *handle = w;
    return 0;
failed:
    const auto cleanup = release(w);
    if (cleanup != cudaSuccess)
        std::fprintf(stderr, "SUBSPACE_CLEANUP_ERROR %d\n", int(cleanup));
    return int(status);
    #undef CHECK
}

extern "C" int salmon_subspace_context_overlap(void *handle, const double2 *a,
        const double2 *b, double hvol, double2 *hmat) {
    auto *w = static_cast<Workspace *>(handle);
    if (!w || w->phase != 0 || !a || !b || !hmat) return int(cudaErrorInvalidValue);
    w->phase = -1;
    cudaError_t status;
    #define CHECK(call) do { status = (call); if (status != cudaSuccess) return int(status); } while (0)
    CHECK(cudaMemcpy(w->a, a, field_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(w->b, b, field_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaError_t(salmon_subspace_overlap_device(w->a, w->b, w->matrix, hvol)));
    CHECK(cudaMemcpy(hmat, w->matrix, matrix_bytes, cudaMemcpyDeviceToHost));
    // The blocking copy to the host completes the CN kernel before the CPU ZHEEV.
    w->phase = 1;
    return 0;
    #undef CHECK
}

extern "C" int salmon_subspace_context_rotate(void *handle, const double2 *evec, double2 *out) {
    auto *w = static_cast<Workspace *>(handle);
    if (!w || w->phase != 1 || !evec || !out) return int(cudaErrorInvalidValue);
    w->phase = -1;
    cudaError_t status;
    #define CHECK(call) do { status = (call); if (status != cudaSuccess) return int(status); } while (0)
    CHECK(cudaMemcpy(w->matrix, evec, matrix_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(w->b, 0, field_bytes)); // original wf2_block=0; NN beta=one
    CHECK(cudaError_t(salmon_subspace_rotation_device(w->a, w->matrix, w->b)));
    CHECK(cudaMemcpy(out, w->b, field_bytes, cudaMemcpyDeviceToHost));
    w->phase = 0;
    return 0;
    #undef CHECK
}

extern "C" int salmon_subspace_context_destroy(void **handle) {
    if (!handle) return int(cudaErrorInvalidValue);
    if (!*handle) return 0;
    auto *w = static_cast<Workspace *>(*handle);
    *handle = nullptr;
    return int(release(w));
}
