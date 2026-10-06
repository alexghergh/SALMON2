// SPDX-License-Identifier: Apache-2.0
// Device workspace for one block pair; the MPI sum between the two products stays on the host.
#include "real_gram_ordered.h"
#include <cstdio>
#include <initializer_list>
#include <new>

namespace {
constexpr size_t field_bytes = 21296 * 1027 * sizeof(double);
constexpr size_t matrix_bytes = 1027 * 1027 * sizeof(double);
struct Workspace {
    double *a = nullptr, *b = nullptr, *matrix = nullptr;
    int phase = 0; // 0: overlap; 1: update; 2: done; -1: destroy only.
};

cudaError_t release(Workspace *w) {
    cudaError_t first = cudaSuccess;
    for (auto p : {w->a, w->b, w->matrix}) if (p) {
        const auto status = cudaFree(p);
        if (first == cudaSuccess) first = status;
    }
    delete w;
    return first;
}
} // namespace

extern "C" int salmon_real_gram_context_create(void **handle) {
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
        std::fprintf(stderr, "REAL_GRAM_CLEANUP_ERROR %d\n", int(cleanup));
    return int(status);
    #undef CHECK
}

extern "C" int salmon_real_gram_context_overlap(void *handle, const double *a,
        const double *b, double hvol, double *overlap) {
    auto *w = static_cast<Workspace *>(handle);
    if (!w || w->phase != 0 || !a || !b || !overlap) return int(cudaErrorInvalidValue);
    w->phase = -1;
    cudaError_t status;
    #define CHECK(call) do { status = (call); if (status != cudaSuccess) return int(status); } while (0)
    CHECK(cudaMemcpy(w->a, a, field_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(w->b, b, field_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaError_t(salmon_real_gram_overlap_ordered(w->a, w->b, w->matrix, hvol)));
    CHECK(cudaMemcpy(overlap, w->matrix, matrix_bytes, cudaMemcpyDeviceToHost));
    // Blocking DtoH completes TN before the original CPU comm_summation.
    w->phase = 1;
    return 0;
    #undef CHECK
}

extern "C" int salmon_real_gram_context_update(void *handle, const double *reduced,
        double *out) {
    auto *w = static_cast<Workspace *>(handle);
    if (!w || w->phase != 1 || !reduced || !out) return int(cudaErrorInvalidValue);
    w->phase = -1;
    cudaError_t status;
    #define CHECK(call) do { status = (call); if (status != cudaSuccess) return int(status); } while (0)
    CHECK(cudaMemcpy(w->matrix, reduced, matrix_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaError_t(salmon_real_gram_update_ordered(w->a, w->matrix, w->b)));
    CHECK(cudaMemcpy(out, w->b, field_bytes, cudaMemcpyDeviceToHost));
    // B must be back on the host before its next CPU normalization/orthogonalization.
    w->phase = 2;
    return 0;
    #undef CHECK
}

extern "C" int salmon_real_gram_context_destroy(void **handle) {
    if (!handle) return int(cudaErrorInvalidValue);
    if (!*handle) return 0;
    auto *w = static_cast<Workspace *>(*handle);
    *handle = nullptr;
    return int(release(w));
}
