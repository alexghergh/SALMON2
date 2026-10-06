// SPDX-License-Identifier: Apache-2.0
// One subspace invocation owns two padded fields and one batch of small matrices.
#include "../common/hpsi_cuda_maps.h"
#include "subspace_gemm_ordered.h"
#include <cstdio>
#include <initializer_list>
#include <new>

namespace {
struct SubspaceCall : HpsiDeviceMaps {
    double2 *psi=nullptr, *hpsi=nullptr, *matrix=nullptr;
    size_t field_bytes=0, matrix_bytes=0;
    int stage=0; // 1: Hpsi ready; 2: overlaps returned; 3: rotation returned; -1: failed.
};
int read_physical(const SubspaceCall &c, const double2 *field, double2 *out) {
    return int(cudaMemcpy2D(out, 16*17*sizeof(double2), field, 16*17*sizeof(double2),
        16*16*sizeof(double2), size_t(16)*80*c.maps.nk, cudaMemcpyDeviceToHost));
}
}

extern "C" int salmon_subspace_connected_close(void *handle) {
    auto *c=static_cast<SubspaceCall*>(handle);
    if (!c) return 0;
    int first=0;
    for (auto *p : {c->psi,c->hpsi,c->matrix}) if (p) {
        const int rc=int(cudaFree(p)); if (!first) first=rc;
    }
    const int rc=hpsi_maps_close(*c); if (!first) first=rc;
    delete c; return first;
}

extern "C" int salmon_subspace_connected_begin(const EpochInput *input,
                                               const double2 *psi, void **handle) {
    if (!handle || *handle || !psi) return -1;
    int rc=hpsi_validate(input); if (rc) return rc;
    if (input->no!=80 || input->nk<1 || input->nk>65535) return -1;
    rc=hpsi_device_guard(); if (rc) return rc;
    auto *c=new(std::nothrow) SubspaceCall; if (!c) return -3;
    c->field_bytes=size_t(4352)*80*input->nk*sizeof(double2);
    c->matrix_bytes=size_t(80)*80*input->nk*sizeof(double2);
#define TRY(cmd) do { rc=int(cmd); if(rc) { \
    const int cleanup=salmon_subspace_connected_close(c); \
    if(cleanup) std::fprintf(stderr,"SUBSPACE_CONNECTED_CLEANUP_ERROR %d\n",cleanup); \
    return rc; } } while(0)
    TRY(hpsi_maps_begin(*input,*c));
    TRY(cudaMalloc(reinterpret_cast<void**>(&c->psi),c->field_bytes));
    TRY(cudaMalloc(reinterpret_cast<void**>(&c->hpsi),c->field_bytes));
    TRY(cudaMalloc(reinterpret_cast<void**>(&c->matrix),c->matrix_bytes));
    TRY(cudaMemcpy(c->psi,psi,c->field_bytes,cudaMemcpyHostToDevice));
    TRY(hpsi_stencil(*c,c->psi,c->hpsi));
    TRY(hpsi_nonlocal(*c,c->psi,c->hpsi));
    c->stage=1; *handle=c; return 0;
#undef TRY
}

extern "C" int salmon_subspace_connected_query(void *handle, int *nk, size_t *fields,
                                               size_t *matrix, size_t *maps, size_t *coeff) {
    auto *c=static_cast<SubspaceCall*>(handle);
    if (!c || c->stage!=1 || !nk || !fields || !matrix || !maps || !coeff) return -1;
    *nk=c->maps.nk; *fields=2*c->field_bytes; *matrix=c->matrix_bytes;
    *maps=c->map_bytes; *coeff=c->coeff_count*sizeof(double2); return 0;
}

extern "C" int salmon_subspace_connected_overlap(void *handle, double hvol,
                                                 double2 *matrices, double2 *hx, int retain_hx) {
    auto *c=static_cast<SubspaceCall*>(handle);
    if (!c || c->stage!=1 || !matrices || (retain_hx!=0 && retain_hx!=1) || (retain_hx && !hx)) return -1;
    c->stage=-1;
    int rc=salmon_subspace_cn80_padded_batch(c->psi,c->hpsi,c->matrix,hvol,0.0,c->maps.nk);
    if (rc) return rc;
    rc=int(cudaMemcpy(matrices,c->matrix,c->matrix_bytes,cudaMemcpyDeviceToHost));
    if (rc) return rc;
    // Hpsi is the unrotated original output. Save it before reusing this field.
    // Physical-only copies preserve every host padding bit in both output modes.
    if (retain_hx) { rc=read_physical(*c,c->hpsi,hx); if (rc) return rc; }
    c->stage=2; return 0;
}

extern "C" int salmon_subspace_connected_rotate(void *handle, const double2 *evec, double2 *psi) {
    auto *c=static_cast<SubspaceCall*>(handle);
    if (!c || c->stage!=2 || !evec || !psi) return -1;
    c->stage=-1;
    int rc=int(cudaMemcpy(c->matrix,evec,c->matrix_bytes,cudaMemcpyHostToDevice)); if (rc) return rc;
    // All overlaps and optional Hpsi output are complete. NN beta=one reads +0.
    rc=int(cudaMemset(c->hpsi,0,c->field_bytes)); if (rc) return rc;
    rc=salmon_subspace_nn80_padded_batch(c->psi,c->matrix,c->hpsi,1.0,0.0,c->maps.nk); if (rc) return rc;
    rc=read_physical(*c,c->hpsi,psi); if (rc) return rc;
    c->stage=3; return 0;
}
