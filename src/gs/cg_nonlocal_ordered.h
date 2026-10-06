// SPDX-License-Identifier: Apache-2.0
#ifndef SALMON_CG_NONLOCAL_ORDERED_H
#define SALMON_CG_NONLOCAL_ORDERED_H
#include <cuda_runtime.h>
#include <cstddef>

// Borrowed device arrays; value shape scalars. Default stream, synchronous return.
// The caller validates/pins real s_pp_grid inverse maps once before device upload.
extern "C" int salmon_cg_nonlocal_ordered(
    double2 *hpsi,int im_s,int im_e,int ik_s,int ik_e,int io_s,int io_e,
    int nspin,int nlma,int nps,int natom,
    int lx,int ux,int ly,int uy,int lz,int uz,
    const int *ia_tbl,const int *mps,const int *jxyz,
    const double2 *uv,const double *rinv,const double2 *psi,
    int max_vi,const int *v2j,const int *v2nlma,
    const int *k2ilma,const int *k2j,double2 *coeff,size_t coeff_count);
#endif
