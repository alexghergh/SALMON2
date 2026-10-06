// CUDA complex stencil in CPU operation order, used by the whole-CG GPU mode and the CUDA Hpsi.
// Borrowed CUDA device arrays, default stream, no allocation/copy/synchronization.
#ifndef SALMON_CG_STENCIL_ORDERED_H
#define SALMON_CG_STENCIL_ORDERED_H
#include <vector_types.h>

extern "C" int salmon_cg_stencil_ordered(
    int isx, int iex, int isy, int iey, int isz, int iez,
    int lax, int uax, int lay, int uay, int laz, int uaz,
    int nspin, int no, int nk,
    const int *idx, const int *idy, const int *idz,
    const double *vlocal, const double *lap0, const double *lapt,
    const double *nabt, const double2 *psi, double2 *hpsi);

// Map storage starts at is(axis)-4 and retains the original Fortran map values.
// psi/hpsi: (lax:uax,lay:uay,laz:uaz,nspin,no,nk), image count one.
// vlocal: physical (isx:iex,isy:iey,isz:iez,nspin), without orbital padding.
// lap0(nk), lapt(12), nabt(12,nk): already evaluated by the original CPU
// kAc/k_lap0/k_nabt expressions, in owned-k order. The leaf does not recompute them.
// Only physical hpsi elements are written. psi/hpsi must be distinct allocations.
// Returns a CUDA launch status; the owner synchronizes after ordered nonlocal work.
#endif
