// CUDA ZGEMMs for 80-orbital subspace diagonalization on a 4096-point grid, in CPU operation order.
#pragma once
#include <cuda_runtime.h>

// Borrowed device arrays, Fortran column-major, disjoint inputs/output.
// CN: A,B(4096,80), C(80,80), beta=0. NN: A(4096,80), E(80,80), C(4096,80), beta=1.
// Default stream, launch only; caller owns allocation, synchronization and lifetime.
extern "C" int salmon_subspace_cn80_ordered(const double2 *a, const double2 *b,
                                           double2 *c, double ar, double ai);
extern "C" int salmon_subspace_nn80_ordered(const double2 *a, const double2 *e,
                                           double2 *c, double ar, double ai);
