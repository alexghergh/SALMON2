#ifndef SALMON_REAL_GRAM_ORDERED_H
#define SALMON_REAL_GRAM_ORDERED_H
#include <cuda_runtime.h>

// Arrays are caller-owned device memory, column-major, in this fixed shape:
// A,B(21296,1027), S,U(1027,1027). Inputs must not alias outputs.
// Launch on the default stream; caller owns allocation/copies/synchronization.
// These are not general DGEMMs; gram_schmidt_col_rblas reaches them through real_gram_context.
extern "C" int salmon_real_gram_overlap_ordered(const double *a, const double *b,
                                               double *s, double hvol);
extern "C" int salmon_real_gram_update_ordered(const double *a, const double *u,
                                              double *b);
#endif
