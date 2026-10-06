// CUDA periodic complex stencil with the operation order of zstencil_core/typical/seq.f90 as
// compiled for the CPU by nvfortran 26.1. It does not follow the X then Y/Z split of the
// OpenACC stencil in zstencil_core/gpu.
#include <cuda_runtime.h>
#include <cstddef>
#include <limits>
#include "cg_stencil_ordered.h"

namespace {
struct Grid {
    int nx, ny, nz, sx, sy, no, nspin;
    int isx, isy, isz, lax, lay, laz;
    size_t plane, field, physical, vectors;
    __device__ size_t at(int x,int y,int z,size_t vector) const {
        return vector*field+size_t(z-laz)*plane+size_t(y-lay)*sx+size_t(x-lax);
    }
};

__device__ __forceinline__ double2 plus(double2 a,double2 b) {
    return make_double2(__dadd_rn(a.x,b.x),__dadd_rn(a.y,b.y));
}
__device__ __forceinline__ double2 minus(double2 a,double2 b) {
    return make_double2(__dsub_rn(a.x,b.x),__dsub_rn(a.y,b.y));
}
__device__ __forceinline__ double2 scale(double c,double2 a) {
    return make_double2(__dmul_rn(c,a.x),__dmul_rn(c,a.y));
}
__device__ __forceinline__ double2 madd(double c,double2 a,double2 b) {
    return make_double2(__fma_rn(c,a.x,b.x),__fma_rn(c,a.y,b.y));
}

// One four-neighbour group, in the original coefficient order 1,2,3,4.
// Explicit intrinsics stop the CUDA compiler from contracting or reordering these operations.
__device__ __forceinline__ void axis_group(const double2 p[4],const double2 m[4],
        const double *lap,const double *nab,double2 &v,double2 &w) {
    v=scale(lap[0],plus(p[0],m[0]));
    w=scale(nab[0],minus(p[0],m[0]));
    v=madd(lap[1],plus(p[1],m[1]),v);
    w=madd(nab[1],minus(p[1],m[1]),w);
    v=madd(lap[2],plus(p[2],m[2]),v);
    w=madd(nab[2],minus(p[2],m[2]),w);
    v=madd(lap[3],plus(p[3],m[3]),v);
    w=madd(nab[3],minus(p[3],m[3]),w);
}

__global__ void stencil(Grid q,const int *idx,const int *idy,const int *idz,
        const double *vlocal,const double *lap0,const double *lapt,const double *nabt,
        const double2 *psi,double2 *hpsi) {
    size_t id=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=q.physical*q.vectors)return;
    size_t vector=id/q.physical,point=id%q.physical;
    int x=int(point%q.nx)+q.isx;
    int y=int((point/q.nx)%q.ny)+q.isy;
    int z=int(point/(size_t(q.nx)*q.ny))+q.isz;
    int spin=vector%q.nspin;
    int k=vector/(size_t(q.nspin)*q.no);
    const double *nab=nabt+12*k;
    double2 p[4],m[4],v,w,axis_v,axis_w;
    #pragma unroll
    for(int j=1;j<=4;++j) {
        p[j-1]=psi[q.at(idx[x-q.isx+4+j],y,z,vector)];
        m[j-1]=psi[q.at(idx[x-q.isx+4-j],y,z,vector)];
    }
    axis_group(p,m,lapt,nab,v,w);
    #pragma unroll
    for(int j=1;j<=4;++j) {
        p[j-1]=psi[q.at(x,idy[y-q.isy+4+j],z,vector)];
        m[j-1]=psi[q.at(x,idy[y-q.isy+4-j],z,vector)];
    }
    axis_group(p,m,lapt+4,nab+4,axis_v,axis_w);
    v=plus(axis_v,v);w=plus(axis_w,w); // Y group + X group, as assigned on CPU.
    #pragma unroll
    for(int j=1;j<=4;++j) {
        p[j-1]=psi[q.at(x,y,idz[z-q.isz+4+j],vector)];
        m[j-1]=psi[q.at(x,y,idz[z-q.isz+4-j],vector)];
    }
    axis_group(p,m,lapt+8,nab+8,axis_v,axis_w);
    v=plus(axis_v,v);w=plus(axis_w,w); // Z group + the prior (Y + X) assignment.
    size_t at=q.at(x,y,z,vector);
    double2 a=psi[at];
    double2 local=scale(vlocal[size_t(spin)*q.physical+point],a);
    local=madd(lap0[k],a,local);
    // nvfortran 26.1 compiles the CPU statement as separate complex (0.5,+0)*v and
    // (+0,+1)*w products, then two subtractions. Keep their zero cross
    // terms: replacing them by real scaling/rotation changes signed zeros.
    // The first subtraction is not fused into the local-potential sum.
    double2 half_v=make_double2(
        __fma_rn(0.5,v.x,__dmul_rn(-0.0,v.y)),
        __fma_rn(0.0,v.x,__dmul_rn(0.5,v.y)));
    local=minus(local,half_v);
    double2 iw=make_double2(__fma_rn(0.0,w.x,-w.y),
                           __fma_rn(0.0,w.y,w.x));
    hpsi[at]=minus(local,iw);
}
} // namespace

extern "C" int salmon_cg_stencil_ordered(
        int isx,int iex,int isy,int iey,int isz,int iez,
        int lax,int uax,int lay,int uay,int laz,int uaz,
        int nspin,int no,int nk,const int *idx,const int *idy,const int *idz,
        const double *vlocal,const double *lap0,const double *lapt,const double *nabt,
        const double2 *psi,double2 *hpsi) {
    // Supported: a 16^3 grid with one padding row in y, one spin, one image, no r-space division,
    // at most 80 orbitals and 64 k points.
    if(static_cast<long long>(iex)-isx!=15||static_cast<long long>(iey)-isy!=15||
       static_cast<long long>(iez)-isz!=15||nspin!=1||no<1||no>80||nk<1||nk>64||
       lax!=isx||uax!=iex||lay!=isy||static_cast<long long>(uay)!=static_cast<long long>(iey)+1||laz!=isz||uaz!=iez||
       !idx||!idy||!idz||!vlocal||!lap0||!lapt||!nabt||!psi||!hpsi||psi==hpsi)
        return int(cudaErrorInvalidValue);
    Grid q{16,16,16,16,17,no,nspin,isx,isy,isz,lax,lay,laz,
           16*17,16*17*16,16*16*16,size_t(nspin)*no*nk};
    constexpr unsigned threads=128;
    size_t blocks=(q.physical*q.vectors+threads-1)/threads;
    if(blocks>std::numeric_limits<unsigned>::max())return int(cudaErrorInvalidValue);
    stencil<<<unsigned(blocks),threads>>>(q,idx,idy,idz,vlocal,lap0,lapt,nabt,psi,hpsi);
    return int(cudaGetLastError());
}
