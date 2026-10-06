// SPDX-License-Identifier: Apache-2.0
// CUDA nonlocal (projector) term of Hpsi in CPU operation order (SALMON_CG_ORDERED_NONLOCAL).
// Layout/order follows SALMON nonlocal_potential::zpseudo and init_uvpsi_blocking.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include "cg_nonlocal_ordered.h"

namespace {
struct Layout {
    int ims, ime, ks, ke, os, oe, ns, nlma, nps;
    int lo[3], hi[3], max_vi;
    size_t nv, field;
    __device__ size_t at(int x,int y,int z,size_t v) const {
        return v*field + size_t(z-lo[2])*(hi[1]-lo[1]+1)*(hi[0]-lo[0]+1)
             + size_t(y-lo[1])*(hi[0]-lo[0]+1) + size_t(x-lo[0]);
    }
    __device__ int local_k(size_t v) const {
        return int((v/(size_t(ns)*(oe-os+1)))%size_t(ke-ks+1));
    }
    __device__ size_t uv(int j,int ilma,int k) const {
        return size_t(j)+size_t(nps)*(ilma+size_t(nlma)*k);
    }
};
// Complex products use the explicit operation tree of the kernels in cg_resident.cu, which
// matches zpseudo as compiled for the CPU by nvfortran.
__device__ __forceinline__ double2 add(double2 a,double2 b){
    return make_double2(__dadd_rn(a.x,b.x),__dadd_rn(a.y,b.y));
}
__device__ __forceinline__ double2 conj_product(double2 a,double2 b){
    return make_double2(__fma_rn(a.x,b.x,__dmul_rn(a.y,b.y)),
                        __fma_rn(a.x,b.y,__dmul_rn(-a.y,b.x)));
}
__device__ __forceinline__ double2 product(double2 a,double2 b){
    return make_double2(__fma_rn(a.x,b.x,__dmul_rn(-a.y,b.y)),
                        __fma_rn(a.x,b.y,__dmul_rn(a.y,b.x)));
}
// One independent (owned orbital,projector) per thread; j is strictly sequential.
__global__ void project(Layout q,const int *ia_tbl,const int *mps,const int *jxyz,
        const double2 *uv,const double *rinv,const double2 *psi,double2 *coeff){
    size_t id=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=q.nv*size_t(q.nlma))return;
    int ilma=int(id%q.nlma),ia=ia_tbl[ilma]-1;
    size_t v=id/q.nlma;
    double2 a=make_double2(0.,0.);
    int k=q.local_k(v);
    for(int j=0;j<mps[ia];++j){
        size_t ij=3*(size_t(j)+size_t(q.nps)*ia);
        size_t p=q.at(jxyz[ij],jxyz[ij+1],jxyz[ij+2],v);
        a=add(a,conj_product(uv[q.uv(j,ilma,k)],psi[p]));
    }
    coeff[id]=make_double2(__dmul_rn(a.x,rinv[ilma]),__dmul_rn(a.y,rinv[ilma]));
}
// Inverse map entries have the original increasing (ilma,j) order, including
// duplicate visits. Start from the stencil result, NOT from zero; no atomics.
__global__ void scatter_ordered(Layout q,const int *v2j,const int *v2nlma,
        const int *k2ilma,const int *k2j,const double2 *uv,
        const double2 *coeff,double2 *hpsi){
    size_t id=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=q.nv*size_t(q.max_vi))return;
    int vi=int(id%q.max_vi);size_t v=id/q.max_vi;
    size_t p=q.at(v2j[3*vi],v2j[3*vi+1],v2j[3*vi+2],v);
    double2 a=hpsi[p];int k=q.local_k(v);
    for(int n=0;n<v2nlma[vi];++n){
        size_t row=vi+size_t(q.max_vi)*n;
        int ilma=k2ilma[row]-1,j=k2j[row]-1;
        a=add(a,product(coeff[ilma+size_t(q.nlma)*v],uv[q.uv(j,ilma,k)]));
    }
    hpsi[p]=a;
}
}

// All array arguments below are already device addresses, including scratch.
// The caller owns allocation, transfers and the final copy back, and validates the maps
// against s_pp_grid and the physical (not padded) spatial bounds once.
// This routine borrows them synchronously and does not create a device context.
extern "C" int salmon_cg_nonlocal_ordered(
        double2 *hpsi,int im_s,int im_e,int ik_s,int ik_e,int io_s,int io_e,
        int nspin,int nlma,int nps,int natom,
        int lx,int ux,int ly,int uy,int lz,int uz,
        const int *ia_tbl,const int *mps,const int *jxyz,
        const double2 *uv,const double *rinv,const double2 *psi,
        int max_vi,const int *v2j,const int *v2nlma,
        const int *k2ilma,const int *k2j,double2 *coeff,size_t coeff_count){
    if(im_s!=1||im_e!=1||nspin!=1||io_s<1||io_e<io_s||ik_s<1||ik_e<ik_s||
       nlma<=0||nps<=0||natom<=0||max_vi<=0||ux<lx||uy<ly||uz<lz||
       !hpsi||!ia_tbl||!mps||!jxyz||!uv||!rinv||!psi||!v2j||!v2nlma||
       !k2ilma||!k2j||!coeff||hpsi==psi||hpsi==coeff||psi==coeff)return -1;
    const std::int64_t nx=std::int64_t(ux)-lx+1;
    const std::int64_t ny=std::int64_t(uy)-ly+1;
    const std::int64_t nz=std::int64_t(uz)-lz+1;
    if(nx>std::numeric_limits<int>::max()||ny>std::numeric_limits<int>::max()||
       nz>std::numeric_limits<int>::max())return -2;
    size_t nv=size_t(io_e-io_s+1)*size_t(ik_e-ik_s+1);
    size_t field=size_t(nx);
    const size_t limit=std::numeric_limits<size_t>::max();
    for(size_t n:{size_t(ny),size_t(nz),nv}){
        if(field>limit/n)return -2;field*=n;
    }
    if(field>limit/sizeof(double2)||nv>limit/size_t(nlma)||
       nv>limit/size_t(max_vi)||coeff_count!=nv*size_t(nlma)||
       coeff_count>limit-127||nv*size_t(max_vi)>limit-127)return -2;
    size_t project_blocks=(coeff_count+127)/128;
    size_t scatter_blocks=(nv*size_t(max_vi)+127)/128;
    if(project_blocks>size_t(std::numeric_limits<int>::max())||
       scatter_blocks>size_t(std::numeric_limits<int>::max()))return -2;
    Layout q{im_s,im_e,ik_s,ik_e,io_s,io_e,nspin,nlma,nps,
             {lx,ly,lz},{ux,uy,uz},max_vi,nv,field/nv};
    project<<<unsigned(project_blocks),128>>>(q,ia_tbl,mps,jxyz,uv,rinv,psi,coeff);
    int rc=int(cudaGetLastError());if(rc)return rc;
    scatter_ordered<<<unsigned(scatter_blocks),128>>>(q,v2j,v2nlma,k2ilma,k2j,uv,coeff,hpsi);
    rc=int(cudaGetLastError());if(rc)return rc;
    return int(cudaDeviceSynchronize());
}
