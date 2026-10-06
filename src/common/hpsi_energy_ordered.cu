//
//  Copyright 2019-2020 SALMON developers
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

#include "hpsi_energy_ordered.h"
#include <cstddef>

namespace {
constexpr int physical=16*16*16,padded=16*17*16;
__device__ __forceinline__ int storage(int point){
    return (point%256)+272*(point/256);
}
__global__ void kinetic_points(int nv,const double *v,const double2 *x,
                               const double2 *h,double2 *t){
    const size_t q=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(q>=size_t(nv)*physical)return;
    const int point=int(q%physical);
    const size_t i=(q/physical)*padded+storage(point);
    const double2 a=x[i],b=h[i];const double p=v[point];
    // nvfortran compiles this complex subtraction as one negative FMA per component.
    t[i]=make_double2(__fma_rn(-p,a.x,b.x),__fma_rn(-p,a.y,b.y));
}
__device__ __forceinline__ double dot_real(double2 a,double2 b){
    return __fma_rn(a.x,b.x,__dmul_rn(a.y,b.y));
}
__global__ void ordered_sums(int nv,const double *v,const double2 *x,
                             const double2 *h,const double2 *t,double *out){
    const int lane=threadIdx.x%32;
    const int vector=blockIdx.x*4+threadIdx.x/32;
    if(vector>=nv)return; // The whole warp leaves together, also when nv is small.
    double es=0.0,ek=0.0,en=0.0;
    // Four independent orbital warps per block. Lanes load contiguous physical
    // points; lane zero consumes each contribution in the CPU's exact x,y,z order.
    // No partial-sum tree, reassociation, padding reads or shared-memory storage.
    for(int base=0;base<physical;base+=32){
        const int point=base+lane;
        const size_t i=size_t(vector)*padded+storage(point);
        const double2 a=x[i],b=h[i],k=t[i];const double p=v[point];
        const double2 nl=make_double2(__dsub_rn(b.x,__fma_rn(p,a.x,k.x)),
                                     __dsub_rn(b.y,__fma_rn(p,a.y,k.y)));
        const double se=dot_real(a,b),sk=dot_real(a,k),sn=dot_real(a,nl);
        for(int j=0;j<32;++j){
            const double ve=__shfl_sync(0xffffffff,se,j);
            const double vk=__shfl_sync(0xffffffff,sk,j);
            const double vn=__shfl_sync(0xffffffff,sn,j);
            if(lane==0){es=__dadd_rn(es,ve);ek=__dadd_rn(ek,vk);en=__dadd_rn(en,vn);}
        }
    }
    if(lane==0){out[3*vector]=es;out[3*vector+1]=ek;out[3*vector+2]=en;}
}
}
int hpsi_energy_kinetic(int nv,const double *v,const double2 *x,const double2 *h,double2 *t){
    if(nv<=0||!v||!x||!h||!t)return -1;
    kinetic_points<<<(size_t(nv)*physical+255)/256,256>>>(nv,v,x,h,t);
    return int(cudaGetLastError());
}
int hpsi_energy_sums(int nv,const double *v,const double2 *x,const double2 *h,const double2 *t,double *out){
    if(nv<=0||!v||!x||!h||!t||!out)return -1;
    ordered_sums<<<(nv+3)/4,128>>>(nv,v,x,h,t,out);
    return int(cudaGetLastError());
}
