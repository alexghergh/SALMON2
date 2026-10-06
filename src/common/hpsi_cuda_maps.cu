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

// Device copies of the Hpsi maps, and the stencil/nonlocal launches shared by CG and Hpsi.
#include "hpsi_cuda_maps.h"
#include "../gs/cg_stencil_ordered.h"
#include "../gs/cg_nonlocal_ordered.h"
#include <limits>
#include <new>
#include <vector>
#include <type_traits>
// Check that the inverse projector map of init_uvpsi_blocking lists every (ilma,j) visit in
// the original order. Validate once per context; CG reuses the maps for its Hpsi calls.
// Invalid maps are an error; unsupported layouts are sent to the CPU path by the bridge.
static bool valid_hpsi_maps(const EpochInput &m){
    if(m.no<=0||m.nk<=0||m.nlma<=0||m.nps<=0||m.natom<=0||m.max_vi<=0||
       m.max_vi>4096||m.max_contrib<=0)return false;
    const void *required[]={m.idx,m.idy,m.idz,m.vlocal,m.lap0,m.lapt,m.nabt,
        m.ia_tbl,m.mps,m.jxyz,m.uv,m.rinv,m.v2j,m.v2nlma,m.k2ilma,m.k2j};
    for(const void *p:required)if(!p)return false;
    const int *index[]={m.idx,m.idy,m.idz};
    for(int axis=0;axis<3;++axis){
        if(m.hi[axis]-m.lo[axis]!=15)return false;
        for(int j=0;j<24;++j)if(index[axis][j]<m.lo[axis]||index[axis][j]>m.hi[axis])return false;
    }
    auto point=[&](const int *xyz){
        for(int axis=0;axis<3;++axis)if(xyz[axis]<m.lo[axis]||xyz[axis]>m.hi[axis])return -1;
        return xyz[0]-m.lo[0]+16*(xyz[1]-m.lo[1]+16*(xyz[2]-m.lo[2]));
    };
    std::vector<int> inverse(4096,-1),seen(size_t(m.max_vi),0);
    for(int vi=0;vi<m.max_vi;++vi){
        int at=point(m.v2j+3*vi);
        if(at<0||inverse[at]!=-1||m.v2nlma[vi]<1||m.v2nlma[vi]>m.max_contrib)return false;
        inverse[at]=vi;
    }
    for(int ia=0;ia<m.natom;++ia)if(m.mps[ia]<0||m.mps[ia]>m.nps)return false;
    for(int ilma=0;ilma<m.nlma;++ilma){
        int ia=m.ia_tbl[ilma]-1;if(ia<0||ia>=m.natom)return false;
        for(int j=0;j<m.mps[ia];++j){
            int at=point(m.jxyz+3*(size_t(j)+size_t(m.nps)*ia));if(at<0)return false;
            int vi=inverse[at];if(vi<0||seen[vi]>=m.v2nlma[vi])return false;
            size_t row=vi+size_t(m.max_vi)*seen[vi]++;
            if(m.k2ilma[row]!=ilma+1||m.k2j[row]!=j+1)return false;
        }
    }
    for(int vi=0;vi<m.max_vi;++vi)if(seen[vi]!=m.v2nlma[vi])return false;
    return true;
}
int hpsi_validate(const EpochInput *input){
    if(!input)return -41;
    try {if(!valid_hpsi_maps(*input))return -41;}
    catch(const std::bad_alloc &){return -3;}
    const auto &m=*input;
    size_t nv=size_t(m.no)*m.nk;
    if(nv>size_t(std::numeric_limits<int>::max())/32)return -2;
    const size_t limit=std::numeric_limits<size_t>::max()/sizeof(double2);
    if(nv>limit/(16*17*16)||nv>limit/size_t(m.nlma)||
       size_t(m.nps)>limit/size_t(m.nlma)/size_t(m.nk)||
       size_t(m.nps)>limit/size_t(m.natom)/3||
       size_t(m.max_vi)>limit/size_t(m.max_contrib))return -2;
    return 0;
}
int hpsi_maps_close(HpsiDeviceMaps &owner){
    int result=0;
    const auto &m=owner.maps;
    const void *owned[]={m.idx,m.idy,m.idz,m.vlocal,m.lap0,m.lapt,m.nabt,
                         m.ia_tbl,m.mps,m.jxyz,m.uv,m.rinv,m.v2j,m.v2nlma,m.k2ilma,m.k2j,owner.coeff};
    for(const void *p:owned)if(p){int rc=int(cudaFree(const_cast<void*>(p)));if(!result)result=rc;}
    owner=HpsiDeviceMaps{};
    return result;
}
int hpsi_maps_begin(const EpochInput &m,HpsiDeviceMaps &owner){
    // Both callers validate once, before any context allocation.
    int rc=0;
    auto *c=&owner;const size_t nv=size_t(m.no)*m.nk;
    // Copy only scalar metadata first: borrowed host pointers must never be freed.
    for(int axis=0;axis<3;++axis){c->maps.lo[axis]=m.lo[axis];c->maps.hi[axis]=m.hi[axis];}
    c->maps.no=m.no;c->maps.nk=m.nk;c->maps.nlma=m.nlma;c->maps.nps=m.nps;
    c->maps.natom=m.natom;c->maps.max_vi=m.max_vi;c->maps.max_contrib=m.max_contrib;
#define TRY(cmd) do{rc=int(cmd);if(rc){hpsi_maps_close(owner);return rc;}}while(0)
    c->coeff_count=nv*size_t(m.nlma);
    TRY(cudaMalloc(reinterpret_cast<void**>(&c->coeff),c->coeff_count*sizeof(double2)));
    auto copy=[&](auto &dst,const auto *src,size_t n){
        using T=std::remove_const_t<std::remove_pointer_t<std::decay_t<decltype(dst)>>>;
        T *device=nullptr;int result=int(cudaMalloc(reinterpret_cast<void**>(&device),n*sizeof(T)));
        if(result)return result;dst=device;c->map_bytes+=n*sizeof(T);
        return int(cudaMemcpy(device,src,n*sizeof(T),cudaMemcpyHostToDevice));
    };
    TRY(copy(c->maps.idx,m.idx,24));TRY(copy(c->maps.idy,m.idy,24));TRY(copy(c->maps.idz,m.idz,24));
    TRY(copy(c->maps.vlocal,m.vlocal,4096));TRY(copy(c->maps.lap0,m.lap0,m.nk));
    TRY(copy(c->maps.lapt,m.lapt,12));TRY(copy(c->maps.nabt,m.nabt,size_t(12)*m.nk));
    TRY(copy(c->maps.ia_tbl,m.ia_tbl,m.nlma));TRY(copy(c->maps.mps,m.mps,m.natom));
    TRY(copy(c->maps.jxyz,m.jxyz,size_t(3)*m.nps*m.natom));
    TRY(copy(c->maps.uv,m.uv,size_t(m.nps)*m.nlma*m.nk));TRY(copy(c->maps.rinv,m.rinv,m.nlma));
    TRY(copy(c->maps.v2j,m.v2j,size_t(3)*m.max_vi));TRY(copy(c->maps.v2nlma,m.v2nlma,m.max_vi));
    TRY(copy(c->maps.k2ilma,m.k2ilma,size_t(m.max_vi)*m.max_contrib));
    TRY(copy(c->maps.k2j,m.k2j,size_t(m.max_vi)*m.max_contrib));
    return 0;
#undef TRY
}
int hpsi_stencil(const HpsiDeviceMaps &owner,const double2 *psi,double2 *hpsi){
    const auto &m=owner.maps;
    return salmon_cg_stencil_ordered(m.lo[0],m.hi[0],m.lo[1],m.hi[1],m.lo[2],m.hi[2],
        m.lo[0],m.hi[0],m.lo[1],m.hi[1]+1,m.lo[2],m.hi[2],1,m.no,m.nk,
        m.idx,m.idy,m.idz,m.vlocal,m.lap0,m.lapt,m.nabt,psi,hpsi);
}
int hpsi_nonlocal(const HpsiDeviceMaps &owner,const double2 *psi,double2 *hpsi){
    const auto &m=owner.maps;
    return salmon_cg_nonlocal_ordered(hpsi,1,1,1,m.nk,1,m.no,1,m.nlma,m.nps,m.natom,
        m.lo[0],m.hi[0],m.lo[1],m.hi[1]+1,m.lo[2],m.hi[2],m.ia_tbl,m.mps,m.jxyz,
        m.uv,m.rinv,psi,m.max_vi,m.v2j,m.v2nlma,m.k2ilma,m.k2j,owner.coeff,owner.coeff_count);
}
