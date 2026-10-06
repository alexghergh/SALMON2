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

#pragma once
// Shared map ownership only; no CG fields and no Hamiltonian arithmetic here.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdlib>
#include <cstring>
// Interoperable with hpsi_input in hpsi_cuda_inputs.f90. Host pointers are
// borrowed during capture; each client owns separate device copies until close.
struct EpochInput {
    int lo[3],hi[3],no,nk,nlma,nps,natom,max_vi,max_contrib;
    const int *idx,*idy,*idz;
    const double *vlocal,*lap0,*lapt,*nabt;
    const int *ia_tbl,*mps,*jxyz;
    const double2 *uv;
    const double *rinv;
    const int *v2j,*v2nlma,*k2ilma,*k2j;
};
struct HpsiDeviceMaps {
    EpochInput maps{};
    double2 *coeff=nullptr;
    size_t map_bytes=0,coeff_count=0;
};
inline int hpsi_device_guard(){
    const char *visible=std::getenv("CUDA_VISIBLE_DEVICES");
    if(!visible||!*visible||std::strchr(visible,','))return -10;
    int count=0; auto rc=cudaGetDeviceCount(&count);
    if(rc!=cudaSuccess)return int(rc);
    return count==1?0:-11;
}
int hpsi_validate(const EpochInput *input);
// Requires a fresh empty owner and an input that passed hpsi_validate.
int hpsi_maps_begin(const EpochInput &input,HpsiDeviceMaps &owner);
int hpsi_maps_close(HpsiDeviceMaps &owner);
int hpsi_stencil(const HpsiDeviceMaps &owner,const double2 *psi,double2 *hpsi);
int hpsi_nonlocal(const HpsiDeviceMaps &owner,const double2 *psi,double2 *hpsi);
