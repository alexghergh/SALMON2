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

// One host Hpsi call owns two padded fields. Map/leaf code is shared with CG.
#include "hpsi_cuda_maps.h"
#include <new>
#include <initializer_list>

namespace {
struct HpsiCall : HpsiDeviceMaps {
    double2 *psi=nullptr,*hpsi=nullptr;
    size_t bytes=0;
    int stage=0;
};
int read_physical(const HpsiCall &c,double2 *out){
    if(!out)return -1;
    // One row is a physical XY plane; skip its extra padding Y row on both sides.
    return int(cudaMemcpy2D(out,16*17*sizeof(double2),c.hpsi,16*17*sizeof(double2),
        16*16*sizeof(double2),size_t(16)*c.maps.no*c.maps.nk,cudaMemcpyDeviceToHost));
}
}
extern "C" int salmon_hpsi_call_close(void *handle){
    auto *c=static_cast<HpsiCall*>(handle);if(!c)return 0;
    int result=0;
    for(auto *p:{c->psi,c->hpsi})if(p){int rc=int(cudaFree(p));if(!result)result=rc;}
    int rc=hpsi_maps_close(*c);if(!result)result=rc;
    delete c;return result;
}
extern "C" int salmon_hpsi_call_begin(const EpochInput *input,const double2 *psi,void **handle){
    if(!handle)return -1;*handle=nullptr;
    if(!psi)return -1;
    int rc=hpsi_validate(input);if(rc)return rc;
    rc=hpsi_device_guard();if(rc)return rc;
    auto *c=new(std::nothrow) HpsiCall;if(!c)return -3;
    c->bytes=size_t(input->no)*input->nk*16*17*16*sizeof(double2);
#define TRY(cmd) do{rc=int(cmd);if(rc){salmon_hpsi_call_close(c);return rc;}}while(0)
    TRY(hpsi_maps_begin(*input,*c));
    TRY(cudaMalloc(reinterpret_cast<void**>(&c->psi),c->bytes));
    TRY(cudaMalloc(reinterpret_cast<void**>(&c->hpsi),c->bytes));
    TRY(cudaMemcpy(c->psi,psi,c->bytes,cudaMemcpyHostToDevice));
    TRY(hpsi_stencil(*c,c->psi,c->hpsi));
    c->stage=1;*handle=c;return 0;
#undef TRY
}
extern "C" int salmon_hpsi_call_query(void *handle,size_t *fields,size_t *maps,size_t *coeff){
    auto *c=static_cast<HpsiCall*>(handle);
    if(!c||!fields||!maps||!coeff||c->stage!=1)return -1;
    *fields=2*c->bytes;*maps=c->map_bytes;*coeff=c->coeff_count*sizeof(double2);return 0;
}
extern "C" int salmon_hpsi_call_stencil_read(void *handle,double2 *out){
    auto *c=static_cast<HpsiCall*>(handle);if(!c||c->stage!=1)return -1;
    return read_physical(*c,out);
}
extern "C" int salmon_hpsi_call_finish(void *handle,double2 *out){
    auto *c=static_cast<HpsiCall*>(handle);if(!c||c->stage!=1)return -1;
    int rc=hpsi_nonlocal(*c,c->psi,c->hpsi);if(rc)return rc;
    rc=read_physical(*c,out);if(!rc)c->stage=2;return rc;
}
