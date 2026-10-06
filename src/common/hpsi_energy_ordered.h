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
#include <cuda_runtime.h>
// Borrowed padded 16x17x16 fields; physical potential has 16x16x16 entries.
// Only physical points are consumed/written. Output is sums(3,no,nk), real FP64.
int hpsi_energy_kinetic(int nv,const double *potential,const double2 *psi,
                        const double2 *stencil,double2 *kinetic);
int hpsi_energy_sums(int nv,const double *potential,const double2 *psi,
                     const double2 *hpsi,const double2 *kinetic,double *sums);
