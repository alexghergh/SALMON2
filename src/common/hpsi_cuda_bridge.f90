!
!  Copyright 2019-2020 SALMON developers
!
!  Licensed under the Apache License, Version 2.0 (the "License");
!  you may not use this file except in compliance with the License.
!  You may obtain a copy of the License at
!
!      http://www.apache.org/licenses/LICENSE-2.0
!
!  Unless required by applicable law or agreed to in writing, software
!  distributed under the License is distributed on an "AS IS" BASIS,
!  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
!  See the License for the specific language governing permissions and
!  limitations under the License.
!

! Optional CUDA Hpsi outside CG (SALMON_SHARED_HPSI), using the CPU-order stencil and nonlocal
! kernels of the CUDA CG.
module hpsi_cuda_bridge
  use iso_c_binding
  use structures, only: s_rgrid,s_dft_system,s_parallel_info,s_orbital,s_stencil,s_pp_grid,s_scalar
  use hpsi_cuda_inputs, only: hpsi_input,prepare_hpsi_inputs,owned_rectangle
  implicit none
  private
  public :: hpsi_cuda_state,hpsi_cuda_begin,hpsi_cuda_finish,hpsi_cuda_active
  type hpsi_cuda_state
    type(c_ptr) :: handle=c_null_ptr
  end type
  logical,save :: initialized=.false.,requested=.false.,energy_requested=.false.
  interface
    function device_guard() bind(C,name='salmon_cg_device_guard') result(rc)
      import c_int
      integer(c_int) :: rc
    end function
    function begin_gpu(input,x,handle) bind(C,name='salmon_hpsi_call_begin') result(rc)
      import hpsi_input,c_int,c_ptr,c_double_complex
      type(hpsi_input),intent(in) :: input
      complex(c_double_complex),intent(in) :: x(*)
      type(c_ptr),intent(out) :: handle
      integer(c_int) :: rc
    end function
    function query_gpu(handle,fields,maps,coeff) bind(C,name='salmon_hpsi_call_query') result(rc)
      import c_ptr,c_size_t,c_int
      type(c_ptr),value :: handle
      integer(c_size_t),intent(out) :: fields,maps,coeff
      integer(c_int) :: rc
    end function
    function stencil_read_gpu(handle,out) bind(C,name='salmon_hpsi_call_stencil_read') result(rc)
      import c_ptr,c_int,c_double_complex
      type(c_ptr),value :: handle
      complex(c_double_complex),intent(inout) :: out(*)
      integer(c_int) :: rc
    end function
    function finish_gpu(handle,out) bind(C,name='salmon_hpsi_call_finish') result(rc)
      import c_ptr,c_int,c_double_complex
      type(c_ptr),value :: handle
      complex(c_double_complex),intent(inout) :: out(*)
      integer(c_int) :: rc
    end function
    function energy_gpu(handle,out) bind(C,name='salmon_hpsi_call_energy') result(rc)
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      real(c_double),intent(out) :: out(*)
      integer(c_int) :: rc
    end function
    function close_gpu(handle) bind(C,name='salmon_hpsi_call_close') result(rc)
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int) :: rc
    end function
  end interface
contains
  logical function hpsi_cuda_active(s)
    type(hpsi_cuda_state),intent(in) :: s
    hpsi_cuda_active=c_associated(s%handle)
  end function

  subroutine require_success(s,rc,stage)
    type(hpsi_cuda_state),intent(inout) :: s
    integer(c_int),intent(in) :: rc
    character(*),intent(in) :: stage
    integer(c_int) :: cleanup_rc
    if(rc==0)return
    cleanup_rc=close_gpu(s%handle);s%handle=c_null_ptr
    write(*,*) 'HPSI_CUDA ERROR ',stage,' rc=',rc,' cleanup=',cleanup_rc
    error stop 'requested shared Hpsi failed'
  end subroutine

  subroutine hpsi_cuda_begin(s,mg,system,info,stencil,ppg,vlocal,x,hx,supported,tx,energy_sums,captured)
    type(hpsi_cuda_state),intent(inout) :: s
    type(s_rgrid),intent(in) :: mg
    type(s_dft_system),intent(in) :: system
    type(s_parallel_info),intent(in) :: info
    type(s_stencil),intent(in),target :: stencil
    type(s_pp_grid),intent(in),target :: ppg
    type(s_scalar),intent(in) :: vlocal(:)
    type(s_orbital),intent(in) :: x
    type(s_orbital),intent(inout) :: hx
    logical,intent(in) :: supported
    type(s_orbital),optional,intent(in) :: tx
    real(c_double),optional,intent(out) :: energy_sums(:,:,:)
    logical,optional,intent(out) :: captured
    type(hpsi_input) :: m
    integer(c_int),target :: idx(24),idy(24),idz(24)
    real(c_double),allocatable,target :: vl(:,:,:),lap0(:),nabt(:,:,:)
    integer(c_int) :: rc
    integer(c_size_t) :: fields,maps,coeff
    integer :: status
    logical :: eligible
    character(16) :: setting
    if(.not.initialized)then
      setting='';call get_environment_variable('SALMON_SHARED_HPSI',setting,status=status)
      if(status/=0.and.status/=1)error stop 'SALMON_SHARED_HPSI unreadable or truncated'
      select case(trim(setting))
      case('','0');requested=.false.
      case('1');requested=.true.
      case default;error stop 'SALMON_SHARED_HPSI must be 0 or 1'
      end select
      setting='';call get_environment_variable('SALMON_HPSI_ENERGY',setting,status=status)
      if(status/=0.and.status/=1)error stop 'SALMON_HPSI_ENERGY unreadable or truncated'
      select case(trim(setting))
      case('','0');energy_requested=.false.
      case('1');energy_requested=.true.
      case default;error stop 'SALMON_HPSI_ENERGY must be 0 or 1'
      end select
      initialized=.true.
    endif
    if(present(captured))captured=.false.
    if(.not.requested)return
    if(hpsi_cuda_active(s))error stop 'shared Hpsi called with live owner'
    if(.not.supported)then
      write(*,*) 'HPSI_CUDA FALLBACK physics rank=',info%id_rko
      return
    endif
    eligible=all(mg%is_array==mg%is).and.all(mg%ie_array==mg%ie+[0,1,0])
    eligible=eligible.and.owned_rectangle(x,mg,system,info,1).and.owned_rectangle(hx,mg,system,info,1)
    if(present(tx))eligible=eligible.and.owned_rectangle(tx,mg,system,info,1)
    if(.not.eligible)then
      write(*,*) 'HPSI_CUDA FALLBACK storage rank=',info%id_rko
      return
    endif
    call prepare_hpsi_inputs(mg,system,info,stencil,ppg,vlocal,m,idx,idy,idz,vl,lap0,nabt,eligible)
    if(.not.eligible)then
      write(*,*) 'HPSI_CUDA FALLBACK inputs rank=',info%id_rko
      return
    endif
    rc=device_guard()
    if(rc/=0)then
      write(*,*) 'HPSI_CUDA FALLBACK device rank=',info%id_rko,' rc=',rc
      return
    endif
    if(present(energy_sums).and.energy_requested)then
      if(.not.present(captured))error stop 'energy capture result missing'
      if(any(shape(energy_sums)/=[3,m%no,m%nk]))error stop 'energy result ownership shape'
    endif
    rc=begin_gpu(m,x%zwf,s%handle);call require_success(s,rc,'capture')
    rc=query_gpu(s%handle,fields,maps,coeff);call require_success(s,rc,'query')
    if(present(energy_sums).and.energy_requested)then
      rc=energy_gpu(s%handle,energy_sums);call require_success(s,rc,'energy sums')
      rc=query_gpu(s%handle,fields,maps,coeff);call require_success(s,rc,'energy query')
      rc=close_gpu(s%handle);s%handle=c_null_ptr
      call require_success(s,rc,'energy close')
      captured=.true.
      write(*,*) 'HPSI_ENERGY CAPTURE rank=',info%id_rko,' vectors=',m%no*m%nk
      write(*,*) 'HPSI_ENERGY BYTES fields maps coeff sums=',fields,maps,coeff,24*m%no*m%nk
      write(*,*) 'HPSI_ENERGY LAUNCH kinetic_threads=256 sums_threads=128 vectors_per_block=4'
      return
    endif
    if(present(tx))then
      ! Preserve the original CPU kinetic subtraction before nonlocal addition.
      rc=stencil_read_gpu(s%handle,hx%zwf);call require_success(s,rc,'stencil read')
    endif
    write(*,*) 'HPSI_CUDA CAPTURE rank=',info%id_rko,' vectors=',m%no*m%nk,' kinetic=',present(tx)
    write(*,*) 'HPSI_CUDA BYTES fields maps coeff=',fields,maps,coeff
  end subroutine

  subroutine hpsi_cuda_finish(s,hx)
    type(hpsi_cuda_state),intent(inout) :: s
    type(s_orbital),intent(inout) :: hx
    integer(c_int) :: rc
    if(.not.hpsi_cuda_active(s))error stop 'shared Hpsi finish without owner'
    rc=finish_gpu(s%handle,hx%zwf);call require_success(s,rc,'nonlocal/final read')
    rc=close_gpu(s%handle);s%handle=c_null_ptr
    call require_success(s,rc,'close')
    write(*,*) 'HPSI_CUDA END'
  end subroutine
end module hpsi_cuda_bridge
