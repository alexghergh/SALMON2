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

! Host-side inputs (grid index maps, k coefficients, projector maps) for the CUDA Hpsi kernels.
! No device allocation here; borrowed pointers survive until the synchronous capture returns.
module hpsi_cuda_inputs
  use iso_c_binding
  use structures, only: s_rgrid,s_dft_system,s_parallel_info,s_orbital,s_stencil,s_pp_grid,s_scalar
  implicit none
  private
  public :: hpsi_input,prepare_hpsi_inputs,owned_rectangle
  ! Host pointers live only for the synchronous capture; CUDA owns its copies thereafter.
  type,bind(C) :: hpsi_input
    integer(c_int) :: lo(3),hi(3),no,nk,nlma,nps,natom,max_vi,max_contrib
    type(c_ptr) :: idx,idy,idz,vlocal,lap0,lapt,nabt,ia_tbl,mps,jxyz,uv,rinv,v2j,v2nlma,k2ilma,k2j
  end type
contains
  logical function owned_rectangle(w,mg,system,info,pady) result(ok)
    type(s_orbital),intent(in) :: w
    type(s_rgrid),intent(in) :: mg
    type(s_dft_system),intent(in) :: system
    type(s_parallel_info),intent(in) :: info
    integer,intent(in) :: pady
    ok=.false.
    if(.not.allocated(w%zwf))return
    ok=all(lbound(w%zwf)==[mg%is,1,info%io_s,info%ik_s,1]).and. &
       all(ubound(w%zwf)==[mg%ie+[0,pady,0],system%nspin,info%io_e,info%ik_e,1])
  end function

  subroutine prepare_hpsi_inputs(mg,system,info,stencil,ppg,vlocal,m,idx,idy,idz,vl,lap0,nabt,ok)
    type(s_rgrid),intent(in) :: mg
    type(s_dft_system),intent(in) :: system
    type(s_parallel_info),intent(in) :: info
    type(s_stencil),intent(in),target :: stencil
    type(s_pp_grid),intent(in),target :: ppg
    type(s_scalar),intent(in) :: vlocal(:)
    type(hpsi_input),intent(out) :: m
    integer(c_int),intent(out),target :: idx(24),idy(24),idz(24)
    real(c_double),allocatable,intent(out),target :: vl(:,:,:),lap0(:),nabt(:,:,:)
    logical,intent(out) :: ok
    real(c_double) :: kAc(3)
    integer :: no,nk,natom,ik,k
    ok=.false.
    if(info%isize_r/=1.or.info%if_divide_rspace.or.system%nspin/=1)return
    if(info%im_s/=1.or.info%im_e/=1)return
    if(.not.stencil%if_orthogonal.or.allocated(system%Ac_micro%v))return
    if(any(mg%ie-mg%is+1/=16))return
    if(info%io_s<1.or.info%io_e>system%no.or.info%ik_s<1.or.info%ik_e>system%nk)return
    no=info%io_e-info%io_s+1;nk=info%ik_e-info%ik_s+1
    if(no<=0.or.nk<=0)return
    if(int(no,c_int64_t)*int(nk,c_int64_t)>int(huge(0_c_int),c_int64_t)/32)return
    if(.not.allocated(mg%idx).or..not.allocated(mg%idy).or..not.allocated(mg%idz))return
    if(lbound(mg%idx,1)>mg%is(1)-4.or.ubound(mg%idx,1)<mg%ie(1)+4.or. &
       lbound(mg%idy,1)>mg%is(2)-4.or.ubound(mg%idy,1)<mg%ie(2)+4.or. &
       lbound(mg%idz,1)>mg%is(3)-4.or.ubound(mg%idz,1)<mg%ie(3)+4)return
    if(size(vlocal)/=1)return
    if(.not.allocated(vlocal(1)%f))return
    if(any(lbound(vlocal(1)%f)>mg%is).or.any(ubound(vlocal(1)%f)<mg%ie))return
    if(.not.allocated(system%vec_k))return
    if(lbound(system%vec_k,1)>1.or.ubound(system%vec_k,1)<3.or. &
       lbound(system%vec_k,2)>info%ik_s.or.ubound(system%vec_k,2)<info%ik_e)return
    if(.not.allocated(ppg%ia_tbl).or..not.allocated(ppg%mps).or..not.allocated(ppg%jxyz).or. &
       .not.allocated(ppg%zekr_uv).or..not.allocated(ppg%rinv_uvu).or. &
       .not.allocated(ppg%v2j).or..not.allocated(ppg%v2nlma).or. &
       .not.allocated(ppg%k2ilma).or..not.allocated(ppg%k2j))return
    natom=size(ppg%mps)
    if(ppg%nps<=0.or.ppg%nlma<=0.or.natom<=0.or.ppg%max_vi<=0)return
    ! init_ps reserves projector capacity; only the first nlma entries are active.
    ! These rank-one arrays are copied by that active count, independent of capacity.
    if(lbound(ppg%ia_tbl,1)/=1.or.size(ppg%ia_tbl)<ppg%nlma.or. &
       lbound(ppg%mps,1)/=1.or.lbound(ppg%rinv_uvu,1)/=1.or.size(ppg%rinv_uvu)<ppg%nlma)return
    ! Multidimensional borrowed sections still require the exact leading dimensions.
    if(any(lbound(ppg%jxyz)/=[1,1,1]).or.any(shape(ppg%jxyz)/=[3,ppg%nps,natom]))return
    if(lbound(ppg%zekr_uv,1)/=1.or.lbound(ppg%zekr_uv,2)/=1)return
    if(size(ppg%zekr_uv,1)/=ppg%nps.or.size(ppg%zekr_uv,2)/=ppg%nlma.or. &
       lbound(ppg%zekr_uv,3)>info%ik_s.or.ubound(ppg%zekr_uv,3)<info%ik_e)return
    if(any(lbound(ppg%v2j)/=[1,0]).or.size(ppg%v2j,1)/=3.or.size(ppg%v2j,2)<ppg%max_vi)return
    if(lbound(ppg%v2nlma,1)/=0.or.size(ppg%v2nlma)/=ppg%max_vi)return
    if(any(lbound(ppg%k2ilma)/=[0,1]).or.any(lbound(ppg%k2j)/=[0,1]))return
    if(size(ppg%k2ilma,1)/=ppg%max_vi.or.any(shape(ppg%k2ilma)/=shape(ppg%k2j)))return
    if(size(ppg%k2ilma,2)<=0)return
    allocate(vl(16,16,16),lap0(nk),nabt(4,3,nk))
    vl=vlocal(1)%f(mg%is(1):mg%ie(1),mg%is(2):mg%ie(2),mg%is(3):mg%ie(3))
    idx=mg%idx(mg%is(1)-4:mg%ie(1)+4)
    idy=mg%idy(mg%is(2)-4:mg%ie(2)+4)
    idz=mg%idz(mg%is(3)-4:mg%ie(3)+4)
    do ik=info%ik_s,info%ik_e
      k=ik-info%ik_s+1
      ! The k-dependent coefficients of hpsi ('zstencil loop B'), evaluated on the host.
      kAc(1:3)=system%vec_k(1:3,ik)+system%vec_Ac(1:3)
      lap0(k)=stencil%coef_lap0+0.5d0*sum(kAc(1:3)**2)
      nabt(:,1,k)=kAc(1)*stencil%coef_nab(:,1)
      nabt(:,2,k)=kAc(2)*stencil%coef_nab(:,2)
      nabt(:,3,k)=kAc(3)*stencil%coef_nab(:,3)
    enddo
    m%lo=mg%is;m%hi=mg%ie;m%no=no;m%nk=nk;m%nlma=ppg%nlma;m%nps=ppg%nps
    m%natom=natom;m%max_vi=ppg%max_vi;m%max_contrib=size(ppg%k2ilma,2)
    m%idx=c_loc(idx);m%idy=c_loc(idy);m%idz=c_loc(idz)
    m%vlocal=c_loc(vl);m%lap0=c_loc(lap0);m%lapt=c_loc(stencil%coef_lap);m%nabt=c_loc(nabt)
    m%ia_tbl=c_loc(ppg%ia_tbl);m%mps=c_loc(ppg%mps);m%jxyz=c_loc(ppg%jxyz)
    m%uv=c_loc(ppg%zekr_uv(1,1,info%ik_s));m%rinv=c_loc(ppg%rinv_uvu)
    m%v2j=c_loc(ppg%v2j);m%v2nlma=c_loc(ppg%v2nlma)
    m%k2ilma=c_loc(ppg%k2ilma);m%k2j=c_loc(ppg%k2j)
    ok=.true.
  end subroutine
end module hpsi_cuda_inputs
