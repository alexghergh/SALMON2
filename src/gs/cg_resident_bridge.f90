! Fortran side of the CUDA offload of gscg_zwf (SALMON_CG_CUDA). SALMON_CG_CUDA_MODE selects off,
! shadow (compute on both, stop on any bit difference, keep the CPU result) or port (use the GPU).
module cg_resident_bridge
  use iso_c_binding
  use structures, only: s_rgrid,s_dft_system,s_parallel_info,s_orbital
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: cg_region_state,cg_region_begin,cg_region_products_check,cg_region_finish
  public :: cg_region_update_check,cg_region_report,cg_region_real_fallback
  type cg_region_state
    type(c_ptr) :: handle=c_null_ptr
    integer :: mode=0
    complex(c_double_complex),allocatable :: six(:,:,:,:),norm(:,:,:)
    complex(c_double_complex),allocatable :: hx(:,:,:,:,:,:,:),g(:,:,:,:,:,:,:)
  end type
  logical,save :: initialized=.false.
  integer,save :: selected=0,fused=0
  ! Call counts per path: off, real orbitals, spin-orbit, OpenACC, divided r-space, layout,
  ! no device, shadow, port
  integer(c_int64_t),save :: paths(9)=0
  interface
    function device_guard() bind(C,name='salmon_cg_device_guard') result(rc)
      import c_int
      integer(c_int) :: rc
    end function
    function begin_gpu(nx,ny,nz,nv,pady,h,x,p,hx,hp,six,handle) bind(C,name='salmon_cg_begin') result(rc)
      import c_int,c_double,c_double_complex,c_ptr
      integer(c_int),value :: nx,ny,nz,nv,pady
      real(c_double),value :: h
      complex(c_double_complex),intent(in) :: x(*),p(*),hx(*),hp(*)
      complex(c_double_complex),intent(out) :: six(*)
      type(c_ptr),intent(out) :: handle
      integer(c_int) :: rc
    end function
    function finish_gpu(handle,u,e,hx,g,norm,fuse) bind(C,name='salmon_cg_finish') result(rc)
      import c_int,c_double,c_double_complex,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: fuse
      complex(c_double_complex),intent(in) :: u(*)
      real(c_double),intent(in) :: e(*)
      complex(c_double_complex),intent(out) :: hx(*),g(*),norm(*)
      integer(c_int) :: rc
    end function
    function close_gpu(handle) bind(C,name='salmon_cg_close') result(rc)
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int) :: rc
    end function
  end interface
contains
  subroutine settings()
    character(32) :: setting
    integer :: status
    if(initialized)return
    setting='';call get_environment_variable('SALMON_CG_CUDA_MODE',setting,status=status)
    if(status/=0.and.status/=1)error stop 'SALMON_CG_CUDA_MODE unreadable or truncated'
    select case(trim(setting))
    case('','off'); selected=0
    case('shadow'); selected=1
    case('port'); selected=2
    case default; error stop 'SALMON_CG_CUDA_MODE must be off, shadow, or port'
    end select
    setting='';call get_environment_variable('SALMON_CG_CUDA_FUSED',setting,status=status)
    if(status/=0.and.status/=1)error stop 'SALMON_CG_CUDA_FUSED unreadable or truncated'
    select case(trim(setting))
    case('','0'); fused=0
    case('1'); error stop 'Fused update/norm is rejected in this source trial'
    case default; error stop 'SALMON_CG_CUDA_FUSED must be 0 or 1'
    end select
    write(*,*) 'CG_CUDA CONFIG mode(0=off,1=shadow,2=port)=',selected,' fused=',fused
    initialized=.true.
  end subroutine

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

  subroutine cg_region_begin(s,mg,system,info,x,p,hx,hp,g,spinorbit,products)
    type(cg_region_state),intent(inout) :: s
    type(s_rgrid),intent(in) :: mg
    type(s_dft_system),intent(in) :: system
    type(s_parallel_info),intent(in) :: info
    type(s_orbital),intent(in) :: x,p,hx,hp,g
    character(*),intent(in) :: spinorbit
    complex(8),intent(inout) :: products(:,:,:,:)
    complex(c_double_complex),allocatable :: packed(:,:)
    integer :: nv,v,ik,io,ispin,ns(3),pady
    integer(c_int) :: rc
    integer(c_int64_t) :: nv64
    call settings()
    if(c_associated(s%handle))error stop 'CG context already active'
    s%mode=0
    if(selected==0)then
      paths(1)=paths(1)+1;return
    endif
    if(spinorbit/='n')then
      paths(3)=paths(3)+1;return
    endif
#ifdef USE_OPENACC
    ! Host arrays may be stale inside an OpenACC data region, so OpenACC builds use the CPU path.
    paths(4)=paths(4)+1;return
#endif
    if(info%isize_r/=1)then
      paths(5)=paths(5)+1;return
    endif
    if(.not.allocated(x%zwf))then
      paths(6)=paths(6)+1;return
    endif
    pady=ubound(x%zwf,2)-mg%ie(2)
    if(pady<0.or.pady>1)then
      paths(6)=paths(6)+1;return
    endif
    if(.not.owned_rectangle(x,mg,system,info,pady).or..not.owned_rectangle(p,mg,system,info,pady).or. &
       .not.owned_rectangle(hx,mg,system,info,pady).or..not.owned_rectangle(hp,mg,system,info,pady).or. &
       .not.owned_rectangle(g,mg,system,info,pady))then
      paths(6)=paths(6)+1;return
    endif
    ns=mg%ie-mg%is+1
    nv64=int(system%nspin,c_int64_t)*int(info%io_e-info%io_s+1,c_int64_t)* &
         int(info%ik_e-info%ik_s+1,c_int64_t)
    if(any(ns<=0).or.nv64<=0.or.nv64>huge(0_c_int))error stop 'invalid CG descriptor'
    if(info%io_s<1.or.info%io_e>system%no.or.info%ik_s<1.or.info%ik_e>system%nk) &
      error stop 'invalid CG ownership bounds'
    rc=device_guard()
    if(rc/=0)then
      paths(7)=paths(7)+1
      if(paths(7)==1)write(*,*) 'CG_CUDA device fallback status=',rc,' rank=',info%id_rko
      return
    endif
    nv=int(nv64);allocate(packed(6,nv))
    rc=begin_gpu(ns(1),ns(2),ns(3),nv,pady,system%hvol,x%zwf,p%zwf,hx%zwf,hp%zwf,packed,s%handle)
    call require_success(rc,'begin')
    allocate(s%six(6,system%nspin,system%no,system%nk));s%six=0.d0
    v=0
    do ik=info%ik_s,info%ik_e
    do io=info%io_s,info%io_e
    do ispin=1,system%nspin
      v=v+1;s%six(:,ispin,io,ik)=packed(:,v)
    enddo
    enddo
    enddo
    ! icomm_r has size one. No orbital/k collective belongs here; unowned slots remain zero.
    if(paths(7+selected)==0)then
      write(*,*) 'CG_CUDA LAYOUT physical=',ns,' pad_y=',pady,' owned_vectors=',nv
      write(*,*) 'CG_CUDA OWNERSHIP orbital=',info%io_s,info%io_e,' k=',info%ik_s,info%ik_e
    endif
    s%mode=selected;paths(7+selected)=paths(7+selected)+1
    if(s%mode==2)products=s%six
  end subroutine

  subroutine cg_region_products_check(s,products)
    type(cg_region_state),intent(inout) :: s
    complex(8),intent(in) :: products(:,:,:,:)
    if(s%mode==1)call exact('six products',reshape(s%six,[size(s%six)]),reshape(products,[size(products)]))
    if(allocated(s%six))deallocate(s%six)
  end subroutine

  subroutine cg_region_finish(s,system,info,u,e,hx,g,norm)
    type(cg_region_state),intent(inout) :: s
    type(s_dft_system),intent(in) :: system
    type(s_parallel_info),intent(in) :: info
    complex(8),intent(in) :: u(:,:,:,:)
    real(8),intent(in) :: e(:,:,:)
    type(s_orbital),intent(inout) :: hx,g
    complex(8),intent(inout) :: norm(:,:,:)
    complex(c_double_complex),allocatable :: up(:,:),np(:)
    real(c_double),allocatable :: ep(:)
    integer :: v,nv,ik,io,ispin
    integer(c_int) :: rc,close_rc
    if(s%mode==0)return
    nv=system%nspin*(info%io_e-info%io_s+1)*(info%ik_e-info%ik_s+1)
    allocate(up(2,nv),ep(nv),np(nv));v=0
    do ik=info%ik_s,info%ik_e
    do io=info%io_s,info%io_e
    do ispin=1,system%nspin
      v=v+1;up(:,v)=u(:,ispin,io,ik);ep(v)=e(ispin,io,ik)
    enddo
    enddo
    enddo
    if(s%mode==1)then
      allocate(s%hx,mold=hx%zwf);allocate(s%g,mold=g%zwf)
      s%g=g%zwf ! Preserve trailing y padding, which CUDA intentionally never writes.
      rc=finish_gpu(s%handle,up,ep,s%hx,s%g,np,fused)
    else
      rc=finish_gpu(s%handle,up,ep,hx%zwf,g%zwf,np,fused)
    endif
    close_rc=close_gpu(s%handle);s%handle=c_null_ptr
    call require_success(rc,'finish');call require_success(close_rc,'close')
    allocate(s%norm(system%nspin,system%no,system%nk));s%norm=0.d0;v=0
    do ik=info%ik_s,info%ik_e
    do io=info%io_s,info%io_e
    do ispin=1,system%nspin
      v=v+1;s%norm(ispin,io,ik)=np(v)
    enddo
    enddo
    enddo
    if(s%mode==2)norm=s%norm
  end subroutine

  subroutine cg_region_update_check(s,hx,g,norm)
    type(cg_region_state),intent(inout) :: s
    type(s_orbital),intent(in) :: hx,g
    complex(8),intent(in) :: norm(:,:,:)
    if(s%mode==1)then
      call exact('Hx',reshape(s%hx,[size(s%hx)]),reshape(hx%zwf,[size(hx%zwf)]),.false.)
      call exact('gradient',reshape(s%g,[size(s%g)]),reshape(g%zwf,[size(g%zwf)]),.false.)
      call exact('norm',reshape(s%norm,[size(s%norm)]),reshape(norm,[size(norm)]))
      deallocate(s%hx,s%g)
    endif
    if(allocated(s%norm))deallocate(s%norm)
  end subroutine

  subroutine exact(label,a,b,check_finite)
    character(*),intent(in) :: label
    complex(8),intent(in) :: a(:),b(:)
    logical,intent(in),optional :: check_finite
    logical :: finite_required
    ! Padding may hold any value, NaN included: compare it bit by bit, but do not require it finite.
    finite_required=.true.
    if(present(check_finite))finite_required=check_finite
    if(size(a)/=size(b))error stop 'CG shadow shape mismatch'
    if(finite_required)then
    if(.not.all(ieee_is_finite(real(a))).or..not.all(ieee_is_finite(aimag(a))).or. &
       .not.all(ieee_is_finite(real(b))).or..not.all(ieee_is_finite(aimag(b)))) &
      error stop 'CG shadow nonfinite'
    endif
    if(any(transfer(a,[0_c_int64_t])/=transfer(b,[0_c_int64_t])))then
      write(*,*) 'CG_CUDA SHADOW_MISMATCH ',label
      error stop 'CG shadow bitwise mismatch; no promotion'
    endif
  end subroutine

  subroutine require_success(rc,stage)
    integer(c_int),intent(in) :: rc
    character(*),intent(in) :: stage
    if(rc==0)return
    write(*,*) 'CG_CUDA ERROR stage=',stage,' status=',rc
    error stop 'CG CUDA failure; no partial CPU recovery'
  end subroutine

  subroutine cg_region_real_fallback(rank)
    integer,intent(in) :: rank
    call settings()
    if(selected/=0)then
      paths(2)=paths(2)+1;call cg_region_report(rank)
    endif
  end subroutine

  subroutine cg_region_report(rank)
    integer,intent(in) :: rank
    if(.not.initialized)return
    write(*,*) 'CG_CUDA PATH_COUNTS rank=',rank,' selected_mode=',selected,' fused=',fused
    write(*,*) 'CG_CUDA off real spinorbit openacc spatial layout device shadow port=',paths
  end subroutine
end module
