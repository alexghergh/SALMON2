! Fortran side of the CUDA offload of gscg_zwf (SALMON_CG_CUDA). SALMON_CG_CUDA_MODE selects off,
! shadow (compute on both, stop on any bit difference, keep the CPU result) or port (use the GPU).
module cg_resident_bridge
  use iso_c_binding
  use hpsi_cuda_inputs, only: epoch_input=>hpsi_input,prepare_hpsi_inputs,owned_rectangle
  use structures, only: s_rgrid,s_dft_system,s_parallel_info,s_orbital,s_stencil,s_pp_grid,s_scalar
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: cg_region_state,cg_region_begin,cg_region_products_check,cg_region_finish
  public :: cg_region_update_check,cg_region_report,cg_region_real_fallback
  public :: cg_region_accept,cg_region_end
  public :: cg_epoch_begin,cg_epoch_hpsi,cg_epoch_initial,cg_epoch_initial_check,cg_epoch_energy_check
  public :: cg_epoch_norm,cg_epoch_norm_check,cg_epoch_direction,cg_epoch_field_check,cg_epoch_end
  type cg_region_state
    type(c_ptr) :: handle=c_null_ptr
    integer :: mode=0
    logical :: resident=.false.,awaiting_accept=.false.,epoch=.false.,scratch_reused=.false.
    integer :: epoch_iterations=0
    integer :: next_iteration=0,layout(18)=0
    integer(c_int64_t) :: hvol_bits=0
    complex(c_double_complex),allocatable :: six(:,:,:,:),norm(:,:,:),initial_energy(:,:,:)
    complex(c_double_complex),allocatable :: hx(:,:,:,:,:,:,:),g(:,:,:,:,:,:,:)
  end type
  logical,save :: initialized=.false.
  integer,save :: selected=0,fused=0,retain=0,epoch_requested=0
  ! Call counts per path: off, real orbitals, spin-orbit, OpenACC, divided r-space, layout,
  ! no device, shadow, port
  integer(c_int64_t),save :: paths(9)=0
  interface
    function epoch_available() bind(C,name='salmon_cg_epoch_available') result(rc)
      import c_int
      integer(c_int) :: rc
    end function
    function epoch_begin_gpu(input,h,x,handle) bind(C,name='salmon_cg_epoch_begin') result(rc)
      import c_int,c_double,c_double_complex,c_ptr,epoch_input
      type(epoch_input),intent(in) :: input
      real(c_double),value :: h
      complex(c_double_complex),intent(in) :: x(*)
      type(c_ptr),intent(out) :: handle
      integer(c_int) :: rc
    end function
    function epoch_begin_scratch_gpu(input,h,x,handle) bind(C,name='salmon_cg_epoch_begin_scratch') result(rc)
      import c_int,c_double,c_double_complex,c_ptr,epoch_input
      type(epoch_input),intent(in) :: input
      real(c_double),value :: h
      complex(c_double_complex),intent(in) :: x(*)
      type(c_ptr),intent(out) :: handle
      integer(c_int) :: rc
    end function
    function epoch_query_gpu(handle,fields,maps,coeff) bind(C,name='salmon_cg_epoch_query') result(rc)
      import c_int,c_ptr,c_size_t
      type(c_ptr),value :: handle
      integer(c_size_t),intent(out) :: fields,maps,coeff
      integer(c_int) :: rc
    end function
    function epoch_hpsi_gpu(handle,initial) bind(C,name='salmon_cg_epoch_hpsi') result(rc)
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: initial
      integer(c_int) :: rc
    end function
    function epoch_initial_gpu(handle,e,norm) bind(C,name='salmon_cg_epoch_initial') result(rc)
      import c_int,c_ptr,c_double_complex
      type(c_ptr),value :: handle
      complex(c_double_complex),intent(out) :: e(*),norm(*)
      integer(c_int) :: rc
    end function
    function epoch_norm_gpu(handle,norm) bind(C,name='salmon_cg_epoch_norm') result(rc)
      import c_int,c_ptr,c_double_complex
      type(c_ptr),value :: handle
      complex(c_double_complex),intent(out) :: norm(*)
      integer(c_int) :: rc
    end function
    function epoch_direction_gpu(handle,iteration,bk) bind(C,name='salmon_cg_epoch_direction') result(rc)
      import c_int,c_ptr,c_double
      type(c_ptr),value :: handle
      integer(c_int),value :: iteration
      real(c_double),intent(in) :: bk(*)
      integer(c_int) :: rc
    end function
    function epoch_products_gpu(handle,six) bind(C,name='salmon_cg_epoch_products') result(rc)
      import c_int,c_ptr,c_double_complex
      type(c_ptr),value :: handle
      complex(c_double_complex),intent(out) :: six(*)
      integer(c_int) :: rc
    end function
    function epoch_finish_gpu(handle,u,e,norm) bind(C,name='salmon_cg_epoch_finish') result(rc)
      import c_int,c_ptr,c_double,c_double_complex
      type(c_ptr),value :: handle
      complex(c_double_complex),intent(in) :: u(*)
      real(c_double),intent(in) :: e(*)
      complex(c_double_complex),intent(out) :: norm(*)
      integer(c_int) :: rc
    end function
    function epoch_read_gpu(handle,field,out) bind(C,name='salmon_cg_epoch_read') result(rc)
      import c_int,c_ptr,c_double_complex
      type(c_ptr),value :: handle
      integer(c_int),value :: field
      complex(c_double_complex),intent(inout) :: out(*)
      integer(c_int) :: rc
    end function
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
    function refresh_gpu(handle,p,hp,six) bind(C,name='salmon_cg_refresh') result(rc)
      import c_int,c_double_complex,c_ptr
      type(c_ptr),value :: handle
      complex(c_double_complex),intent(in) :: p(*),hp(*)
      complex(c_double_complex),intent(out) :: six(*)
      integer(c_int) :: rc
    end function
    function retained_finish_gpu(handle,u,e,hx,g,norm,copy_hx) bind(C,name='salmon_cg_retained_finish') result(rc)
      import c_int,c_double,c_double_complex,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: copy_hx
      complex(c_double_complex),intent(in) :: u(*)
      real(c_double),intent(in) :: e(*)
      complex(c_double_complex),intent(inout) :: hx(*),g(*)
      complex(c_double_complex),intent(out) :: norm(*)
      integer(c_int) :: rc
    end function
    function accept_gpu(handle,bk) bind(C,name='salmon_cg_accept') result(rc)
      import c_int,c_double,c_ptr
      type(c_ptr),value :: handle
      real(c_double),intent(in) :: bk(*)
      integer(c_int) :: rc
    end function
    function snapshot_gpu(handle,x,hx) bind(C,name='salmon_cg_snapshot') result(rc)
      import c_int,c_double_complex,c_ptr
      type(c_ptr),value :: handle
      complex(c_double_complex),intent(out) :: x(*),hx(*)
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
    setting='';call get_environment_variable('SALMON_CG_CUDA_RETAIN',setting,status=status)
    if(status/=0.and.status/=1)error stop 'SALMON_CG_CUDA_RETAIN unreadable or truncated'
    select case(trim(setting))
    case('','0'); retain=0
    case('1'); retain=1
    case default; error stop 'SALMON_CG_CUDA_RETAIN must be 0 or 1'
    end select
    setting='';call get_environment_variable('SALMON_CG_CUDA_EPOCH',setting,status=status)
    if(status/=0.and.status/=1)error stop 'SALMON_CG_CUDA_EPOCH unreadable or truncated'
    select case(trim(setting))
    case('','0'); epoch_requested=0
    case('1'); epoch_requested=1
    case default; error stop 'SALMON_CG_CUDA_EPOCH must be 0 or 1'
    end select
    write(*,*) 'CG_CUDA EPOCH requested=',epoch_requested,' compiled=',epoch_available()
    write(*,*) 'CG_CUDA RETAIN=',retain
    write(*,*) 'CG_CUDA CONFIG mode(0=off,1=shadow,2=port)=',selected,' fused=',fused
    initialized=.true.
  end subroutine

  subroutine cg_region_begin(s,mg,system,info,x,p,hx,hp,g,spinorbit,products,iteration)
    type(cg_region_state),intent(inout) :: s
    type(s_rgrid),intent(in) :: mg
    type(s_dft_system),intent(in) :: system
    type(s_parallel_info),intent(in) :: info
    type(s_orbital),intent(in) :: x,p,hx,hp,g
    character(*),intent(in) :: spinorbit
    complex(8),intent(inout) :: products(:,:,:,:)
    integer,intent(in),optional :: iteration
    complex(c_double_complex),allocatable :: packed(:,:)
    integer :: nv,v,ik,io,ispin,ns(3),pady,key(18)
    integer(c_int) :: rc
    integer(c_int64_t) :: nv64
    call settings()
    if(c_associated(s%handle))then
      if(.not.s%resident.or..not.present(iteration))call context_error(s,'CG context already active')
      if(iteration/=s%next_iteration.or.s%awaiting_accept)call context_error(s,'CG iteration sequence changed')
      ! These values and arrays do not change within one gscg_zwf call. Once x/Hx live on the GPU,
      ! a mismatch is an error: a CPU fallback would read stale host copies of x/Hx.
      pady=s%layout(18)
      key=[mg%is,mg%ie,system%nspin,system%no,system%nk,info%io_s,info%io_e, &
           info%ik_s,info%ik_e,info%isize_r,info%id_rko,info%im_s,info%im_e,pady]
      if(any(key/=s%layout).or.transfer(system%hvol,0_c_int64_t)/=s%hvol_bits.or. &
         spinorbit/='n'.or.selected/=s%mode)call context_error(s,'CG resident descriptor changed')
      if(.not.owned_rectangle(x,mg,system,info,pady).or..not.owned_rectangle(p,mg,system,info,pady).or. &
         .not.owned_rectangle(hx,mg,system,info,pady).or..not.owned_rectangle(hp,mg,system,info,pady).or. &
         .not.owned_rectangle(g,mg,system,info,pady))call context_error(s,'CG resident storage changed')
    endif
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
      if(c_associated(s%handle))call context_error(s,'CG resident device changed')
      paths(7)=paths(7)+1
      if(paths(7)==1)write(*,*) 'CG_CUDA device fallback status=',rc,' rank=',info%id_rko
      return
    endif
    nv=int(nv64);allocate(packed(6,nv))
    if(c_associated(s%handle))then
      if(s%epoch)then
        rc=epoch_products_gpu(s%handle,packed)
      else
        rc=refresh_gpu(s%handle,p%zwf,hp%zwf,packed)
      endif
    else
      rc=begin_gpu(ns(1),ns(2),ns(3),nv,pady,system%hvol,x%zwf,p%zwf,hx%zwf,hp%zwf,packed,s%handle)
      s%resident=retain==1.and.present(iteration)
      if(s%resident)then
        if(iteration<1)call context_error(s,'CG invalid initial iteration')
        s%next_iteration=iteration
        s%layout=[mg%is,mg%ie,system%nspin,system%no,system%nk,info%io_s,info%io_e, &
                  info%ik_s,info%ik_e,info%isize_r,info%id_rko,info%im_s,info%im_e,pady]
        s%hvol_bits=transfer(system%hvol,0_c_int64_t)
      endif
    endif
    call require_success(rc,'begin/refresh',s)
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
    if(s%resident.and.s%awaiting_accept)call context_error(s,'CG repeated resident finish')
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
      if(s%epoch)then
        s%hx=hx%zwf
        rc=epoch_finish_gpu(s%handle,up,ep,np);call require_success(rc,'epoch finish',s)
        rc=epoch_read_gpu(s%handle,1_c_int,s%hx);call require_success(rc,'shadow epoch Hx',s)
        rc=epoch_read_gpu(s%handle,2_c_int,s%g)
      else if(s%resident)then
        rc=retained_finish_gpu(s%handle,up,ep,s%hx,s%g,np,1_c_int)
      else
        rc=finish_gpu(s%handle,up,ep,s%hx,s%g,np,fused)
      endif
    else
      if(s%epoch)then
        rc=epoch_finish_gpu(s%handle,up,ep,np)
      else if(s%resident)then
        rc=retained_finish_gpu(s%handle,up,ep,hx%zwf,g%zwf,np,0_c_int)
      else
        rc=finish_gpu(s%handle,up,ep,hx%zwf,g%zwf,np,fused)
      endif
    endif
    if(s%resident)then
      call require_success(rc,'retained finish',s)
      s%awaiting_accept=.true.
    else
      close_rc=close_gpu(s%handle);s%handle=c_null_ptr
      call require_success(rc,'finish');call require_success(close_rc,'close')
    endif
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

  subroutine cg_region_accept(s,system,info,bk,x)
    type(cg_region_state),intent(inout) :: s
    type(s_dft_system),intent(in) :: system
    type(s_parallel_info),intent(in) :: info
    real(8),intent(in) :: bk(:,:,:)
    type(s_orbital),intent(in) :: x
    real(c_double),allocatable :: mask(:)
    complex(c_double_complex),allocatable :: xp(:,:,:,:,:,:,:),hp(:,:,:,:,:,:,:)
    integer :: nv,v,ik,io,ispin
    integer(c_int) :: rc
    if(.not.s%resident)return
    if(.not.s%awaiting_accept)call context_error(s,'CG missing resident finish')
    nv=system%nspin*(info%io_e-info%io_s+1)*(info%ik_e-info%ik_s+1)
    allocate(mask(nv));v=0
    do ik=info%ik_s,info%ik_e
    do io=info%io_s,info%io_e
    do ispin=1,system%nspin
      v=v+1;mask(v)=bk(ispin,io,ik)
    enddo
    enddo
    enddo
    rc=accept_gpu(s%handle,mask);call require_success(rc,'accepted x',s)
    if(s%mode==1)then
      ! Diagnostic copies are intentional in shadow mode, including padding bits.
      allocate(xp,mold=x%zwf);allocate(hp,mold=x%zwf)
      if(s%epoch)then
        rc=epoch_read_gpu(s%handle,0_c_int,xp)
      else
        rc=snapshot_gpu(s%handle,xp,hp)
      endif
      call require_success(rc,'shadow x',s)
      call exact('accepted x',reshape(xp,[size(xp)]),reshape(x%zwf,[size(x%zwf)]),.false.)
    endif
    s%awaiting_accept=.false.;s%next_iteration=s%next_iteration+1
  end subroutine

  subroutine cg_region_end(s,x,hx)
    type(cg_region_state),intent(inout) :: s
    type(s_orbital),intent(inout) :: x,hx
    complex(c_double_complex),allocatable :: xp(:,:,:,:,:,:,:),hp(:,:,:,:,:,:,:)
    integer(c_int) :: rc,close_rc
    if(.not.c_associated(s%handle))return
    if(s%epoch)call context_error(s,'epoch requires complete field flush')
    if(.not.s%resident.or.s%awaiting_accept)call context_error(s,'CG incomplete resident iteration')
    if(s%mode==1)then
      allocate(xp,mold=x%zwf);allocate(hp,mold=hx%zwf)
      rc=snapshot_gpu(s%handle,xp,hp)
    else
      rc=snapshot_gpu(s%handle,x%zwf,hx%zwf)
    endif
    close_rc=close_gpu(s%handle);s%handle=c_null_ptr;s%resident=.false.
    call require_success(rc,'final x/Hx');call require_success(close_rc,'final close')
    if(s%mode==1)then
      call exact('final x',reshape(xp,[size(xp)]),reshape(x%zwf,[size(x%zwf)]),.false.)
      call exact('final Hx',reshape(hp,[size(hp)]),reshape(hx%zwf,[size(hx%zwf)]),.false.)
    endif
    s%mode=0;s%next_iteration=0;s%layout=0;s%hvol_bits=0
  end subroutine

  ! Capture is allowed only before the first Hpsi. Every unsupported route returns
  ! with all host fields untouched. Once captured, all failures are fatal.
  subroutine cg_epoch_begin(s,mg,system,info,stencil,ppg,vlocal,x,p,hx,hp,g,pre_g,hpsi_supported, &
                            retain_workspace_outputs)
    type(cg_region_state),intent(inout) :: s
    type(s_rgrid),intent(in) :: mg
    type(s_dft_system),intent(in) :: system
    type(s_parallel_info),intent(in) :: info
    type(s_stencil),intent(in),target :: stencil
    type(s_pp_grid),intent(in),target :: ppg
    type(s_scalar),intent(in) :: vlocal(:)
    type(s_orbital),intent(in) :: x,p,hx,hp,g,pre_g
    logical,intent(in) :: hpsi_supported
    logical,optional,intent(in) :: retain_workspace_outputs
    logical :: reuse_scratch
    type(epoch_input) :: m
    integer(c_int),target :: idx(24),idy(24),idz(24)
    real(c_double),allocatable,target :: vl(:,:,:),lap0(:),nabt(:,:,:)
    integer :: nk,no,pady
    logical :: supported
    integer(c_int) :: rc
    integer(c_size_t) :: field_bytes,map_bytes,coeff_bytes
    call settings()
    if(epoch_requested==0.or.selected==0)return
    if(c_associated(s%handle))call context_error(s,'epoch capture on live context')
    if(epoch_available()/=1)return
#ifdef USE_OPENACC
    return
#endif
    if(.not.hpsi_supported)return
    if(.not.allocated(x%zwf))return
    pady=ubound(x%zwf,2)-mg%ie(2)
    if(pady/=1)return
    if(.not.owned_rectangle(x,mg,system,info,pady).or..not.owned_rectangle(p,mg,system,info,pady).or. &
       .not.owned_rectangle(hx,mg,system,info,pady).or..not.owned_rectangle(hp,mg,system,info,pady).or. &
       .not.owned_rectangle(g,mg,system,info,pady).or..not.owned_rectangle(pre_g,mg,system,info,pady))return
    call prepare_hpsi_inputs(mg,system,info,stencil,ppg,vlocal,m,idx,idy,idz,vl,lap0,nabt,supported)
    if(.not.supported)return
    no=m%no;nk=m%nk
    rc=device_guard()
    if(rc/=0)then
      write(*,*) 'CG_CUDA EPOCH device fallback=',rc,' rank=',info%id_rko
      return
    endif
    reuse_scratch=.false.
    if(selected==2.and.present(retain_workspace_outputs))then
      reuse_scratch=.not.retain_workspace_outputs
    endif
    if(reuse_scratch)then
      rc=epoch_begin_scratch_gpu(m,system%hvol,x%zwf,s%handle)
    else
      rc=epoch_begin_gpu(m,system%hvol,x%zwf,s%handle)
    endif
    call require_success(rc,'epoch capture',s)
    s%scratch_reused=reuse_scratch
    s%epoch=.true.;s%resident=.true.;s%mode=selected;s%next_iteration=1;s%epoch_iterations=0
    s%layout=[mg%is,mg%ie,system%nspin,system%no,system%nk,info%io_s,info%io_e, &
              info%ik_s,info%ik_e,info%isize_r,info%id_rko,info%im_s,info%im_e,pady]
    s%hvol_bits=transfer(system%hvol,0_c_int64_t)
    rc=epoch_query_gpu(s%handle,field_bytes,map_bytes,coeff_bytes)
    if(rc/=1)call context_error(s,'epoch context query mismatch')
    write(*,*) 'CG_CUDA EPOCH_CAPTURE rank=',info%id_rko,' mode=',s%mode,' vectors=',no*nk
    write(*,*) 'CG_CUDA EPOCH_BYTES fields maps coeff=',field_bytes,map_bytes,coeff_bytes
  end subroutine

  subroutine cg_epoch_hpsi(s,initial)
    type(cg_region_state),intent(inout) :: s
    logical,intent(in) :: initial
    integer(c_int) :: rc
    if(.not.s%epoch)return
    rc=epoch_hpsi_gpu(s%handle,merge(1_c_int,0_c_int,initial))
    call require_success(rc,'epoch Hpsi',s)
  end subroutine

  subroutine unpack_epoch(system,info,packed,out)
    type(s_dft_system),intent(in) :: system
    type(s_parallel_info),intent(in) :: info
    complex(c_double_complex),intent(in) :: packed(:)
    complex(c_double_complex),intent(out) :: out(:,:,:)
    integer :: v,ik,io,ispin
    out=0.d0;v=0
    do ik=info%ik_s,info%ik_e
    do io=info%io_s,info%io_e
    do ispin=1,system%nspin
      v=v+1;out(ispin,io,ik)=packed(v)
    enddo
    enddo
    enddo
    if(v/=size(packed))error stop 'epoch packed shape changed'
  end subroutine

  subroutine cg_epoch_initial(s,system,info,e,norm)
    type(cg_region_state),intent(inout) :: s
    type(s_dft_system),intent(in) :: system
    type(s_parallel_info),intent(in) :: info
    real(8),intent(inout) :: e(:,:,:)
    complex(8),intent(inout) :: norm(:,:,:)
    complex(c_double_complex),allocatable :: ep(:),np(:)
    integer :: nv
    integer(c_int) :: rc
    if(.not.s%epoch)return
    nv=system%nspin*(info%io_e-info%io_s+1)*(info%ik_e-info%ik_s+1)
    allocate(ep(nv),np(nv),s%initial_energy(system%nspin,system%no,system%nk), &
             s%norm(system%nspin,system%no,system%nk))
    rc=epoch_initial_gpu(s%handle,ep,np);call require_success(rc,'epoch initial gradient/norm',s)
    call unpack_epoch(system,info,ep,s%initial_energy)
    call unpack_epoch(system,info,np,s%norm)
    if(s%mode==2)then
      e=dble(s%initial_energy);norm=s%norm
    endif
  end subroutine

  subroutine cg_epoch_energy_check(s,energy)
    type(cg_region_state),intent(inout) :: s
    complex(8),intent(in) :: energy(:,:,:)
    if(.not.s%epoch.or.s%mode/=1)return
    call exact('initial complex product',reshape(s%initial_energy,[size(s%initial_energy)]), &
               reshape(energy,[size(energy)]))
  end subroutine

  subroutine cg_epoch_initial_check(s,e,g,norm)
    type(cg_region_state),intent(inout) :: s
    real(8),intent(in) :: e(:,:,:)
    type(s_orbital),intent(in) :: g
    complex(8),intent(in) :: norm(:,:,:)
    if(.not.s%epoch)return
    if(s%mode==1)then
      call exact('initial E',reshape(cmplx(dble(s%initial_energy),0.d0,kind=8),[size(e)]), &
                 reshape(cmplx(e,0.d0,kind=8),[size(e)]))
      call cg_epoch_field_check(s,2,g,'initial gradient')
    endif
    deallocate(s%initial_energy)
    call cg_epoch_norm_check(s,norm)
  end subroutine

  subroutine cg_epoch_norm(s,system,info,norm)
    type(cg_region_state),intent(inout) :: s
    type(s_dft_system),intent(in) :: system
    type(s_parallel_info),intent(in) :: info
    complex(8),intent(inout) :: norm(:,:,:)
    complex(c_double_complex),allocatable :: np(:)
    integer(c_int) :: rc
    if(.not.s%epoch)return
    allocate(np(system%nspin*(info%io_e-info%io_s+1)*(info%ik_e-info%ik_s+1)))
    allocate(s%norm(system%nspin,system%no,system%nk))
    rc=epoch_norm_gpu(s%handle,np);call require_success(rc,'epoch pre_g/g product',s)
    call unpack_epoch(system,info,np,s%norm)
    if(s%mode==2)norm=s%norm
  end subroutine

  subroutine cg_epoch_norm_check(s,norm)
    type(cg_region_state),intent(inout) :: s
    complex(8),intent(in) :: norm(:,:,:)
    if(.not.s%epoch)return
    if(s%mode==1)call exact('epoch norm',reshape(s%norm,[size(s%norm)]),reshape(norm,[size(norm)]))
    deallocate(s%norm)
  end subroutine

  subroutine cg_epoch_direction(s,system,info,iteration,bk,p)
    type(cg_region_state),intent(inout) :: s
    type(s_dft_system),intent(in) :: system
    type(s_parallel_info),intent(in) :: info
    integer,intent(in) :: iteration
    real(8),intent(in) :: bk(:,:,:)
    type(s_orbital),intent(in) :: p
    real(c_double),allocatable :: bp(:)
    integer :: v,ik,io,ispin
    integer(c_int) :: rc
    if(.not.s%epoch)return
    if(s%awaiting_accept.or.iteration/=s%next_iteration)call context_error(s,'epoch direction sequence')
    allocate(bp(system%nspin*(info%io_e-info%io_s+1)*(info%ik_e-info%ik_s+1)));bp=0.d0
    if(iteration>1)then
      v=0
      do ik=info%ik_s,info%ik_e
      do io=info%io_s,info%io_e
      do ispin=1,system%nspin
        v=v+1;bp(v)=bk(ispin,io,ik)
      enddo
      enddo
      enddo
    endif
    rc=epoch_direction_gpu(s%handle,int(iteration,c_int),bp);call require_success(rc,'epoch direction',s)
    s%epoch_iterations=iteration
    call cg_epoch_field_check(s,3,p,'direction p')
  end subroutine

  subroutine cg_epoch_field_check(s,field,w,label)
    type(cg_region_state),intent(inout) :: s
    integer,intent(in) :: field
    type(s_orbital),intent(in) :: w
    character(*),intent(in) :: label
    complex(c_double_complex),allocatable :: copy(:,:,:,:,:,:,:)
    integer(c_int) :: rc
    if(.not.s%epoch.or.s%mode/=1)return
    allocate(copy,mold=w%zwf);copy=w%zwf
    rc=epoch_read_gpu(s%handle,int(field,c_int),copy);call require_success(rc,'epoch shadow field',s)
    call exact(label,reshape(copy,[size(copy)]),reshape(w%zwf,[size(w%zwf)]),.false.)
  end subroutine

  subroutine epoch_output(s,field,w,label)
    type(cg_region_state),intent(inout) :: s
    integer,intent(in) :: field
    type(s_orbital),intent(inout) :: w
    character(*),intent(in) :: label
    integer(c_int) :: rc
    if(s%mode==1)then
      call cg_epoch_field_check(s,field,w,label)
    else
      rc=epoch_read_gpu(s%handle,int(field,c_int),w%zwf);call require_success(rc,label,s)
    endif
  end subroutine

  subroutine cg_epoch_end(s,x,hx,g,pre_g,p,hp,retain_workspace_outputs)
    type(cg_region_state),intent(inout) :: s
    type(s_orbital),intent(inout) :: x,hx,g,pre_g,p,hp
    logical,optional,intent(in) :: retain_workspace_outputs
    integer(c_int) :: rc
    logical :: keep_workspaces
    if(.not.s%epoch)return
    if(s%awaiting_accept)call context_error(s,'epoch exit before acceptance')
    keep_workspaces=.true.
    if(s%mode==2)then
      if(present(retain_workspace_outputs))keep_workspaces=retain_workspace_outputs
    endif
    if(s%scratch_reused.and.keep_workspaces)call context_error(s,'scratch reuse cannot restore workspace outputs')
    call epoch_output(s,0,x,'final epoch x')
    if(keep_workspaces)then
      call epoch_output(s,1,hx,'final epoch Hx')
      call epoch_output(s,2,g,'final epoch g')
      ! The original loop copies pre_g=g BEFORE every exit test, including Ncg=0.
      call epoch_output(s,2,pre_g,'final epoch pre_g')
      if(s%epoch_iterations>0)then
        call epoch_output(s,3,p,'final epoch p')
        call epoch_output(s,4,hp,'final epoch Hp')
      endif
    endif
    write(*,*) 'CG_CUDA EPOCH_WORKSPACE_OUTPUTS retained=',keep_workspaces
    rc=close_gpu(s%handle);s%handle=c_null_ptr;call require_success(rc,'epoch close')
    write(*,*) 'CG_CUDA EPOCH_END iterations=',s%epoch_iterations,' mode=',s%mode
    s%mode=0;s%epoch=.false.;s%resident=.false.;s%scratch_reused=.false.;s%next_iteration=0;s%epoch_iterations=0
    s%layout=0;s%hvol_bits=0
  end subroutine

  subroutine context_error(s,message)
    type(cg_region_state),intent(inout) :: s
    character(*),intent(in) :: message
    integer(c_int) :: rc
    rc=close_gpu(s%handle);s%handle=c_null_ptr;s%resident=.false.
    write(*,*) 'CG_CUDA CONTEXT_ERROR ',message,' cleanup=',rc
    error stop 'CG resident contract failed; no stale CPU fallback'
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

  subroutine require_success(rc,stage,s)
    integer(c_int),intent(in) :: rc
    character(*),intent(in) :: stage
    type(cg_region_state),intent(inout),optional :: s
    if(rc==0)return
    if(present(s))call context_error(s,stage)
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
