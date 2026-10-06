! SPDX-License-Identifier: Apache-2.0
! Call-local fields; one host matrix batch is overwritten by ascending CPU ZHEEV results.
module subspace_connected_context
  use iso_c_binding
  use structures, only: s_rgrid,s_dft_system,s_parallel_info,s_orbital,s_stencil,s_pp_grid,s_scalar
  use hpsi_cuda_inputs, only: hpsi_input,prepare_hpsi_inputs,owned_rectangle
  implicit none
  private
  public :: connected_subspace,subspace_connected_requested,subspace_connected_active
  public :: subspace_connected_begin,subspace_connected_overlap,subspace_connected_matrix
  public :: subspace_connected_eigenvectors,subspace_connected_end
  type connected_subspace
    private
    type(c_ptr) :: handle=c_null_ptr
    complex(c_double_complex),allocatable :: matrices(:,:,:)
  end type
  interface
    function device_guard() bind(C,name='salmon_cg_device_guard') result(rc)
      import c_int
      integer(c_int) :: rc
    end function
    function begin_gpu(input,x,handle) bind(C,name='salmon_subspace_connected_begin') result(rc)
      import hpsi_input,c_int,c_ptr,c_double_complex
      type(hpsi_input),intent(in) :: input
      complex(c_double_complex),intent(in) :: x(*)
      type(c_ptr),intent(inout) :: handle
      integer(c_int) :: rc
    end function
    function query_gpu(handle,nk,fields,matrix,maps,coeff) bind(C,name='salmon_subspace_connected_query') result(rc)
      import c_ptr,c_size_t,c_int
      type(c_ptr),value :: handle
      integer(c_int),intent(out) :: nk
      integer(c_size_t),intent(out) :: fields,matrix,maps,coeff
      integer(c_int) :: rc
    end function
    function overlap_gpu(handle,hvol,matrices,hx,retain_hx) bind(C,name='salmon_subspace_connected_overlap') result(rc)
      import c_ptr,c_int,c_double,c_double_complex
      type(c_ptr),value :: handle
      real(c_double),value :: hvol
      complex(c_double_complex),intent(out) :: matrices(*)
      complex(c_double_complex),intent(inout) :: hx(*)
      integer(c_int),value :: retain_hx
      integer(c_int) :: rc
    end function
    function rotate_gpu(handle,evec,x) bind(C,name='salmon_subspace_connected_rotate') result(rc)
      import c_ptr,c_int,c_double_complex
      type(c_ptr),value :: handle
      complex(c_double_complex),intent(in) :: evec(*)
      complex(c_double_complex),intent(inout) :: x(*)
      integer(c_int) :: rc
    end function
    function close_gpu(handle) bind(C,name='salmon_subspace_connected_close') result(rc)
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int) :: rc
    end function
  end interface
contains
  logical function subspace_connected_requested() result(enabled)
    character(8) :: setting
    integer :: status
    setting='';call get_environment_variable('SALMON_SUBSPACE_CONNECTED',setting,status=status)
    if(status/=0.and.status/=1)error stop 'SALMON_SUBSPACE_CONNECTED unreadable or truncated'
    select case(trim(setting))
    case('','0');enabled=.false.
    case('1');enabled=.true.
    case default;error stop 'SALMON_SUBSPACE_CONNECTED must be 0 or 1'
    end select
  end function

  logical function subspace_connected_active(s)
    type(connected_subspace),intent(in) :: s
    subspace_connected_active=c_associated(s%handle)
  end function

  subroutine checked(s,rc,where)
    type(connected_subspace),intent(inout) :: s
    integer(c_int),intent(in) :: rc
    character(*),intent(in) :: where
    integer(c_int) :: cleanup
    if(rc==0)return
    cleanup=close_gpu(s%handle);s%handle=c_null_ptr
    if(allocated(s%matrices))deallocate(s%matrices)
    write(*,*) 'SUBSPACE_CONNECTED_ERROR ',where,' rc=',rc,' cleanup=',cleanup
    error stop 'connected subspace failed; no partial CPU fallback'
  end subroutine

  subroutine subspace_connected_begin(s,mg,system,info,stencil,ppg,vlocal,x,hx)
    use salmon_global, only: yn_spinorbit,yn_periodic,yn_jm,absorbing_boundary
    use pseudo_pt_plusU_sub, only: PLUS_U_ON
    use code_optimization, only: optimized_stencil_is_callable,stencil_is_parallelized_by_omp
    type(connected_subspace),intent(inout) :: s
    type(s_rgrid),intent(in) :: mg
    type(s_dft_system),intent(in) :: system
    type(s_parallel_info),intent(in) :: info
    type(s_stencil),intent(in),target :: stencil
    type(s_pp_grid),intent(in),target :: ppg
    type(s_scalar),intent(in) :: vlocal(:)
    type(s_orbital),intent(in) :: x,hx
    type(hpsi_input) :: m
    integer(c_int),target :: idx(24),idy(24),idz(24)
    real(c_double),allocatable,target :: vl(:,:,:),lap0(:),nabt(:,:,:)
    integer(c_int) :: rc,nk
    integer(c_size_t) :: fields,matrix,maps,coeff
    integer :: status
    logical :: eligible
    if(subspace_connected_active(s).or.allocated(s%matrices))error stop 'connected subspace already owned'
    ! Same conditions as the CUDA Hpsi: no spin-orbit, plain periodic case, sequential stencil.
    eligible=.not.allocated(x%rwf).and.yn_spinorbit=='n'.and.yn_periodic=='y'.and.yn_jm=='n'.and. &
      absorbing_boundary/='z'.and..not.PLUS_U_ON.and..not.system%xc_payload%use_tau_operator.and. &
      .not.optimized_stencil_is_callable.and..not.stencil_is_parallelized_by_omp
    eligible=eligible.and.all(mg%is_array==mg%is).and.all(mg%ie_array==mg%ie+[0,1,0])
    eligible=eligible.and.owned_rectangle(x,mg,system,info,1).and.owned_rectangle(hx,mg,system,info,1)
    if(.not.eligible)then
      write(*,*) 'SUBSPACE_CONNECTED_FALLBACK physics/storage rank=',info%id_rko
      return
    endif
    call prepare_hpsi_inputs(mg,system,info,stencil,ppg,vlocal,m,idx,idy,idz,vl,lap0,nabt,eligible)
    if(.not.eligible)then
      write(*,*) 'SUBSPACE_CONNECTED_FALLBACK inputs rank=',info%id_rko
      return
    endif
    if(m%no/=80.or.m%nk<1.or.m%nk>65535)then
      write(*,*) 'SUBSPACE_CONNECTED_FALLBACK batch rank=',info%id_rko
      return
    endif
    rc=device_guard()
    if(rc/=0)then
      write(*,*) 'SUBSPACE_CONNECTED_FALLBACK device rank=',info%id_rko,' rc=',rc
      return
    endif
    allocate(s%matrices(80,80,m%nk),stat=status)
    if(status/=0)error stop 'connected subspace host matrix allocation failed'
    rc=begin_gpu(m,x%zwf,s%handle);call checked(s,rc,'capture')
    rc=query_gpu(s%handle,nk,fields,matrix,maps,coeff);call checked(s,rc,'query')
    if(nk/=m%nk)call checked(s,-1_c_int,'captured batch')
    write(*,*) 'SUBSPACE_CONNECTED_CAPTURE rank=',info%id_rko,' vectors=',m%no*nk,' nk=',nk
    write(*,*) 'SUBSPACE_CONNECTED_BYTES fields matrix maps coeff=',fields,matrix,maps,coeff
  end subroutine

  subroutine subspace_connected_overlap(s,hvol,hx,retain_hx)
    type(connected_subspace),intent(inout) :: s
    real(c_double),intent(in) :: hvol
    type(s_orbital),intent(inout) :: hx
    logical,intent(in) :: retain_hx
    integer(c_int) :: rc
    if(.not.subspace_connected_active(s))error stop 'connected overlap without owner'
    rc=overlap_gpu(s%handle,hvol,s%matrices,hx%zwf,merge(1_c_int,0_c_int,retain_hx))
    call checked(s,rc,'overlap')
    write(*,*) 'SUBSPACE_CONNECTED_WORKSPACE_OUTPUTS retained=',retain_hx
  end subroutine

  subroutine subspace_connected_matrix(s,k,hmat)
    type(connected_subspace),intent(inout) :: s
    integer,intent(in) :: k
    complex(c_double_complex),intent(out) :: hmat(80,80)
    if(.not.subspace_connected_active(s))error stop 'connected matrix without owner'
    if(k<1.or.k>size(s%matrices,3))call checked(s,-1_c_int,'matrix index')
    hmat=s%matrices(:,:,k)
  end subroutine

  subroutine subspace_connected_eigenvectors(s,k,evec)
    type(connected_subspace),intent(inout) :: s
    integer,intent(in) :: k
    complex(c_double_complex),intent(in) :: evec(80,80)
    if(.not.subspace_connected_active(s))error stop 'connected eigenvectors without owner'
    if(k<1.or.k>size(s%matrices,3))call checked(s,-1_c_int,'eigenvector index')
    s%matrices(:,:,k)=evec
  end subroutine

  subroutine subspace_connected_end(s,x)
    type(connected_subspace),intent(inout) :: s
    type(s_orbital),intent(inout) :: x
    integer(c_int) :: rc
    if(.not.subspace_connected_active(s))error stop 'connected end without owner'
    rc=rotate_gpu(s%handle,s%matrices,x%zwf);call checked(s,rc,'rotation/final read')
    rc=close_gpu(s%handle);s%handle=c_null_ptr
    call checked(s,rc,'close')
    write(*,*) 'SUBSPACE_CONNECTED_END nk=',size(s%matrices,3)
    deallocate(s%matrices)
  end subroutine
end module subspace_connected_context
