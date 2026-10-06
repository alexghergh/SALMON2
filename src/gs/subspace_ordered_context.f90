! SPDX-License-Identifier: Apache-2.0
! Explicit host handoffs around the unchanged CPU ZHEEV. No saved device cache.
module subspace_ordered_context
  use iso_c_binding, only: c_ptr,c_null_ptr,c_associated,c_int,c_double,c_double_complex
  implicit none
  private
  public :: subspace_workspace,subspace_requested,subspace_begin,subspace_overlap,subspace_rotate,subspace_end
  type subspace_workspace
    private
    type(c_ptr) :: handle=c_null_ptr
  end type
  interface
    function create_gpu(handle) bind(C,name='salmon_subspace_context_create') result(rc)
      import c_ptr,c_int
      type(c_ptr),intent(inout) :: handle
      integer(c_int) :: rc
    end function
    function overlap_gpu(handle,a,b,hvol,hmat) bind(C,name='salmon_subspace_context_overlap') result(rc)
      import c_ptr,c_int,c_double,c_double_complex
      type(c_ptr),value :: handle
      complex(c_double_complex),intent(in) :: a(*),b(*)
      real(c_double),value :: hvol
      complex(c_double_complex),intent(out) :: hmat(*)
      integer(c_int) :: rc
    end function
    function rotate_gpu(handle,evec,out) bind(C,name='salmon_subspace_context_rotate') result(rc)
      import c_ptr,c_int,c_double_complex
      type(c_ptr),value :: handle
      complex(c_double_complex),intent(in) :: evec(*)
      complex(c_double_complex),intent(out) :: out(*)
      integer(c_int) :: rc
    end function
    function destroy_gpu(handle) bind(C,name='salmon_subspace_context_destroy') result(rc)
      import c_ptr,c_int
      type(c_ptr),intent(inout) :: handle
      integer(c_int) :: rc
    end function
  end interface
contains
  logical function subspace_requested() result(enabled)
    character(8) :: setting
    integer :: status
    setting=''
    call get_environment_variable('SALMON_SUBSPACE_ORDERED',setting,status=status)
    if(status/=0.and.status/=1) error stop 'SALMON_SUBSPACE_ORDERED unreadable or truncated'
    select case(trim(setting))
    case('','0'); enabled=.false.
    case('1'); enabled=.true.
    case default; error stop 'SALMON_SUBSPACE_ORDERED must be 0 or 1'
    end select
  end function

  subroutine checked(state,rc,where)
    type(subspace_workspace),intent(inout) :: state
    integer(c_int),intent(in) :: rc
    character(*),intent(in) :: where
    integer(c_int) :: cleanup
    if(rc==0) return
    cleanup=destroy_gpu(state%handle)
    write(*,*) 'SUBSPACE_ORDERED_ERROR ',where,rc,' cleanup=',cleanup
    error stop 'ordered subspace failed; no partial CPU fallback'
  end subroutine

  subroutine subspace_begin(state)
    type(subspace_workspace),intent(inout) :: state
    integer(c_int) :: rc
    if(c_associated(state%handle)) then
      rc=destroy_gpu(state%handle)
      write(*,*) 'SUBSPACE_ORDERED_ERROR duplicate begin; cleanup=',rc
      error stop 'subspace workspace already active'
    endif
    rc=create_gpu(state%handle)
    call checked(state,rc,'begin')
  end subroutine

  subroutine subspace_overlap(state,a,b,hvol,hmat)
    type(subspace_workspace),intent(inout) :: state
    complex(c_double_complex),intent(in) :: a(*),b(*)
    real(c_double),intent(in) :: hvol
    complex(c_double_complex),intent(out) :: hmat(*)
    integer(c_int) :: rc
    rc=overlap_gpu(state%handle,a,b,hvol,hmat)
    call checked(state,rc,'overlap')
  end subroutine

  subroutine subspace_rotate(state,evec,out)
    type(subspace_workspace),intent(inout) :: state
    complex(c_double_complex),intent(in) :: evec(*)
    complex(c_double_complex),intent(out) :: out(*)
    integer(c_int) :: rc
    rc=rotate_gpu(state%handle,evec,out)
    call checked(state,rc,'rotate')
  end subroutine

  subroutine subspace_end(state)
    type(subspace_workspace),intent(inout) :: state
    integer(c_int) :: rc
    rc=destroy_gpu(state%handle)
    call checked(state,rc,'end')
  end subroutine
end module
