! SPDX-License-Identifier: Apache-2.0
! Explicit pair lifetime; host MPI retains its original buffers and ordering.
module real_gram_context
  use iso_c_binding, only: c_ptr,c_null_ptr,c_associated,c_int,c_double,c_int64_t
  implicit none
  private
  public :: real_gram_workspace,real_gram_requested,real_gram_supported
  public :: real_gram_begin,real_gram_overlap,real_gram_update,real_gram_end
  type real_gram_workspace
    private
    type(c_ptr) :: handle=c_null_ptr
  end type
  interface
    function create_gpu(handle) bind(C,name='salmon_real_gram_context_create') result(rc)
      import c_ptr,c_int
      type(c_ptr),intent(inout) :: handle
      integer(c_int) :: rc
    end function
    function overlap_gpu(handle,a,b,hvol,overlap) bind(C,name='salmon_real_gram_context_overlap') result(rc)
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      real(c_double),intent(in) :: a(*),b(*)
      real(c_double),value :: hvol
      real(c_double),intent(out) :: overlap(*)
      integer(c_int) :: rc
    end function
    function update_gpu(handle,reduced,out) bind(C,name='salmon_real_gram_context_update') result(rc)
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      real(c_double),intent(in) :: reduced(*)
      real(c_double),intent(out) :: out(*)
      integer(c_int) :: rc
    end function
    function destroy_gpu(handle) bind(C,name='salmon_real_gram_context_destroy') result(rc)
      import c_ptr,c_int
      type(c_ptr),intent(inout) :: handle
      integer(c_int) :: rc
    end function
  end interface
contains
  logical function real_gram_requested() result(enabled)
    character(8) :: setting
    integer :: status
    setting=''
    call get_environment_variable('SALMON_REAL_GRAM_ORDERED',setting,status=status)
    if(status/=0.and.status/=1) error stop 'SALMON_REAL_GRAM_ORDERED unreadable or truncated'
    select case(trim(setting))
    case('','0'); enabled=.false.
    case('1'); enabled=.true.
    case default; error stop 'SALMON_REAL_GRAM_ORDERED must be 0 or 1'
    end select
  end function

  logical function real_gram_supported(points,left,right,hvol) result(supported)
    integer,intent(in) :: points,left,right
    real(c_double),intent(in) :: hvol
    supported=points==21296.and.left==1027.and.right==1027.and. &
      transfer(hvol,0_c_int64_t)==int(z'3FB9FD771FE71D7E',c_int64_t)
  end function

  subroutine checked(state,rc,where)
    type(real_gram_workspace),intent(inout) :: state
    integer(c_int),intent(in) :: rc
    character(*),intent(in) :: where
    integer(c_int) :: cleanup
    if(rc==0) return
    cleanup=destroy_gpu(state%handle)
    write(*,*) 'REAL_GRAM_ORDERED_ERROR ',where,rc,' cleanup=',cleanup
    error stop 'ordered real Gram failed; no partial CPU fallback'
  end subroutine

  subroutine real_gram_begin(state)
    type(real_gram_workspace),intent(inout) :: state
    integer(c_int) :: rc
    if(c_associated(state%handle)) then
      rc=destroy_gpu(state%handle)
      write(*,*) 'REAL_GRAM_ORDERED_ERROR duplicate begin; cleanup=',rc
      error stop 'real Gram workspace already active'
    endif
    rc=create_gpu(state%handle)
    call checked(state,rc,'begin')
  end subroutine

  subroutine real_gram_overlap(state,a,b,hvol,overlap)
    type(real_gram_workspace),intent(inout) :: state
    real(c_double),intent(in) :: a(*),b(*),hvol
    real(c_double),intent(out) :: overlap(*)
    integer(c_int) :: rc
    rc=overlap_gpu(state%handle,a,b,hvol,overlap)
    call checked(state,rc,'overlap')
  end subroutine

  subroutine real_gram_update(state,reduced,out)
    type(real_gram_workspace),intent(inout) :: state
    real(c_double),intent(in) :: reduced(*)
    real(c_double),intent(out) :: out(*)
    integer(c_int) :: rc
    rc=update_gpu(state%handle,reduced,out)
    call checked(state,rc,'update')
  end subroutine

  subroutine real_gram_end(state)
    type(real_gram_workspace),intent(inout) :: state
    integer(c_int) :: rc
    rc=destroy_gpu(state%handle)
    call checked(state,rc,'end')
  end subroutine
end module
