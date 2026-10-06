! Overlap (alpha = hvol) and rotation (alpha = 1) entry points for the subspace ZGEMM kernels.
module subspace_ordered_bridge
  use iso_c_binding, only: c_ptr,c_int,c_double
  implicit none
  private
  public :: salmon_subspace_overlap_device, salmon_subspace_rotation_device
  interface
    function salmon_subspace_cn80_ordered(a,b,c,ar,ai) bind(C) result(status)
      import c_ptr,c_int,c_double
      type(c_ptr),value :: a,b,c
      real(c_double),value :: ar,ai
      integer(c_int) :: status
    end function
    function salmon_subspace_nn80_ordered(a,e,c,ar,ai) bind(C) result(status)
      import c_ptr,c_int,c_double
      type(c_ptr),value :: a,e,c
      real(c_double),value :: ar,ai
      integer(c_int) :: status
    end function
  end interface
contains
  function salmon_subspace_overlap_device(a,b,c,hvol) bind(C) result(status)
    type(c_ptr),value :: a,b,c
    real(c_double),value :: hvol
    integer(c_int) :: status
    status=salmon_subspace_cn80_ordered(a,b,c,hvol,0d0)
  end function
  function salmon_subspace_rotation_device(a,e,c) bind(C) result(status)
    type(c_ptr),value :: a,e,c
    integer(c_int) :: status
    status=salmon_subspace_nn80_ordered(a,e,c,1d0,0d0)
  end function
end module
