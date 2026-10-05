! SPDX-License-Identifier: Apache-2.0
! CUDA Fortran Gram-Schmidt of 16 complex orbitals on an undivided grid (USE_ORDERED_SMALL_GRAM).
module small_gram_ordered_cuda
  implicit none
  private
  public :: small_gram_ordered
contains
  ! The block stays on the GPU. Dots, scaling and updates follow the operation order of the CPU
  ! ZDOTC/ZDSCAL/ZAXPY kernels (eight FMA chains reduced as ((1+3)+(5+7))+((2+4)+(6+8)), then the
  ! remainder); norms and coefficients are formed on the host with the original expressions.
  integer(4) function small_gram_ordered(n, m, hvol, block) result(status)
    use cudafor
    use openacc, only: acc_wait, acc_get_cuda_stream, acc_async_sync
    implicit none
    integer, intent(in) :: n, m
    real(8), intent(in) :: hvol
    complex(8), device, intent(inout) :: block(n,m)
    complex(8), device, allocatable :: dots(:), coefficients_device(:)
    complex(8), parameter :: one=1d0
    complex(8) :: dot, coefficients(m)
    real(8) :: norm2_tmp, norm2, scale_real
    integer :: j, k, allocation_status
    integer(kind=cuda_stream_kind) :: stream

    status=-1
    if (n<=0 .or. m/=16) return
    allocate(dots(m), coefficients_device(m), stat=allocation_status)
    if (allocation_status/=0) return
    call acc_wait(acc_async_sync)
    stream=acc_get_cuda_stream(acc_async_sync)
    do j=1,m
      call ordered_gram_dot<<<1,128,0,stream>>>(n,m,j,j,block,dots)
      status=cudaStreamSynchronize(stream)
      if (status/=0) exit
      dot=dots(j)
      norm2_tmp=real(dot)*hvol
      norm2=norm2_tmp
      scale_real=real(one/sqrt(norm2),kind=8)
      call ordered_gram_scale<<<(n+255)/256,256,0,stream>>>(n,m,j,scale_real,block)
      if (j<m) then
        call ordered_gram_dot<<<m-j,128,0,stream>>>(n,m,j,j+1,block,dots)
        status=cudaStreamSynchronize(stream)
        if (status/=0) exit
        coefficients=0d0
        do k=j+1,m
          dot=dots(k)
          coefficients(k)=dot*hvol
        enddo
        coefficients_device=coefficients
        call ordered_gram_axpy<<<dim3((n+255)/256,m-j,1),256,0,stream>>>(n,m,j,block,coefficients_device)
      endif
    enddo
    if (status==0) status=cudaStreamSynchronize(stream)
    deallocate(dots, coefficients_device)
  end function small_gram_ordered

  attributes(device) real(8) function ordered_sum8(a) result(value)
    use cudadevice, only: __dadd_rn
    implicit none
    real(8), intent(in) :: a(8)
    real(8) :: even, odd
    even=__dadd_rn(__dadd_rn(a(1),a(3)),__dadd_rn(a(5),a(7)))
    odd=__dadd_rn(__dadd_rn(a(2),a(4)),__dadd_rn(a(6),a(8)))
    value=__dadd_rn(even,odd)
  end function ordered_sum8

  attributes(global) subroutine ordered_gram_dot(n,m,left,right_first,block,dots)
    use cudafor
    use cudadevice, only: __fma_rn, __dadd_rn, __dmul_rn
    implicit none
    integer, value :: n,m,left,right_first
    complex(8), device, intent(in) :: block(n,m)
    complex(8), device, intent(out) :: dots(m)
    integer, parameter :: tile_points=1024
    complex(8), shared :: tile_left(tile_points),tile_right(tile_points)
    real(8), shared :: partial(8,4)
    real(8) :: rr,ii,ri,ir,xr,xi,yr,yi,imaginary,real_part
    integer :: lane,i,right,bulk,first,tile_count,tile_i
    lane=threadIdx%x
    right=right_first+blockIdx%x-1
    bulk=n-mod(n,8)
    if (lane<=8) then
      rr=0d0; ii=0d0; ri=0d0; ir=0d0
    endif
    ! Bulk and each tile contain complete groups of eight; chains span tiles.
    do first=1,bulk,tile_points
      tile_count=min(tile_points,bulk-first+1)
      do tile_i=lane,tile_count,blockDim%x
        tile_left(tile_i)=block(first+tile_i-1,left)
        tile_right(tile_i)=block(first+tile_i-1,right)
      enddo
      call syncthreads()
      if (lane<=8) then
        do tile_i=lane,tile_count,8
          xr=real(tile_left(tile_i)); xi=aimag(tile_left(tile_i))
          yr=real(tile_right(tile_i)); yi=aimag(tile_right(tile_i))
          rr=__fma_rn(xr,yr,rr)
          ii=__fma_rn(xi,yi,ii)
          ri=__fma_rn(xr,yi,ri)
          ir=__fma_rn(xi,yr,ir)
        enddo
      endif
      call syncthreads()
    enddo
    if (lane<=8) then
      partial(lane,1)=rr; partial(lane,2)=ii
      partial(lane,3)=ri; partial(lane,4)=ir
    endif
    call syncthreads()
    if (lane==1) then
      rr=ordered_sum8(partial(:,1)); ii=ordered_sum8(partial(:,2))
      ri=ordered_sum8(partial(:,3)); ir=ordered_sum8(partial(:,4))
      ! The CPU remainder uses separate multiplies and adds, after the bulk tree.
      do i=bulk+1,n
        xr=real(block(i,left)); xi=aimag(block(i,left))
        yr=real(block(i,right)); yi=aimag(block(i,right))
        rr=__dadd_rn(rr,__dmul_rn(xr,yr))
        ri=__dadd_rn(ri,__dmul_rn(xr,yi))
        ii=__dadd_rn(ii,__dmul_rn(xi,yi))
        ir=__dadd_rn(ir,__dmul_rn(xi,yr))
      enddo
      imaginary=__dadd_rn(ri,-ir)
      real_part=__dadd_rn(__dadd_rn(rr,ii),__dmul_rn(0d0,imaginary))
      dots(right)=cmplx(real_part,imaginary,kind=8)
    endif
  end subroutine ordered_gram_dot

  attributes(global) subroutine ordered_gram_scale(n,m,column,alpha,block)
    use cudafor
    use cudadevice, only: __dmul_rn
    implicit none
    integer, value :: n,m,column
    real(8), value :: alpha
    complex(8), device, intent(inout) :: block(n,m)
    integer :: i
    i=(blockIdx%x-1)*blockDim%x+threadIdx%x
    if (i<=n) block(i,column)=cmplx(__dmul_rn(real(block(i,column)),alpha), &
                                  __dmul_rn(aimag(block(i,column)),alpha),kind=8)
  end subroutine ordered_gram_scale

  attributes(global) subroutine ordered_gram_axpy(n,m,left,block,coefficients)
    use cudafor
    use cudadevice, only: __fma_rn, __dadd_rn, __dmul_rn
    implicit none
    integer, value :: n,m,left
    complex(8), device, intent(inout) :: block(n,m)
    complex(8), device, intent(in) :: coefficients(m)
    integer :: i,right,bulk
    real(8) :: ar,ai,xr,xi,yr,yi
    i=(blockIdx%x-1)*blockDim%x+threadIdx%x
    right=left+blockIdx%y
    if (i>n) return
    ar=-real(coefficients(right)); ai=-aimag(coefficients(right))
    ! As in ZAXPY, a zero alpha (either sign of zero) leaves Y untouched.
    if (ar==0d0 .and. ai==0d0) return
    xr=real(block(i,left)); xi=aimag(block(i,left))
    yr=real(block(i,right)); yi=aimag(block(i,right))
    bulk=n-mod(n,16)
    if (i<=bulk) then
      yr=__fma_rn(-ai,xi,__fma_rn(ar,xr,yr))
      yi=__fma_rn(ai,xr,__fma_rn(ar,xi,yi))
    else
      yr=__dadd_rn(yr,__dadd_rn(__dmul_rn(ar,xr),-__dmul_rn(ai,xi)))
      yi=__dadd_rn(yi,__dadd_rn(__dmul_rn(ai,xr),__dmul_rn(ar,xi)))
    endif
    block(i,right)=cmplx(yr,yi,kind=8)
  end subroutine ordered_gram_axpy
end module small_gram_ordered_cuda
