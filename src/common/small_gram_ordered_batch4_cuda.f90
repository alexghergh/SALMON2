! SPDX-License-Identifier: Apache-2.0
! The Gram-Schmidt of small_gram_ordered_cuda for one to four independent k-point blocks per call.
module small_gram_ordered_batch4_cuda
  implicit none
  private
  public :: small_gram_ordered_batch4
contains
  integer(4) function small_gram_ordered_batch4(n,m,nb,hvol,block) result(status)
    use cudafor
    use openacc, only: acc_wait,acc_get_cuda_stream,acc_async_sync
    implicit none
    integer,intent(in) :: n,m,nb
    real(8),intent(in) :: hvol
    complex(8),device,intent(inout) :: block(n,m,nb)
    complex(8),device,allocatable :: dots(:,:),coefficients_device(:,:)
    real(8),device,allocatable :: scales_device(:)
    complex(8),parameter :: one=1d0
    complex(8) :: dot,dots_host(m,nb),coefficients(m,nb)
    real(8) :: norm2_tmp,norm2,scale_real,scales(nb)
    integer :: j,k,ib,allocation_status
    integer(kind=cuda_stream_kind) :: stream
    status=-1
    if(n<=0 .or. m/=16 .or. nb<1 .or. nb>4) return
    allocate(dots(m,nb),coefficients_device(m,nb),scales_device(nb),stat=allocation_status)
    if(allocation_status/=0) return
    call acc_wait(acc_async_sync)
    stream=acc_get_cuda_stream(acc_async_sync)
    dots=0d0
    do j=1,m
      call batch_gram_dot<<<dim3(1,nb,1),128,0,stream>>>(n,m,nb,j,j,block,dots)
      status=cudaStreamSynchronize(stream)
      if(status/=0) exit
      dots_host=dots
      do ib=1,nb
        dot=dots_host(j,ib)
        norm2_tmp=real(dot)*hvol
        norm2=norm2_tmp
        scale_real=real(one/sqrt(norm2),kind=8)
        scales(ib)=scale_real
      enddo
      scales_device=scales
      call batch_gram_scale<<<dim3((n+255)/256,nb,1),256,0,stream>>>(n,m,nb,j,scales_device,block)
      if(j<m) then
        call batch_gram_dot<<<dim3(m-j,nb,1),128,0,stream>>>(n,m,nb,j,j+1,block,dots)
        status=cudaStreamSynchronize(stream)
        if(status/=0) exit
        dots_host=dots
        coefficients=0d0
        do ib=1,nb
          do k=j+1,m
            dot=dots_host(k,ib)
            coefficients(k,ib)=dot*hvol
          enddo
        enddo
        coefficients_device=coefficients
        call batch_gram_axpy<<<dim3((n+255)/256,m-j,nb),256,0,stream>>>(n,m,nb,j,block,coefficients_device)
      endif
    enddo
    if(status==0) status=cudaStreamSynchronize(stream)
    deallocate(dots,coefficients_device,scales_device)
  end function small_gram_ordered_batch4

  attributes(device) real(8) function ordered_sum8(a) result(value)
    use cudadevice, only: __dadd_rn
    implicit none
    real(8), intent(in) :: a(8)
    real(8) :: even, odd
    even=__dadd_rn(__dadd_rn(a(1),a(3)),__dadd_rn(a(5),a(7)))
    odd=__dadd_rn(__dadd_rn(a(2),a(4)),__dadd_rn(a(6),a(8)))
    value=__dadd_rn(even,odd)
  end function ordered_sum8

  attributes(global) subroutine batch_gram_dot(n,m,nb,left,right_first,block,dots)
    use cudafor
    use cudadevice, only: __fma_rn, __dadd_rn, __dmul_rn
    implicit none
    integer, value :: n,m,nb,left,right_first
    complex(8), device, intent(in) :: block(n,m,nb)
    complex(8), device, intent(out) :: dots(m,nb)
    integer, parameter :: tile_points=1024
    complex(8), shared :: tile_left(tile_points),tile_right(tile_points)
    real(8), shared :: partial(8,4)
    real(8) :: rr,ii,ri,ir,xr,xi,yr,yi,imaginary,real_part
    integer :: lane,i,right,bulk,first,tile_count,tile_i,ib
    ib=blockIdx%y
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
        tile_left(tile_i)=block(first+tile_i-1,left,ib)
        tile_right(tile_i)=block(first+tile_i-1,right,ib)
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
        xr=real(block(i,left,ib)); xi=aimag(block(i,left,ib))
        yr=real(block(i,right,ib)); yi=aimag(block(i,right,ib))
        rr=__dadd_rn(rr,__dmul_rn(xr,yr))
        ri=__dadd_rn(ri,__dmul_rn(xr,yi))
        ii=__dadd_rn(ii,__dmul_rn(xi,yi))
        ir=__dadd_rn(ir,__dmul_rn(xi,yr))
      enddo
      imaginary=__dadd_rn(ri,-ir)
      real_part=__dadd_rn(__dadd_rn(rr,ii),__dmul_rn(0d0,imaginary))
      dots(right,ib)=cmplx(real_part,imaginary,kind=8)
    endif
  end subroutine batch_gram_dot

  attributes(global) subroutine batch_gram_scale(n,m,nb,column,alpha,block)
    use cudafor
    use cudadevice, only: __dmul_rn
    implicit none
    integer, value :: n,m,nb,column
    real(8), device, intent(in) :: alpha(nb)
    complex(8), device, intent(inout) :: block(n,m,nb)
    integer :: i,ib
    ib=blockIdx%y
    i=(blockIdx%x-1)*blockDim%x+threadIdx%x
    if (i<=n) block(i,column,ib)=cmplx(__dmul_rn(real(block(i,column,ib)),alpha(ib)), &
                                  __dmul_rn(aimag(block(i,column,ib)),alpha(ib)),kind=8)
  end subroutine batch_gram_scale

  attributes(global) subroutine batch_gram_axpy(n,m,nb,left,block,coefficients)
    use cudafor
    use cudadevice, only: __fma_rn, __dadd_rn, __dmul_rn
    implicit none
    integer, value :: n,m,nb,left
    complex(8), device, intent(inout) :: block(n,m,nb)
    complex(8), device, intent(in) :: coefficients(m,nb)
    integer :: i,right,bulk,ib
    real(8) :: ar,ai,xr,xi,yr,yi
    i=(blockIdx%x-1)*blockDim%x+threadIdx%x
    ib=blockIdx%z
    right=left+blockIdx%y
    if (i>n) return
    ar=-real(coefficients(right,ib)); ai=-aimag(coefficients(right,ib))
    ! As in ZAXPY, a zero alpha (either sign of zero) leaves Y untouched.
    if (ar==0d0 .and. ai==0d0) return
    xr=real(block(i,left,ib)); xi=aimag(block(i,left,ib))
    yr=real(block(i,right,ib)); yi=aimag(block(i,right,ib))
    bulk=n-mod(n,16)
    if (i<=bulk) then
      yr=__fma_rn(-ai,xi,__fma_rn(ar,xr,yr))
      yi=__fma_rn(ai,xr,__fma_rn(ar,xi,yi))
    else
      yr=__dadd_rn(yr,__dadd_rn(__dmul_rn(ar,xr),-__dmul_rn(ai,xi)))
      yi=__dadd_rn(yi,__dadd_rn(__dmul_rn(ai,xr),__dmul_rn(ar,xi)))
    endif
    block(i,right,ib)=cmplx(yr,yi,kind=8)
  end subroutine batch_gram_axpy
end module small_gram_ordered_batch4_cuda
