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
!=======================================================================
module gram_schmidt_orth
  implicit none

contains

  subroutine gram_schmidt(sys, rg, wfi, wf)
    use structures, only: s_dft_system, s_rgrid, s_parallel_info, s_orbital
    use gram_schmidt_so_sub, only: gram_schmidt_so
    use salmon_global, only: yn_spinorbit, yn_gramschmidt_blas
    use timer
    use nvtx_wrapper
    implicit none
    type(s_dft_system),   intent(in)    :: sys
    type(s_rgrid),        intent(in)    :: rg
    type(s_parallel_info),intent(in)    :: wfi
    type(s_orbital),      intent(inout) :: wf
    call nvtxStartRange('gram_schmidt', __LINE__)
    
    if ( yn_spinorbit=='y' ) then
      call gram_schmidt_so(sys, rg, wfi, wf)
      return
    end if

    call timer_begin(LOG_CALC_GRAM_SCHMIDT)

    if (sys%if_real_orbital) then
      if(yn_gramschmidt_blas =='y')then
        call gram_schmidt_col_rblas(sys, rg, wfi, wf)
      else
        call gram_schmidt_col_real8(sys, rg, wfi, wf)
      end if
    else
      if(yn_gramschmidt_blas =='y')then
        call gram_schmidt_col_cblas(sys, rg, wfi, wf)
      else
        call gram_schmidt_col_complex8(sys, rg, wfi, wf)
      end if
    end if

    call timer_end(LOG_CALC_GRAM_SCHMIDT)
    call nvtxEndRange
    return
  end subroutine

  subroutine gram_schmidt_col_real8(sys, rg, wfi, wf)
    ! Only for the colinear L(S)DA:
    use structures, only: s_dft_system, s_rgrid, s_parallel_info, s_orbital
    use pack_unpack, only: copy_data
    use timer
    use communication, only: comm_bcast, comm_summation
    implicit none
    type(s_dft_system),   intent(in)    :: sys
    type(s_rgrid),        intent(in)    :: rg
    type(s_parallel_info),intent(in)    :: wfi
    type(s_orbital),      intent(inout) :: wf

    integer :: nsize_rg
    integer :: ik, im, ispin
    integer :: io1, io2
    real(8) :: coeff(1:sys%no), coeff_tmp(1:sys%no)
    real(8) :: norm2, norm2_tmp
    real(8), dimension( &
      & rg%is_array(1):rg%ie_array(1), &
      & rg%is_array(2):rg%ie_array(2), &
      & rg%is_array(3):rg%ie_array(3)) &
      & :: wf_io1, wf_exc, wf_exc_tmp

    nsize_rg =  (rg%ie_array(1) - rg%is_array(1) + 1) &
      & * (rg%ie_array(2) - rg%is_array(2) + 1) &
      & * (rg%ie_array(3) - rg%is_array(3) + 1)

    do im = wfi%im_s, wfi%im_e
    do ik = wfi%ik_s, wfi%ik_e
    do ispin = 1, sys%nspin

      ! Loop for all orbit #io1:
      do io1 = 1, sys%no

        ! Retrieve orbit #io1 into "wf_io1":
        if (wfi%io_s<= io1 .and. io1 <= wfi%io_e) then
          call copy_data( &
            & wf%rwf(:, :, :, ispin, io1, ik, im), &
            & wf_io1)
        end if
        if (wfi%if_divide_orbit) &
          & call comm_bcast(wf_io1, wfi%icomm_o, wfi%irank_io(io1))

        ! Calculate overlap coefficients:
        coeff_tmp = 0d0
        do io2 = 1, io1 - 1
          if (wfi%io_s<= io2 .and. io2 <= wfi%io_e) then
            coeff_tmp(io2) = dot_wf( &
              & wf%rwf(:, :, :, ispin, io2, ik, im), &
              & wf_io1)
          end if
        end do
        if (wfi%if_divide_rspace .or. wfi%if_divide_orbit) then
          call comm_summation(coeff_tmp, coeff, sys%no, wfi%icomm_ro)
        else
          coeff = coeff_tmp
        endif

        ! Calculate exclusion term "wf_exc":
        wf_exc_tmp = 0d0
        do io2 = 1, io1 - 1
          if (wfi%io_s<= io2 .and. io2 <= wfi%io_e) then
            call axpy_wf( &
              & coeff(io2), wf%rwf(:, :, :, ispin, io2, ik, im), &
              & wf_exc_tmp)
          end if
        end do
        if (wfi%if_divide_orbit) then
          call comm_summation(wf_exc_tmp, wf_exc, nsize_rg, wfi%icomm_o)
        else
          call copy_data(wf_exc_tmp, wf_exc)
        end if

        if (wfi%io_s<= io1 .and. io1 <= wfi%io_e) then
          ! Exclude non-orthonormal component:
          call axpy_wf(-1d0, wf_exc, wf_io1)
          ! Normalization:
          norm2_tmp = dot_wf(wf_io1, wf_io1)
          if (wfi%if_divide_rspace) then
            call comm_summation(norm2_tmp, norm2, wfi%icomm_r)
          else
            norm2 = norm2_tmp
          endif
          call scal_wf(1d0 / sqrt(norm2), wf_io1)
          ! Write back to "rwf":
          call copy_data( &
            & wf_io1, &
            & wf%rwf(:, :, :, ispin, io1, ik, im))
        end if
      end do !io1

    end do !ispin
    end do !ik
    end do !im

    return
  contains

  ! Dot product of two wavefunctions:
  real(8) function dot_wf(x, y) result(p)
    implicit none
    real(8), intent(in) :: x( &
      & rg%is_array(1):rg%ie_array(1), &
      & rg%is_array(2):rg%ie_array(2), &
      & rg%is_array(3):rg%ie_array(3))
    real(8), intent(in) :: y( &
      & rg%is_array(1):rg%ie_array(1), &
      & rg%is_array(2):rg%ie_array(2), &
      & rg%is_array(3):rg%ie_array(3))
    integer :: i1, i2, i3
    p = 0d0
    !$omp parallel do collapse(2) default(shared) private(i1,i2,i3) reduction(+:p)
    do i3 = rg%is(3), rg%ie(3)
      do i2 = rg%is(2), rg%ie(2)
        do i1 = rg%is(1), rg%ie(1)
          p = p + x(i1, i2, i3) * y(i1, i2, i3)
        end do
      end do
    end do
    !$omp end parallel do
    p = p * sys%hvol
    return
  end function dot_wf

  ! Constant times a wavefunction plus a wavefunction:
  subroutine axpy_wf(a, x, y)
    implicit none
    real(8), intent(in) :: a
    real(8), intent(in) :: x( &
      & rg%is_array(1):rg%ie_array(1), &
      & rg%is_array(2):rg%ie_array(2), &
      & rg%is_array(3):rg%ie_array(3))
    real(8), intent(inout) :: y( &
      & rg%is_array(1):rg%ie_array(1), &
      & rg%is_array(2):rg%ie_array(2), &
      & rg%is_array(3):rg%ie_array(3))
    integer :: i1, i2, i3
    !$omp parallel do collapse(2) default(shared) private(i1,i2,i3)
    do i3 = rg%is(3), rg%ie(3)
      do i2 = rg%is(2), rg%ie(2)
        do i1 = rg%is(1), rg%ie(1)
          y(i1, i2, i3) = a * x(i1, i2, i3) + y(i1, i2, i3)
        end do
      end do
    end do
    !$omp end parallel do
    return
  end subroutine axpy_wf


  ! Scales a wavefunction by a constant
  subroutine scal_wf(a, x)
    implicit none
    real(8), intent(in) :: a
    real(8), intent(inout) :: x( &
      & rg%is_array(1):rg%ie_array(1), &
      & rg%is_array(2):rg%ie_array(2), &
      & rg%is_array(3):rg%ie_array(3))
    integer :: i1, i2, i3
    !$omp parallel do collapse(2) default(shared) private(i1,i2,i3)
    do i3 = rg%is(3), rg%ie(3)
      do i2 = rg%is(2), rg%ie(2)
        do i1 = rg%is(1), rg%ie(1)
          x(i1, i2, i3) = a * x(i1, i2, i3)
        end do
      end do
    end do
    !$omp end parallel do
    return
  end subroutine scal_wf

  end subroutine gram_schmidt_col_real8


!=======================================================================

  subroutine gram_schmidt_col_cblas(sys, rg, wfi, wf)
    ! Only for the colinear L(S)DA:
    use structures, only: s_dft_system, s_rgrid, s_parallel_info, s_orbital
    use pack_unpack, only: copy_data
    use timer
    use communication, only: comm_bcast, comm_summation
#if defined(USE_OPENACC) && defined(USE_CUDA_CUBLASV2)
    use cudafor
    use cublas_v2, only: cublaszgemm, cublaszdotc, cublaszdscal, cublaszaxpy, &
                         cublasHandle, cublasCreate, cublasSetStream, cublasDestroy, &
                         CUBLAS_OP_C, CUBLAS_OP_N
#endif
    implicit none
    type(s_dft_system),   intent(in)    :: sys
    type(s_rgrid),        intent(in)    :: rg
    type(s_parallel_info),intent(in)    :: wfi
    type(s_orbital),      intent(inout) :: wf

    complex(8), parameter :: zero = 0d0
    complex(8), parameter :: one = 1d0
    integer, parameter :: n_one = 1
    character(1),parameter :: TRANSA='C', TRANSB='N'
#if defined(USE_OPENACC) && defined(USE_CUDA_CUBLASV2)
    integer, parameter :: TRANSA2=CUBLAS_OP_C, TRANSB2=CUBLAS_OP_N
    type(cublasHandle) :: handle
    integer(kind=cuda_stream_kind) :: custream
    integer :: istat
    complex(8) :: norm2_tmp_comp
#endif

    integer :: nsize_rg
    integer :: ik, im, ispin, m
    integer :: io, jo, jo1, jo2
    integer :: idiv1, idiv2, idiv3, io0_s, io0_e, io1_s, io1_e, io2_s, io2_e, io3_s, io3_e
    integer :: numo0, numo1, numo1_0, numo1_1, numo2, numo2_0, numo2_1, numo3, numo3_0, numo3_1
    complex(8) :: coeff(wfi%numo), coeff_tmp(wfi%numo)
    real(8) :: norm2, norm2_tmp
    complex(8) :: wf_block(rg%is(1):rg%ie(1), rg%is(2):rg%ie(2), rg%is(3):rg%ie(3), wfi%numo)
    complex(8) :: wf_block1(rg%is(1):rg%ie(1), rg%is(2):rg%ie(2), rg%is(3):rg%ie(3), wfi%numo)
    complex(8) :: wf_block_send(rg%is(1):rg%ie(1), rg%is(2):rg%ie(2), rg%is(3):rg%ie(3), wfi%numo_max)
    complex(8) :: umat(wfi%numo_max,wfi%numo), umat_tmp(wfi%numo_max,wfi%numo)
    complex(8) :: ZDOTC

#if defined(USE_OPENACC) && defined(USE_CUDA_CUBLASV2)
    istat = cublasCreate(handle)
    istat = cudaStreamCreate(custream)
    istat = cublasSetStream(handle, custream)
#endif

    nsize_rg = (rg%ie(1)-rg%is(1)+1)*(rg%ie(2)-rg%is(2)+1)*(rg%ie(3)-rg%is(3)+1)

    do im = wfi%im_s, wfi%im_e
    do ik = wfi%ik_s, wfi%ik_e
    do ispin = 1, sys%nspin

      ! Copy wave function
      do io = wfi%io_s, wfi%io_e
        jo = io - wfi%io_s + 1
        call copy_data( &
          & wf%zwf(rg%is(1):rg%ie(1), rg%is(2):rg%ie(2), rg%is(3):rg%ie(3), ispin, io, ik, im), &
          & wf_block(:, :, :, jo))
      end do

      ! Loop for all:
      do m = 0, wfi%nporbital-1
        if(wfi%id_o==m)then

          if(wfi%numo < 32) then

        ! Loop for orbit #io1 in my rank:
            do jo1 = 1, wfi%numo

              ! normalize orbital jo1
              norm2_tmp = real( ZDOTC( &
               &      nsize_rg, wf_block(:,:,:,jo1), n_one, &
               &                wf_block(:,:,:,jo1), n_one )) &
               &     * sys%hvol
              if (wfi%if_divide_rspace) then
                call comm_summation(norm2_tmp, norm2, wfi%icomm_r)
              else
                norm2 = norm2_tmp
              endif
              call ZDSCAL(nsize_rg, one/sqrt(norm2), wf_block(:,:,:,jo1), n_one)

              ! Calculate overlap coefficients:
              coeff_tmp = 0d0
              do jo2 = jo1+1, wfi%numo
                coeff_tmp(jo2) = ZDOTC( &
                 &      nsize_rg, wf_block(:,:,:,jo1), n_one, &
                 &                wf_block(:,:,:,jo2), n_one ) &
                 &     * sys%hvol
              end do
              if (wfi%if_divide_rspace) then
                call comm_summation(coeff_tmp, coeff, wfi%numo, wfi%icomm_r)
              else
                coeff = coeff_tmp
              endif

              ! Exclude non-orthonormal component:
              do jo2 = jo1+1, wfi%numo
                call ZAXPY( &
                  &  nsize_rg, - coeff(jo2), wf_block(:,:,:,jo1), n_one, &
                  &                          wf_block(:,:,:,jo2), n_one)
              end do

            end do !jo1

          else

            io0_s = 1
            io0_e = wfi%numo
            numo0 = wfi%numo

            numo1_0 = (numo0 + 1)/2
            numo1_1 = numo0 - numo1_0
            do idiv1 = 0, 1
              if( idiv1 == 0 ) then
                io1_s = io0_s
                io1_e = io0_s - 1 + numo1_0
              else
                io1_s = io0_s     + numo1_0
                io1_e = io0_e
              end if
              numo1 = io1_e - io1_s + 1

              numo2_0 = (numo1 + 1)/2
              numo2_1 = numo1 - numo2_0
              do idiv2 = 0, 1
                if( idiv2 == 0 ) then
                  io2_s = io1_s
                  io2_e = io1_s - 1 + numo2_0
                else
                  io2_s = io1_s     + numo2_0
                  io2_e = io1_e
                end if
                numo2 = io2_e - io2_s + 1

                numo3_0 = (numo2 + 1)/2
                numo3_1 = numo2 - numo3_0
                do idiv3 = 0, 1
                  if( idiv3 == 0 ) then
                    io3_s = io2_s
                    io3_e = io2_s - 1 + numo3_0
                  else
                    io3_s = io2_s     + numo3_0
                    io3_e = io2_e
                  end if
                  numo3 = io3_e - io3_s + 1

                  do jo1 = io3_s, io3_e
                    ! normaliza the orbital jo1
#if defined(USE_OPENACC) && defined(USE_CUDA_CUBLASV2)

!$acc data copyin(wf_block(:,:,:,jo1))
!$acc host_data use_device(wf_block(:,:,:,jo1))
                    istat = cublasZdotc(handle, &
                      &     nsize_rg, wf_block(:,:,:,jo1), n_one, &
                      &               wf_block(:,:,:,jo1), n_one, &
                      &     norm2_tmp_comp)
                    istat = cudaStreamSynchronize(custream)
!$acc end host_data
!$acc end data
                    norm2_tmp = real(norm2_tmp_comp) * sys%hvol
#else
                    norm2_tmp = real( ZDOTC( &
                     &      nsize_rg, wf_block(:,:,:,jo1), n_one, &
                     &                wf_block(:,:,:,jo1), n_one )) &
                     &     * sys%hvol
#endif

                    if (wfi%if_divide_rspace) then
                      call comm_summation(norm2_tmp, norm2, wfi%icomm_r)
                    else
                      norm2 = norm2_tmp
                    endif
#if defined(USE_OPENACC) && defined(USE_CUDA_CUBLASV2)

!$acc data copy(wf_block(:,:,:,jo1))
!$acc host_data use_device(wf_block(:,:,:,jo1))
                    istat = cublasZdscal(handle, &
                      & nsize_rg, one/sqrt(norm2), wf_block(:,:,:,jo1), n_one)
                    istat = cudaStreamSynchronize(custream)
!$acc end host_data
!$acc end data
#else
                    call ZDSCAL(nsize_rg, one/sqrt(norm2), wf_block(:,:,:,jo1), n_one)
#endif

                ! Calculate overlap coefficients:
                    coeff_tmp = 0d0
                    do jo2 = jo1+1, io3_e

#if defined(USE_OPENACC) && defined(USE_CUDA_CUBLASV2)
!$acc data copyin(wf_block(:,:,:,jo1), wf_block(:,:,:,jo2))
!$acc host_data use_device(wf_block(:,:,:,jo1), wf_block(:,:,:,jo2))
                      istat = cublasZdotc(handle, &
                       &      nsize_rg, wf_block(:,:,:,jo1), n_one, &
                       &                wf_block(:,:,:,jo2), n_one, &
                       &      coeff_tmp(jo2))
                      istat = cudaStreamSynchronize(custream)
!$acc end host_data
!$acc end data
                      coeff_tmp(jo2) = coeff_tmp(jo2) * sys%hvol
#else
                      coeff_tmp(jo2) = ZDOTC( &
                       &      nsize_rg, wf_block(:,:,:,jo1), n_one, &
                       &                wf_block(:,:,:,jo2), n_one ) &
                       &     * sys%hvol
#endif
                    end do
                    if (wfi%if_divide_rspace) then
                      call comm_summation(coeff_tmp(io3_s:io3_e), coeff(io3_s:io3_e), numo3, wfi%icomm_r)
                    else
                      coeff(io3_s:io3_e) = coeff_tmp(io3_s:io3_e)
                    endif

                    ! Exclude non-orthonormal component:
                    do jo2 = jo1+1, io3_e
#if defined(USE_OPENACC) && defined(USE_CUDA_CUBLASV2)
!$acc data copy(wf_block(:,:,:,jo1), wf_block(:,:,:,jo2))
!$acc host_data use_device(wf_block(:,:,:,jo1), wf_block(:,:,:,jo2))
                      istat = cublasZaxpy(handle, &
                        &  nsize_rg, - coeff(jo2), wf_block(:,:,:,jo1), n_one, &
                        &                          wf_block(:,:,:,jo2), n_one)
                      istat = cudaStreamSynchronize(custream)
!$acc end host_data
!$acc end data
#else
                      call ZAXPY( &
                        &  nsize_rg, - coeff(jo2), wf_block(:,:,:,jo1), n_one, &
                        &                          wf_block(:,:,:,jo2), n_one)
#endif
                    end do

                  end do !jo1

                  if(idiv3 == 0) then

                    umat_tmp = 0.d0
                    wf_block_send(:,:,:,1:numo3_0) = wf_block(:,:,:,io2_s:io2_s-1+numo3_0)
                    wf_block1(:,:,:,1:numo3_1) = wf_block(:,:,:,io2_s+numo3_0:io2_e)
#if defined(USE_OPENACC) && defined(USE_CUDA_CUBLASV2)
!$acc data copyin(wf_block_send, wf_block1) copy(umat_tmp)
!$acc host_data use_device(wf_block_send, wf_block1, umat_tmp)
                    istat = cublaszgemm(handle,  &
                       &  TRANSA2, TRANSB2, numo3_0, numo3_1, nsize_rg,  &
                       &  one*sys%hvol, wf_block_send, nsize_rg,  &
                       &                wf_block1, nsize_rg,  &
                       &          zero, umat_tmp, wfi%numo_max)
                    istat = cudaStreamSynchronize(custream)
!$acc end host_data
!$acc end data
#else
                    call zgemm(TRANSA, TRANSB, numo3_0, numo3_1, nsize_rg,  &
                        &  one*sys%hvol, wf_block_send, nsize_rg,  &
                        &                wf_block1, nsize_rg,  &
                        &          zero, umat_tmp, wfi%numo_max)
#endif

                   if(wfi%if_divide_rspace) then
                      call comm_summation(umat_tmp, umat, wfi%numo_max*wfi%numo, wfi%icomm_r)
                    else
                      umat = umat_tmp
                    end if

#if defined(USE_OPENACC) && defined(USE_CUDA_CUBLASV2)
!$acc data copyin(wf_block_send, umat) copy(wf_block1)
!$acc host_data use_device(wf_block_send, wf_block1, umat)
                    istat = cublaszgemm(handle,  &
                      &  TRANSB2, TRANSB2, nsize_rg, numo3_1, numo3_0,  &
                      &         - one, wf_block_send, nsize_rg,  &
                      &                umat, wfi%numo_max,  &
                      &           one, wf_block1, nsize_rg)
                    istat = cudaStreamSynchronize(custream)
!$acc end host_data
!$acc end data
#else
                    call zgemm(TRANSB, TRANSB, nsize_rg, numo3_1, numo3_0,  &
                      &         - one, wf_block_send, nsize_rg,  &
                      &                umat, wfi%numo_max,  &
                      &           one, wf_block1, nsize_rg)
#endif

                    wf_block(:,:,:,io2_s+numo3_0:io2_e) = wf_block1(:,:,:,1:numo3_1)
                  end if ! idiv3 == 0

                end do !idiv3

                if(idiv2 == 0) then

                  umat_tmp = 0.d0
                  wf_block_send(:,:,:,1:numo2_0) = wf_block(:,:,:,io1_s:io1_s-1+numo2_0)
                  wf_block1(:,:,:,1:numo2_1) = wf_block(:,:,:,io1_s+numo2_0:io1_e)

#if defined(USE_OPENACC) && defined(USE_CUDA_CUBLASV2)
!$acc data copyin(wf_block_send, wf_block1) copy(umat_tmp)
!$acc host_data use_device(wf_block_send, wf_block1, umat_tmp)
                  istat = cublaszgemm(handle, &
                      &  TRANSA2, TRANSB2, numo2_0, numo2_1, nsize_rg,  &
                      &  one*sys%hvol, wf_block_send, nsize_rg,  &
                      &                wf_block1, nsize_rg,  &
                      &          zero, umat_tmp, wfi%numo_max)
                  istat = cudaStreamSynchronize(custream)
!$acc end host_data
!$acc end data
#else
                  call zgemm(TRANSA, TRANSB, numo2_0, numo2_1, nsize_rg,  &
                      &  one*sys%hvol, wf_block_send, nsize_rg,  &
                      &                wf_block1, nsize_rg,  &
                      &          zero, umat_tmp, wfi%numo_max)
#endif

                  if(wfi%if_divide_rspace) then
                    call comm_summation(umat_tmp, umat, wfi%numo_max*wfi%numo, wfi%icomm_r)
                  else
                    umat = umat_tmp
                  end if

#if defined(USE_OPENACC) && defined(USE_CUDA_CUBLASV2)
!$acc data copyin(wf_block_send, umat) copy(wf_block1)
!$acc host_data use_device(wf_block_send, wf_block1, umat)
                  istat = cublaszgemm(handle,  &
                    &  TRANSB2, TRANSB2, nsize_rg, numo2_1, numo2_0,  &
                    &         - one, wf_block_send, nsize_rg,  &
                    &                umat, wfi%numo_max,  &
                    &           one, wf_block1, nsize_rg)
                  istat = cudaStreamSynchronize(custream)
!$acc end host_data
!$acc end data
#else
                  call zgemm(TRANSB, TRANSB, nsize_rg, numo2_1, numo2_0,  &
                    &         - one, wf_block_send, nsize_rg,  &
                    &                umat, wfi%numo_max,  &
                    &           one, wf_block1, nsize_rg)
#endif

                  wf_block(:,:,:,io1_s+numo2_0:io1_e) = wf_block1(:,:,:,1:numo2_1)
                end if ! idiv2 == 0

              end do !idiv2

              if(idiv1 == 0) then

                umat_tmp = 0.d0
                wf_block_send(:,:,:,1:numo1_0) = wf_block(:,:,:,io0_s:io0_s-1+numo1_0)
                wf_block1(:,:,:,1:numo1_1) = wf_block(:,:,:,io0_s+numo1_0:io0_e)

#if defined(USE_OPENACC) && defined(USE_CUDA_CUBLASV2)
!$acc data copyin(wf_block_send, wf_block1) copy(umat_tmp)
!$acc host_data use_device(wf_block_send, wf_block1, umat_tmp)
                istat = cublaszgemm(handle,  &
                    &  TRANSA2, TRANSB2, numo1_0, numo1_1, nsize_rg,  &
                    &  one*sys%hvol, wf_block_send, nsize_rg,  &
                    &                wf_block1, nsize_rg,  &
                    &          zero, umat_tmp, wfi%numo_max)
                istat = cudaStreamSynchronize(custream)
!$acc end host_data
!$acc end data
#else
                call zgemm(TRANSA, TRANSB, numo1_0, numo1_1, nsize_rg,  &
                    &  one*sys%hvol, wf_block_send, nsize_rg,  &
                    &                wf_block1, nsize_rg,  &
                    &          zero, umat_tmp, wfi%numo_max)
#endif

                if(wfi%if_divide_rspace) then
                  call comm_summation(umat_tmp, umat, wfi%numo_max*wfi%numo, wfi%icomm_r)
                else
                  umat = umat_tmp
                end if

#if defined(USE_OPENACC) && defined(USE_CUDA_CUBLASV2)
!$acc data copyin(wf_block_send, umat) copy(wf_block1)
!$acc host_data use_device(wf_block_send, wf_block1, umat)
                istat = cublaszgemm(handle,  &
                  &  TRANSB2, TRANSB2, nsize_rg, numo1_1, numo1_0,  &
                  &         - one, wf_block_send, nsize_rg,  &
                  &                umat, wfi%numo_max,  &
                  &           one, wf_block1, nsize_rg)
                istat = cudaStreamSynchronize(custream)
!$acc end host_data
!$acc end data
#else
                call zgemm(TRANSB, TRANSB, nsize_rg, numo1_1, numo1_0,  &
                  &         - one, wf_block_send, nsize_rg,  &
                  &                umat, wfi%numo_max,  &
                  &           one, wf_block1, nsize_rg)
#endif
                wf_block(:,:,:,io0_s+numo1_0:io0_e) = wf_block1(:,:,:,1:numo1_1)
              end if ! idiv1 == 0

            end do !idiv1

          end if ! if(wfi%numo < 32) else

        end if !m==wfi%id_o

        if (wfi%if_divide_orbit) then
          call copy_data( wf_block(:,:,:,:), wf_block_send(:,:,:,1:wfi%numo) )
          wf_block_send(:,:,:,wfi%numo+1:wfi%numo_max) = 0d0
          call comm_bcast( wf_block_send, wfi%icomm_o, wfi%irank_io(wfi%io_s_all(m)))

          umat_tmp=0.d0
          if( wfi%id_o > m )then
            call zgemm(TRANSA, TRANSB, wfi%numo_max, wfi%numo, nsize_rg,  &
              &   one*sys%hvol, wf_block_send(:,:,:,1), nsize_rg,  &
              &                wf_block(:,:,:,1), nsize_rg,  &
              &          zero, umat_tmp, wfi%numo_max )

            if(wfi%if_divide_rspace) then
              call comm_summation(umat_tmp, umat, wfi%numo_max*wfi%numo, wfi%icomm_r)
            else
              umat = umat_tmp
            end if

            call zgemm(TRANSB, TRANSB, nsize_rg, wfi%numo, wfi%numo_max,  &
              &         - one, wf_block_send(:,:,:,1), nsize_rg,  &
              &                umat, wfi%numo_max,  &
              &           one, wf_block(:,:,:,1), nsize_rg)
          end if
        end if ! wfi%if_divide_orbit

      end do !m

      do io = wfi%io_s, wfi%io_e
        jo = io - wfi%io_s + 1
        call copy_data( &
          & wf_block(:, :, :, jo), &
          & wf%zwf(rg%is(1):rg%ie(1), rg%is(2):rg%ie(2), rg%is(3):rg%ie(3), ispin, io, ik, im))
      end do

    end do !ispin
    end do !ik
    end do !im

#if defined(USE_OPENACC) && defined(USE_CUDA_CUBLASV2)
    istat = cudaStreamDestroy(custream)
    istat = cublasDestroy(handle)
#endif
    return

  end subroutine gram_schmidt_col_cblas

  !=======================================================================

    subroutine gram_schmidt_col_cblas_old(sys, rg, wfi, wf)
      ! Only for the colinear L(S)DA:
      use structures, only: s_dft_system, s_rgrid, s_parallel_info, s_orbital
      use pack_unpack, only: copy_data
      use timer
      use communication, only: comm_bcast, comm_summation
      implicit none
      type(s_dft_system),   intent(in)    :: sys
      type(s_rgrid),        intent(in)    :: rg
      type(s_parallel_info),intent(in)    :: wfi
      type(s_orbital),      intent(inout) :: wf

      complex(8), parameter :: zero = 0d0
      complex(8), parameter :: one = 1d0
      integer, parameter :: n_one = 1
      character(1),parameter :: TRANSA='C', TRANSB='N'

      integer :: nsize_rg
      integer :: ik, im, ispin, m
      integer :: io, jo, jo1, jo2
      complex(8) :: coeff(wfi%numo), coeff_tmp(wfi%numo)
      real(8) :: norm2, norm2_tmp
      complex(8) :: wf_block(rg%is(1):rg%ie(1), rg%is(2):rg%ie(2), rg%is(3):rg%ie(3), wfi%numo)
      complex(8) :: wf_block_send(rg%is(1):rg%ie(1), rg%is(2):rg%ie(2), rg%is(3):rg%ie(3), wfi%numo_max)
      complex(8) :: umat(wfi%numo_max,wfi%numo), umat_tmp(wfi%numo_max,wfi%numo)
      complex(8) :: ZDOTC

      nsize_rg = (rg%ie(1)-rg%is(1)+1)*(rg%ie(2)-rg%is(2)+1)*(rg%ie(3)-rg%is(3)+1)

      do im = wfi%im_s, wfi%im_e
      do ik = wfi%ik_s, wfi%ik_e
      do ispin = 1, sys%nspin

        ! Copy wave function
        do io = wfi%io_s, wfi%io_e
          jo = io - wfi%io_s + 1
          call copy_data( &
            & wf%zwf(rg%is(1):rg%ie(1), rg%is(2):rg%ie(2), rg%is(3):rg%ie(3), ispin, io, ik, im), &
            & wf_block(:, :, :, jo))
        end do

        ! Loop for all:
        do m = 0, wfi%nporbital-1
          if(wfi%id_o==m)then
          ! Loop for orbit #io1 in my rank:
            do jo1 = 1, wfi%numo

              ! normalize orbital jo1
              norm2_tmp = real( ZDOTC( &
               &      nsize_rg, wf_block(:,:,:,jo1), n_one, &
               &                wf_block(:,:,:,jo1), n_one )) &
               &     * sys%hvol
              if (wfi%if_divide_rspace) then
                call comm_summation(norm2_tmp, norm2, wfi%icomm_r)
              else
                norm2 = norm2_tmp
              endif
              call ZDSCAL(nsize_rg, one/sqrt(norm2), wf_block(:,:,:,jo1), n_one)

              ! Calculate overlap coefficients:
              coeff_tmp = 0d0
              do jo2 = jo1+1, wfi%numo
                coeff_tmp(jo2) = ZDOTC( &
                 &      nsize_rg, wf_block(:,:,:,jo1), n_one, &
                 &                wf_block(:,:,:,jo2), n_one ) &
                 &     * sys%hvol
              end do
              if (wfi%if_divide_rspace) then
                call comm_summation(coeff_tmp, coeff, wfi%numo, wfi%icomm_r)
              else
                coeff = coeff_tmp
              endif

              ! Exclude non-orthonormal component:
              do jo2 = jo1+1, wfi%numo
                call ZAXPY( &
                  &  nsize_rg, - coeff(jo2), wf_block(:,:,:,jo1), n_one, &
                  &                          wf_block(:,:,:,jo2), n_one)
              end do

            end do !jo1

          end if !m==wfi%id_o

          if (wfi%if_divide_orbit) then
            call copy_data( wf_block(:,:,:,:), wf_block_send(:,:,:,1:wfi%numo) )
            wf_block_send(:,:,:,wfi%numo+1:wfi%numo_max) = 0d0
            call comm_bcast( wf_block_send, wfi%icomm_o, wfi%irank_io(wfi%io_s_all(m)))

            umat_tmp=0.d0
            if( wfi%id_o > m )then
              call zgemm(TRANSA, TRANSB, wfi%numo_max, wfi%numo, nsize_rg,  &
                &   one*sys%hvol, wf_block_send(:,:,:,1), nsize_rg,  &
                &                wf_block(:,:,:,1), nsize_rg,  &
                &          zero, umat_tmp, wfi%numo_max )

              if(wfi%if_divide_rspace) then
                call comm_summation(umat_tmp, umat, wfi%numo_max*wfi%numo, wfi%icomm_r)
              else
                umat = umat_tmp
              end if

              call zgemm(TRANSB, TRANSB, nsize_rg, wfi%numo, wfi%numo_max,  &
                &         - one, wf_block_send(:,:,:,1), nsize_rg,  &
                &                umat, wfi%numo_max,  &
                &           one, wf_block(:,:,:,1), nsize_rg)
            end if
          end if ! wfi%if_divide_orbit

        end do !m

        do io = wfi%io_s, wfi%io_e
          jo = io - wfi%io_s + 1
          call copy_data( &
            & wf_block(:, :, :, jo), &
            & wf%zwf(rg%is(1):rg%ie(1), rg%is(2):rg%ie(2), rg%is(3):rg%ie(3), ispin, io, ik, im))
        end do

      end do !ispin
      end do !ik
      end do !im

      return

    end subroutine gram_schmidt_col_cblas_old

  !=======================================================================


    subroutine gram_schmidt_col_rblas(sys, rg, wfi, wf)
      ! Only for the colinear L(S)DA:
      use structures, only: s_dft_system, s_rgrid, s_parallel_info, s_orbital
      use pack_unpack, only: copy_data
      use timer
      use communication, only: comm_bcast, comm_summation
      implicit none
      type(s_dft_system),   intent(in)    :: sys
      type(s_rgrid),        intent(in)    :: rg
      type(s_parallel_info),intent(in)    :: wfi
      type(s_orbital),      intent(inout) :: wf

      real(8), parameter :: zero = 0d0
      real(8), parameter :: one = 1d0
      integer, parameter :: n_one = 1
      character(1),parameter :: TRANSA='T', TRANSB='N'

      integer :: nsize_rg
      integer :: ik, im, ispin, m
      integer :: io, jo, jo1, jo2
      integer :: idiv1, idiv2, idiv3, io0_s, io0_e, io1_s, io1_e, io2_s, io2_e, io3_s, io3_e
      integer :: numo0, numo1, numo1_0, numo1_1, numo2, numo2_0, numo2_1, numo3, numo3_0, numo3_1
      real(8) :: coeff(wfi%numo), coeff_tmp(wfi%numo)
      real(8) :: norm2, norm2_tmp
      real(8) :: wf_block(rg%is(1):rg%ie(1), rg%is(2):rg%ie(2), rg%is(3):rg%ie(3), wfi%numo)
      real(8) :: wf_block1(rg%is(1):rg%ie(1), rg%is(2):rg%ie(2), rg%is(3):rg%ie(3), wfi%numo)
      real(8) :: wf_block_send(rg%is(1):rg%ie(1), rg%is(2):rg%ie(2), rg%is(3):rg%ie(3), wfi%numo_max)
      real(8) :: umat(wfi%numo_max,wfi%numo), umat_tmp(wfi%numo_max,wfi%numo)
      real(8) :: DDOT

#ifdef USE_OPENACC
      call gram_schmidt_col_rblas_gpu(sys, rg, wfi, wf)
      return
#endif

      nsize_rg = (rg%ie(1)-rg%is(1)+1)*(rg%ie(2)-rg%is(2)+1)*(rg%ie(3)-rg%is(3)+1)

      do im = wfi%im_s, wfi%im_e
      do ik = wfi%ik_s, wfi%ik_e
      do ispin = 1, sys%nspin

        ! Copy wave function
        do io = wfi%io_s, wfi%io_e
          jo = io - wfi%io_s + 1
          call copy_data( &
            &  wf%rwf(rg%is(1):rg%ie(1), rg%is(2):rg%ie(2), rg%is(3):rg%ie(3), ispin, io, ik, im) , &
            & wf_block(:, :, :, jo))
        end do

        ! Loop for all:
        do m = 0, wfi%nporbital-1
          if(wfi%id_o==m)then

          if(wfi%numo < 32) then

          ! Loop for orbit #jo1 in my rank:
            do jo1 = 1, wfi%numo

              ! normaliza the orbital jo1
              norm2_tmp = DDOT( &
               &      nsize_rg, wf_block(:,:,:,jo1), n_one, &
               &                wf_block(:,:,:,jo1), n_one ) &
               &     * sys%hvol
              if (wfi%if_divide_rspace) then
                call comm_summation(norm2_tmp, norm2, wfi%icomm_r)
              else
                norm2 = norm2_tmp
              endif
              call DSCAL(nsize_rg, one/sqrt(norm2), wf_block(:,:,:,jo1), n_one)

              ! Calculate overlap coefficients:
              coeff_tmp = 0d0
              do jo2 = jo1+1, wfi%numo
                coeff_tmp(jo2) = DDOT( &
                 &      nsize_rg, wf_block(:,:,:,jo1), n_one, &
                 &                wf_block(:,:,:,jo2), n_one ) &
                 &     * sys%hvol
              end do
              if (wfi%if_divide_rspace) then
                call comm_summation(coeff_tmp, coeff, wfi%numo, wfi%icomm_r)
              else
                coeff = coeff_tmp
              endif

              ! Exclude non-orthonormal component:
              do jo2 = jo1+1, wfi%numo
                call DAXPY( &
                  &  nsize_rg, - coeff(jo2), wf_block(:,:,:,jo1), n_one, &
                  &                          wf_block(:,:,:,jo2), n_one)
              end do

            end do !jo1

          else

            io0_s = 1
            io0_e = wfi%numo
            numo0 = wfi%numo

            numo1_0 = (numo0 + 1)/2
            numo1_1 = numo0 - numo1_0
            do idiv1 = 0, 1
              if( idiv1 == 0 ) then
                io1_s = io0_s
                io1_e = io0_s - 1 + numo1_0
              else
                io1_s = io0_s     + numo1_0
                io1_e = io0_e
              end if
              numo1 = io1_e - io1_s + 1

              numo2_0 = (numo1 + 1)/2
              numo2_1 = numo1 - numo2_0
              do idiv2 = 0, 1
                if( idiv2 == 0 ) then
                  io2_s = io1_s
                  io2_e = io1_s - 1 + numo2_0
                else
                  io2_s = io1_s     + numo2_0
                  io2_e = io1_e
                end if
                numo2 = io2_e - io2_s + 1

                numo3_0 = (numo2 + 1)/2
                numo3_1 = numo2 - numo3_0
                do idiv3 = 0, 1
                  if( idiv3 == 0 ) then
                    io3_s = io2_s
                    io3_e = io2_s - 1 + numo3_0
                  else
                    io3_s = io2_s     + numo3_0
                    io3_e = io2_e
                  end if
                  numo3 = io3_e - io3_s + 1

                  do jo1 = io3_s, io3_e
                    ! normaliza the orbital jo1
                    norm2_tmp = DDOT( &
                     &      nsize_rg, wf_block(:,:,:,jo1), n_one, &
                     &                wf_block(:,:,:,jo1), n_one ) &
                     &     * sys%hvol
                    if (wfi%if_divide_rspace) then
                      call comm_summation(norm2_tmp, norm2, wfi%icomm_r)
                    else
                      norm2 = norm2_tmp
                    endif
                    call DSCAL(nsize_rg, one/sqrt(norm2), wf_block(:,:,:,jo1), n_one)

                ! Calculate overlap coefficients:
                    coeff_tmp = 0d0
                    do jo2 = jo1+1, io3_e
                      coeff_tmp(jo2) = DDOT( &
                       &      nsize_rg, wf_block(:,:,:,jo1), n_one, &
                       &                wf_block(:,:,:,jo2), n_one ) &
                       &     * sys%hvol
                    end do
                    if (wfi%if_divide_rspace) then
                      call comm_summation(coeff_tmp(io3_s:io3_e), coeff(io3_s:io3_e), numo3, wfi%icomm_r)
                    else
                      coeff(io3_s:io3_e) = coeff_tmp(io3_s:io3_e)
                    endif

                    ! Exclude non-orthonormal component:
                    do jo2 = jo1+1, io3_e
                      call DAXPY( &
                        &  nsize_rg, - coeff(jo2), wf_block(:,:,:,jo1), n_one, &
                        &                          wf_block(:,:,:,jo2), n_one)
                    end do

                  end do !jo1

                  if(idiv3 == 0) then
                    umat_tmp = 0.d0
                    wf_block_send(:,:,:,1:numo3_0) = wf_block(:,:,:,io2_s:io2_s-1+numo3_0)
                    wf_block1(:,:,:,1:numo3_1) = wf_block(:,:,:,io2_s+numo3_0:io2_e)
                    call dgemm(TRANSA, TRANSB, numo3_0, numo3_1, nsize_rg,  &
                        &      sys%hvol, wf_block_send, nsize_rg,  &
                        &                wf_block1, nsize_rg,  &
                        &          zero, umat_tmp, wfi%numo_max)
                    if(wfi%if_divide_rspace) then
                      call comm_summation(umat_tmp, umat, wfi%numo_max*wfi%numo, wfi%icomm_r)
                    else
                      umat = umat_tmp
                    end if

                    call dgemm(TRANSB, TRANSB, nsize_rg, numo3_1, numo3_0,  &
                      &         - one, wf_block_send, nsize_rg,  &
                      &                umat, wfi%numo_max,  &
                      &           one, wf_block1, nsize_rg)
                    wf_block(:,:,:,io2_s+numo3_0:io2_e) = wf_block1(:,:,:,1:numo3_1)
                  end if ! idiv3 == 0

                end do !idiv3

                if(idiv2 == 0) then

                  umat_tmp = 0.d0
                  wf_block_send(:,:,:,1:numo2_0) = wf_block(:,:,:,io1_s:io1_s-1+numo2_0)
                  wf_block1(:,:,:,1:numo2_1) = wf_block(:,:,:,io1_s+numo2_0:io1_e)
                  call dgemm(TRANSA, TRANSB, numo2_0, numo2_1, nsize_rg,  &
                      &      sys%hvol, wf_block_send, nsize_rg,  &
                      &                wf_block1, nsize_rg,  &
                      &          zero, umat_tmp, wfi%numo_max)

                  if(wfi%if_divide_rspace) then
                    call comm_summation(umat_tmp, umat, wfi%numo_max*wfi%numo, wfi%icomm_r)
                  else
                    umat = umat_tmp
                  end if

                  call dgemm(TRANSB, TRANSB, nsize_rg, numo2_1, numo2_0,  &
                    &         - one, wf_block_send, nsize_rg,  &
                    &                umat, wfi%numo_max,  &
                    &           one, wf_block1, nsize_rg)
                  wf_block(:,:,:,io1_s+numo2_0:io1_e) = wf_block1(:,:,:,1:numo2_1)
                end if ! idiv2 == 0

              end do !idiv2

              if(idiv1 == 0) then
                umat_tmp = 0.d0
                wf_block_send(:,:,:,1:numo1_0) = wf_block(:,:,:,io0_s:io0_s-1+numo1_0)
                wf_block1(:,:,:,1:numo1_1) = wf_block(:,:,:,io0_s+numo1_0:io0_e)
                call dgemm(TRANSA, TRANSB, numo1_0, numo1_1, nsize_rg,  &
                    &      sys%hvol, wf_block_send, nsize_rg,  &
                    &                wf_block1, nsize_rg,  &
                    &          zero, umat_tmp, wfi%numo_max)

                if(wfi%if_divide_rspace) then
                  call comm_summation(umat_tmp, umat, wfi%numo_max*wfi%numo, wfi%icomm_r)
                else
                  umat = umat_tmp
                end if

                call dgemm(TRANSB, TRANSB, nsize_rg, numo1_1, numo1_0,  &
                  &         - one, wf_block_send, nsize_rg,  &
                  &                umat, wfi%numo_max,  &
                  &           one, wf_block1, nsize_rg)
                wf_block(:,:,:,io0_s+numo1_0:io0_e) = wf_block1(:,:,:,1:numo1_1)
              end if ! idiv1 == 0

            end do !idiv1

          end if ! if(wfi%numo < 32) else

          end if !m==wfi%id_o

          if (wfi%if_divide_orbit) then
            call copy_data( wf_block(:,:,:,:), wf_block_send(:,:,:,1:wfi%numo) )
            wf_block_send(:,:,:,wfi%numo+1:wfi%numo_max) = 0d0
            call comm_bcast( wf_block_send, wfi%icomm_o, wfi%irank_io(wfi%io_s_all(m)))

            umat_tmp=0.d0
            if( wfi%id_o > m )then
              call dgemm(TRANSA, TRANSB, wfi%numo_max, wfi%numo, nsize_rg,  &
                &      sys%hvol, wf_block_send(:,:,:,1), nsize_rg,  &
                &                wf_block(:,:,:,1), nsize_rg,  &
                &          zero, umat_tmp, wfi%numo_max )

              if(wfi%if_divide_rspace) then
                call comm_summation(umat_tmp, umat, wfi%numo_max*wfi%numo, wfi%icomm_r)
              else
                umat = umat_tmp
              end if

              call dgemm(TRANSB, TRANSB, nsize_rg, wfi%numo, wfi%numo_max,  &
                &         - one, wf_block_send(:,:,:,1), nsize_rg,  &
                &                umat, wfi%numo_max,  &
                &           one, wf_block(:,:,:,1), nsize_rg)
            end if
          end if ! wfi%if_divide_orbit

        end do !m

        do io = wfi%io_s, wfi%io_e
          jo = io - wfi%io_s + 1
          call copy_data( &
            & wf_block(:, :, :, jo), &
            & wf%rwf(rg%is(1):rg%ie(1), rg%is(2):rg%ie(2), rg%is(3):rg%ie(3), ispin, io, ik, im))
        end do

      end do !ispin
      end do !ik
      end do !im

      return

    end subroutine gram_schmidt_col_rblas

#ifdef USE_OPENACC
    !=======================================================================
    ! gram_schmidt_col_rblas on the GPU (OpenACC builds).
    !
    ! The algorithm and its order of operations are those of gram_schmidt_col_rblas:
    !  - within the orbitals of a rank, three levels of halving; modified Gram-Schmidt
    !    (normalize, overlaps, subtract) on the leaves and one block projection per halving;
    !    fewer than 32 local orbitals are a single leaf;
    !  - across orbital ranks (if_divide_orbit), rank m broadcasts its orthonormalized block
    !    and the ranks id_o > m project it out of their own block.
    ! The orbital blocks stay on the device: they are packed from wf%rwf by an OpenACC kernel,
    ! updated with cuBLAS, broadcast as device data (CUDA-aware MPI, as for the halo exchange
    ! in sendrecv_grid), and unpacked into wf%rwf. Only the norms and overlap matrices that
    ! need an r-space summation (if_divide_rspace) pass through the host.
    ! The leaf DDOT/DSCAL/DAXPY operations are OpenACC loops and the block products are cuBLAS
    ! DGEMMs, as in the complex-orbital GPU subspace diagonalization (ssdg_zwf_cblas_gpu).
    subroutine gram_schmidt_col_rblas_gpu(sys, rg, wfi, wf)
      use structures, only: s_dft_system, s_rgrid, s_parallel_info, s_orbital
      use communication, only: comm_bcast_device
      use openacc, only: acc_wait, acc_async_sync
      implicit none
      type(s_dft_system),   intent(in)    :: sys
      type(s_rgrid),        intent(in)    :: rg
      type(s_parallel_info),intent(in)    :: wfi
      type(s_orbital),      intent(inout) :: wf

      integer :: nsize_rg, numo, numo_max, nx, nxy, io_s
      integer :: is1, is2, is3, ie1, ie2, ie3
      integer :: ik, im, ispin, m
      integer :: i, jo, ix, iy, iz
      integer :: idiv1, idiv2, idiv3, io0_s, io0_e, io1_s, io1_e, io2_s, io2_e, io3_s, io3_e
      integer :: numo0, numo1, numo1_0, numo1_1, numo2, numo2_0, numo2_1, numo3_0, numo3_1
      ! Device work arrays. They are used only inside the OpenACC data region below, so their
      ! host storage is never touched.
      real(8) :: wf_block((rg%ie(1)-rg%is(1)+1)*(rg%ie(2)-rg%is(2)+1)*(rg%ie(3)-rg%is(3)+1), wfi%numo)
      real(8) :: wf_block_send((rg%ie(1)-rg%is(1)+1)*(rg%ie(2)-rg%is(2)+1)*(rg%ie(3)-rg%is(3)+1), wfi%numo_max)
      ! Small matrices: the device copy is used by cuBLAS, the host copy by the r-space summation.
      real(8) :: umat(wfi%numo_max,wfi%numo), umat_tmp(wfi%numo_max,wfi%numo)
      real(8) :: coeff(wfi%numo), coeff_tmp(wfi%numo)

      is1 = rg%is(1); is2 = rg%is(2); is3 = rg%is(3)
      ie1 = rg%ie(1); ie2 = rg%ie(2); ie3 = rg%ie(3)
      nx  = ie1-is1+1
      nxy = nx*(ie2-is2+1)
      nsize_rg = nxy*(ie3-is3+1)
      numo = wfi%numo
      numo_max = wfi%numo_max
      io_s = wfi%io_s

!$acc data create(wf_block, wf_block_send, umat, coeff)
      do im = wfi%im_s, wfi%im_e
      do ik = wfi%ik_s, wfi%ik_e
      do ispin = 1, sys%nspin

        ! Copy wave function
!$acc parallel loop collapse(4) private(jo,iz,iy,ix,i)
        do jo = 1, numo
        do iz = is3, ie3
        do iy = is2, ie2
        do ix = is1, ie1
          i = 1 + (ix-is1) + nx*(iy-is2) + nxy*(iz-is3)
          wf_block(i,jo) = wf%rwf(ix,iy,iz,ispin,io_s-1+jo,ik,im)
        end do
        end do
        end do
        end do

        ! Loop for all:
        do m = 0, wfi%nporbital-1
          if(wfi%id_o==m)then

          if(numo < 32) then

            call rblas_gpu_leaf(nsize_rg, numo, wf_block, coeff, coeff_tmp, 1, numo, sys%hvol, &
              &                 wfi%if_divide_rspace, wfi%icomm_r)

          else

            io0_s = 1
            io0_e = numo
            numo0 = numo

            numo1_0 = (numo0 + 1)/2
            numo1_1 = numo0 - numo1_0
            do idiv1 = 0, 1
              if( idiv1 == 0 ) then
                io1_s = io0_s
                io1_e = io0_s - 1 + numo1_0
              else
                io1_s = io0_s     + numo1_0
                io1_e = io0_e
              end if
              numo1 = io1_e - io1_s + 1

              numo2_0 = (numo1 + 1)/2
              numo2_1 = numo1 - numo2_0
              do idiv2 = 0, 1
                if( idiv2 == 0 ) then
                  io2_s = io1_s
                  io2_e = io1_s - 1 + numo2_0
                else
                  io2_s = io1_s     + numo2_0
                  io2_e = io1_e
                end if
                numo2 = io2_e - io2_s + 1

                numo3_0 = (numo2 + 1)/2
                numo3_1 = numo2 - numo3_0
                do idiv3 = 0, 1
                  if( idiv3 == 0 ) then
                    io3_s = io2_s
                    io3_e = io2_s - 1 + numo3_0
                  else
                    io3_s = io2_s     + numo3_0
                    io3_e = io2_e
                  end if

                  call rblas_gpu_leaf(nsize_rg, numo, wf_block, coeff, coeff_tmp, io3_s, io3_e, sys%hvol, &
                    &                 wfi%if_divide_rspace, wfi%icomm_r)

                  if(idiv3 == 0) then
                    call rblas_gpu_project(nsize_rg, numo3_0, numo3_1, &
                      &  wf_block(:,io2_s:io2_s-1+numo3_0), wf_block(:,io2_s+numo3_0:io2_e), &
                      &  numo_max, umat, umat_tmp, sys%hvol, wfi%if_divide_rspace, wfi%icomm_r)
                  end if ! idiv3 == 0

                end do !idiv3

                if(idiv2 == 0) then
                  call rblas_gpu_project(nsize_rg, numo2_0, numo2_1, &
                    &  wf_block(:,io1_s:io1_s-1+numo2_0), wf_block(:,io1_s+numo2_0:io1_e), &
                    &  numo_max, umat, umat_tmp, sys%hvol, wfi%if_divide_rspace, wfi%icomm_r)
                end if ! idiv2 == 0

              end do !idiv2

              if(idiv1 == 0) then
                call rblas_gpu_project(nsize_rg, numo1_0, numo1_1, &
                  &  wf_block(:,io0_s:io0_s-1+numo1_0), wf_block(:,io0_s+numo1_0:io0_e), &
                  &  numo_max, umat, umat_tmp, sys%hvol, wfi%if_divide_rspace, wfi%icomm_r)
              end if ! idiv1 == 0

            end do !idiv1

          end if ! if(numo < 32) else

          end if !m==wfi%id_o

          if (wfi%if_divide_orbit) then
            if (wfi%id_o == m) then
!$acc parallel loop collapse(2) private(jo,i)
              do jo = 1, numo_max
              do i = 1, nsize_rg
                if (jo <= numo) then
                  wf_block_send(i,jo) = wf_block(i,jo)
                else
                  wf_block_send(i,jo) = 0d0
                end if
              end do
              end do
            end if
            ! Pending cuBLAS work may still read wf_block_send: finish it before MPI writes.
            call acc_wait(acc_async_sync)
!$acc host_data use_device(wf_block_send)
            call comm_bcast_device( wf_block_send, wfi%icomm_o, wfi%irank_io(wfi%io_s_all(m)))
!$acc end host_data

            if( wfi%id_o > m )then
              call rblas_gpu_project(nsize_rg, numo_max, numo, wf_block_send, wf_block, &
                &  numo_max, umat, umat_tmp, sys%hvol, wfi%if_divide_rspace, wfi%icomm_r)
            end if
          end if ! wfi%if_divide_orbit

        end do !m

        call acc_wait(acc_async_sync)
!$acc parallel loop collapse(4) private(jo,iz,iy,ix,i)
        do jo = 1, numo
        do iz = is3, ie3
        do iy = is2, ie2
        do ix = is1, ie1
          i = 1 + (ix-is1) + nx*(iy-is2) + nxy*(iz-is3)
          wf%rwf(ix,iy,iz,ispin,io_s-1+jo,ik,im) = wf_block(i,jo)
        end do
        end do
        end do
        end do

      end do !ispin
      end do !ik
      end do !im
!$acc end data

      return

    end subroutine gram_schmidt_col_rblas_gpu

    !-----------------------------------------------------------------------
    ! Modified Gram-Schmidt of the columns js..je of w (device data), as in the leaves of
    ! gram_schmidt_col_rblas: normalize column jo1 (DDOT, DSCAL), take its overlaps with the
    ! columns jo1+1..je (one DDOT each), and subtract its component from them (one DAXPY each).
    subroutine rblas_gpu_leaf(nsize_rg, ncol, w, coeff, coeff_tmp, js, je, hvol, if_divide_rspace, icomm_r)
      use communication, only: comm_summation
      implicit none
      integer, intent(in)    :: nsize_rg, ncol, js, je, icomm_r
      real(8), intent(inout) :: w(nsize_rg, ncol)
      real(8), intent(inout) :: coeff(ncol), coeff_tmp(ncol)
      real(8), intent(in)    :: hvol
      logical, intent(in)    :: if_divide_rspace

      real(8), parameter :: one = 1d0
      integer :: i, jo1, jo2, nrest
      real(8) :: norm2, norm2_tmp, rnorm, s

      do jo1 = js, je

        ! normalize the orbital jo1
        norm2_tmp = 0d0
!$acc parallel loop reduction(+:norm2_tmp)
        do i = 1, nsize_rg
          norm2_tmp = norm2_tmp + w(i,jo1)*w(i,jo1)
        end do
        norm2_tmp = norm2_tmp * hvol
        if (if_divide_rspace) then
          call comm_summation(norm2_tmp, norm2, icomm_r)
        else
          norm2 = norm2_tmp
        endif
        rnorm = one/sqrt(norm2)
!$acc parallel loop
        do i = 1, nsize_rg
          w(i,jo1) = rnorm*w(i,jo1)
        end do

        nrest = je - jo1
        if (nrest <= 0) cycle

        ! Calculate overlap coefficients:
!$acc parallel loop gang private(s)
        do jo2 = jo1+1, je
          s = 0d0
!$acc loop vector reduction(+:s)
          do i = 1, nsize_rg
            s = s + w(i,jo1)*w(i,jo2)
          end do
          coeff(jo2) = s * hvol
        end do
        if (if_divide_rspace) then
!$acc update self(coeff(jo1+1:je))
          call comm_summation(coeff(jo1+1:je), coeff_tmp(jo1+1:je), nrest, icomm_r)
          coeff(jo1+1:je) = coeff_tmp(jo1+1:je)
!$acc update device(coeff(jo1+1:je))
        endif

        ! Exclude non-orthonormal component:
!$acc parallel loop collapse(2)
        do jo2 = jo1+1, je
        do i = 1, nsize_rg
          w(i,jo2) = w(i,jo2) - coeff(jo2)*w(i,jo1)
        end do
        end do

      end do !jo1

    end subroutine rblas_gpu_leaf

    !-----------------------------------------------------------------------
    ! Block projection of gram_schmidt_col_rblas on device data:
    !   umat = hvol * a^T b  (summed over icomm_r when the r-space is divided),  b = b - a umat.
    subroutine rblas_gpu_project(nsize_rg, na, nb, a, b, ldu, umat, umat_tmp, hvol, if_divide_rspace, icomm_r)
      use communication, only: comm_summation
      use cublas, only: CUBLAS_OP_N, CUBLAS_OP_T
      use cublas_wrapper, only: cublasDgemm_wrapper
      use openacc, only: acc_wait, acc_async_sync
      implicit none
      integer, intent(in)    :: nsize_rg, na, nb, ldu, icomm_r
      real(8), intent(in)    :: a(nsize_rg, na)
      real(8), intent(inout) :: b(nsize_rg, nb)
      real(8), intent(inout) :: umat(ldu, nb), umat_tmp(ldu, nb)
      real(8), intent(in)    :: hvol
      logical, intent(in)    :: if_divide_rspace

      real(8), parameter :: zero = 0d0
      real(8), parameter :: one = 1d0
      integer :: istat

      if (na <= 0 .or. nb <= 0) return

!$acc host_data use_device(a, b, umat)
      istat = cublasDgemm_wrapper(CUBLAS_OP_T, CUBLAS_OP_N, na, nb, nsize_rg, &
        &                         hvol, a, nsize_rg, b, nsize_rg, zero, umat, ldu)
!$acc end host_data
      call check_cublas(istat, 'gram_schmidt_col_rblas_gpu: DGEMM (overlap)')

      if (if_divide_rspace) then
        call acc_wait(acc_async_sync)
!$acc update self(umat)
        call comm_summation(umat, umat_tmp, ldu*nb, icomm_r)
        umat = umat_tmp
!$acc update device(umat)
      end if

!$acc host_data use_device(a, b, umat)
      istat = cublasDgemm_wrapper(CUBLAS_OP_N, CUBLAS_OP_N, nsize_rg, nb, na, &
        &                         -one, a, nsize_rg, umat, ldu, one, b, nsize_rg)
!$acc end host_data
      call check_cublas(istat, 'gram_schmidt_col_rblas_gpu: DGEMM (update)')
      ! Finish the update before b, a or umat are used outside cuBLAS (leaf loops, host updates,
      ! MPI): cuBLAS returns before the product is done.
      call acc_wait(acc_async_sync)

    end subroutine rblas_gpu_project

    subroutine check_cublas(istat, where)
      use iso_fortran_env, only: error_unit
      implicit none
      integer, intent(in) :: istat
      character(*), intent(in) :: where
      if (istat /= 0) then
        write(error_unit,'(a,": cuBLAS status = ",i0)') trim(where), istat
        error stop
      end if
    end subroutine check_cublas
#endif

    !=======================================================================

      subroutine gram_schmidt_col_rblas_old(sys, rg, wfi, wf)
        ! Only for the colinear L(S)DA:
        use structures, only: s_dft_system, s_rgrid, s_parallel_info, s_orbital
        use pack_unpack, only: copy_data
        use timer
        use communication, only: comm_bcast, comm_summation
        implicit none
        type(s_dft_system),   intent(in)    :: sys
        type(s_rgrid),        intent(in)    :: rg
        type(s_parallel_info),intent(in)    :: wfi
        type(s_orbital),      intent(inout) :: wf

        real(8), parameter :: zero = 0d0
        real(8), parameter :: one = 1d0
        integer, parameter :: n_one = 1
        character(1),parameter :: TRANSA='T', TRANSB='N'

        integer :: nsize_rg
        integer :: ik, im, ispin, m
        integer :: io, jo, jo1, jo2
        real(8) :: coeff(wfi%numo), coeff_tmp(wfi%numo)
        real(8) :: norm2, norm2_tmp
        real(8) :: wf_block(rg%is(1):rg%ie(1), rg%is(2):rg%ie(2), rg%is(3):rg%ie(3), wfi%numo)
        real(8) :: wf_block_send(rg%is(1):rg%ie(1), rg%is(2):rg%ie(2), rg%is(3):rg%ie(3), wfi%numo_max)
        real(8) :: umat(wfi%numo_max,wfi%numo), umat_tmp(wfi%numo_max,wfi%numo)
        real(8) :: DDOT

        nsize_rg = (rg%ie(1)-rg%is(1)+1)*(rg%ie(2)-rg%is(2)+1)*(rg%ie(3)-rg%is(3)+1)

        do im = wfi%im_s, wfi%im_e
        do ik = wfi%ik_s, wfi%ik_e
        do ispin = 1, sys%nspin

          ! Copy wave function
          do io = wfi%io_s, wfi%io_e
            jo = io - wfi%io_s + 1
            call copy_data( &
              &  wf%rwf(rg%is(1):rg%ie(1), rg%is(2):rg%ie(2), rg%is(3):rg%ie(3), ispin, io, ik, im) , &
              & wf_block(:, :, :, jo))
          end do

          ! Loop for all:
          do m = 0, wfi%nporbital-1
            if(wfi%id_o==m)then
            ! Loop for orbit #jo1 in my rank:
              do jo1 = 1, wfi%numo

                ! normaliza the orbital jo1
                norm2_tmp = DDOT( &
                 &      nsize_rg, wf_block(:,:,:,jo1), n_one, &
                 &                wf_block(:,:,:,jo1), n_one ) &
                 &     * sys%hvol
                if (wfi%if_divide_rspace) then
                  call comm_summation(norm2_tmp, norm2, wfi%icomm_r)
                else
                  norm2 = norm2_tmp
                endif
                call DSCAL(nsize_rg, one/sqrt(norm2), wf_block(:,:,:,jo1), n_one)

                ! Calculate overlap coefficients:
                coeff_tmp = 0d0
                do jo2 = jo1+1, wfi%numo
                  coeff_tmp(jo2) = DDOT( &
                   &      nsize_rg, wf_block(:,:,:,jo1), n_one, &
                   &                wf_block(:,:,:,jo2), n_one ) &
                   &     * sys%hvol
                end do
                if (wfi%if_divide_rspace) then
                  call comm_summation(coeff_tmp, coeff, wfi%numo, wfi%icomm_r)
                else
                  coeff = coeff_tmp
                endif

                ! Exclude non-orthonormal component:
                do jo2 = jo1+1, wfi%numo
                  call DAXPY( &
                    &  nsize_rg, - coeff(jo2), wf_block(:,:,:,jo1), n_one, &
                    &                          wf_block(:,:,:,jo2), n_one)
                end do

              end do !jo1

            end if !m==wfi%id_o

            if (wfi%if_divide_orbit) then
              call copy_data( wf_block(:,:,:,:), wf_block_send(:,:,:,1:wfi%numo) )
              wf_block_send(:,:,:,wfi%numo+1:wfi%numo_max) = 0d0
              call comm_bcast( wf_block_send, wfi%icomm_o, wfi%irank_io(wfi%io_s_all(m)))

              umat_tmp=0.d0
              if( wfi%id_o > m )then
                call dgemm(TRANSA, TRANSB, wfi%numo_max, wfi%numo, nsize_rg,  &
                  &      sys%hvol, wf_block_send(:,:,:,1), nsize_rg,  &
                  &                wf_block(:,:,:,1), nsize_rg,  &
                  &          zero, umat_tmp, wfi%numo_max )

                if(wfi%if_divide_rspace) then
                  call comm_summation(umat_tmp, umat, wfi%numo_max*wfi%numo, wfi%icomm_r)
                else
                  umat = umat_tmp
                end if

                call dgemm(TRANSB, TRANSB, nsize_rg, wfi%numo, wfi%numo_max,  &
                  &         - one, wf_block_send(:,:,:,1), nsize_rg,  &
                  &                umat, wfi%numo_max,  &
                  &           one, wf_block(:,:,:,1), nsize_rg)
              end if
            end if ! wfi%if_divide_orbit

          end do !m

          do io = wfi%io_s, wfi%io_e
            jo = io - wfi%io_s + 1
            call copy_data( &
              & wf_block(:, :, :, jo), &
              & wf%rwf(rg%is(1):rg%ie(1), rg%is(2):rg%ie(2), rg%is(3):rg%ie(3), ispin, io, ik, im))
          end do

        end do !ispin
        end do !ik
        end do !im

        return

      end subroutine gram_schmidt_col_rblas_old

    !=======================================================================

    subroutine gram_schmidt_col_complex8(sys, rg, wfi, wf)
      ! Only for the colinear L(S)DA:
      use structures, only: s_dft_system, s_rgrid, s_parallel_info, s_orbital
      use pack_unpack, only: copy_data
      use timer
      use communication, only: comm_bcast, comm_summation
      implicit none
      type(s_dft_system),   intent(in)    :: sys
      type(s_rgrid),        intent(in)    :: rg
      type(s_parallel_info),intent(in)    :: wfi
      type(s_orbital),      intent(inout) :: wf

      integer :: nsize_rg
      integer :: ik, im, ispin
      integer :: io1, io2
      complex(8) :: coeff(1:sys%no), coeff_tmp(1:sys%no)
      real(8) :: norm2, norm2_tmp
      complex(8), dimension( &
        & rg%is_array(1):rg%ie_array(1), &
        & rg%is_array(2):rg%ie_array(2), &
        & rg%is_array(3):rg%ie_array(3)) &
        & :: wf_io1, wf_exc, wf_exc_tmp

      complex(8), parameter :: one = 1d0

      nsize_rg =  (rg%ie_array(1) - rg%is_array(1) + 1) &
        & * (rg%ie_array(2) - rg%is_array(2) + 1) &
        & * (rg%ie_array(3) - rg%is_array(3) + 1)

      do im = wfi%im_s, wfi%im_e
      do ik = wfi%ik_s, wfi%ik_e
      do ispin = 1, sys%nspin

        ! Loop for all orbit #io1:
        do io1 = 1, sys%no

          ! Retrieve orbit #io1 into "wf_io1":
          if (wfi%io_s<= io1 .and. io1 <= wfi%io_e) then
            call copy_data( &
              & wf%zwf(:, :, :, ispin, io1, ik, im), &
              & wf_io1)
          end if
          if (wfi%if_divide_orbit) &
            & call comm_bcast(wf_io1, wfi%icomm_o, wfi%irank_io(io1))

          ! Calculate overlap coefficients:
          coeff_tmp = 0d0
          do io2 = 1, io1 - 1
            if (wfi%io_s<= io2 .and. io2 <= wfi%io_e) then
              coeff_tmp(io2) = dot_wf( &
                & wf%zwf(:, :, :, ispin, io2, ik, im), &
                & wf_io1)
            end if
          end do
          if (wfi%if_divide_rspace .or. wfi%if_divide_orbit) then
            call comm_summation(coeff_tmp, coeff, sys%no, wfi%icomm_ro)
          else
            coeff = coeff_tmp
          endif

          ! Calculate exclusion term "wf_exc":
          wf_exc_tmp = 0d0
          do io2 = 1, io1 - 1
            if (wfi%io_s<= io2 .and. io2 <= wfi%io_e) then
              call axpy_wf( &
                & coeff(io2), wf%zwf(:, :, :, ispin, io2, ik, im), &
                & wf_exc_tmp)
            end if
          end do
          if (wfi%if_divide_orbit) then
            call comm_summation(wf_exc_tmp, wf_exc, nsize_rg, wfi%icomm_o)
          else
            call copy_data(wf_exc_tmp, wf_exc)
          endif

          if (wfi%io_s<= io1 .and. io1 <= wfi%io_e) then
            ! Exclude non-orthonormal component:
            call axpy_wf(-one, wf_exc, wf_io1)
            ! Normalization:
            norm2_tmp = real(dot_wf(wf_io1, wf_io1))
            if (wfi%if_divide_rspace) then
              call comm_summation(norm2_tmp, norm2, wfi%icomm_r)
            else
              norm2 = norm2_tmp
            endif
            call scal_wf(one / sqrt(norm2), wf_io1)
            ! Write back to "zwf":
            call copy_data( &
              & wf_io1, &
              & wf%zwf(:, :, :, ispin, io1, ik, im))
          end if
        end do !io1

      end do !ispin
      end do !ik
      end do !im

      return
    contains

    ! Dot product of two wavefunctions:
    complex(8) function dot_wf(x, y) result(p)
      implicit none
      complex(8), intent(in) :: x( &
        & rg%is_array(1):rg%ie_array(1), &
        & rg%is_array(2):rg%ie_array(2), &
        & rg%is_array(3):rg%ie_array(3))
      complex(8), intent(in) :: y( &
        & rg%is_array(1):rg%ie_array(1), &
        & rg%is_array(2):rg%ie_array(2), &
        & rg%is_array(3):rg%ie_array(3))
      integer :: i1, i2, i3
      p = 0d0
      !$omp parallel do collapse(2) default(shared) private(i1,i2,i3) reduction(+:p)
      do i3 = rg%is(3), rg%ie(3)
        do i2 = rg%is(2), rg%ie(2)
          do i1 = rg%is(1), rg%ie(1)
            p = p + conjg(x(i1, i2, i3)) * y(i1, i2, i3)
          end do
        end do
      end do
      !$omp end parallel do
      p = p * sys%hvol
      return
    end function dot_wf

    ! Constant times a wavefunction plus a wavefunction:
    subroutine axpy_wf(a, x, y)
      implicit none
      complex(8), intent(in) :: a
      complex(8), intent(in) :: x( &
        & rg%is_array(1):rg%ie_array(1), &
        & rg%is_array(2):rg%ie_array(2), &
        & rg%is_array(3):rg%ie_array(3))
      complex(8), intent(inout) :: y( &
        & rg%is_array(1):rg%ie_array(1), &
        & rg%is_array(2):rg%ie_array(2), &
        & rg%is_array(3):rg%ie_array(3))
      integer :: i1, i2, i3
      !$omp parallel do collapse(2) default(shared) private(i1,i2,i3)
      do i3 = rg%is(3), rg%ie(3)
        do i2 = rg%is(2), rg%ie(2)
          do i1 = rg%is(1), rg%ie(1)
            y(i1, i2, i3) = a * x(i1, i2, i3) + y(i1, i2, i3)
          end do
        end do
      end do
      !$omp end parallel do
      return
    end subroutine axpy_wf


    ! Scales a wavefunction by a constant
    subroutine scal_wf(a, x)
      implicit none
      complex(8), intent(in) :: a
      complex(8), intent(inout) :: x( &
        & rg%is_array(1):rg%ie_array(1), &
        & rg%is_array(2):rg%ie_array(2), &
        & rg%is_array(3):rg%ie_array(3))
      integer :: i1, i2, i3
      !$omp parallel do collapse(2) default(shared) private(i1,i2,i3)
      do i3 = rg%is(3), rg%ie(3)
        do i2 = rg%is(2), rg%ie(2)
          do i1 = rg%is(1), rg%ie(1)
            x(i1, i2, i3) = a * x(i1, i2, i3)
          end do
        end do
      end do
      !$omp end parallel do
      return
    end subroutine scal_wf

  end subroutine gram_schmidt_col_complex8

end module gram_schmidt_orth
