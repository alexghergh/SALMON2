module cusolver_wrapper
  use iso_c_binding
  use iso_fortran_env
  use cudafor, only: cuda_stream_kind
  use openacc

  use cublas, only: &
       CUBLAS_FILL_MODE_UPPER

  use cusolverDn, only: &
       cusolverDnHandle, cusolverDnParams, &
       cusolverDnCreate, cusolverDnDestroy, &
       cusolverDnCreateParams, cusolverDnDestroyParams, &
       cusolverDnSetStream, &
       CUSOLVER_STATUS_SUCCESS, &
       CUSOLVER_EIG_MODE_VECTOR, &
       cudaDataType, CUDA_R_64F, CUDA_C_64F

  implicit none
  private

  public :: eigen_zheev_batched_gpu
  public :: eigen_dsyev_gpu

  type(cusolverDnHandle), save :: handle
  type(cusolverDnParams), save :: params
  logical, save :: initialized = .false.

  integer(c_int64_t), save :: saved_n     = -1_c_int64_t
  integer(c_int64_t), save :: saved_batch = -1_c_int64_t

  integer(c_size_t), save :: ws_dev_bytes  = 0_c_size_t
  integer(c_size_t), save :: ws_host_bytes = 0_c_size_t

  integer(c_int8_t), allocatable, target, save :: work_dev(:)
  integer(c_int8_t), allocatable, target, save :: work_host(:)
  integer(c_int),    allocatable, target, save :: info(:)

  ! Workspace of eigen_dsyev_gpu (kept apart from the batched complex solver's workspace).
  ! The matrix and eigenvalue copies are allocatable, hence managed memory in OpenACC builds,
  ! as for the complex solver above.
  integer(c_int64_t), save :: saved_n_real = -1_c_int64_t
  integer(c_size_t),  save :: ws_dev_bytes_real  = 0_c_size_t
  integer(c_size_t),  save :: ws_host_bytes_real = 0_c_size_t
  integer(c_int8_t), allocatable, target, save :: work_dev_real(:)
  integer(c_int8_t), allocatable, target, save :: work_host_real(:)
  integer(c_int),    allocatable, target, save :: info_real(:)
  real(8),           allocatable, target, save :: a_real(:,:)
  real(8),           allocatable, target, save :: w_real(:)

  logical, save :: work_dev_present = .false.
  logical, save :: info_present     = .false.

  interface
    integer(c_int) function cusolverDnXsyevBatched_bufferSize_f( &
        handle_c, params_c, jobz, uplo, n, &
        dataTypeA, A, lda, &
        dataTypeW, W, computeType, &
        workspaceInBytesOnDevice, workspaceInBytesOnHost, &
        batchSize) bind(C, name="cusolverDnXsyevBatched_bufferSize")
      import :: c_int, c_int64_t, c_size_t, c_ptr, cudaDataType
      implicit none

      type(c_ptr), value :: handle_c
      type(c_ptr), value :: params_c
      integer(c_int), value :: jobz, uplo
      integer(c_int64_t), value :: n, lda, batchSize

      type(cudaDataType), value :: dataTypeA
      type(c_ptr), value :: A

      type(cudaDataType), value :: dataTypeW
      type(c_ptr), value :: W

      type(cudaDataType), value :: computeType

      integer(c_size_t) :: workspaceInBytesOnDevice
      integer(c_size_t) :: workspaceInBytesOnHost
    end function cusolverDnXsyevBatched_bufferSize_f


    integer(c_int) function cusolverDnXsyevBatched_f( &
        handle_c, params_c, jobz, uplo, n, &
        dataTypeA, A, lda, &
        dataTypeW, W, computeType, &
        bufferOnDevice, workspaceInBytesOnDevice, &
        bufferOnHost, workspaceInBytesOnHost, &
        info, batchSize) bind(C, name="cusolverDnXsyevBatched")
      import :: c_int, c_int64_t, c_size_t, c_ptr, cudaDataType
      implicit none

      type(c_ptr), value :: handle_c
      type(c_ptr), value :: params_c
      integer(c_int), value :: jobz, uplo
      integer(c_int64_t), value :: n, lda, batchSize

      type(cudaDataType), value :: dataTypeA
      type(c_ptr), value :: A

      type(cudaDataType), value :: dataTypeW
      type(c_ptr), value :: W

      type(cudaDataType), value :: computeType

      type(c_ptr), value :: bufferOnDevice
      integer(c_size_t), value :: workspaceInBytesOnDevice

      type(c_ptr), value :: bufferOnHost
      integer(c_size_t), value :: workspaceInBytesOnHost

      type(c_ptr), value :: info
    end function cusolverDnXsyevBatched_f


    integer(c_int) function cusolverDnXsyevd_bufferSize_f( &
        handle_c, params_c, jobz, uplo, n, &
        dataTypeA, A, lda, &
        dataTypeW, W, computeType, &
        workspaceInBytesOnDevice, workspaceInBytesOnHost) bind(C, name="cusolverDnXsyevd_bufferSize")
      import :: c_int, c_int64_t, c_size_t, c_ptr, cudaDataType
      implicit none

      type(c_ptr), value :: handle_c
      type(c_ptr), value :: params_c
      integer(c_int), value :: jobz, uplo
      integer(c_int64_t), value :: n, lda

      type(cudaDataType), value :: dataTypeA
      type(c_ptr), value :: A

      type(cudaDataType), value :: dataTypeW
      type(c_ptr), value :: W

      type(cudaDataType), value :: computeType

      integer(c_size_t) :: workspaceInBytesOnDevice
      integer(c_size_t) :: workspaceInBytesOnHost
    end function cusolverDnXsyevd_bufferSize_f


    integer(c_int) function cusolverDnXsyevd_f( &
        handle_c, params_c, jobz, uplo, n, &
        dataTypeA, A, lda, &
        dataTypeW, W, computeType, &
        bufferOnDevice, workspaceInBytesOnDevice, &
        bufferOnHost, workspaceInBytesOnHost, &
        info) bind(C, name="cusolverDnXsyevd")
      import :: c_int, c_int64_t, c_size_t, c_ptr, cudaDataType
      implicit none

      type(c_ptr), value :: handle_c
      type(c_ptr), value :: params_c
      integer(c_int), value :: jobz, uplo
      integer(c_int64_t), value :: n, lda

      type(cudaDataType), value :: dataTypeA
      type(c_ptr), value :: A

      type(cudaDataType), value :: dataTypeW
      type(c_ptr), value :: W

      type(cudaDataType), value :: computeType

      type(c_ptr), value :: bufferOnDevice
      integer(c_size_t), value :: workspaceInBytesOnDevice

      type(c_ptr), value :: bufferOnHost
      integer(c_size_t), value :: workspaceInBytesOnHost

      type(c_ptr), value :: info
    end function cusolverDnXsyevd_f
  end interface

contains

  ! The batched eigen_zheev subroutine operation in the eigen_lapack module on GPU
  subroutine eigen_zheev_batched_gpu(h, e, v, n, im_s, im_e, ik_s, ik_e, nspin)
    implicit none

    integer, intent(in) :: n, im_s, im_e, ik_s, ik_e, nspin

    complex(8), intent(in) :: h(n,n,nspin,ik_s:ik_e,im_s:im_e)
    real(8), target, intent(out) :: e(n,nspin,ik_s:ik_e,im_s:im_e)
    complex(8), target, intent(out) :: v(n,n,nspin,ik_s:ik_e,im_s:im_e)

    integer(c_int) :: stat
    integer(c_int) :: jobz, uplo
    type(cudaDataType) :: dtype_A, dtype_W, dtype_compute
    integer(c_int64_t) :: n64, lda64, batch64, n_eig64
    integer :: batch_size, n_eig

    type(c_ptr) :: A_dev
    type(c_ptr) :: W_dev
    type(c_ptr) :: work_dev_ptr
    type(c_ptr) :: work_host_ptr
    type(c_ptr) :: info_dev_ptr

    integer(c_size_t) :: ws_dev_need
    integer(c_size_t) :: ws_host_need

    if (n <= 0) then
      write(error_unit,*) "eigen_zheev_batched_gpu: n must be positive."
      error stop
    end if

    batch64 = int(im_e - im_s + 1, c_int64_t) * &
              int(ik_e - ik_s + 1, c_int64_t) * &
              int(nspin,           c_int64_t)

    if (batch64 < 1_c_int64_t) then
      write(error_unit,*) "eigen_zheev_batched_gpu: batch size must be positive."
      error stop
    end if

    n64       = int(n, c_int64_t)
    lda64     = n64
    n_eig64   = n64 * batch64

    if (size(e, kind=c_int64_t) < n_eig64) then
      write(error_unit,*) "eigen_zheev_batched_gpu: e(:) is too small. need =", n_eig64, "but size(e) = ", size(e, kind=c_int64_t)
      error stop
    end if

    if (n64 * lda64 * batch64 > int(huge(0_c_int), c_int64_t)) then
      write(error_unit,*) "eigen_zheev_batched_gpu: n*lda*batchSize exceeds INT32_MAX."
      error stop
    end if

    batch_size = int(batch64)
    n_eig      = int(n_eig64)

    jobz = CUSOLVER_EIG_MODE_VECTOR
    uplo = CUBLAS_FILL_MODE_UPPER

    dtype_A       = cudaDataType(CUDA_C_64F)
    dtype_W       = cudaDataType(CUDA_R_64F)
    dtype_compute = cudaDataType(CUDA_C_64F)

    call init_handle_if_needed()

    !$acc kernels present(h, v)
    v(:,:,:,:,:) = h(:,:,:,:,:)
    !$acc end kernels

    if (.not. allocated(work_dev) .or. &
        saved_n /= n64 .or. saved_batch /= batch64) then

      A_dev = c_loc(v(1,1,1,ik_s,im_s))
      W_dev = c_loc(e(1,1,ik_s,im_s))

      stat = cusolverDnXsyevBatched_bufferSize_f( &
          handle%handle, params%params, &
          jobz, uplo, n64, &
          dtype_A, A_dev, lda64, &
          dtype_W, W_dev, dtype_compute, &
          ws_dev_need, ws_host_need, batch64)

      call check_cusolver(stat, "cusolverDnXsyevBatched_bufferSize")

      call allocate_workspace(ws_dev_need, ws_host_need, batch_size)

      saved_n     = n64
      saved_batch = batch64
    end if

    !$acc kernels
    info(:) = 0_c_int
    !$acc end kernels

    if (ws_host_bytes > 0_c_size_t) then
      work_host_ptr = c_loc(work_host(1))
    else
      work_host_ptr = c_null_ptr
    end if

    A_dev        = c_loc(v(1,1,1,ik_s,im_s))
    W_dev        = c_loc(e(1,1,ik_s,im_s))
    work_dev_ptr = c_loc(work_dev(1))
    info_dev_ptr = c_loc(info(1))

    stat = cusolverDnXsyevBatched_f( &
        handle%handle, params%params, &
        jobz, uplo, n64, &
        dtype_A, A_dev, lda64, &
        dtype_W, W_dev, dtype_compute, &
        work_dev_ptr, ws_dev_bytes, &
        work_host_ptr, ws_host_bytes, &
        info_dev_ptr, batch64)

    call check_cusolver(stat, "cusolverDnXsyevBatched")
    call acc_wait(acc_async_sync)
    call check_cusolver_info(info, batch_size, "cusolverDnXsyevBatched")

  end subroutine eigen_zheev_batched_gpu


  ! The eigen_dsyev operation of the eigen_lapack module (all eigenpairs of a real symmetric
  ! matrix, upper triangle referenced, eigenvalues in ascending order) computed on the GPU with
  ! cusolverDnXsyevd. h, e and v are host arrays with the same shapes as in eigen_dsyev.
  subroutine eigen_dsyev_gpu(h, e, v)
    implicit none
    real(8), intent(in)  :: h(:,:)
    real(8), intent(out) :: e(:)
    real(8), intent(out) :: v(:,:)

    integer(c_int) :: stat
    integer(c_int) :: jobz, uplo
    type(cudaDataType) :: dtype
    integer(c_int64_t) :: n64
    integer :: n

    type(c_ptr) :: A_dev
    type(c_ptr) :: W_dev
    type(c_ptr) :: work_dev_ptr
    type(c_ptr) :: work_host_ptr
    type(c_ptr) :: info_dev_ptr

    integer(c_size_t) :: ws_dev_need
    integer(c_size_t) :: ws_host_need

    n = ubound(h,1)
    if (n <= 0) then
      write(error_unit,*) "eigen_dsyev_gpu: n must be positive."
      error stop
    end if
    n64 = int(n, c_int64_t)

    jobz  = CUSOLVER_EIG_MODE_VECTOR
    uplo  = CUBLAS_FILL_MODE_UPPER
    dtype = cudaDataType(CUDA_R_64F)

    call init_handle_if_needed()

    if (.not. allocated(a_real) .or. saved_n_real /= n64) then
      if (allocated(a_real)) deallocate(a_real)
      if (allocated(w_real)) deallocate(w_real)
      allocate(a_real(n,n), w_real(n))
    end if

    a_real(:,:) = h(1:n,1:n)

    A_dev = c_loc(a_real(1,1))
    W_dev = c_loc(w_real(1))

    if (.not. allocated(work_dev_real) .or. saved_n_real /= n64) then
      stat = cusolverDnXsyevd_bufferSize_f( &
          handle%handle, params%params, &
          jobz, uplo, n64, &
          dtype, A_dev, n64, &
          dtype, W_dev, dtype, &
          ws_dev_need, ws_host_need)
      call check_cusolver(stat, "cusolverDnXsyevd_bufferSize")

      if (allocated(work_dev_real))  deallocate(work_dev_real)
      if (allocated(work_host_real)) deallocate(work_host_real)
      if (allocated(info_real))      deallocate(info_real)
      ws_dev_bytes_real  = ws_dev_need
      ws_host_bytes_real = ws_host_need
      allocate(work_dev_real(max(1_c_size_t, ws_dev_bytes_real)))
      allocate(work_host_real(max(1_c_size_t, ws_host_bytes_real)))
      allocate(info_real(1))
      saved_n_real = n64
    end if

    info_real(1) = 0_c_int

    if (ws_host_bytes_real > 0_c_size_t) then
      work_host_ptr = c_loc(work_host_real(1))
    else
      work_host_ptr = c_null_ptr
    end if
    work_dev_ptr = c_loc(work_dev_real(1))
    info_dev_ptr = c_loc(info_real(1))

    stat = cusolverDnXsyevd_f( &
        handle%handle, params%params, &
        jobz, uplo, n64, &
        dtype, A_dev, n64, &
        dtype, W_dev, dtype, &
        work_dev_ptr, ws_dev_bytes_real, &
        work_host_ptr, ws_host_bytes_real, &
        info_dev_ptr)
    call check_cusolver(stat, "cusolverDnXsyevd")
    call acc_wait(acc_async_sync)
    call check_cusolver_info(info_real, 1, "cusolverDnXsyevd")

    e(1:n)     = w_real(1:n)
    v(1:n,1:n) = a_real(1:n,1:n)

  end subroutine eigen_dsyev_gpu


  subroutine init_handle_if_needed()
    implicit none
    integer(c_int) :: stat
    integer(cuda_stream_kind) :: stream

    if (initialized) return

    stat = cusolverDnCreate(handle)
    call check_cusolver(stat, "cusolverDnCreate")

    stat = cusolverDnCreateParams(params)
    call check_cusolver(stat, "cusolverDnCreateParams")

    stream = acc_get_cuda_stream(acc_async_sync)
    stat = cusolverDnSetStream(handle, stream)
    call check_cusolver(stat, "cusolverDnSetStream")

    initialized = .true.
  end subroutine init_handle_if_needed

  subroutine allocate_workspace(ws_dev_need, ws_host_need, batch_size)
    implicit none
    integer(c_size_t), intent(in) :: ws_dev_need
    integer(c_size_t), intent(in) :: ws_host_need
    integer, intent(in) :: batch_size

    call release_workspace()

    ws_dev_bytes  = ws_dev_need
    ws_host_bytes = ws_host_need

    allocate(work_dev(max(1_c_size_t, ws_dev_bytes)))
    work_dev_present = .true.

    allocate(work_host(max(1_c_size_t, ws_host_bytes)))

    allocate(info(batch_size))
    info_present = .true.
  end subroutine allocate_workspace

  subroutine release_workspace()
    implicit none

    if (work_dev_present) then
      work_dev_present = .false.
    end if

    if (info_present) then
      info_present = .false.
    end if

    if (allocated(work_dev))  deallocate(work_dev)
    if (allocated(work_host)) deallocate(work_host)
    if (allocated(info))      deallocate(info)

    ws_dev_bytes  = 0_c_size_t
    ws_host_bytes = 0_c_size_t
    saved_n       = -1_c_int64_t
    saved_batch   = -1_c_int64_t
  end subroutine release_workspace

  subroutine check_cusolver(stat, where)
    implicit none
    integer(c_int), intent(in) :: stat
    character(*), intent(in) :: where

    if (stat /= CUSOLVER_STATUS_SUCCESS) then
      write(error_unit,'(a,": cuSOLVER status = ",i0)') trim(where), stat
      error stop
    end if
  end subroutine check_cusolver

  subroutine check_cusolver_info(info, batch_size, where)
    implicit none
    integer(c_int), intent(in) :: info(:)
    integer, intent(in) :: batch_size
    character(*), intent(in) :: where

    integer :: ib

    do ib = 1, batch_size
      if (info(ib) /= 0_c_int) then
        write(error_unit,*) trim(where), ": info(", ib, ") = ", info(ib)
        error stop
      end if
    end do
  end subroutine check_cusolver_info

end module cusolver_wrapper
