cmake_minimum_required(VERSION 3.14)

if ("${CMAKE_CURRENT_SOURCE_DIR}" MATCHES "${CMAKE_CURRENT_BINARY_DIR}")
  message(FATAL_ERROR
  [[in-source build does not support
    Example:
      $ mkdir ${SALMON_BUILD_DIR}
      $ cd ${SALMON_BUILD_DIR}
      $ ${SALMON_SOURCE_DIR}/configure.py]]
  )
endif ()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/platforms)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmakefiles/Modules)
include(${CMAKE_SOURCE_DIR}/cmakefiles/misc.cmake)


### Project settings
enable_testing()
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD          99)
project(SALMON
        VERSION   2.3.0
        LANGUAGES Fortran C
        )

# Optional CUDA kernels for parts of the ground-state solver in CPU (non-OpenACC) builds.
# They keep the operation order of the CPU code they replace. All are OFF by default.
option(SALMON_CG_CUDA "Offload the complex CG solver gscg_zwf to CUDA in CPU builds (see SALMON_CG_CUDA_MODE)" OFF)
option(SALMON_CG_ORDERED_NONLOCAL "Also apply Hpsi of SALMON_CG_CUDA on the GPU (periodic 16^3 grids)" OFF)
option(SALMON_SUBSPACE_ORDERED "Compute the subspace-diagonalization ZGEMMs with CUDA (80 orbitals, 16^3 grid)" OFF)
option(SALMON_SHARED_HPSI "Use the CUDA Hpsi of SALMON_CG_ORDERED_NONLOCAL also outside CG" OFF)
option(SALMON_SUBSPACE_CONNECTED "Keep psi and Hpsi on the GPU through subspace diagonalization" OFF)
if(SALMON_SUBSPACE_CONNECTED AND (NOT SALMON_SHARED_HPSI OR NOT SALMON_SUBSPACE_ORDERED))
  message(FATAL_ERROR "SALMON_SUBSPACE_CONNECTED requires SALMON_SHARED_HPSI and SALMON_SUBSPACE_ORDERED")
endif()
if(SALMON_SUBSPACE_CONNECTED AND NOT CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
  message(FATAL_ERROR "SALMON_SUBSPACE_CONNECTED requires NVHPC Fortran: its kernels follow the ZGEMM order of the NVHPC BLAS")
endif()
if(SALMON_SHARED_HPSI AND NOT SALMON_CG_ORDERED_NONLOCAL)
  message(FATAL_ERROR "SALMON_SHARED_HPSI requires SALMON_CG_ORDERED_NONLOCAL")
endif()
if(SALMON_SUBSPACE_ORDERED AND NOT SALMON_CG_CUDA)
  message(FATAL_ERROR "SALMON_SUBSPACE_ORDERED requires SALMON_CG_CUDA")
endif()
if(SALMON_CG_ORDERED_NONLOCAL AND NOT SALMON_CG_CUDA)
  message(FATAL_ERROR "SALMON_CG_ORDERED_NONLOCAL requires SALMON_CG_CUDA")
endif()
set(SALMON_CG_CUDA_ARCH "" CACHE STRING "Explicit allocated CUDA architecture, e.g. 90")
if(SALMON_CG_CUDA)
  if(CMAKE_VERSION VERSION_LESS 3.18)
    message(FATAL_ERROR "SALMON_CG_CUDA requires CMake >= 3.18")
  endif()
  if(SALMON_CG_CUDA_ARCH STREQUAL "")
    message(FATAL_ERROR "Set SALMON_CG_CUDA_ARCH explicitly")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

message(STATUS "user-defined FFLAGS  = ${CMAKE_Fortran_FLAGS}")
message(STATUS "user-defined CFLAGS  = ${CMAKE_C_FLAGS}")
message(STATUS "user-defined LDFLAGS = ${CMAKE_EXE_LINKER_FLAGS}")

include(cmakefiles/check_build_environments.cmake)

if(SALMON_CG_CUDA AND USE_OPENACC)
  message(FATAL_ERROR "SALMON_CG_CUDA works only in CPU builds; do not combine it with an OpenACC platform file")
endif()

if(USE_OPENACC)
  if(USE_CUDA)
    option(CMAKE_CUDA_ARCHITECTURES "80") # Dummy variable
    enable_language(CUDA)
  endif()
endif()


# Bulid options
## For third party libraries
option_set(USE_MPI       "Use MPI parallelization" OFF)
option_set(USE_SCALAPACK "Use ScaLAPACK Library"   OFF)
option_set(USE_EIGENEXA  "Use EigenExa Library"    OFF)
option_set(USE_LIBXC     "Use Libxc library"       OFF)
option_set(USE_FFTW      "Use FFTW library"        OFF)

# The CUDA Fortran kernels follow the operation order of ZDOTC/ZDSCAL/ZAXPY in the BLAS bundled
# with NVHPC (OpenBLAS kernels for AMD Zen), which the OpenACC build otherwise calls on the CPU.
option(USE_ORDERED_SMALL_GRAM "Run Gram-Schmidt for 16 undivided complex orbitals with CUDA Fortran (OpenACC builds)" OFF)
if(USE_ORDERED_SMALL_GRAM)
  if(NOT USE_OPENACC OR NOT CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
    message(FATAL_ERROR "USE_ORDERED_SMALL_GRAM requires NVHPC Fortran and USE_OPENACC")
  endif()
endif()

## Optimization for stencil compitations
option_set(USE_OPT_ARRAY_PADDING           "Enable array padding for the stencil"          ON)
option_set(USE_OPT_EXPLICIT_VECTORIZATION  "Enable explicit vectorization for the stencil" OFF)


# variables
set(TARGET_NAME "salmon")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR})

include(${CMAKE_SOURCE_DIR}/cmakefiles/check_compiler_features.cmake)


# for third party libraries
include(${CMAKE_SOURCE_DIR}/cmakefiles/build_required_packages.cmake)
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${EXTERNAL_FLAGS}")
set(CMAKE_C_FLAGS       "${CMAKE_C_FLAGS} ${EXTERNAL_FLAGS}")


# Generate configure header
configure_file(
  ${CMAKE_SOURCE_DIR}/src/version.h.in
  ${CMAKE_BINARY_DIR}/version.h
)

configure_file(
  ${CMAKE_SOURCE_DIR}/src/versionf.h.in
  ${CMAKE_BINARY_DIR}/versionf.h
)

configure_file(
  ${CMAKE_SOURCE_DIR}/src/config.h.in
  ${CMAKE_BINARY_DIR}/config.h
)


add_subdirectory(${CMAKE_SOURCE_DIR}/src)        # sources
add_subdirectory(${CMAKE_SOURCE_DIR}/testsuites) # tests



# debug log
message(STATUS "Target executable file  = ${TARGET_NAME}")
message(STATUS "CMake generated FFLAGS  = ${CMAKE_Fortran_FLAGS}")
message(STATUS "CMake generated CFLAGS  = ${CMAKE_C_FLAGS}")
message(STATUS "CMake generated LDFLAGS = ${CMAKE_EXE_LINKER_FLAGS} ${EXTERNAL_LIBS}")
