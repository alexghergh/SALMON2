# SALMON sources
set(SALMON_SOURCES)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/parallel)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/io)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/math)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gs)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/rt)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/xc)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/atom)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/poisson)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/maxwell)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/common)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/misc)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/symmetry)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/so)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/plusu)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ms)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ttm)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/paw)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ssbe)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/analysis)

# Executable binary
add_executable(${TARGET_NAME} main.f90 ${SALMON_SOURCES})
target_link_libraries(${TARGET_NAME} ${EXTERNAL_LIBS})

if(SALMON_REAL_GRAM_ORDERED)
  add_library(salmon_real_gram_cuda STATIC common/real_gram_ordered.cu common/real_gram_context.cu)
  set_target_properties(salmon_real_gram_cuda PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${SALMON_REAL_GRAM_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)
  target_compile_options(salmon_real_gram_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/gs/gram_schmidt_orth.f90
               APPEND PROPERTY COMPILE_DEFINITIONS SALMON_REAL_GRAM_ORDERED)
  target_link_libraries(${TARGET_NAME} salmon_real_gram_cuda CUDA::cudart stdc++)
endif()

if(USE_ORDERED_SMALL_GRAM)
  # Keep CUDA Fortran syntax and its dispatch definition local to these sources.
  set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/common/small_gram_ordered_cuda.f90
                      ${CMAKE_CURRENT_SOURCE_DIR}/common/small_gram_ordered_batch4_cuda.f90
               APPEND PROPERTY COMPILE_OPTIONS -cuda)
  set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/gs/gram_schmidt_orth.f90
               APPEND PROPERTY COMPILE_DEFINITIONS USE_ORDERED_SMALL_GRAM)
  target_link_options(${TARGET_NAME} PRIVATE -cuda)
endif()

if(SALMON_CG_CUDA)
  add_library(salmon_cg_cuda STATIC gs/cg_resident.cu)
  if(SALMON_SUBSPACE_ORDERED)
    target_sources(salmon_cg_cuda PRIVATE gs/subspace_gemm_ordered.cu gs/subspace_ordered_context.cu)
    set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/gs/subspace_diagonalization.f90
                 APPEND PROPERTY COMPILE_DEFINITIONS SALMON_SUBSPACE_ORDERED)
  endif()
  if(SALMON_CG_ORDERED_NONLOCAL)
    target_sources(salmon_cg_cuda PRIVATE gs/cg_nonlocal_ordered.cu gs/cg_stencil_ordered.cu common/hpsi_cuda_maps.cu)
    target_compile_definitions(salmon_cg_cuda PRIVATE SALMON_CG_ORDERED_NONLOCAL)
    set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/atom/pp/prep_pp.f90
                 APPEND PROPERTY COMPILE_DEFINITIONS SALMON_CG_ORDERED_NONLOCAL)
  endif()
  if(SALMON_SHARED_HPSI)
    target_sources(salmon_cg_cuda PRIVATE common/hpsi_cuda_call.cu common/hpsi_energy_ordered.cu)
    set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/common/hamiltonian.f90
                 APPEND PROPERTY COMPILE_DEFINITIONS SALMON_SHARED_HPSI)
  endif()
  if(SALMON_SUBSPACE_CONNECTED)
    target_sources(salmon_cg_cuda PRIVATE gs/subspace_connected_context.cu)
    set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/gs/subspace_diagonalization.f90
                 APPEND PROPERTY COMPILE_DEFINITIONS SALMON_SUBSPACE_CONNECTED)
  endif()
  set_target_properties(salmon_cg_cuda PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${SALMON_CG_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)
  target_compile_options(salmon_cg_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/gs/conjugate_gradient.f90
               APPEND PROPERTY COMPILE_DEFINITIONS SALMON_CG_CUDA)
  target_link_libraries(${TARGET_NAME} salmon_cg_cuda CUDA::cudart stdc++)
endif()

# Install binary
install(TARGETS ${TARGET_NAME} DESTINATION "bin")
