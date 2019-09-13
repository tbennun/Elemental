add_subdirectory(blas)

if (HYDROGEN_HAVE_GPU)
  set(CUDA_SOURCES "${CUDA_SOURCES}" PARENT_SCOPE)
endif ()
