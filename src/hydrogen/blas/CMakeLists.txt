if (HYDROGEN_HAVE_GPU)
  add_subdirectory(gpu)

  set(CUDA_SOURCES "${CUDA_SOURCES}" PARENT_SCOPE)
endif ()
