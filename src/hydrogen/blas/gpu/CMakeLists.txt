set_full_path(THIS_DIR_CUDA_SOURCES
  Axpy.cu
  Copy.cu
  Fill.cu
  Hadamard.cu
  Scale.cu
  Transpose.cu
  )

# Propagate the files up the tree
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CUDA_SOURCES}" PARENT_SCOPE)
