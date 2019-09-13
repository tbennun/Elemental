#ifndef HYDROGEN_BLAS_GPU_HADAMARD_HPP_
#define HYDROGEN_BLAS_GPU_HADAMARD_HPP_

/** @file
 *  @todo Write documentation!
 */

#include <hydrogen/Device.hpp>
#include <hydrogen/meta/MetaUtilities.hpp>

#include <cuda_runtime.h>

#include <stdexcept>

namespace hydrogen
{

template <typename T, typename=EnableWhen<IsComputeType<T,Device::GPU>>>
void Hadamard_GPU_impl(
    size_t height, size_t width,
    T const* A, size_t row_stride_A, size_t lda,
    T const* B, size_t row_stride_B, size_t ldb,
    T* C, size_t row_stride_C, size_t ldc,
    cudaStream_t stream);

template <typename T,
          typename=EnableUnless<IsComputeType<T,Device::GPU>>,
          typename=void>
void Hadamard_GPU_impl(
    size_t const&, size_t const&,
    T const* const&, size_t const&, size_t const&,
    T const* const&, size_t const&, size_t const&,
    T* const&, size_t const&, size_t const&,
    cudaStream_t const&)
{
    throw std::logic_error("Hadamard: Type not valid on GPU.");
}

}// namespace hydrogen
#endif // HYDROGEN_BLAS_GPU_HADAMARD_HPP_
