#ifndef HYDROGEN_BLAS_GPU_AXPY_HPP_
#define HYDROGEN_BLAS_GPU_AXPY_HPP_

#include <hydrogen/Device.hpp>
#include <hydrogen/meta/MetaUtilities.hpp>

#include <cuda_runtime.h>

#include <stdexcept>

namespace hydrogen
{

/** @brief Execute a 2-D AXPY operation on the GPU
 *
 *  Writes dest = alpha*src + dest, taking into account 2D stride
 *  information.
 *
 *  @tparam T (Inferred) The type of data. Must be the same for source
 *      and destination matrices.
 *
 *  @param num_rows The number of rows in the matrix
 *  @param num_cols The number of columns in the matrix
 *  @param alpha The scaling factor
 *  @param src The source matrix, in column-major ordering. Must not
 *      overlap with the destination matrix.
 *  @param src_row_stride The number of `T`s between rows in a column
 *      of the source matrix. For "traditional" packed matrices, this
 *      will be "1".
 *  @param src_col_stride The number of `T`s between columns in a row
 *      of the source matrix. For "traditional" packed matrices, this
 *      will be the leading dimension.
 *  @param dest The destination matrix, in column-major ordering. Must not
 *      overlap with the source matrix.
 *  @param dest_row_stride The number of `T`s between rows in a column
 *      of the destination matrix. For "traditional" packed matrices,
 *      this will be "1".
 *  @param dest_col_stride The number of `T`s between columns in a row
 *      of the destination matrix. For "traditional" packed matrices,
 *      this will be the leading dimension.
 *  @param stream The CUDA stream on which the kernel should be
 *      launched.
 */
template <typename T, typename SizeT,
          typename=EnableWhen<IsComputeType<T,Device::GPU>>>
void Axpy_GPU_impl(
    SizeT num_rows, SizeT num_cols, T alpha,
    T const* src, SizeT src_row_stride, SizeT src_col_stride,
    T* dest, SizeT dest_row_stride, SizeT dest_col_stride,
    cudaStream_t stream);

template <typename T, typename SizeT,
          typename=EnableUnless<IsComputeType<T,Device::GPU>>,
          typename=void>
void Axpy_GPU_impl(
    SizeT, SizeT, T,
    T const*, SizeT, SizeT, T*, SizeT, SizeT,
    cudaStream_t)
{
    throw std::logic_error("Axpy: Type not valid on GPU.");
}

}// namespace hydrogen
#endif // HYDROGEN_BLAS_GPU_AXPY_HPP_
