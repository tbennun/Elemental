#ifndef HYDROGEN_UTILS_HALFPRECISION_HPP_
#define HYDROGEN_UTILS_HALFPRECISION_HPP_

/** @file
 *
 *  Here we handle a few high-level metaprogramming tasks for 16-bit
 *  floating point support. This currently just includes IEEE-754
 *  support but will likely soon expand to include "bfloat16" as well.
 */

#include <El/hydrogen_config.h>
#include <hydrogen/meta/TypeTraits.hpp>

#ifdef HYDROGEN_HAVE_HALF
#include <half.hpp>
#endif // HYDROGEN_HAVE_HALF

#ifdef HYDROGEN_GPU_USE_FP16
#if defined(HYDROGEN_HAVE_CUDA)
#include <cuda_fp16.h>
#elif defined(HYDROGEN_HAVE_AMDGPU)
#include <rocblas-types.h>
#endif // HYDROGEN_HAVE_CUDA
#endif // HYDROGEN_GPU_USE_FP16

namespace hydrogen
{

#ifdef HYDROGEN_HAVE_HALF
/** @brief Unified name for the FP16 type on CPU */
using cpu_half_type = half_float::half;
#endif // HYDROGEN_HAVE_HALF


#ifdef HYDROGEN_GPU_USE_FP16
#if defined(HYDROGEN_HAVE_CUDA)
/** @brief Unified name for the FP16 type on GPU */
using gpu_half_type = __half;

/** @brief Enable "update" functionality for __half. */
inline gpu_half_type& operator+=(gpu_half_type& val, gpu_half_type const& rhs)
{
    val = float(val) + float(rhs);
    return val;
}

/** @brief Enable "scale" functionality for __half. */
inline gpu_half_type& operator*=(gpu_half_type& val, gpu_half_type const& rhs)
{
    val = float(val) * float(rhs);
    return val;
}

template <>
struct TypeTraits<gpu_half_type>
{
    static gpu_half_type One() noexcept { return 1.f; }
    static gpu_half_type Zero() noexcept { return 0.f; }
    static std::string Name() { return typeid(gpu_half_type).name(); }
};// struct TypeTraits<gpu_half_type>

#elif defined(HYDROGEN_HAVE_AMDGPU)
/** @brief Unified name for the FP16 type on GPU */
using gpu_half_type = rocblas_half;
#endif // HYDROGEN_HAVE_CUDA
#endif // HYDROGEN_GPU_USE_FP16

}// namespace hydrogen
#endif // HYDROGEN_UTILS_HALFPRECISION_HPP_
