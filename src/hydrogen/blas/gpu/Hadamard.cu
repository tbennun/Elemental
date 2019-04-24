#include <hydrogen/blas/gpu/Hadamard.hpp>

#include <El/hydrogen_config.h>
#include <hydrogen/device/gpu/CUDA.hpp>
#include <cuda_runtime.h>

namespace
{

template <typename T>
__global__ void Hadamard1D_kernel(size_t size,
                                  T const* __restrict__ X,
                                  T const* __restrict__ Y,
                                  T* __restrict__ Z)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t numThreads = blockDim.x * gridDim.x;
    for (size_t pos = tid; pos < size; pos += numThreads)
    {
        Z[pos] = X[pos] * Y[pos];
    }
}

template <typename T>
__global__ void MultAssign_kernel(size_t size, T const* X, T* Y)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t numThreads = blockDim.x * gridDim.x;
    for (size_t pos = tid; pos < size; pos += numThreads)
    {
        Y[pos] *= X[pos];
    }
}

template <typename T>
__global__ void Hadamard2D_kernel(size_t height, size_t width,
                                  T const* X, size_t colStrideX, size_t rowStrideX,
                                  T const* Y, size_t colStrideY, size_t rowStrideY,
                                  T* Z, size_t colStrideZ, size_t rowStrideZ)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t numThreads = blockDim.x * gridDim.x;
    for (size_t pos = tid; pos < height * width; pos += numThreads)
    {
        const size_t i = pos % height;
        const size_t j = pos / height;
        const auto& x_local = X[i*colStrideX+j*rowStrideX];
        const auto& y_local = Y[i*colStrideY+j*rowStrideY];
        Z[i*colStrideZ+j*rowStrideZ] = x_local * y_local;
    }
}

}// namespace <anon>

namespace hydrogen
{

template <typename T, typename>
void Hadamard_GPU_impl(
    size_t height, size_t width,
    T const* X, size_t colStrideX, size_t rowStrideX,
    T const* Y, size_t colStrideY, size_t rowStrideY,
    T* Z, size_t colStrideZ, size_t rowStrideZ,
    cudaStream_t stream)
{
    if (height <= 0 || width <= 0) { return; }
    size_t size = height * width;
    size_t const blockDim = 256;
    size_t const gridDim = (size + blockDim - 1) / blockDim;
    if (colStrideX == 1 && rowStrideX == height
        && colStrideY == 1 && rowStrideY == height
        && colStrideZ == 1 && rowStrideZ == height)
    {
        if (X == Z)
        {
            void* args[] = { &size, &Y, &Z };
            H_CHECK_CUDA(
                cudaLaunchKernel(
                    (void const*)&MultAssign_kernel<T>,
                    gridDim, blockDim, args, 0, stream));
        }
        else if (Y == Z)
        {
            void* args[] = { &size, &X, &Z };
            H_CHECK_CUDA(
                cudaLaunchKernel(
                    (void const*)&MultAssign_kernel<T>,
                    gridDim, blockDim, args, 0, stream));
        }
        else
        {
            void* args[] = { &size, &X, &Y, &Z };
            H_CHECK_CUDA(
                cudaLaunchKernel(
                    (void const*)&Hadamard1D_kernel<T>,
                    gridDim, blockDim, args, 0, stream));
        }
    }
    else
    {
        void* args[] = { &height, &width,
                         &X, &colStrideX, &rowStrideX,
                         &Y, &colStrideY, &rowStrideY,
                         &Z, &colStrideZ, &rowStrideZ };
        H_CHECK_CUDA(
            cudaLaunchKernel(
                (void const*)&Hadamard2D_kernel<T>,
                gridDim, blockDim, args, 0, stream));
    }

}

#define ETI(T)                                  \
    template void Hadamard_GPU_impl(            \
        size_t, size_t,                         \
        T const*, size_t, size_t,               \
        T const*, size_t, size_t,               \
        T*, size_t, size_t, cudaStream_t)

#ifdef HYDROGEN_GPU_USE_FP16
ETI(gpu_half_type);
#endif

ETI(float);
ETI(double);

}// namespace hydrogen
