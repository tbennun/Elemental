/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   Copyright (c) 2013, Jeff Hammond
   All rights reserved.

   Copyright (c) 2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/


#ifndef EL_IMPORTS_MPIUTILS_HPP
#define EL_IMPORTS_MPIUTILS_HPP

#ifdef HYDROGEN_HAVE_CUDA
#include <El/core/imports/cuda.hpp>
#define EL_CHECK_MPI(mpi_call)                                          \
    do                                                                  \
    {                                                                   \
       EL_CHECK_CUDA(cudaStreamSynchronize(GPUManager::Stream()));      \
       CheckMpi( mpi_call );                                            \
    }                                                                   \
    while( 0 )
#else
#define EL_CHECK_MPI(mpi_call) CheckMpi( mpi_call )
#endif

#define EL_CHECK_MPI_NO_DATA(mpi_call) CheckMpi( mpi_call )

namespace {

inline void
CheckMpi( int error ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_ONLY(
      if( error != MPI_SUCCESS )
      {
          char errorString[MPI_MAX_ERROR_STRING];
          int lengthOfErrorString;
          MPI_Error_string( error, errorString, &lengthOfErrorString );
          El::RuntimeError( std::string(errorString) );
      }
    )
}

template<typename T>
MPI_Op NativeOp( const El::mpi::Op& op )
{
    MPI_Op opC;
    if( op == El::mpi::SUM )
        opC = El::mpi::SumOp<T>().op;
    else if( op == El::mpi::PROD )
        opC = El::mpi::ProdOp<T>().op;
    else if( op == El::mpi::MAX )
        opC = El::mpi::MaxOp<T>().op;
    else if( op == El::mpi::MIN )
        opC = El::mpi::MinOp<T>().op;
    else
        opC = op.op;
    return opC;
}

}// namespace <anon>

// This is for handling the host-blocking host-transfer stuff
#ifndef HYDROGEN_ASSUME_CUDA_AWARE_MPI
namespace El
{
namespace mpi
{
namespace internal
{

template <typename T>
class PassthroughMemoryWrapper
{
public:
    PassthroughMemoryWrapper(T* buf) noexcept : data_{buf} {}
    T* data() const noexcept { return data_; }
private:
    T* data_;
};// struct PassThroughMemoryWrapper

/**
 * \class ManagedHostMemoryWrapper
 * \brief Transfer memory to the host on construction, to the device
 *     on destruction.
 */
template <typename T, Device D>
class ManagedHostMemoryWrapper;

#ifdef HYDROGEN_HAVE_CUDA
template <typename T>
class ManagedHostMemoryWrapper<T,Device::GPU>
{
public:

    ManagedHostMemoryWrapper(T* buf, size_t size,
                             SyncInfo<Device::GPU> const& syncInfo,
                             bool copy_data_to_host = false)
        : host_data_{size, SyncInfo<Device::CPU>{}, /*mode=*/ 1},
          syncInfo_{syncInfo},
          device_data_{buf}
    {
        if (copy_data_to_host && (size > 0))
            InterDeviceCopy<Device::GPU, Device::CPU>::MemCopy1DAsync(
                host_data_.data(), device_data_, size, syncInfo_.stream_);
    }

    ManagedHostMemoryWrapper(T* buf, size_t totalsize,
                             size_t local_offset, size_t local_size,
                             SyncInfo<Device::GPU> const& syncInfo)
        : host_data_{totalsize, SyncInfo<Device::CPU>{}, /*mode=*/1},
          syncInfo_{syncInfo},
          device_data_{buf}
    {
        InterDeviceCopy<Device::GPU, Device::CPU>::MemCopy1DAsync(
            host_data_+local_offset, device_data_+local_offset,
            local_size, syncInfo_.stream_);
    }

    ~ManagedHostMemoryWrapper()
    {
        // Transfer stuff back to device
        if (host_data_.size() > 0)
        {
            InterDeviceCopy<Device::CPU, Device::GPU>::MemCopy1DAsync(
                device_data_, host_data_.data(), host_data_.size(),
                syncInfo_.stream_);
            Synchronize(syncInfo_);
        }
    }

    // Enable move construction/assignment
    ManagedHostMemoryWrapper(
        ManagedHostMemoryWrapper<T,Device::GPU>&&) = default;
    // Disable copy construction/assignment
    ManagedHostMemoryWrapper(
        ManagedHostMemoryWrapper<T,Device::GPU> const&) = delete;

    T* data() noexcept { return host_data_.data(); }
private:
    simple_buffer<T,Device::CPU> host_data_;
    SyncInfo<Device::GPU> syncInfo_;
    DevicePtr<T> device_data_;
};// class ManagedHostMemoryWrapper<T,Device::GPU>
#endif // HYDROGEN_HAVE_CUDA

// Helper functions that make all this stuff useful.
template <typename T>
auto
MakeHostBuffer(T* buf, size_t const& /*size*/, SyncInfo<Device::CPU> const&)
    -> PassthroughMemoryWrapper<T>
{
    return PassthroughMemoryWrapper<T>(buf);
}

template <typename T>
auto
MakeManagedHostBuffer(T* buf, size_t const& /*size*/,
                      SyncInfo<Device::CPU> const&, bool)
    -> PassthroughMemoryWrapper<T>
{
    return PassthroughMemoryWrapper<T>(buf);
}

template <typename T>
auto
MakeManagedHostBuffer(T* buf, size_t const& , size_t const&, size_t const&,
                      SyncInfo<Device::CPU> const&)
    -> PassthroughMemoryWrapper<T>
{
    return PassthroughMemoryWrapper<T>(buf);
}
template <typename T>
struct type_check;

#ifdef HYDROGEN_HAVE_CUDA
// This can't (shouldn't) just be std::vector<T> because we want
// pinned memory for GPUs. And I don't want to write a new Allocator
// for std::vector that uses pinned memory through CUDA. We can access
// pinned memory through the simple_buffer.
template <typename T>
auto MakeHostBuffer(T const* buf, size_t const& size,
                    SyncInfo<Device::GPU> const& syncInfo)
    -> simple_buffer<T,Device::CPU>
{
    simple_buffer<T,Device::CPU> locbuf(
        size, SyncInfo<Device::CPU>{}, /*mode=*/ 1);
    InterDeviceCopy<Device::GPU, Device::CPU>::MemCopy1DAsync(
        locbuf.data(), buf, size, syncInfo.stream_);
    return locbuf;
}
#endif // HYDROGEN_HAVE_CUDA

template <typename T, Device D>
auto MakeManagedHostBuffer(T* buf, size_t const& size,
                           SyncInfo<D> const& syncInfo,
                           bool copy_to_host_on_construction)
    -> ManagedHostMemoryWrapper<T,D>
{
    return ManagedHostMemoryWrapper<T,D>(buf, size, syncInfo,
                                         copy_to_host_on_construction);
}

template <typename T, Device D>
auto MakeManagedHostBuffer(T* buf, size_t const& totalsize,
                           size_t const& local_offset, size_t const& local_size,
                           SyncInfo<D> const& syncInfo)
    -> ManagedHostMemoryWrapper<T,D>
{
    return ManagedHostMemoryWrapper<T,D>(
        buf, totalsize, local_offset, local_size, syncInfo);
}

#define ENSURE_HOST_SEND_BUFFER(buf, size, syncinfo)                \
    auto host_sbuf = internal::MakeHostBuffer(buf, size, syncinfo); \
    buf = host_sbuf.data()

#define ENSURE_HOST_RECV_BUFFER(buf, size, syncinfo)                 \
    auto host_rbuf =                                                 \
        internal::MakeManagedHostBuffer(buf, size, syncinfo, false); \
    buf = host_rbuf.data()

#define ENSURE_HOST_INPLACE_BUFFER(buf, totalsize, offset, size, syncinfo) \
    auto host_buf =                                                     \
        internal::MakeManagedHostBuffer(                                \
            buf, totalsize, offset, size, syncinfo);                    \
    buf = host_buf.data()

#define ENSURE_HOST_INPLACE_BUFFER_ALL_XFER(buf, size, syncinfo)     \
    auto host_buf =                                                  \
        internal::MakeManagedHostBuffer(buf, size, syncinfo, true);  \
    buf = host_buf.data()

}// namespace internal
}// namespace mpi
}// namespace El
#endif // HYDROGEN_ASSUME_CUDA_AWARE_MPI

#endif // ifndef EL_IMPORTS_MPIUTILS_HPP
