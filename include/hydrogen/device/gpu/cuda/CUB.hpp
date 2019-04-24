#ifndef HYDROGEN_IMPORTS_CUB_HPP_
#define HYDROGEN_IMPORTS_CUB_HPP_

#include <cuda_runtime.h>
#include <cub/util_allocator.cuh>

namespace hydrogen
{
namespace cub
{

    /** Get singleton instance of CUB memory pool. */
    ::cub::CachingDeviceAllocator& MemoryPool();
    /** Destroy singleton instance of CUB memory pool. */
    void DestroyMemoryPool();

} // namespace cub
} // namespace hydrogen

#endif // HYDROGEN_IMPORTS_CUB_HPP_
