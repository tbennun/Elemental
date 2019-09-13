#include <hydrogen/device/gpu/cuda/CUB.hpp>

#include <memory>

namespace
{
/** Singleton instance of CUB memory pool. */
std::unique_ptr<::cub::CachingDeviceAllocator> memoryPool_;
} // namespace <anon>

namespace hydrogen
{
namespace cub
{

::cub::CachingDeviceAllocator& MemoryPool()
{
    if (!memoryPool_)
        memoryPool_.reset(new ::cub::CachingDeviceAllocator(2u));
    return *memoryPool_;
}

void DestroyMemoryPool()
{ memoryPool_.reset(); }

} // namespace CUBMemoryPool
} // namespace hydrogen
