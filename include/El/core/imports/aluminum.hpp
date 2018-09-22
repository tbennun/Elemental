#ifndef HYDROGEN_IMPORTS_ALUMINUM_HPP_
#define HYDROGEN_IMPORTS_ALUMINUM_HPP_

#ifdef HYDROGEN_HAVE_ALUMINUM
#include <Al.hpp>
#endif // HYDROGEN_HAVE_ALUMINUM

namespace El
{

// FIXME: This is a lame shortcut to save some
// metaprogramming. Deadlines are the worst.
enum class Collective
{
    ALLGATHER,
    ALLREDUCE,
    ALLTOALL,
    BROADCAST,
    GATHER,
    REDUCE,
    REDUCESCATTER,
    SCATTER,
    SENDRECV
};// enum class Collective

#ifndef HYDROGEN_HAVE_ALUMINUM

template <typename T> struct IsAluminumTypeT : std::false_type {};
template <typename T, Device D>
struct IsAluminumDeviceType : std::false_type {};
template <typename T, Device D, Collective C>
struct IsAluminumSupported : std::false_type {};

#else

// A function to convert an MPI MPI_Op into an Aluminum operator
Al::ReductionOperator MPI_Op2ReductionOperator(MPI_Op op);

template <typename T, typename BackendT>
struct IsAlTypeT : std::false_type {};

template <>
struct IsAlTypeT<              char, Al::MPIBackend> : std::true_type {};
template <>
struct IsAlTypeT<       signed char, Al::MPIBackend> : std::true_type {};
template <>
struct IsAlTypeT<     unsigned char, Al::MPIBackend> : std::true_type {};
template <>
struct IsAlTypeT<             short, Al::MPIBackend> : std::true_type {};
template <>
struct IsAlTypeT<    unsigned short, Al::MPIBackend> : std::true_type {};
template <>
struct IsAlTypeT<               int, Al::MPIBackend> : std::true_type {};
template <>
struct IsAlTypeT<      unsigned int, Al::MPIBackend> : std::true_type {};
template <>
struct IsAlTypeT<          long int, Al::MPIBackend> : std::true_type {};
template <>
struct IsAlTypeT<     long long int, Al::MPIBackend> : std::true_type {};
template <>
struct IsAlTypeT< unsigned long int, Al::MPIBackend> : std::true_type {};
template <>
struct IsAlTypeT<unsigned long long, Al::MPIBackend> : std::true_type {};
template <>
struct IsAlTypeT<             float, Al::MPIBackend> : std::true_type {};
template <>
struct IsAlTypeT<            double, Al::MPIBackend> : std::true_type {};
template <>
struct IsAlTypeT<       long double, Al::MPIBackend> : std::true_type {};

#ifdef HYDROGEN_HAVE_NCCL2
template <>
struct IsAlTypeT<                  char, Al::NCCLBackend> : std::true_type {};
template <>
struct IsAlTypeT<         unsigned char, Al::NCCLBackend> : std::true_type {};
template <>
struct IsAlTypeT<                   int, Al::NCCLBackend> : std::true_type {};
template <>
struct IsAlTypeT<          unsigned int, Al::NCCLBackend> : std::true_type {};
template <>
struct IsAlTypeT<         long long int, Al::NCCLBackend> : std::true_type {};
template <>
struct IsAlTypeT<unsigned long long int, Al::NCCLBackend> : std::true_type {};
template <>
struct IsAlTypeT<                 float, Al::NCCLBackend> : std::true_type {};
template <>
struct IsAlTypeT<                double, Al::NCCLBackend> : std::true_type {};
#endif // HYDROGEN_HAVE_NCCL2

#ifdef HYDROGEN_HAVE_AL_MPI_CUDA
template <typename T>
struct IsAlTypeT<T, Al::MPICUDABackend> : IsAlTypeT<T, Al::MPIBackend> {};
#endif // HYDROGEN_HAVE_AL_MPI_CUDA

template <typename T, typename BackendT>
constexpr bool IsAlType() { return IsAlTypeT<T,BackendT>::value; }

/** \class IsAluminumTypeT
 *  \brief A predicate to determine if a type is a valid Aluminum type.
 *
 *  In contrast to the previous predicate, this checks all available backends.
 */
template <typename T>
struct IsAluminumTypeT
{
    static constexpr bool value = IsAlType<T, Al::MPIBackend>()
#ifdef HYDROGEN_HAVE_NCCL2
        || IsAlType<T, Al::NCCLBackend>()
#endif // HYDROGEN_HAVE_NCCL2
#ifdef HYDROGEN_HAVE_AL_MPI_CUDA
        || IsAlType<T, Al::MPICUDABackend>()
#endif // HYDROGEN_HAVE_AL_MPI_CUDA
        ;
};

template <typename T>
constexpr bool IsAluminumType() { return IsAluminumTypeT<T>::value; }

/** \class IsHostMemoryCompatibleT
 *  \brief A traits class for whether host memory can be used with a
 *      given backend.
 */
template <typename BackendT>
struct IsHostMemoryCompatibleT : std::false_type {};

// Only the MPI backend is compatible with host memory
template <>
struct IsHostMemoryCompatibleT<Al::MPIBackend> : std::true_type {};

/** \brief Helper function for IsHostMemoryCompatibleT trait. */
template <typename BackendT>
constexpr bool IsHostMemCompatible()
{
    return IsHostMemoryCompatibleT<BackendT>::value;
}

/** \class IsGPUMemoryCompatibleT
 *  \brief A traits class for whether GPU memory can be used with a
 *      given backend.
 */
template <typename BackendT>
struct IsGPUMemoryCompatibleT : std::false_type {};

// The MPI-CUDA and NCCL backends are compatible with GPU memory
#ifdef HYDROGEN_HAVE_AL_MPI_CUDA
template <>
struct IsGPUMemoryCompatibleT<Al::MPICUDABackend> : std::true_type {};
#endif // HYDROGEN_HAVE_AL_MPI_CUDA

#ifdef HYDROGEN_HAVE_NCCL2
template <>
struct IsGPUMemoryCompatibleT<Al::NCCLBackend> : std::true_type {};
#endif // HYDROGEN_HAVE_NCCL2

/** \brief Helper function for IsGPUMemoryCompatibleT trait. */
template <typename BackendT>
constexpr bool IsGPUMemCompatible()
{
    return IsGPUMemoryCompatibleT<BackendT>::value;
}

// FIXME: We need to account for the fact that CUDA might be enabled
// in Hydrogen but Aluminum might not have a GPU backend enabled.

template <Device D>
struct BackendsForDeviceT;

template <>
struct BackendsForDeviceT<Device::CPU>
{
    using type = TypeList<Al::MPIBackend>;
};// struct BackendsForDeviceT<Device::CPU>

// Prefer the NCCL2 backend
#ifdef HYDROGEN_HAVE_CUDA
template <>
struct BackendsForDeviceT<Device::GPU>
{
    using type = TypeList<
#ifdef HYDROGEN_HAVE_NCCL2
        Al::NCCLBackend
#ifdef HYDROGEN_HAVE_AL_MPI_CUDA
        ,
#endif // HYDROGEN_HAVE_AL_MPI_CUDA
#endif // HYDROGEN_HAVE_NCCL2
#ifdef HYDROGEN_HAVE_AL_MPI_CUDA
        Al::MPICUDABackend
#endif // HYDROGEN_HAVE_AL_MPI_CUDA
        >;
};// struct BackendsForDeviceT<Device::GPU>
#endif // HYDROGEN_HAVE_CUDA

// Helper using statement
template <Device D>
using BackendsForDevice = typename BackendsForDeviceT<D>::type;

//
// Aluminum-specific predicates/metafunctions
//

// Predicate for checking if T is a valid Aluminum type on Device D
//
// A type T is a valid Aluminum type iff IsAlType<T,BackendT> is true
// for some BackendT in BackendsForDevice<D>.
template <typename T, Device D>
struct IsAluminumDeviceType
    : IsTrueForAny<BackendsForDevice<D>, T, IsAlTypeT>
{};

template <Collective C, typename BackendT>
struct IsBackendSupported : std::false_type {};

// MPI backend only supports AllReduce
template <>
struct IsBackendSupported<Collective::ALLREDUCE, Al::MPIBackend>
    : std::true_type {};

#ifdef HYDROGEN_HAVE_NCCL2
// NCCL backend supports these
template <>
struct IsBackendSupported<Collective::ALLGATHER, Al::NCCLBackend>
    : std::true_type {};
template <>
struct IsBackendSupported<Collective::ALLREDUCE, Al::NCCLBackend>
    : std::true_type {};
template <>
struct IsBackendSupported<Collective::BROADCAST, Al::NCCLBackend>
    : std::true_type {};
template <>
struct IsBackendSupported<Collective::REDUCE, Al::NCCLBackend>
    : std::true_type {};
template <>
struct IsBackendSupported<Collective::REDUCESCATTER, Al::NCCLBackend>
    : std::true_type {};
#endif // HYDROGEN_HAVE_NCCL2

#ifdef HYDROGEN_HAVE_AL_MPI_CUDA
// MPICUDA backend only supports AllReduce
template <>
struct IsBackendSupported<Collective::ALLREDUCE, Al::MPICUDABackend>
    : std::true_type {};
template <>
struct IsBackendSupported<Collective::ALLTOALL, Al::MPICUDABackend>
    : std::true_type {};
template <>
struct IsBackendSupported<Collective::GATHER, Al::MPICUDABackend>
    : std::true_type {};
template <>
struct IsBackendSupported<Collective::SENDRECV, Al::MPICUDABackend>
    : std::true_type {};
#endif // HYDROGEN_HAVE_AL_MPI_CUDA

template <Collective C, typename BackendList>
struct IsBackendSupportedByAny
    : Or<IsBackendSupported<C,Head<BackendList>>,
         IsBackendSupportedByAny<C,Tail<BackendList>>>
{};

template <Collective C>
struct IsBackendSupportedByAny<C,TypeList<>>
    : std::false_type {};

template <typename T, Device D, Collective C>
struct IsAluminumSupported
    : And<IsBackendSupportedByAny<C,BackendsForDevice<D>>,
          IsAluminumDeviceType<T,D>>
{};

template <typename T, typename BackendT, Collective C>
struct AluminumSupportsBackendAndCollective
    : And<IsBackendSupported<C,BackendT>, IsAlTypeT<T,BackendT>>
{};

template <typename List, typename U,
          Collective C, template <class,class,Collective> class Pred>
struct SelectFirstOkBackend
    : Select<Pred<U,Head<List>,C>, HeadT<List>,
             SelectFirstOkBackend<Tail<List>,U,C,Pred>>
{};

// The "best" backend is the first one in the list that supports our
// type T and implements our collective C.
template <typename T, Device D, Collective C>
struct BestBackendT
    : SelectFirstOkBackend<BackendsForDevice<D>,T,C,AluminumSupportsBackendAndCollective>
{};

template <typename T, Device D, Collective C>
using BestBackend = typename BestBackendT<T,D,C>::type;

#endif // ndefined(HYDROGEN_HAVE_ALUMINUM)

} // namespace El

#endif // HYDROGEN_IMPORTS_ALUMINUM_HPP_
