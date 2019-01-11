#pragma once
#ifndef EL_IMPORTS_MPI_COMM_IMPL_HPP_
#define EL_IMPORTS_MPI_COMM_IMPL_HPP_

#include <El/config.h>
#include <mpi.h>

namespace El
{
namespace mpi
{

template <typename SpecificCommImpl>
class CommImpl
{
public:

    /** @name Constructors */
    ///@{

    CommImpl() = default;
    CommImpl(MPI_Comm mpi_comm)
    {
        MPI_Comm_dup(mpi_comm, &comm_);
    }

    // *NOT* copyable
    CommImpl(CommImpl<SpecificCommImpl> const&) = delete;
    CommImpl<SpecificCommImpl>& operator=(
        CommImpl<SpecificCommImpl> const&) = delete;

    // Moveable
    CommImpl(CommImpl<SpecificCommImpl>&&) EL_NO_EXCEPT = default;
    CommImpl<SpecificCommImpl>& operator=(
        CommImpl<SpecificCommImpl>&&) EL_NO_EXCEPT = default;

    ///@}
    /** @name Query metadata */
    ///@{

    /** @brief Get the process rank in the communicator. */
    int Rank() const EL_NO_RELEASE_EXCEPT
    {
        if (comm_ == MPI_COMM_NULL)
            return MPI_UNDEFINED;

        int rank;
        MPI_Comm_rank(comm_, &rank);
        return rank;
    }

    /** @brief Get the number of processes in the communicator. */
    int Size() const EL_NO_RELEASE_EXCEPT
    {
        if (comm_ == MPI_COMM_NULL)
            return MPI_UNDEFINED;

        int size;
        MPI_Comm_size(comm_, &size);
        return size;
    }

    ///@}
    /** @name Resource access */
    ///@{

    /** @brief Get the raw MPI handle. */
    MPI_Comm GetMPIComm() const EL_NO_EXCEPT { return comm_; };

    ///@}
    /** @name Modifiers */
    ///@{

    /** @brief Take ownership of an existing communicator. */
    void Control(MPI_Comm comm)
    {
        Reset();
        comm_ = comm;
    }

    /** @brief Clear all the state of the communicator. */
    void Reset()
    {
        static_cast<SpecificCommImpl*>(this)->DoReset();
        if (comm_ != MPI_COMM_NULL)
        {
            MPI_Comm_free(&comm_);
            comm_ = MPI_COMM_NULL;
        }
    }

    /** @brief Clear current state and recreate based on new comm */
    void Reset(MPI_comm comm)
    {
        Reset();
        MPI_Comm_dup(comm, &comm_);
    }

    /** @brief Relinquish ownership of the underlying resource */
    MPI_Comm Release()
    {
        MPI_Comm ret = comm_;
        comm_ = MPI_COMM_NULL;
        Reset();
        return ret;
    }

    /** @brief Swap the interal state */
    void Swap(CommImpl<SpecificCommImpl>& other)
    {
        static_cast<SpecificCommImpl*>(this)->DoSwap(
            static_cast<SpecificCommImpl&>(other));
        std::swap(comm_, other.comm_);
    }

    ///@}

protected:

    ~CommImpl()
    {
        Reset();
    }

private:

    /** @brief The raw MPI handle. */
    MPI_Comm comm_ = MPI_COMM_NULL;

};// class CommImpl

// NOTE (trb): With how this class has evolved, I'm not really
// comfortable with this anymore. Nonetheless, I'll keep them around
// for back-compatiblity until I can dig further.
//
// Combined with Jack's comment below, I think this is actually a
// truly horrific abomination, but semantic perfection is a steep goal
// at the moment.
template <typename ImplT>
bool operator==(
    CommImpl<ImplT> const& a, CommImpl<ImplT> const& b) EL_NO_EXCEPT
{ return a.GetMPIComm() == b.GetMPIComm(); }

template <typename ImplT>
bool operator!=(
    CommImpl<ImplT> const& a, CommImpl<ImplT> const& b) EL_NO_EXCEPT
{ return !(a == b); }

// Hopefully, despite the fact that MPI_Comm is opaque, the following will
// reliably hold (otherwise it must be extended). Typically, MPI_Comm is
// either 'int' or 'void*'.
template <typename ImplT>
bool operator<(
    CommImpl<ImplT> const& a, CommImpl<ImplT> const& b ) EL_NO_EXCEPT
{ return a.GetMPIComm() < b.GetMPIComm(); }

}// namespace mpi
}// namespace El
#endif /* EL_IMPORTS_MPI_COMM_IMPL_HPP_ */
