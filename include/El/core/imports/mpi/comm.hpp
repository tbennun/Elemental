#pragma once
#ifndef EL_IMPORTS_MPI_COMM_HPP_
#define EL_IMPORTS_MPI_COMM_HPP_

#include <El/config.h>
#include <El/core/imports/mpi/plain_comm.hpp>
#ifdef HYDROGEN_HAVE_ALUMINUM
#include <El/core/imports/mpi/aluminum_comm.hpp>
#endif // HYDROGEN_HAVE_ALUMINUM

#include <type_traits>

namespace El
{
namespace mpi
{

#ifdef HYDROGEN_HAVE_ALUMINUM
using Comm = AluminumComm;
#else
using Comm = PlainComm;
#endif // HYDROGEN_HAVE_ALUMINUM

using CommRef = std::reference_wrapper<Comm>;
using CommCRef = std::reference_wrapper<Comm const>;

}// namespace mpi
}// namespace El
#endif /* EL_IMPORTS_MPI_COMM_HPP_ */
