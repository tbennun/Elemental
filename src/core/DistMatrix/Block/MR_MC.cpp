/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

#define COLDIST MR
#define ROWDIST MC

#include "./setup.hpp"

namespace El {

// Public section
// ##############

// Assignment and reconfiguration
// ==============================

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,MC,MR,BLOCK>& A )
{
    DEBUG_ONLY(CSE cse("[MR,MC] = [MC,MR]"))
    // TODO: More efficient implementation
    copy::GeneralPurpose( A, *this );
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,MC,STAR,BLOCK>& A )
{
    DEBUG_ONLY(CSE cse("[MR,MC] = [MC,STAR]"))
    DistMatrix<T,VC,STAR,BLOCK> A_VC_STAR( A );
    DistMatrix<T,VR,STAR,BLOCK> A_VR_STAR( this->Grid() );
    A_VR_STAR.AlignColsWith(*this);
    A_VR_STAR = A_VC_STAR;
    A_VC_STAR.Empty(); 

    *this = A_VR_STAR;
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,STAR,MR,BLOCK>& A )
{ 
    DEBUG_ONLY(CSE cse("[MR,MC] = [STAR,MR]"))
    DistMatrix<T,STAR,VR,BLOCK> A_STAR_VR( A );
    DistMatrix<T,STAR,VC,BLOCK> A_STAR_VC( this->Grid() );
    A_STAR_VR.AlignRowsWith(*this);
    A_STAR_VC = A_STAR_VR;
    A_STAR_VR.Empty(); 

    *this = A_STAR_VC;
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,MD,STAR,BLOCK>& A )
{
    DEBUG_ONLY(CSE cse("[MR,MC] = [MD,STAR]"))
    // TODO: More efficient implementation
    copy::GeneralPurpose( A, *this );
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,STAR,MD,BLOCK>& A )
{
    DEBUG_ONLY(CSE cse("[MR,MC] = [STAR,MD]"))
    // TODO: More efficient implementation
    copy::GeneralPurpose( A, *this );
    return *this;
}

template<typename T>
BDM& BDM::operator=( const BDM& A )
{ 
    DEBUG_ONLY(CSE cse("[MR,MC] = [MR,MC]"))
    copy::Translate( A, *this );
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,MR,STAR,BLOCK>& A )
{ 
    DEBUG_ONLY(CSE cse("[MR,MC] = [MR,STAR]"))
    copy::RowFilter( A, *this );
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,STAR,MC,BLOCK>& A )
{ 
    DEBUG_ONLY(CSE cse("[MR,MC] = [STAR,MC]"))
    copy::ColFilter( A, *this );
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,VC,STAR,BLOCK>& A )
{ 
    DEBUG_ONLY(CSE cse("[MR,MC] = [VC,STAR]"))
    DistMatrix<T,VR,STAR,BLOCK> A_VR_STAR( A );
    *this = A_VR_STAR;
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,STAR,VC,BLOCK>& A )
{ 
    DEBUG_ONLY(CSE cse("[MR,MC] = [STAR,VC]"))
    copy::RowAllToAllPromote( A, *this );
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,VR,STAR,BLOCK>& A )
{ 
    DEBUG_ONLY(CSE cse("[MR,MC] = [VR,STAR]"))
    copy::ColAllToAllPromote( A, *this );
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,STAR,VR,BLOCK>& A )
{ 
    DEBUG_ONLY(CSE cse("[MR,MC] = [STAR,VR]"))
    DistMatrix<T,STAR,VC,BLOCK> A_STAR_VC( A );
    *this = A_STAR_VC;
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,STAR,STAR,BLOCK>& A )
{
    DEBUG_ONLY(CSE cse("[MR,MC] = [STAR,STAR]"))
    copy::Filter( A, *this );
    return *this;
}

template<typename T>
BDM& BDM::operator=( const DistMatrix<T,CIRC,CIRC,BLOCK>& A )
{
    DEBUG_ONLY(CSE cse("[MR,MC] = [CIRC,CIRC]"))
    // TODO: More efficient implementation
    copy::GeneralPurpose( A, *this );
    return *this;
}

template<typename T>
BDM& BDM::operator=( const BlockMatrix<T>& A )
{
    DEBUG_ONLY(CSE cse("BDM = ABDM"))
    #define GUARD(CDIST,RDIST) \
      A.DistData().colDist == CDIST && A.DistData().rowDist == RDIST
    #define PAYLOAD(CDIST,RDIST) \
      auto& ACast = \
        dynamic_cast<const DistMatrix<T,CDIST,RDIST,BLOCK>&>(A); \
      *this = ACast;
    #include "El/macros/GuardAndPayload.h"
    return *this;
}

// Basic queries
// =============
template<typename T>
mpi::Comm BDM::DistComm() const EL_NO_EXCEPT { return this->grid_->VRComm(); }

template<typename T>
mpi::Comm BDM::CrossComm() const EL_NO_EXCEPT
{ return ( this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL ); }

template<typename T>
mpi::Comm BDM::RedundantComm() const EL_NO_EXCEPT
{ return ( this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL ); }

template<typename T>
mpi::Comm BDM::ColComm() const EL_NO_EXCEPT { return this->grid_->MRComm(); }
template<typename T>
mpi::Comm BDM::RowComm() const EL_NO_EXCEPT { return this->grid_->MCComm(); }

template<typename T>
mpi::Comm BDM::PartialColComm() const EL_NO_EXCEPT { return this->ColComm(); }
template<typename T>
mpi::Comm BDM::PartialRowComm() const EL_NO_EXCEPT { return this->RowComm(); }
template<typename T>
mpi::Comm BDM::PartialUnionColComm() const EL_NO_EXCEPT
{ return ( this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL ); }
template<typename T>
mpi::Comm BDM::PartialUnionRowComm() const EL_NO_EXCEPT
{ return ( this->Grid().InGrid() ? mpi::COMM_SELF : mpi::COMM_NULL ); }

template<typename T>
int BDM::ColStride() const EL_NO_EXCEPT { return this->grid_->MRSize(); }
template<typename T>
int BDM::RowStride() const EL_NO_EXCEPT { return this->grid_->MCSize(); }
template<typename T>
int BDM::DistSize() const EL_NO_EXCEPT { return this->grid_->VRSize(); }
template<typename T>
int BDM::CrossSize() const EL_NO_EXCEPT { return 1; }
template<typename T>
int BDM::RedundantSize() const EL_NO_EXCEPT { return 1; }
template<typename T>
int BDM::PartialColStride() const EL_NO_EXCEPT { return this->ColStride(); }
template<typename T>
int BDM::PartialRowStride() const EL_NO_EXCEPT { return this->RowStride(); }
template<typename T>
int BDM::PartialUnionColStride() const EL_NO_EXCEPT { return 1; }
template<typename T>
int BDM::PartialUnionRowStride() const EL_NO_EXCEPT { return 1; }

template<typename T>
int BDM::ColRank() const EL_NO_EXCEPT { return this->grid_->MRRank(); }
template<typename T>
int BDM::RowRank() const EL_NO_EXCEPT { return this->grid_->MCRank(); }
template<typename T>
int BDM::DistRank() const EL_NO_EXCEPT { return this->grid_->VRRank(); }
template<typename T>
int BDM::CrossRank() const EL_NO_EXCEPT
{ return ( this->Grid().InGrid() ? 0 : mpi::UNDEFINED ); }
template<typename T>
int BDM::RedundantRank() const EL_NO_EXCEPT
{ return ( this->Grid().InGrid() ? 0 : mpi::UNDEFINED ); }
template<typename T>
int BDM::PartialColRank() const EL_NO_EXCEPT { return this->ColRank(); }
template<typename T>
int BDM::PartialRowRank() const EL_NO_EXCEPT { return this->RowRank(); }
template<typename T>
int BDM::PartialUnionColRank() const EL_NO_EXCEPT
{ return ( this->Grid().InGrid() ? 0 : mpi::UNDEFINED ); }
template<typename T>
int BDM::PartialUnionRowRank() const EL_NO_EXCEPT
{ return ( this->Grid().InGrid() ? 0 : mpi::UNDEFINED ); }

// Instantiate {Int,Real,Complex<Real>} for each Real in {float,double}
// ####################################################################

#define SELF(T,U,V) \
  template DistMatrix<T,COLDIST,ROWDIST,BLOCK>::DistMatrix \
  ( const DistMatrix<T,U,V,BLOCK>& A );
#define OTHER(T,U,V) \
  template DistMatrix<T,COLDIST,ROWDIST,BLOCK>::DistMatrix \
  ( const DistMatrix<T,U,V>& A ); \
  template DistMatrix<T,COLDIST,ROWDIST,BLOCK>& \
           DistMatrix<T,COLDIST,ROWDIST,BLOCK>::operator= \
           ( const DistMatrix<T,U,V>& A )
#define BOTH(T,U,V) \
  SELF(T,U,V); \
  OTHER(T,U,V)
#define PROTO(T) \
  template class DistMatrix<T,COLDIST,ROWDIST,BLOCK>; \
  BOTH( T,CIRC,CIRC); \
  BOTH( T,MC,  MR  ); \
  BOTH( T,MC,  STAR); \
  BOTH( T,MD,  STAR); \
  OTHER(T,MR,  MC  ); \
  BOTH( T,MR,  STAR); \
  BOTH( T,STAR,MC  ); \
  BOTH( T,STAR,MD  ); \
  BOTH( T,STAR,MR  ); \
  BOTH( T,STAR,STAR); \
  BOTH( T,STAR,VC  ); \
  BOTH( T,STAR,VR  ); \
  BOTH( T,VC,  STAR); \
  BOTH( T,VR,  STAR);

#define EL_ENABLE_QUAD
#include "El/macros/Instantiate.h"

} // namespace El
