/*
   Copyright (c) 2009-2014, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ELEM_DISTMATRIX_MD_STAR_DECL_HPP
#define ELEM_DISTMATRIX_MD_STAR_DECL_HPP

namespace elem {

// Partial specialization to A[MD,* ].
// 
// The columns of these distributed matrices will be distributed like 
// "Matrix Diagonals" (MD). It is important to recognize that the diagonal
// of a sufficiently large distributed matrix is distributed amongst the 
// entire process grid if and only if the dimensions of the process grid
// are coprime.
template<typename T>
class DistMatrix<T,MD,STAR> : public AbstractDistMatrix<T,MD,STAR>
{
public:
    // Typedefs
    // ========
    typedef AbstractDistMatrix<T,MD,STAR> admType;
    typedef DistMatrix<T,MD,STAR> type;

    // Constructors and destructors
    // ============================
    // Create a 0 x 0 distributed matrix
    DistMatrix( const elem::Grid& g=DefaultGrid() );
    // Create a height x width distributed matrix
    DistMatrix( Int height, Int width, const elem::Grid& g=DefaultGrid() );
    // Create a height x width distributed matrix with specified alignments
    DistMatrix
    ( Int height, Int width, Int colAlign, Int root, const elem::Grid& g );
    // Create a height x width distributed matrix with specified alignments
    // and leading dimension
    DistMatrix
    ( Int height, Int width, 
      Int colAlign, Int root, Int ldim, const elem::Grid& g );
    // View a constant distributed matrix's buffer
    DistMatrix
    ( Int height, Int width, Int colAlign, Int root,
      const T* buffer, Int ldim, const elem::Grid& g );
    // View a mutable distributed matrix's buffer
    DistMatrix
    ( Int height, Int width, Int colAlign, Int root,
      T* buffer, Int ldim, const elem::Grid& g );
    // Create a copy of distributed matrix A
    DistMatrix( const type& A );
    template<Dist U,Dist V> DistMatrix( const DistMatrix<T,U,V>& A );
#ifndef SWIG
    // Move constructor
    DistMatrix( type&& A );
#endif
    // Destructor
    ~DistMatrix();

    // Assignment and reconfiguration
    // ==============================
    const type& operator=( const DistMatrix<T,MC,  MR  >& A );
    const type& operator=( const DistMatrix<T,MC,  STAR>& A );
    const type& operator=( const DistMatrix<T,STAR,MR  >& A );
    const type& operator=( const DistMatrix<T,MD,  STAR>& A );
    const type& operator=( const DistMatrix<T,STAR,MD  >& A );
    const type& operator=( const DistMatrix<T,MR,  MC  >& A );
    const type& operator=( const DistMatrix<T,MR,  STAR>& A );
    const type& operator=( const DistMatrix<T,STAR,MC  >& A );
    const type& operator=( const DistMatrix<T,VC,  STAR>& A );
    const type& operator=( const DistMatrix<T,STAR,VC  >& A );
    const type& operator=( const DistMatrix<T,VR,  STAR>& A );
    const type& operator=( const DistMatrix<T,STAR,VR  >& A );
    const type& operator=( const DistMatrix<T,STAR,STAR>& A );
    const type& operator=( const DistMatrix<T,CIRC,CIRC>& A );
#ifndef SWIG
    // Move assignment
    type& operator=( type&& A );
#endif

    // Buffer attachment
    // -----------------
    // (Immutable) view of a distributed matrix's buffer
    void Attach
    ( Int height, Int width, Int colAlign, Int root,
      T* buffer, Int ldim, const elem::Grid& grid );
    void LockedAttach
    ( Int height, Int width, Int colAlign, Int root, 
      const T* buffer, Int ldim, const elem::Grid& grid );
    void Attach
    ( Matrix<T>& A, Int colAlign, Int root, const elem::Grid& grid );
    void LockedAttach
    ( const Matrix<T>& A, Int colAlign, Int root, const elem::Grid& grid );

    // Realignment
    // -----------
    virtual void AlignWith( const elem::DistData& data );
    virtual void AlignColsWith( const elem::DistData& data );

    // Basic queries
    // =============
    virtual elem::DistData DistData() const;
    virtual mpi::Comm DistComm() const;
    virtual mpi::Comm CrossComm() const;
    virtual mpi::Comm RedundantComm() const;
    virtual mpi::Comm ColComm() const;
    virtual mpi::Comm RowComm() const;
    virtual Int RowStride() const;
    virtual Int ColStride() const;

private:
    // Exchange metadata with another matrix
    // =====================================
    virtual void ShallowSwap( type& A );

    // Friend declarations
    // ===================
#ifndef SWIG
    template<typename S,Dist U,Dist V> friend class DistMatrix;
    friend void HandleDiagPath<>( type& A, const type& B );
#endif 
};

} // namespace elem

#endif // ifndef ELEM_DISTMATRIX_MD_STAR_DECL_HPP
