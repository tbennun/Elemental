/*
   This file is part of elemental, a library for distributed-memory dense 
   linear algebra.

   Copyright (C) 2009-2010 Jack Poulson <jack.poulson@gmail.com>

   This program is released under the terms of the license contained in the 
   file LICENSE.
*/
#include "elemental/blas_internal.hpp"
using namespace std;
using namespace elemental;

namespace {

#ifndef RELEASE
template<typename T>
void 
CheckInput
( const DistMatrix<T,MC,  Star>& A1, 
  const DistMatrix<T,MC,  Star>& A2,
  const DistMatrix<T,Star,MR  >& B1,
  const DistMatrix<T,Star,MR  >& B2,
  const DistMatrix<T,MC,  MR  >& C  )
{
    if( A1.GetGrid() != A2.GetGrid() || A2.GetGrid() != B1.GetGrid() ||
        B1.GetGrid() != B2.GetGrid() || B2.GetGrid() != C.GetGrid() )
        throw logic_error
        ( "A, B, and C must be distributed over the same grid." );
    if( A1.Height() != C.Height() || B1.Width() != C.Width() ||
        A1.Height() != A2.Height() || A1.Width() != A2.Width() ||
        B1.Height() != B2.Height() || B1.Width() != B2.Width() ||
        A1.Width() != B1.Height() )
    {
        ostringstream msg;
        msg << "Nonconformal TriangularRank2K: " << endl
            << "  A1[MC,* ] ~ " << A1.Height() << " x "
                                << A1.Width()  << endl
            << "  A2[MC,* ] ~ " << A2.Height() << " x "
                                << A2.Width()  << endl
            << "  B1[* ,MR] ~ " << B1.Height() << " x "
                                << B1.Width()  << endl
            << "  B2[* ,MR] ~ " << B2.Height() << " x "
                                << B2.Width()  << endl
            << "  C[MC,MR] ~ " << C.Height() << " x " << C.Width() << endl;
        throw logic_error( msg.str() );
    }
    if( A1.ColAlignment() != C.ColAlignment() ||
        B1.RowAlignment() != C.RowAlignment() ||
        A1.ColAlignment() != A2.ColAlignment() ||
        B1.RowAlignment() != B2.RowAlignment() )
    {
        ostringstream msg;
        msg << "Misaligned TriangularRank2K: " << endl
            << "  A1[MC,* ] ~ " << A1.ColAlignment() << endl
            << "  A2[MC,* ] ~ " << A2.ColAlignment() << endl
            << "  B1[* ,MR] ~ " << B1.RowAlignment() << endl
            << "  B2[* ,MR] ~ " << B2.RowAlignment() << endl
            << "  C[MC,MR] ~ " << C.ColAlignment() << " , " <<
                                  C.RowAlignment() << endl;
        throw logic_error( msg.str() );
    }
}

template<typename T>
void 
CheckInput
( Orientation orientationOfB1,
  const DistMatrix<T,MC,  Star>& A1, 
  const DistMatrix<T,MC,  Star>& A2,
  const DistMatrix<T,MR,  Star>& B1,
  const DistMatrix<T,Star,MR  >& B2,
  const DistMatrix<T,MC,  MR  >& C )
{
    if( orientationOfB1 == Normal )
        throw logic_error( "B1[MR,* ] must be (Conjugate)Transpose'd." );
    if( A1.GetGrid() != A2.GetGrid() || A2.GetGrid() != B1.GetGrid() ||
        B1.GetGrid() != B2.GetGrid() || B2.GetGrid() != C.GetGrid() )
        throw logic_error
        ( "A, B, and C must be distributed over the same grid." );
    if( A1.Height() != C.Height() || B1.Height() != C.Width() ||
        A1.Height() != A2.Height() || A1.Width() != A2.Width() ||
        B1.Width() != B2.Height() || B1.Height() != B2.Width() ||
        A1.Width() != B1.Width() )
    {
        ostringstream msg;
        msg << "Nonconformal TriangularRank2K: " << endl
            << "  A1[MC,* ] ~ " << A1.Height() << " x "
                                << A1.Width()  << endl
            << "  A2[MC,* ] ~ " << A2.Height() << " x "
                                << A2.Width()  << endl
            << "  B1[MR,* ] ~ " << B1.Height() << " x "
                                << B1.Width()  << endl
            << "  B2[* ,MR] ~ " << B2.Height() << " x "
                                << B2.Width()  << endl
            << "  C[MC,MR] ~ " << C.Height() << " x " << C.Width() << endl;
        throw logic_error( msg.str() );
    }
    if( A1.ColAlignment() != C.ColAlignment() ||
        B1.ColAlignment() != C.RowAlignment() ||
        A1.ColAlignment() != A2.ColAlignment() ||
        B1.ColAlignment() != B2.RowAlignment() )
    {
        ostringstream msg;
        msg << "Misaligned TriangularRank2K: " << endl
            << "  A1[MC,* ] ~ " << A1.ColAlignment() << endl
            << "  A2[MC,* ] ~ " << A2.ColAlignment() << endl
            << "  B1[MR,* ] ~ " << B1.ColAlignment() << endl
            << "  B2[* ,MR] ~ " << B2.RowAlignment() << endl
            << "  C[MC,MR] ~ " << C.ColAlignment() << " , " <<
                                  C.RowAlignment() << endl;
        throw logic_error( msg.str() );
    }
}

template<typename T>
void 
CheckInput
( Orientation orientationOfA1,
  Orientation orientationOfA2,
  const DistMatrix<T,Star,MC>& A1, 
  const DistMatrix<T,Star,MC>& A2,
  const DistMatrix<T,Star,MR>& B1,
  const DistMatrix<T,Star,MR>& B2,
  const DistMatrix<T,MC,  MR>& C )
{
    if( orientationOfA1 == Normal )
        throw logic_error( "A1[* ,MC] must be (Conjugate)Transpose'd." );
    if( orientationOfA2 == Normal )
        throw logic_error( "A2[* ,MC] must be (Conjugate)Transpose'd." );
    if( A1.GetGrid() != A2.GetGrid() || A2.GetGrid() != B1.GetGrid() ||
        B1.GetGrid() != B2.GetGrid() || B2.GetGrid() != C.GetGrid() )
        throw logic_error
        ( "A, B, and C must be distributed over the same grid." );
    if( A1.Width() != C.Height() || B1.Width() != C.Width() ||
        A1.Height() != A2.Height() || A1.Width() != A2.Width() ||
        B1.Height() != B2.Height() || B1.Width() != B2.Width() ||
        A1.Height() != B1.Height() )
    {
        ostringstream msg;
        msg << "Nonconformal TriangularRank2K: " << endl
            << "  A1[* ,MC] ~ " << A1.Height() << " x "
                                << A1.Width()  << endl
            << "  A2[* ,MC] ~ " << A2.Height() << " x "
                                << A2.Width()  << endl
            << "  B1[* ,MR] ~ " << B1.Height() << " x "
                                << B1.Width()  << endl
            << "  B2[* ,MR] ~ " << B2.Height() << " x "
                                << B2.Width()  << endl
            << "  C[MC,MR] ~ " << C.Height() << " x " << C.Width() << endl;
        throw logic_error( msg.str() );
    }
    if( A1.RowAlignment() != C.ColAlignment() ||
        B1.RowAlignment() != C.RowAlignment() ||
        A1.RowAlignment() != A2.RowAlignment() ||
        B1.RowAlignment() != B2.RowAlignment() )
    {
        ostringstream msg;
        msg << "Misaligned TriangularRank2K: " << endl
            << "  A1[* ,MC] ~ " << A1.RowAlignment() << endl
            << "  A2[* ,MC] ~ " << A2.RowAlignment() << endl
            << "  B1[* ,MR] ~ " << B1.RowAlignment() << endl
            << "  B2[* ,MR] ~ " << B2.RowAlignment() << endl
            << "  C[MC,MR] ~ " << C.ColAlignment() << " , " <<
                                  C.RowAlignment() << endl;
        throw logic_error( msg.str() );
    }
}

template<typename T>
void 
CheckInput
( Orientation orientationOfA1,
  Orientation orientationOfA2,
  Orientation orientationOfB1,
  Orientation orientationOfB2,
  const DistMatrix<T,Star,MC  >& A1, 
  const DistMatrix<T,Star,MC  >& A2,
  const DistMatrix<T,MR,  Star>& B1,
  const DistMatrix<T,MR,  Star>& B2,
  const DistMatrix<T,MC,  MR  >& C )
{
    if( orientationOfA1 == Normal )
        throw logic_error( "A1[* ,MC] must be (Conjugate)Transpose'd." );
    if( orientationOfA2 == Normal )
        throw logic_error( "A2[* ,MC] must be (Conjugate)Transpose'd." );
    if( orientationOfB1 == Normal )
        throw logic_error( "B1[MR,* ] must be (Conjugate)Transpose'd." );
    if( orientationOfB2 == Normal )
        throw logic_error( "B2[MR,* ] must be (Conjugate)Transpose'd." );
    if( A1.GetGrid() != A2.GetGrid() || A2.GetGrid() != B1.GetGrid() ||
        B1.GetGrid() != B2.GetGrid() || B2.GetGrid() != C.GetGrid() )
        throw logic_error
        ( "A, B, and C must be distributed over the same grid." );
    if( A1.Width() != C.Height() || B1.Height() != C.Width() ||
        A1.Height() != A2.Height() || A1.Width() != A2.Width() ||
        B1.Height() != B2.Height() || B1.Width() != B2.Width() ||
        A1.Height() != B1.Width() )
    {
        ostringstream msg;
        msg << "Nonconformal TriangularRank2K: " << endl
            << "  A1[* ,MC] ~ " << A1.Height() << " x "
                                << A1.Width()  << endl
            << "  A2[* ,MC] ~ " << A2.Height() << " x "
                                << A2.Width()  << endl
            << "  B1[MR,* ] ~ " << B1.Height() << " x "
                                << B1.Width()  << endl
            << "  B2[MR,* ] ~ " << B2.Height() << " x "
                                << B2.Width()  << endl
            << "  C[MC,MR] ~ " << C.Height() << " x " << C.Width() << endl;
        throw logic_error( msg.str() );
    }
    if( A1.RowAlignment() != C.ColAlignment() ||
        B1.ColAlignment() != C.RowAlignment() ||
        A1.RowAlignment() != A2.RowAlignment() ||
        B1.ColAlignment() != B2.ColAlignment() )
    {
        ostringstream msg;
        msg << "Misaligned TriangularRank2K: " << endl
            << "  A1[* ,MC] ~ " << A1.RowAlignment() << endl
            << "  A2[* ,MC] ~ " << A2.RowAlignment() << endl
            << "  B1[MR,* ] ~ " << B1.ColAlignment() << endl
            << "  B2[MR,* ] ~ " << B2.ColAlignment() << endl
            << "  C[MC,MR] ~ " << C.ColAlignment() << " , " <<
                                  C.RowAlignment() << endl;
        throw logic_error( msg.str() );
    }
}
#endif

template<typename T>
void
TriangularRank2KKernel
( Shape shape,
  T alpha, const DistMatrix<T,MC,  Star>& A1,
           const DistMatrix<T,MC,  Star>& A2,
           const DistMatrix<T,Star,MR  >& B1,
           const DistMatrix<T,Star,MR  >& B2,
  T beta,        DistMatrix<T,MC,  MR  >& C )
{
#ifndef RELEASE
    PushCallStack("TriangularRank2KKernel");
    CheckInput( A1, A2, B1, B2, C );
#endif
    const Grid& grid = C.GetGrid();

    DistMatrix<T,MC,Star> A1T(grid),  A2T(grid),
                          A1B(grid),  A2B(grid);

    DistMatrix<T,Star,MR> B1L(grid), B1R(grid),
                          B2L(grid), B2R(grid);

    DistMatrix<T,MC,MR> CTL(grid), CTR(grid),
                        CBL(grid), CBR(grid);

    DistMatrix<T,MC,MR> DTL(grid), DBR(grid);

    const unsigned half = C.Height()/2;

    blas::Scal( beta, C );

    LockedPartitionDown
    ( A1, A1T,
          A1B, half );
    LockedPartitionDown
    ( A2, A2T,
          A2B, half );

    LockedPartitionRight( B1, B1L, B1R, half );
    LockedPartitionRight( B2, B2L, B2R, half );

    PartitionDownDiagonal
    ( C, CTL, CTR,
         CBL, CBR, half );

    DTL.AlignWith( CTL );
    DBR.AlignWith( CBR );
    DTL.ResizeTo( CTL.Height(), CTL.Width() );
    DBR.ResizeTo( CBR.Height(), CBR.Width() );
    //------------------------------------------------------------------------//
    if( shape == Lower )
    {
        blas::Gemm
        ( Normal, Normal,
          alpha, A1B.LockedLocalMatrix(),
                 B2L.LockedLocalMatrix(),
          (T)1,  CBL.LocalMatrix() );
        blas::Gemm
        ( Normal, Normal,
          alpha, A2B.LockedLocalMatrix(),
                 B1L.LockedLocalMatrix(),
          (T)1,  CBL.LocalMatrix() );
    }
    else
    {
        blas::Gemm
        ( Normal, Normal,
          alpha, A1T.LockedLocalMatrix(),
                 B2R.LockedLocalMatrix(),
          (T)1,  CTR.LocalMatrix() );
        blas::Gemm
        ( Normal, Normal,
          alpha, A2T.LockedLocalMatrix(),
                 B1R.LockedLocalMatrix(),
          (T)1,  CTR.LocalMatrix() );
    }

    blas::Gemm
    ( Normal, Normal,
      alpha, A1T.LockedLocalMatrix(),
             B2L.LockedLocalMatrix(),
      (T)0,  DTL.LocalMatrix() );
    blas::Gemm
    ( Normal, Normal,
      alpha, A2T.LockedLocalMatrix(),
             B1L.LockedLocalMatrix(),
      (T)1,  DTL.LocalMatrix() );
    DTL.MakeTrapezoidal( Left, shape );
    blas::Axpy( (T)1, DTL, CTL );

    blas::Gemm
    ( Normal, Normal,
      alpha, A1B.LockedLocalMatrix(),
             B2R.LockedLocalMatrix(),
      (T)0,  DBR.LocalMatrix() );
    blas::Gemm
    ( Normal, Normal,
      alpha, A2B.LockedLocalMatrix(),
             B1R.LockedLocalMatrix(),
      (T)1,  DBR.LocalMatrix() );
    DBR.MakeTrapezoidal( Left, shape );
    blas::Axpy( (T)1, DBR, CBR );
    //------------------------------------------------------------------------//
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
TriangularRank2KKernel
( Shape shape,
  Orientation orientationOfB1,
  T alpha, const DistMatrix<T,MC,  Star>& A1,
           const DistMatrix<T,MC,  Star>& A2,
           const DistMatrix<T,MR,  Star>& B1,
           const DistMatrix<T,Star,MR  >& B2,
  T beta,        DistMatrix<T,MC,  MR  >& C )
{
#ifndef RELEASE
    PushCallStack("TriangularRank2KKernel");
    CheckInput( orientationOfB1, A1, A2, B1, B2, C );
#endif
    const Grid& grid = C.GetGrid();

    DistMatrix<T,MC,Star> A1T(grid),  A2T(grid),
                          A1B(grid),  A2B(grid);

    DistMatrix<T,MR,Star> B1T(grid), 
                          B1B(grid);

    DistMatrix<T,Star,MR> B2L(grid), B2R(grid);

    DistMatrix<T,MC,MR> CTL(grid), CTR(grid),
                        CBL(grid), CBR(grid);

    DistMatrix<T,MC,MR> DTL(grid), DBR(grid);

    const unsigned half = C.Height()/2;

    blas::Scal( beta, C );

    LockedPartitionDown
    ( A1, A1T,
          A1B, half );
    LockedPartitionDown
    ( A2, A2T,
          A2B, half );

    LockedPartitionDown
    ( B1, B1T,
          B1B, half );

    LockedPartitionRight( B2, B2L, B2R, half );

    PartitionDownDiagonal
    ( C, CTL, CTR,
         CBL, CBR, half );

    DTL.AlignWith( CTL );
    DBR.AlignWith( CBR );
    DTL.ResizeTo( CTL.Height(), CTL.Width() );
    DBR.ResizeTo( CBR.Height(), CBR.Width() );
    //------------------------------------------------------------------------//
    if( shape == Lower )
    {
        blas::Gemm
        ( Normal, Normal,
          alpha, A1B.LockedLocalMatrix(),
                 B2L.LockedLocalMatrix(),
          (T)1,  CBL.LocalMatrix() );
        blas::Gemm
        ( Normal, orientationOfB1,
          alpha, A2B.LockedLocalMatrix(),
                 B1T.LockedLocalMatrix(),
          (T)1,  CBL.LocalMatrix() );
    }
    else
    {
        blas::Gemm
        ( Normal, Normal,
          alpha, A1T.LockedLocalMatrix(),
                 B2R.LockedLocalMatrix(),
          (T)1,  CTR.LocalMatrix() );
        blas::Gemm
        ( Normal, orientationOfB1,
          alpha, A2T.LockedLocalMatrix(),
                 B1B.LockedLocalMatrix(),
          (T)1,  CTR.LocalMatrix() );
    }

    blas::Gemm
    ( Normal, Normal,
      alpha, A1T.LockedLocalMatrix(),
             B2L.LockedLocalMatrix(),
      (T)0,  DTL.LocalMatrix() );
    blas::Gemm
    ( Normal, orientationOfB1,
      alpha, A2T.LockedLocalMatrix(),
             B1T.LockedLocalMatrix(),
      (T)1,  DTL.LocalMatrix() );
    DTL.MakeTrapezoidal( Left, shape );
    blas::Axpy( (T)1, DTL, CTL );

    blas::Gemm
    ( Normal, Normal,
      alpha, A1B.LockedLocalMatrix(),
             B2R.LockedLocalMatrix(),
      (T)0,  DBR.LocalMatrix() );
    blas::Gemm
    ( Normal, orientationOfB1,
      alpha, A2B.LockedLocalMatrix(),
             B1B.LockedLocalMatrix(),
      (T)1,  DBR.LocalMatrix() );
    DBR.MakeTrapezoidal( Left, shape );
    blas::Axpy( (T)1, DBR, CBR );
    //------------------------------------------------------------------------//
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
TriangularRank2KKernel
( Shape shape,
  Orientation orientationOfA1,
  Orientation orientationOfA2,
  T alpha, const DistMatrix<T,Star,MC>& A1,
           const DistMatrix<T,Star,MC>& A2,
           const DistMatrix<T,Star,MR>& B1,
           const DistMatrix<T,Star,MR>& B2,
  T beta,        DistMatrix<T,MC,  MR>& C )
{
#ifndef RELEASE
    PushCallStack("TriangularRank2KKernel");
    CheckInput( orientationOfA1, orientationOfA2, A1, A2, B1, B2, C );
#endif
    const Grid& grid = C.GetGrid();

    DistMatrix<T,Star,MC> A1L(grid), A1R(grid),
                          A2L(grid), A2R(grid);

    DistMatrix<T,Star,MR> B1L(grid), B1R(grid),
                          B2L(grid), B2R(grid);

    DistMatrix<T,MC,MR> CTL(grid), CTR(grid),
                        CBL(grid), CBR(grid);

    DistMatrix<T,MC,MR> DTL(grid), DBR(grid);

    const unsigned half = C.Height()/2;

    blas::Scal( beta, C );

    LockedPartitionRight( A1, A1L, A1R, half );
    LockedPartitionRight( A2, A2L, A2R, half );

    LockedPartitionRight( B1, B1L, B1R, half );
    LockedPartitionRight( B2, B2L, B2R, half );

    PartitionDownDiagonal
    ( C, CTL, CTR,
         CBL, CBR, half );

    DTL.AlignWith( CTL );
    DBR.AlignWith( CBR );
    DTL.ResizeTo( CTL.Height(), CTL.Width() );
    DBR.ResizeTo( CBR.Height(), CBR.Width() );
    //------------------------------------------------------------------------//
    if( shape == Lower )
    {
        blas::Gemm
        ( orientationOfA1, Normal,
          alpha, A1R.LockedLocalMatrix(),
                 B2L.LockedLocalMatrix(),
          (T)1,  CBL.LocalMatrix() );
        blas::Gemm
        ( orientationOfA2, Normal,
          alpha, A2R.LockedLocalMatrix(),
                 B1L.LockedLocalMatrix(),
          (T)1,  CBL.LocalMatrix() );
    }
    else
    {
        blas::Gemm
        ( orientationOfA1, Normal,
          alpha, A1L.LockedLocalMatrix(),
                 B2R.LockedLocalMatrix(),
          (T)1,  CTR.LocalMatrix() );
        blas::Gemm
        ( orientationOfA2, Normal,
          alpha, A2L.LockedLocalMatrix(),
                 B1R.LockedLocalMatrix(),
          (T)1,  CTR.LocalMatrix() );
    }

    blas::Gemm
    ( orientationOfA1, Normal,
      alpha, A1L.LockedLocalMatrix(),
             B2L.LockedLocalMatrix(),
      (T)0,  DTL.LocalMatrix() );
    blas::Gemm
    ( orientationOfA2, Normal,
      alpha, A2L.LockedLocalMatrix(),
             B1L.LockedLocalMatrix(),
      (T)1,  DTL.LocalMatrix() );
    DTL.MakeTrapezoidal( Left, shape );
    blas::Axpy( (T)1, DTL, CTL );

    blas::Gemm
    ( orientationOfA1, Normal,
      alpha, A1R.LockedLocalMatrix(),
             B2R.LockedLocalMatrix(),
      (T)0,  DBR.LocalMatrix() );
    blas::Gemm
    ( orientationOfA2, Normal,
      alpha, A2R.LockedLocalMatrix(),
             B1R.LockedLocalMatrix(),
      (T)1,  DBR.LocalMatrix() );
    DBR.MakeTrapezoidal( Left, shape );
    blas::Axpy( (T)1, DBR, CBR );
    //------------------------------------------------------------------------//
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
TriangularRank2KKernel
( Shape shape,
  Orientation orientationOfA1,
  Orientation orientationOfA2,
  Orientation orientationOfB1,
  Orientation orientationOfB2,
  T alpha, const DistMatrix<T,Star,MC  >& A1,
           const DistMatrix<T,Star,MC  >& A2,
           const DistMatrix<T,MR,  Star>& B1,
           const DistMatrix<T,MR,  Star>& B2,
  T beta,        DistMatrix<T,MC,  MR  >& C )
{
#ifndef RELEASE
    PushCallStack("TriangularRank2KKernel");
    CheckInput
    ( orientationOfA1, orientationOfA2, orientationOfB1, orientationOfB2, 
      A1, A2, B1, B2, C );
#endif
    const Grid& grid = C.GetGrid();

    DistMatrix<T,Star,MC> A1L(grid), A1R(grid),
                          A2L(grid), A2R(grid);

    DistMatrix<T,MR,Star> B1T(grid),  B2T(grid),
                          B1B(grid),  B2B(grid);

    DistMatrix<T,MC,MR> CTL(grid), CTR(grid),
                        CBL(grid), CBR(grid);

    DistMatrix<T,MC,MR> DTL(grid), DBR(grid);

    const unsigned half = C.Height()/2;

    blas::Scal( beta, C );

    LockedPartitionRight( A1, A1L, A1R, half );
    LockedPartitionRight( A2, A2L, A2R, half );

    LockedPartitionDown
    ( B1, B1T,
          B1B, half );

    LockedPartitionDown
    ( B2, B2T,
          B2B, half );

    PartitionDownDiagonal
    ( C, CTL, CTR,
         CBL, CBR, half );

    DTL.AlignWith( CTL );
    DBR.AlignWith( CBR );
    DTL.ResizeTo( CTL.Height(), CTL.Width() );
    DBR.ResizeTo( CBR.Height(), CBR.Width() );
    //------------------------------------------------------------------------//
    if( shape == Lower )
    {
        blas::Gemm
        ( orientationOfA1, orientationOfB2,
          alpha, A1R.LockedLocalMatrix(),
                 B2T.LockedLocalMatrix(),
          (T)1,  CBL.LocalMatrix() );
        blas::Gemm
        ( orientationOfA2, orientationOfB1,
          alpha, A2R.LockedLocalMatrix(),
                 B1T.LockedLocalMatrix(),
          (T)1,  CBL.LocalMatrix() );
    }
    else
    {
        blas::Gemm
        ( orientationOfA1, orientationOfB2,
          alpha, A1L.LockedLocalMatrix(),
                 B2B.LockedLocalMatrix(),
          (T)1,  CTR.LocalMatrix() );
        blas::Gemm
        ( orientationOfA2, orientationOfB1,
          alpha, A2L.LockedLocalMatrix(),
                 B1B.LockedLocalMatrix(),
          (T)1,  CTR.LocalMatrix() );
    }

    blas::Gemm
    ( orientationOfA1, orientationOfB2,
      alpha, A1L.LockedLocalMatrix(),
             B2T.LockedLocalMatrix(),
      (T)0,  DTL.LocalMatrix() );
    blas::Gemm
    ( orientationOfA2, orientationOfB1,
      alpha, A2L.LockedLocalMatrix(),
             B1T.LockedLocalMatrix(),
      (T)1,  DTL.LocalMatrix() );
    DTL.MakeTrapezoidal( Left, shape );
    blas::Axpy( (T)1, DTL, CTL );

    blas::Gemm
    ( orientationOfA1, orientationOfB2,
      alpha, A1R.LockedLocalMatrix(),
             B2B.LockedLocalMatrix(),
      (T)0,  DBR.LocalMatrix() );
    blas::Gemm
    ( orientationOfA2, orientationOfB1,
      alpha, A2R.LockedLocalMatrix(),
             B1B.LockedLocalMatrix(),
      (T)1,  DBR.LocalMatrix() );
    DBR.MakeTrapezoidal( Left, shape );
    blas::Axpy( (T)1, DBR, CBR );
    //------------------------------------------------------------------------//
#ifndef RELEASE
    PopCallStack();
#endif
}

} // anonymous namespace

template<typename T>
void
elemental::blas::internal::TriangularRank2K
( Shape shape,
  T alpha, const DistMatrix<T,MC,  Star>& A1,
           const DistMatrix<T,MC,  Star>& A2,
           const DistMatrix<T,Star,MR  >& B1,
           const DistMatrix<T,Star,MR  >& B2,
  T beta,        DistMatrix<T,MC,  MR  >& C )
{
#ifndef RELEASE
    PushCallStack("blas::internal::TriangularRank2K");
    CheckInput( A1, A2, B1, B2, C );
#endif
    const Grid& grid = C.GetGrid();

    if( C.Height() < 2*grid.Width()*Blocksize() )
    {
        TriangularRank2KKernel
        ( shape, alpha, A1, A2, B1, B2, beta, C );
    }
    else
    {
        // Split C in four roughly equal pieces, perform a large gemm on corner
        // and recurse on CTL and CBR.

        DistMatrix<T,MC,Star> A1T(grid),  A2T(grid),
                              A1B(grid),  A2B(grid);

        DistMatrix<T,Star,MR> B1L(grid), B1R(grid),
                              B2L(grid), B2R(grid);

        DistMatrix<T,MC,MR> CTL(grid), CTR(grid),
                            CBL(grid), CBR(grid);

        const unsigned half = C.Height() / 2;

        LockedPartitionDown
        ( A1, A1T,
              A1B, half );
        LockedPartitionDown
        ( A2, A2T,
              A2B, half );

        LockedPartitionRight( B1, B1L, B1R, half );
        LockedPartitionRight( B2, B2L, B2R, half );

        PartitionDownDiagonal
        ( C, CTL, CTR,
             CBL, CBR, half );

        if( shape == Lower )
        { 
            blas::Gemm
            ( Normal, Normal, 
              alpha, A1B.LockedLocalMatrix(),
                     B2L.LockedLocalMatrix(),
              beta,  CBL.LocalMatrix() );
            blas::Gemm
            ( Normal, Normal, 
              alpha, A2B.LockedLocalMatrix(),
                     B1L.LockedLocalMatrix(),
              (T)1,  CBL.LocalMatrix() );
        }
        else
        {
            blas::Gemm
            ( Normal, Normal,
              alpha, A1T.LockedLocalMatrix(),
                     B2R.LockedLocalMatrix(),
              beta,  CTR.LocalMatrix() );
            blas::Gemm
            ( Normal, Normal,
              alpha, A2T.LockedLocalMatrix(),
                     B1R.LockedLocalMatrix(),
              (T)1,  CTR.LocalMatrix() );
        }

        // Recurse
        blas::internal::TriangularRank2K
        ( shape, alpha, A1T, A2T, B1L, B2L, beta, CTL );

        blas::internal::TriangularRank2K
        ( shape, alpha, A1B, A2B, B1R, B2R, beta, CBR );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::blas::internal::TriangularRank2K
( Shape shape,
  Orientation orientationOfB1,
  T alpha, const DistMatrix<T,MC,  Star>& A1,
           const DistMatrix<T,MC,  Star>& A2,
           const DistMatrix<T,MR,  Star>& B1,
           const DistMatrix<T,Star,MR  >& B2,
  T beta,        DistMatrix<T,MC,  MR  >& C  )
{
#ifndef RELEASE
    PushCallStack("blas::internal::TriangularRank2K");
    CheckInput( orientationOfB1, A1, A2, B1, B2, C );
#endif
    const Grid& grid = C.GetGrid();

    if( C.Height() < 2*grid.Width()*Blocksize() )
    {
        TriangularRank2KKernel
        ( shape, orientationOfB1, alpha, A1, A2, B1, B2, beta, C );
    }
    else
    {
        // Split C in four roughly equal pieces, perform a large gemm on corner
        // and recurse on CTL and CBR.

        DistMatrix<T,MC,Star> A1T(grid),  A2T(grid),
                              A1B(grid),  A2B(grid);

        DistMatrix<T,MR,Star> B1T(grid), 
                              B1B(grid);

        DistMatrix<T,Star,MR> B2L(grid), B2R(grid);

        DistMatrix<T,MC,MR> CTL(grid), CTR(grid),
                            CBL(grid), CBR(grid);

        const unsigned half = C.Height() / 2;

        LockedPartitionDown
        ( A1, A1T,
              A1B, half );
        LockedPartitionDown
        ( A2, A2T,
              A2B, half );

        LockedPartitionDown
        ( B1, B1T,
              B1B, half );
        LockedPartitionRight( B2, B2L, B2R, half );

        PartitionDownDiagonal
        ( C, CTL, CTR,
             CBL, CBR, half );

        if( shape == Lower )
        { 
            blas::Gemm
            ( Normal, Normal, 
              alpha, A1B.LockedLocalMatrix(),
                     B2L.LockedLocalMatrix(),
              beta,  CBL.LocalMatrix() );
            blas::Gemm
            ( Normal, orientationOfB1, 
              alpha, A2B.LockedLocalMatrix(),
                     B1T.LockedLocalMatrix(),
              (T)1,  CBL.LocalMatrix() );
        }
        else
        {
            blas::Gemm
            ( Normal, Normal,
              alpha, A1T.LockedLocalMatrix(),
                     B2R.LockedLocalMatrix(),
              beta,  CTR.LocalMatrix() );
            blas::Gemm
            ( Normal, orientationOfB1,
              alpha, A2T.LockedLocalMatrix(),
                     B1B.LockedLocalMatrix(),
              (T)1,  CTR.LocalMatrix() );
        }

        // Recurse
        blas::internal::TriangularRank2K
        ( shape, orientationOfB1, alpha, A1T, A2T, B1T, B2L, beta, CTL );

        blas::internal::TriangularRank2K
        ( shape, orientationOfB1, alpha, A1B, A2B, B1B, B2R, beta, CBR );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::blas::internal::TriangularRank2K
( Shape shape,
  Orientation orientationOfA1,
  Orientation orientationOfA2,
  T alpha, const DistMatrix<T,Star,MC>& A1,
           const DistMatrix<T,Star,MC>& A2,
           const DistMatrix<T,Star,MR>& B1,
           const DistMatrix<T,Star,MR>& B2,
  T beta,        DistMatrix<T,MC,  MR>& C  )
{
#ifndef RELEASE
    PushCallStack("blas::internal::TriangularRank2K");
    CheckInput
    ( orientationOfA1, orientationOfA2, A1, A2, B1, B2, C );
#endif
    const Grid& grid = C.GetGrid();

    if( C.Height() < 2*grid.Width()*Blocksize() )
    {
        TriangularRank2KKernel
        ( shape, orientationOfA1, orientationOfA2, 
          alpha, A1, A2, B1, B2, beta, C );
    }
    else
    {
        // Split C in four roughly equal pieces, perform a large gemm on corner
        // and recurse on CTL and CBR.

        DistMatrix<T,Star,MC> A1L(grid), A1R(grid),
                              A2L(grid), A2R(grid);

        DistMatrix<T,Star,MR> B1L(grid), B1R(grid),
                              B2L(grid), B2R(grid);

        DistMatrix<T,MC,MR> CTL(grid), CTR(grid),
                            CBL(grid), CBR(grid);

        const unsigned half = C.Height() / 2;

        LockedPartitionRight( A1, A1L, A1R, half );
        LockedPartitionRight( B1, B1L, B1R, half );

        LockedPartitionRight( B1, B1L, B1R, half );
        LockedPartitionRight( B2, B2L, B2R, half );

        PartitionDownDiagonal
        ( C, CTL, CTR,
             CBL, CBR, half );

        if( shape == Lower )
        { 
            blas::Gemm
            ( orientationOfA1, Normal, 
              alpha, A1R.LockedLocalMatrix(),
                     B2L.LockedLocalMatrix(),
              beta,  CBL.LocalMatrix() );
            blas::Gemm
            ( orientationOfA2, Normal, 
              alpha, A2R.LockedLocalMatrix(),
                     B1L.LockedLocalMatrix(),
              (T)1,  CBL.LocalMatrix() );
        }
        else
        {
            blas::Gemm
            ( orientationOfA1, Normal,
              alpha, A1L.LockedLocalMatrix(),
                     B2R.LockedLocalMatrix(),
              beta,  CTR.LocalMatrix() );
            blas::Gemm
            ( orientationOfA2, Normal,
              alpha, A2L.LockedLocalMatrix(),
                     B1R.LockedLocalMatrix(),
              (T)1,  CTR.LocalMatrix() );
        }

        // Recurse
        blas::internal::TriangularRank2K
        ( shape, orientationOfA1, orientationOfA2,
          alpha, A1L, A2L, B1L, B2L, beta, CTL );

        blas::internal::TriangularRank2K
        ( shape, orientationOfA1, orientationOfA2, 
          alpha, A1R, A2R, B1R, B2R, beta, CBR );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::blas::internal::TriangularRank2K
( Shape shape,
  Orientation orientationOfA1,
  Orientation orientationOfA2,
  Orientation orientationOfB1,
  Orientation orientationOfB2,
  T alpha, const DistMatrix<T,Star,MC  >& A1,
           const DistMatrix<T,Star,MC  >& A2,
           const DistMatrix<T,MR,  Star>& B1,
           const DistMatrix<T,MR,  Star>& B2,
  T beta,        DistMatrix<T,MC,  MR  >& C  )
{
#ifndef RELEASE
    PushCallStack("blas::internal::TriangularRank2K");
    CheckInput
    ( orientationOfA1, orientationOfA2, orientationOfB1, orientationOfB2, 
      A1, A2, B1, B2, C );
#endif
    const Grid& grid = C.GetGrid();

    if( C.Height() < 2*grid.Width()*Blocksize() )
    {
        TriangularRank2KKernel
        ( shape, 
          orientationOfA1, orientationOfA2, 
          orientationOfB1, orientationOfB2, 
          alpha, A1, A2, B1, B2, beta, C );
    }
    else
    {
        // Split C in four roughly equal pieces, perform a large gemm on corner
        // and recurse on CTL and CBR.

        DistMatrix<T,Star,MC> A1L(grid), A1R(grid),
                              A2L(grid), A2R(grid);

        DistMatrix<T,MR,Star> B1T(grid),  B2T(grid),
                              B1B(grid),  B2B(grid);

        DistMatrix<T,MC,MR> CTL(grid), CTR(grid),
                            CBL(grid), CBR(grid);

        const unsigned half = C.Height() / 2;

        LockedPartitionRight( A1, A1L, A1R, half );
        LockedPartitionRight( A2, A2L, A2R, half );

        LockedPartitionDown
        ( B1, B1T,
              B1B, half );
        LockedPartitionDown
        ( B2, B2T,
              B2B, half );

        PartitionDownDiagonal
        ( C, CTL, CTR,
             CBL, CBR, half );

        if( shape == Lower )
        { 
            blas::Gemm
            ( orientationOfA1, orientationOfB2, 
              alpha, A1R.LockedLocalMatrix(),
                     B2T.LockedLocalMatrix(),
              beta,  CBL.LocalMatrix() );
            blas::Gemm
            ( orientationOfA2, orientationOfB1, 
              alpha, A2R.LockedLocalMatrix(),
                     B1T.LockedLocalMatrix(),
              (T)1,  CBL.LocalMatrix() );
        }
        else
        {
            blas::Gemm
            ( orientationOfA1, orientationOfB2,
              alpha, A1L.LockedLocalMatrix(),
                     B2B.LockedLocalMatrix(),
              beta,  CTR.LocalMatrix() );
            blas::Gemm
            ( orientationOfA2, orientationOfB1,
              alpha, A2L.LockedLocalMatrix(),
                     B1B.LockedLocalMatrix(),
              (T)1,  CTR.LocalMatrix() );
        }

        // Recurse
        blas::internal::TriangularRank2K
        ( shape, 
          orientationOfA1, orientationOfA2, 
          orientationOfB1, orientationOfB2, 
          alpha, A1L, A2L, B1T, B2T, beta, CTL );

        blas::internal::TriangularRank2K
        ( shape, 
          orientationOfA1, orientationOfA2,
          orientationOfB1, orientationOfB2,
          alpha, A1R, A2R, B1B, B2B, beta, CBR );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template void
elemental::blas::internal::TriangularRank2K
( Shape shape, 
  float alpha, const DistMatrix<float,MC,  Star>& A1,
               const DistMatrix<float,MC,  Star>& A2,
               const DistMatrix<float,Star,MR  >& B1,
               const DistMatrix<float,Star,MR  >& B2,
  float beta,        DistMatrix<float,MC,  MR  >& C );

template void
elemental::blas::internal::TriangularRank2K
( Shape shape, 
  Orientation orientationOfB1,
  float alpha, const DistMatrix<float,MC,  Star>& A1,
               const DistMatrix<float,MC,  Star>& A2,
               const DistMatrix<float,MR,  Star>& B1,
               const DistMatrix<float,Star,MR  >& B2,
  float beta,        DistMatrix<float,MC,  MR  >& C );


template void
elemental::blas::internal::TriangularRank2K
( Shape shape, 
  Orientation orientationOfA1,
  Orientation orientationOfA2,
  float alpha, const DistMatrix<float,Star,MC>& A1,
               const DistMatrix<float,Star,MC>& A2,
               const DistMatrix<float,Star,MR>& B1,
               const DistMatrix<float,Star,MR>& B2,
  float beta,        DistMatrix<float,MC,  MR>& C );

template void
elemental::blas::internal::TriangularRank2K
( Shape shape, 
  Orientation orientationOfA1,
  Orientation orientationOfA2,
  Orientation orientationOfB1,
  Orientation orientationOfB2,
  float alpha, const DistMatrix<float,Star,MC  >& A1,
               const DistMatrix<float,Star,MC  >& A2,
               const DistMatrix<float,MR,  Star>& B1,
               const DistMatrix<float,MR,  Star>& B2,
  float beta,        DistMatrix<float,MC,  MR  >& C );

template void
elemental::blas::internal::TriangularRank2K
( Shape shape,
  double alpha, const DistMatrix<double,MC,  Star>& A1,
                const DistMatrix<double,MC,  Star>& A2,
                const DistMatrix<double,Star,MR  >& B1,
                const DistMatrix<double,Star,MR  >& B2,
  double beta,        DistMatrix<double,MC,  MR  >& C );

template void
elemental::blas::internal::TriangularRank2K
( Shape shape,
  Orientation orientationOfB1,
  double alpha, const DistMatrix<double,MC,  Star>& A1,
                const DistMatrix<double,MC,  Star>& A2,
                const DistMatrix<double,MR,  Star>& B1,
                const DistMatrix<double,Star,MR  >& B2,
  double beta,        DistMatrix<double,MC,  MR  >& C );

template void
elemental::blas::internal::TriangularRank2K
( Shape shape,
  Orientation orientationOfA1,
  Orientation orientationOfA2,
  double alpha, const DistMatrix<double,Star,MC>& A1,
                const DistMatrix<double,Star,MC>& A2,
                const DistMatrix<double,Star,MR>& B1,
                const DistMatrix<double,Star,MR>& B2,
  double beta,        DistMatrix<double,MC,  MR>& C );

template void
elemental::blas::internal::TriangularRank2K
( Shape shape,
  Orientation orientationOfA1,
  Orientation orientationOfA2,
  Orientation orientationOfB1,
  Orientation orientationOfB2,
  double alpha, const DistMatrix<double,Star,MC  >& A1,
                const DistMatrix<double,Star,MC  >& A2,
                const DistMatrix<double,MR,  Star>& B1,
                const DistMatrix<double,MR,  Star>& B2,
  double beta,        DistMatrix<double,MC,  MR  >& C );

#ifndef WITHOUT_COMPLEX
template void
elemental::blas::internal::TriangularRank2K
( Shape shape,
  scomplex alpha, const DistMatrix<scomplex,MC,  Star>& A1,
                  const DistMatrix<scomplex,MC,  Star>& A2,
                  const DistMatrix<scomplex,Star,MR  >& B1,
                  const DistMatrix<scomplex,Star,MR  >& B2,
  scomplex beta,        DistMatrix<scomplex,MC,  MR  >& C );

template void
elemental::blas::internal::TriangularRank2K
( Shape shape,
  Orientation orientationOfB1,
  scomplex alpha, const DistMatrix<scomplex,MC,  Star>& A1,
                  const DistMatrix<scomplex,MC,  Star>& A2,
                  const DistMatrix<scomplex,MR,  Star>& B1,
                  const DistMatrix<scomplex,Star,MR  >& B2,
  scomplex beta,        DistMatrix<scomplex,MC,  MR  >& C );

template void
elemental::blas::internal::TriangularRank2K
( Shape shape,
  Orientation orientationOfA1,
  Orientation orientationOfA2,
  scomplex alpha, const DistMatrix<scomplex,Star,MC>& A1,
                  const DistMatrix<scomplex,Star,MC>& A2,
                  const DistMatrix<scomplex,Star,MR>& B1,
                  const DistMatrix<scomplex,Star,MR>& B2,
  scomplex beta,        DistMatrix<scomplex,MC,  MR>& C );

template void
elemental::blas::internal::TriangularRank2K
( Shape shape,
  Orientation orientationOfA1,
  Orientation orientationOfA2,
  Orientation orientationOfB1,
  Orientation orientationOfB2,
  scomplex alpha, const DistMatrix<scomplex,Star,MC  >& A1,
                  const DistMatrix<scomplex,Star,MC  >& A2,
                  const DistMatrix<scomplex,MR,  Star>& B1,
                  const DistMatrix<scomplex,MR,  Star>& B2,
  scomplex beta,        DistMatrix<scomplex,MC,  MR  >& C );

template void
elemental::blas::internal::TriangularRank2K
( Shape shape,
  dcomplex alpha, const DistMatrix<dcomplex,MC,  Star>& A1,
                  const DistMatrix<dcomplex,MC,  Star>& A2,
                  const DistMatrix<dcomplex,Star,MR  >& B1,
                  const DistMatrix<dcomplex,Star,MR  >& B2,
  dcomplex beta,        DistMatrix<dcomplex,MC,  MR  >& C );

template void
elemental::blas::internal::TriangularRank2K
( Shape shape,
  Orientation orientationOfB1,
  dcomplex alpha, const DistMatrix<dcomplex,MC,  Star>& A1,
                  const DistMatrix<dcomplex,MC,  Star>& A2,
                  const DistMatrix<dcomplex,MR,  Star>& B1,
                  const DistMatrix<dcomplex,Star,MR  >& B2,
  dcomplex beta,        DistMatrix<dcomplex,MC,  MR  >& C );

template void
elemental::blas::internal::TriangularRank2K
( Shape shape,
  Orientation orientationOfA1,
  Orientation orientationOfA2,
  dcomplex alpha, const DistMatrix<dcomplex,Star,MC>& A1,
                  const DistMatrix<dcomplex,Star,MC>& A2,
                  const DistMatrix<dcomplex,Star,MR>& B1,
                  const DistMatrix<dcomplex,Star,MR>& B2,
  dcomplex beta,        DistMatrix<dcomplex,MC,  MR>& C );

template void
elemental::blas::internal::TriangularRank2K
( Shape shape,
  Orientation orientationOfA1,
  Orientation orientationOfA2,
  Orientation orientationOfB1,
  Orientation orientationOfB2,
  dcomplex alpha, const DistMatrix<dcomplex,Star,MC  >& A1,
                  const DistMatrix<dcomplex,Star,MC  >& A2,
                  const DistMatrix<dcomplex,MR,  Star>& B1,
                  const DistMatrix<dcomplex,MR,  Star>& B2,
  dcomplex beta,        DistMatrix<dcomplex,MC,  MR  >& C );
#endif

