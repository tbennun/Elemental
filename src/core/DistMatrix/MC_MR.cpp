/*
   This file is part of elemental, a library for distributed-memory dense 
   linear algebra.

   Copyright (C) 2009-2010 Jack Poulson <jack.poulson@gmail.com>

   This program is released under the terms of the license contained in the 
   file LICENSE.
*/
#include "elemental/dist_matrix.hpp"
using namespace std;
using namespace elemental;
using namespace elemental::utilities;
using namespace elemental::wrappers::mpi;

//----------------------------------------------------------------------------//
// DistMatrixBase                                                             //
//----------------------------------------------------------------------------//

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::Print( const string& s ) const
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::Print");
#endif
    const Grid& grid = this->GetGrid();
    const int r = grid.Height();
    const int c = grid.Width();

    if( grid.VCRank() == 0 && s != "" )
        cout << s << endl;

    const int height = this->Height();
    const int width  = this->Width();
    const int localHeight = this->LocalHeight();
    const int localWidth  = this->LocalWidth();
    const int colShift = this->ColShift();
    const int rowShift = this->RowShift();

    if( height == 0 || width == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    // Fill the send buffer: zero it then place our entries into their 
    // appropriate locations
    T* sendBuf = new T[height*width];
    for( int i=0; i<height*width; ++i )
        sendBuf[i] = (T)0;
    for( int i=0; i<localHeight; ++i )
        for( int j=0; j<localWidth; ++j )
            sendBuf[colShift+i*r + (rowShift+j*c)*height] = 
                this->LocalEntry(i,j);

    // If we are the root, fill the receive buffer
    T* recvBuf = 0;
    if( grid.VCRank() == 0 )
    {
        recvBuf = new T[height*width];
        for( int i=0; i<height*width; ++i )
            recvBuf[i] = (T)0;
    }

    // Sum the contributions and send to the root
    Reduce( sendBuf, recvBuf, height*width, MPI_SUM, 0, grid.VCComm() );
    delete[] sendBuf;

    if( grid.VCRank() == 0 )
    {
        // Print the data
        for( int i=0; i<height; ++i )
        {
            for( int j=0; j<width; ++j )
                cout << recvBuf[i+j*height] << " ";
            cout << endl;
        }
        cout << endl;
        delete recvBuf;
    }
    Barrier( grid.VCComm() );

#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignWith
( const DistMatrixBase<T,MC,MR>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignWith([MC,MR])");
    this->AssertFreeColAlignment();
    this->AssertFreeRowAlignment();
    this->AssertSameGrid( A );
#endif
    this->_colAlignment = A.ColAlignment();
    this->_rowAlignment = A.RowAlignment();
    this->_colShift     = A.ColShift();
    this->_rowShift     = A.RowShift();
    this->_constrainedColAlignment = true;
    this->_constrainedRowAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignWith
( const DistMatrixBase<T,MC,Star>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignWith([MC,* ])");
    this->AssertFreeColAlignment();
    this->AssertSameGrid( A );
#endif
    this->_colAlignment = A.ColAlignment();
    this->_colShift     = A.ColShift();
    this->_constrainedColAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignWith
( const DistMatrixBase<T,Star,MR>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignWith([* ,MR])");
    this->AssertFreeRowAlignment();
    this->AssertSameGrid( A );
#endif
    this->_rowAlignment = A.RowAlignment();
    this->_rowShift     = A.RowShift();
    this->_constrainedRowAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignWith
( const DistMatrixBase<T,MR,MC>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignWith([MR,MC])");
    this->AssertFreeColAlignment();
    this->AssertFreeRowAlignment();
    this->AssertSameGrid( A );
#endif
    this->_colAlignment = A.RowAlignment();
    this->_rowAlignment = A.ColAlignment();
    this->_colShift     = A.RowShift();
    this->_rowShift     = A.ColShift();
    this->_constrainedColAlignment = true;
    this->_constrainedRowAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignWith
( const DistMatrixBase<T,MR,Star>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignWith([MR,* ])");
    this->AssertFreeRowAlignment();
    this->AssertSameGrid( A );
#endif
    this->_rowAlignment = A.ColAlignment();
    this->_rowShift     = A.ColShift();
    this->_constrainedRowAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignWith
( const DistMatrixBase<T,Star,MC>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignWith([* ,MC])");
    this->AssertFreeColAlignment();
    this->AssertSameGrid( A );
#endif
    this->_colAlignment = A.RowAlignment();
    this->_colShift     = A.RowShift();
    this->_constrainedColAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignWith
( const DistMatrixBase<T,VC,Star>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignWith([VC,* ])");
    this->AssertFreeColAlignment();
    this->AssertSameGrid( A );
#endif
    const Grid& grid = this->GetGrid();
    this->_colAlignment = A.ColAlignment() % grid.Height();
    this->_colShift = 
        Shift( grid.MCRank(), this->ColAlignment(), grid.Height() );
    this->_constrainedColAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignWith
( const DistMatrixBase<T,Star,VC>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignWith([* ,VC])");
    this->AssertFreeColAlignment();
    this->AssertSameGrid( A );
#endif
    const Grid& grid = this->GetGrid();
    this->_colAlignment = A.RowAlignment() % grid.Height();
    this->_colShift = 
        Shift( grid.MCRank(), this->ColAlignment(), grid.Height() );
    this->_constrainedColAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignWith
( const DistMatrixBase<T,VR,Star>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignWith([VR,* ])");
    this->AssertFreeRowAlignment();
    this->AssertSameGrid( A );
#endif
    const Grid& grid = this->GetGrid();
    this->_rowAlignment = A.ColAlignment() % grid.Width();
    this->_rowShift = 
        Shift( grid.MRRank(), this->RowAlignment(), grid.Width() );
    this->_constrainedRowAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignWith
( const DistMatrixBase<T,Star,VR>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignWith([* ,VR])");
    this->AssertFreeRowAlignment();
    this->AssertSameGrid( A );
#endif
    const Grid& grid = this->GetGrid();
    this->_rowAlignment = A.RowAlignment() % grid.Width();
    this->_rowShift = 
        Shift( grid.MRRank(), this->RowAlignment(), grid.Width() );
    this->_constrainedRowAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignColsWith
( const DistMatrixBase<T,MC,MR>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignColsWith([MC,MR])");
    this->AssertFreeColAlignment();
    this->AssertSameGrid( A );
#endif
    this->_colAlignment = A.ColAlignment();
    this->_colShift     = A.ColShift();
    this->_constrainedColAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignColsWith
( const DistMatrixBase<T,MC,Star>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignColsWith([MC,* ])");
    this->AssertFreeColAlignment();
    this->AssertSameGrid( A );
#endif
    this->_colAlignment = A.ColAlignment();
    this->_colShift     = A.ColShift();
    this->_constrainedColAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignColsWith
( const DistMatrixBase<T,MR,MC>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignColsWith([MR,MC])");
    this->AssertFreeColAlignment();
    this->AssertSameGrid( A );
#endif
    this->_colAlignment = A.RowAlignment();
    this->_colShift     = A.RowShift();
    this->_constrainedColAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignColsWith
( const DistMatrixBase<T,Star,MC>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignColsWith([* ,MC])");
    this->AssertFreeColAlignment();
    this->AssertSameGrid( A );
#endif
    this->_colAlignment = A.RowAlignment();
    this->_colShift     = A.RowShift();
    this->_constrainedColAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignColsWith
( const DistMatrixBase<T,VC,Star>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignColsWith([VC,* ])");
    this->AssertFreeColAlignment();
    this->AssertSameGrid( A );
#endif
    const Grid& grid = this->GetGrid();
    this->_colAlignment = A.ColAlignment() % grid.Height();
    this->_colShift = 
        Shift( grid.MCRank(), this->ColAlignment(), grid.Height() );
    this->_constrainedColAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignColsWith
( const DistMatrixBase<T,Star,VC>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignColsWith([* ,VC])");
    this->AssertFreeColAlignment();
    this->AssertSameGrid( A );
#endif
    const Grid& grid = this->GetGrid();
    this->_colAlignment = A.RowAlignment() % grid.Height();
    this->_colShift = 
        Shift( grid.MCRank(), this->ColAlignment(), grid.Height() );
    this->_constrainedColAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignRowsWith
( const DistMatrixBase<T,MC,MR>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignRowsWith([MC,MR])");
    this->AssertFreeRowAlignment();
    this->AssertSameGrid( A );
#endif
    this->_rowAlignment = A.RowAlignment();
    this->_rowShift     = A.RowShift();
    this->_constrainedRowAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignRowsWith
( const DistMatrixBase<T,Star,MR>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignRowsWith([* ,MR])");
    this->AssertFreeRowAlignment();
    this->AssertSameGrid( A );
#endif
    this->_rowAlignment = A.RowAlignment();
    this->_rowShift     = A.RowShift();
    this->_constrainedRowAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignRowsWith
( const DistMatrixBase<T,MR,MC>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignRowsWith([MR,MC])");
    this->AssertFreeRowAlignment();
    this->AssertSameGrid( A );
#endif
    this->_rowAlignment = A.ColAlignment();
    this->_rowShift     = A.ColShift();
    this->_constrainedRowAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignRowsWith
( const DistMatrixBase<T,MR,Star>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignRowsWith([MR,* ])");
    this->AssertFreeRowAlignment();
    this->AssertSameGrid( A );
#endif
    this->_rowAlignment = A.ColAlignment();
    this->_rowShift     = A.ColShift();
    this->_constrainedRowAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignRowsWith
( const DistMatrixBase<T,VR,Star>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignRowsWith([VR,* ])");
    this->AssertFreeRowAlignment();
    this->AssertSameGrid( A );
#endif
    const Grid& grid = this->GetGrid();
    this->_rowAlignment = A.ColAlignment() % grid.Width();
    this->_rowShift = 
        Shift( grid.MRRank(), this->RowAlignment(), grid.Width() );
    this->_constrainedRowAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::AlignRowsWith
( const DistMatrixBase<T,Star,VR>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::AlignRowsWith([* ,VR])");
    this->AssertFreeRowAlignment();
    this->AssertSameGrid( A );
#endif
    const Grid& grid = this->GetGrid();
    this->_rowAlignment = A.RowAlignment() % grid.Width();
    this->_rowShift = 
        Shift( grid.MRRank(), this->RowAlignment(), grid.Width() );
    this->_constrainedRowAlignment = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::View
( DistMatrixBase<T,MC,MR>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::View");
    this->AssertFreeColAlignment();
    this->AssertFreeRowAlignment();
    this->AssertNotStoringData();
    this->AssertSameGrid( A );
#endif
    this->_height = A.Height();
    this->_width  = A.Width();
    this->_colAlignment = A.ColAlignment();
    this->_rowAlignment = A.RowAlignment();
    this->_colShift     = A.ColShift();
    this->_rowShift     = A.RowShift();
    this->_localMatrix.View( A.LocalMatrix() );
    this->_viewing = true;
    this->_lockedView = false;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::LockedView
( const DistMatrixBase<T,MC,MR>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::LockedView");
    this->AssertFreeColAlignment();
    this->AssertFreeRowAlignment();
    this->AssertNotStoringData();
    this->AssertSameGrid( A );
#endif
    this->_height = A.Height();
    this->_width  = A.Width();
    this->_colAlignment = A.ColAlignment();
    this->_rowAlignment = A.RowAlignment();
    this->_colShift     = A.ColShift();
    this->_rowShift     = A.RowShift();
    this->_localMatrix.LockedView( A.LockedLocalMatrix() );
    this->_viewing = true;
    this->_lockedView = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::View
( DistMatrixBase<T,MC,MR>& A, 
  int i, int j, int height, int width )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::View");
    this->AssertFreeColAlignment();
    this->AssertFreeRowAlignment();
    this->AssertNotStoringData();
    this->AssertSameGrid( A );
    this->AssertValidSubmatrix( A, i, j, height, width );
#endif
    this->_height = height;
    this->_width  = width;
    {
        const Grid& grid = this->GetGrid();
        const int r   = grid.Height();
        const int c   = grid.Width();
        const int row = grid.MCRank();
        const int col = grid.MRRank();

        this->_colAlignment = (A.ColAlignment()+i) % r;
        this->_rowAlignment = (A.RowAlignment()+j) % c;
  
        this->_colShift = Shift( row, this->ColAlignment(), r );
        this->_rowShift = Shift( col, this->RowAlignment(), c );

        const int localHeightBehind = LocalLength(i,A.ColShift(),r);
        const int localWidthBehind  = LocalLength(j,A.RowShift(),c);

        const int localHeight = LocalLength( height, this->ColShift(), r );
        const int localWidth  = LocalLength( width,  this->RowShift(), c );

        this->_localMatrix.View
        ( A.LocalMatrix(), localHeightBehind, localWidthBehind,
                           localHeight,       localWidth );
    }
    this->_viewing = true;
    this->_lockedView = false;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::LockedView
( const DistMatrixBase<T,MC,MR>& A, 
  int i, int j, int height, int width )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::LockedView");
    this->AssertFreeColAlignment();
    this->AssertFreeRowAlignment();
    this->AssertNotStoringData();
    this->AssertSameGrid( A );
    this->AssertValidSubmatrix( A, i, j, height, width );
#endif
    this->_height = height;
    this->_width  = width;
    {
        const Grid& grid = this->GetGrid();
        const int r   = grid.Height();
        const int c   = grid.Width();
        const int row = grid.MCRank();
        const int col = grid.MRRank();

        this->_colAlignment = (A.ColAlignment()+i) % r;
        this->_rowAlignment = (A.RowAlignment()+j) % c;
  
        this->_colShift = Shift( row, this->ColAlignment(), r );
        this->_rowShift = Shift( col, this->RowAlignment(), c );

        const int localHeightBehind = LocalLength(i,A.ColShift(),r);
        const int localWidthBehind  = LocalLength(j,A.RowShift(),c);

        const int localHeight = LocalLength( height, this->ColShift(), r );
        const int localWidth  = LocalLength( width,  this->RowShift(), c );

        this->_localMatrix.LockedView
        ( A.LockedLocalMatrix(), localHeightBehind, localWidthBehind,
                                 localHeight,       localWidth );
    }
    this->_viewing = true;
    this->_lockedView = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::View1x2
( DistMatrixBase<T,MC,MR>& AL, 
  DistMatrixBase<T,MC,MR>& AR )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::View1x2");
    this->AssertFreeColAlignment();
    this->AssertFreeRowAlignment();
    this->AssertNotStoringData();
    this->AssertSameGrid( AL );
    this->AssertSameGrid( AR );
    this->AssertConforming1x2( AL, AR );
#endif
    this->_height = AL.Height();
    this->_width  = AL.Width() + AR.Width();
    this->_colAlignment = AL.ColAlignment();
    this->_rowAlignment = AL.RowAlignment();
    this->_colShift     = AL.ColShift();
    this->_rowShift     = AL.RowShift();
    this->_localMatrix.View1x2( AL.LocalMatrix(), AR.LocalMatrix() );
    this->_viewing = true;
    this->_lockedView = false;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::LockedView1x2
( const DistMatrixBase<T,MC,MR>& AL, 
  const DistMatrixBase<T,MC,MR>& AR )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::LockedView1x2");
    this->AssertFreeColAlignment();
    this->AssertFreeRowAlignment();
    this->AssertNotStoringData();
    this->AssertSameGrid( AL );
    this->AssertSameGrid( AR );
    this->AssertConforming1x2( AL, AR );
#endif
    this->_height = AL.Height();
    this->_width  = AL.Width() + AR.Width();
    this->_colAlignment = AL.ColAlignment();
    this->_rowAlignment = AL.RowAlignment();
    this->_colShift     = AL.ColShift();
    this->_rowShift     = AL.RowShift();
    this->_localMatrix.LockedView1x2
    ( AL.LockedLocalMatrix(), 
      AR.LockedLocalMatrix() );
    this->_viewing = true;
    this->_lockedView = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::View2x1
( DistMatrixBase<T,MC,MR>& AT,
  DistMatrixBase<T,MC,MR>& AB )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::View2x1");
    this->AssertFreeColAlignment();
    this->AssertFreeRowAlignment();
    this->AssertNotStoringData();
    this->AssertSameGrid( AT );
    this->AssertSameGrid( AB );
    this->AssertConforming2x1( AT, AB );
#endif
    this->_height = AT.Height() + AB.Height();
    this->_width  = AT.Width();
    this->_colAlignment = AT.ColAlignment();
    this->_rowAlignment = AT.RowAlignment();
    this->_colShift     = AT.ColShift();
    this->_rowShift     = AT.RowShift();
    this->_localMatrix.View2x1
    ( AT.LocalMatrix(), 
      AB.LocalMatrix() );
    this->_viewing = true;
    this->_lockedView = false;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::LockedView2x1
( const DistMatrixBase<T,MC,MR>& AT,
  const DistMatrixBase<T,MC,MR>& AB )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::LockedView2x1");
    this->AssertFreeColAlignment();
    this->AssertFreeRowAlignment();
    this->AssertNotStoringData();
    this->AssertSameGrid( AT );
    this->AssertSameGrid( AB );
    this->AssertConforming2x1( AT, AB );
#endif
    this->_height = AT.Height() + AB.Height();
    this->_width  = AT.Width();
    this->_colAlignment = AT.ColAlignment();
    this->_rowAlignment = AT.RowAlignment();
    this->_colShift     = AT.ColShift();
    this->_rowShift     = AT.RowShift();
    this->_localMatrix.LockedView2x1
    ( AT.LockedLocalMatrix(), 
      AB.LockedLocalMatrix() );
    this->_viewing = true;
    this->_lockedView = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::View2x2
( DistMatrixBase<T,MC,MR>& ATL, 
  DistMatrixBase<T,MC,MR>& ATR,
  DistMatrixBase<T,MC,MR>& ABL,
  DistMatrixBase<T,MC,MR>& ABR )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::View2x2");
    this->AssertFreeColAlignment();
    this->AssertFreeRowAlignment();
    this->AssertNotStoringData();
    this->AssertSameGrid( ATL );
    this->AssertSameGrid( ATR );
    this->AssertSameGrid( ABL );
    this->AssertSameGrid( ABR );
    this->AssertConforming2x2( ATL, ATR, ABL, ABR );
#endif
    this->_height = ATL.Height() + ABL.Height();
    this->_width  = ATL.Width() + ATR.Width();
    this->_colAlignment = ATL.ColAlignment();
    this->_rowAlignment = ATL.RowAlignment();
    this->_colShift     = ATL.ColShift();
    this->_rowShift     = ATL.RowShift();
    this->_localMatrix.View2x2
    ( ATL.LocalMatrix(), ATR.LocalMatrix(),
      ABL.LocalMatrix(), ABR.LocalMatrix() );
    this->_viewing = true;
    this->_lockedView = false;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::LockedView2x2
( const DistMatrixBase<T,MC,MR>& ATL, 
  const DistMatrixBase<T,MC,MR>& ATR,
  const DistMatrixBase<T,MC,MR>& ABL,
  const DistMatrixBase<T,MC,MR>& ABR )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::LockedView2x2");
    this->AssertFreeColAlignment();
    this->AssertFreeRowAlignment();
    this->AssertNotStoringData();
    this->AssertSameGrid( ATL );
    this->AssertSameGrid( ATR );
    this->AssertSameGrid( ABL );
    this->AssertSameGrid( ABR );
    this->AssertConforming2x2( ATL, ATR, ABL, ABR );
#endif
    this->_height = ATL.Height() + ABL.Height();
    this->_width  = ATL.Width() + ATR.Width();
    this->_colAlignment = ATL.ColAlignment();
    this->_rowAlignment = ATL.RowAlignment();
    this->_colShift     = ATL.ColShift();
    this->_rowShift     = ATL.RowShift();
    this->_localMatrix.LockedView2x2
    ( ATL.LockedLocalMatrix(), ATR.LockedLocalMatrix(),
      ABL.LockedLocalMatrix(), ABR.LockedLocalMatrix() );
    this->_viewing = true;
    this->_lockedView = true;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::ResizeTo
( int height, int width )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::ResizeTo");
    this->AssertNotLockedView();
#endif
    this->_height = height;
    this->_width  = width;
    this->_localMatrix.ResizeTo
    ( LocalLength(height,this->ColShift(),this->GetGrid().Height()),
      LocalLength(width, this->RowShift(),this->GetGrid().Width()) );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
T
elemental::DistMatrixBase<T,MC,MR>::Get
( int i, int j ) const
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::Get");
    this->AssertValidEntry( i, j );
#endif
    // We will determine the owner of the (i,j) entry and have him Broadcast
    // throughout the entire process grid
    const Grid& grid = this->GetGrid();
    const int ownerRow = (i + this->ColAlignment()) % grid.Height();
    const int ownerCol = (j + this->RowAlignment()) % grid.Width();
    const int ownerRank = ownerRow + ownerCol * grid.Height();

    T u;
    if( grid.VCRank() == ownerRank )
    {
        const int iLoc = (i-this->ColShift()) / grid.Height();
        const int jLoc = (j-this->RowShift()) / grid.Width();
        u = this->LocalEntry(iLoc,jLoc);
    }
    Broadcast( &u, 1, ownerRank, grid.VCComm() );

#ifndef RELEASE
    PopCallStack();
#endif
    return u;
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::Set
( int i, int j, T u )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::Set");
    this->AssertValidEntry( i, j );
#endif
    const Grid& grid = this->GetGrid();
    const int ownerRow = (i + this->ColAlignment()) % grid.Height();
    const int ownerCol = (j + this->RowAlignment()) % grid.Width();
    const int ownerRank = ownerRow + ownerCol * grid.Height();

    if( grid.VCRank() == ownerRank )
    {
        const int iLoc = (i-this->ColShift()) / grid.Height();
        const int jLoc = (j-this->RowShift()) / grid.Width();
        this->LocalEntry(iLoc,jLoc) = u;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::GetDiagonal
( DistMatrixBase<T,MD,Star>& d, int offset )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::GetDiagonal");
    this->AssertNotLockedView();
#endif
    int width = this->Width();
    int height = this->Height();
    int length;
    if( offset > 0 )
    {
        const int remainingWidth = max(width-offset,0);
        length = min(height,remainingWidth);
    }
    else
    {
        const int remainingHeight = max(height+offset,0);
        length = min(remainingHeight,width);
    }
#ifndef RELEASE
    if( d.Viewing() && length != d.Height() )
    {
        ostringstream msg;
        msg << "d is not of the same length as the diagonal:" << endl
            << "  A ~ " << this->Height() << " x " << this->Width() << endl
            << "  d ~ " << d.Height() << " x " << d.Width() << endl
            << "  A diag length: " << length << endl;
        throw logic_error( msg.str() );
    }
#endif

    if( !d.Viewing() )
    {
        if( !d.ConstrainedColAlignment() )
        {
            d.AlignWithDiag( *this, offset );
        }
        d.ResizeTo( length, 1 );
    }

    if( d.InDiagonal() )
    {
        const Grid& grid = this->GetGrid();
        const int r = grid.Height();
        const int c = grid.Width();
        const int lcm = grid.LCM();
        const int colShift = this->ColShift();
        const int rowShift = this->RowShift();
        const int diagShift = d.ColShift();

        int iStart, jStart;
        if( offset >= 0 )
        {
            iStart = diagShift;
            jStart = diagShift+offset;
        }
        else
        {
            iStart = diagShift-offset;
            jStart = diagShift;
        }

        const int iLocStart = (iStart-colShift) / r;
        const int jLocStart = (jStart-rowShift) / c;

        const int localDiagLength = d.LocalHeight();
        for( int k=0; k<localDiagLength; ++k )
            d.LocalEntry(k,0) = 
                this->LocalEntry(iLocStart+k*(lcm/r),jLocStart+k*(lcm/c));
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::GetDiagonal
( DistMatrixBase<T,Star,MD>& d, int offset )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::GetDiagonal");
    this->AssertNotLockedView();
#endif
    int height = this->Height();
    int width = this->Width();
    int length;
    if( offset > 0 )
    {
        const int remainingWidth = max(width-offset,0);
        length = min(height,remainingWidth);
    }
    else
    {
        const int remainingHeight = max(height+offset,0);
        length = min(remainingHeight,width);
    }
#ifndef RELEASE
    if( d.Viewing() && length != d.Width() )
    {
        ostringstream msg;
        msg << "d is not of the same length as the diagonal:" << endl
            << "  A ~ " << this->Height() << " x " << this->Width() << endl
            << "  d ~ " << d.Height() << " x " << d.Width() << endl
            << "  A diag length: " << length << endl;
        throw logic_error( msg.str() );
    }
#endif

    if( !d.Viewing() )
    {
        if( !d.ConstrainedRowAlignment() )
        {
            d.AlignWithDiag( *this, offset );
        }
        d.ResizeTo( 1, length );
    }

    if( d.InDiagonal() )
    {
        const Grid& grid = this->GetGrid();
        const int r = grid.Height();
        const int c = grid.Width();
        const int lcm = grid.LCM();
        const int colShift = this->ColShift();
        const int rowShift = this->RowShift();
        const int diagShift = d.RowShift();

        int iStart, jStart;
        if( offset >= 0 )
        {
            iStart = diagShift;
            jStart = diagShift+offset;
        }
        else
        {
            iStart = diagShift-offset;
            jStart = diagShift;
        }

        const int iLocStart = (iStart-colShift) / r;
        const int jLocStart = (jStart-rowShift) / c;

        const int localDiagLength = d.LocalWidth();
        for( int k=0; k<localDiagLength; ++k )
            d.LocalEntry(0,k) = 
                this->LocalEntry(iLocStart+k*(lcm/r),jLocStart+k*(lcm/c));
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::SetDiagonal
( const DistMatrixBase<T,MD,Star>& d, int offset )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SetDiagonal");
    if( d.Width() != 1 )
        throw logic_error( "d must be a column vector." );
    {
        int height = this->Height();
        int width = this->Width();
        int length;
        if( offset >= 0 )
        {
            const int remainingWidth = max(width-offset,0);
            length = min(remainingWidth,height);
        }
        else
        {
            const int remainingHeight = max(height+offset,0);
            length = min(remainingHeight,width);
        }
        if( length != d.Height() )
        {
            ostringstream msg;
            msg << "d is not of the same length as the diagonal:" << endl
                << "  A ~ " << this->Height() << " x " << this->Width() << endl
                << "  d ~ " << d.Height() << " x " << d.Width() << endl
                << "  A diag length: " << length << endl;
            throw logic_error( msg.str() );
        }
    }
#endif
    if( d.InDiagonal() )
    {
        const Grid& grid = this->GetGrid();
        const int r = grid.Height();
        const int c = grid.Width();
        const int lcm = grid.LCM();
        const int colShift = this->ColShift();
        const int rowShift = this->RowShift();
        const int diagShift = d.ColShift();

        int iStart,jStart;
        if( offset >= 0 )
        {
            iStart = diagShift;
            jStart = diagShift+offset;
        }
        else
        {
            iStart = diagShift-offset;
            jStart = diagShift;
        }

        const int iLocStart = (iStart-colShift) / r;
        const int jLocStart = (jStart-rowShift) / c;

        const int localDiagLength = d.LocalHeight();
        for( int k=0; k<localDiagLength; ++k )
            this->LocalEntry(iLocStart+k*(lcm/r),jLocStart+k*(lcm/c)) = 
                d.LocalEntry(k,0);
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::SetDiagonal
( const DistMatrixBase<T,Star,MD>& d, int offset )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SetDiagonal");
    if( d.Height() != 1 )
        throw logic_error( "d must be a row vector." );
    {
        int height = this->Height();
        int width = this->Width();
        int length;
        if( offset >= 0 )
        {
            const int remainingWidth = max(width-offset,0);
            length = min(remainingWidth,height);
        }
        else
        {
            const int remainingHeight = max(height+offset,0);
            length = min(remainingHeight,width);
        }
        if( length != d.Width() )
        {
            ostringstream msg;
            msg << "d is not of the same length as the diagonal:" << endl
                << "  A ~ " << this->Height() << " x " << this->Width() << endl
                << "  d ~ " << d.Height() << " x " << d.Width() << endl
                << "  A diag length: " << length << endl;
            throw logic_error( msg.str() );
        }
    }
#endif
    if( d.InDiagonal() )
    {
        const Grid& grid = this->GetGrid();
        const int r = grid.Height();
        const int c = grid.Width();
        const int lcm = grid.LCM();
        const int colShift = this->ColShift();
        const int rowShift = this->RowShift();
        const int diagShift = d.RowShift();

        int iStart,jStart;
        if( offset >= 0 )
        {
            iStart = diagShift;
            jStart = diagShift+offset;
        }
        else
        {
            iStart = diagShift-offset;
            jStart = diagShift;
        }

        const int iLocStart = (iStart-colShift) / r;
        const int jLocStart = (jStart-rowShift) / c;

        const int localDiagLength = d.LocalWidth();
        for( int k=0; k<localDiagLength; ++k )
            this->LocalEntry(iLocStart+k*(lcm/r),jLocStart+k*(lcm/c)) = 
                d.LocalEntry(0,k);
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

//
// Utility functions, e.g., SetToIdentity and MakeTrapezoidal 
//

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::MakeTrapezoidal
( Side side, Shape shape, int offset )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::MakeTrapezoidal");
    this->AssertNotLockedView();
#endif
    const int height = this->Height();
    const int width = this->Width();
    const int localHeight = this->LocalHeight();
    const int localWidth = this->LocalWidth();
    const int r = this->GetGrid().Height();
    const int c = this->GetGrid().Width();
    const int colShift = this->ColShift();
    const int rowShift = this->RowShift();

    if( shape == Lower )
    {
        for( int jLoc=0; jLoc<localWidth; ++jLoc )
        {
            const int j = rowShift + jLoc*c;
            int lastZero_i;
            if( side == Left )
                lastZero_i = j-offset-1;
            else
                lastZero_i = j-offset+height-width-1;
            if( lastZero_i >= 0 )
            {
                const int boundary = min( lastZero_i+1, height );
                const int numZeros = LocalLength( boundary, colShift, r );
                for( int iLoc=0; iLoc<numZeros; ++iLoc )
                    this->LocalEntry(iLoc,jLoc) = (T)0;
            }
        }
    }
    else
    {
        for( int jLoc=0; jLoc<localWidth; ++jLoc )
        {
            const int j = rowShift + jLoc*c;
            int firstZero_i;
            if( side == Left )
                firstZero_i = max(j-offset+1,0);
            else
                firstZero_i = max(j+height-width-offset+1,0);
            const int nonzeroLength = LocalLength(firstZero_i,colShift,r);
            for( int iLoc=nonzeroLength; iLoc<localHeight; ++iLoc )
                this->LocalEntry(iLoc,jLoc) = (T)0;
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::SetToIdentity()
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SetToIdentity");
    this->AssertNotLockedView();
#endif
    const int localHeight = this->LocalHeight();
    const int localWidth = this->LocalWidth();
    const int r = this->GetGrid().Height();
    const int c = this->GetGrid().Width();
    const int colShift = this->ColShift();
    const int rowShift = this->RowShift();

    this->_localMatrix.SetToZero();
    for( int iLoc=0; iLoc<localHeight; ++iLoc )
    {
        const int i = colShift + iLoc*r;                
        if( i % c == rowShift )
        {
            const int jLoc = (i-rowShift) / c;
            if( jLoc < localWidth )
                this->LocalEntry(iLoc,jLoc) = (T)1;
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::SetToRandom()
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SetToRandom");
    this->AssertNotLockedView();
#endif
    const int localHeight = this->LocalHeight();
    const int localWidth = this->LocalWidth();
    for( int i=0; i<localHeight; ++i )
        for( int j=0; j<localWidth; ++j )
            this->LocalEntry(i,j) = Random<T>();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::ConjugateTransposeFrom
( const DistMatrixBase<T,Star,MC>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::ConjugateTransposeFrom");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    if( this->Viewing() )
        this->AssertSameSizeAsTranspose( A );
#endif
    const Grid& grid = this->GetGrid();
    if( !this->Viewing() )
    {
        if( !this->ConstrainedColAlignment() )
        {
            this->_colAlignment = A.RowAlignment();
            this->_colShift = 
                Shift( grid.MCRank(), this->ColAlignment(), grid.Height() );
        }
        this->ResizeTo( A.Width(), A.Height() );
    }

    if( this->ColAlignment() == A.RowAlignment() )
    {
        const int c = grid.Width();
        const int rowShift = this->RowShift();

        const int localHeight = this->LocalHeight();
        const int localWidth = this->LocalWidth();
        for( int j=0; j<localWidth; ++j )
            for( int i=0; i<localHeight; ++i )
                this->LocalEntry(i,j) = Conj( A.LocalEntry(rowShift+j*c,i) );
    }
    else
    {
#ifndef RELEASE
        if( grid.VCRank() == 0 )
            cout << "Unaligned [MC,MR]::ConjugateTransposeFrom." << endl;
#endif
        const int r = grid.Height();
        const int c = grid.Width();
        const int rank = grid.MCRank();
        const int rowShift = this->RowShift();
        const int colAlignment = this->ColAlignment();
        const int rowAlignmentOfA = A.RowAlignment();

        const int sendRank = (rank+r+colAlignment-rowAlignmentOfA) % r;
        const int recvRank = (rank+r+rowAlignmentOfA-colAlignment) % r;

        const int localHeight = this->LocalHeight();
        const int localWidth = this->LocalWidth();
        const int localWidthOfA = A.LocalWidth();

        const int sendSize = localWidthOfA * localWidth;
        const int recvSize = localHeight * localWidth;

        this->_auxMemory.Require( sendSize + recvSize );

        T* buffer = this->_auxMemory.Buffer();
        T* sendBuffer = &buffer[0];
        T* recvBuffer = &buffer[sendSize];

        // Pack
        for( int j=0; j<localWidth; ++j )
            for( int i=0; i<localWidthOfA; ++i )
                sendBuffer[i+j*localWidth] = 
                    Conj( A.LocalEntry(rowShift+j*c,i) );

        // Communicate
        SendRecv
        ( sendBuffer, sendSize, sendRank, 0,
          recvBuffer, recvSize, recvRank, MPI_ANY_TAG, grid.MCComm() );

        // Unpack
        for( int j=0; j<localWidth; ++j )
            for( int i=0; i<localHeight; ++i )
                this->LocalEntry(i,j) = recvBuffer[i+j*localHeight];

        this->_auxMemory.Release();
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::TransposeFrom
( const DistMatrixBase<T,Star,MC>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::TransposeFrom");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    if( this->Viewing() )
        this->AssertSameSizeAsTranspose( A );
#endif
    const Grid& grid = this->GetGrid();
    if( !this->Viewing() )
    {
        if( !this->ConstrainedColAlignment() )
        {
            this->_colAlignment = A.RowAlignment();
            this->_colShift = 
                Shift( grid.MCRank(), this->ColAlignment(), grid.Height() );
        }
        this->ResizeTo( A.Width(), A.Height() );
    }

    if( this->ColAlignment() == A.RowAlignment() )
    {
        const int c = grid.Width();
        const int rowShift = this->RowShift();

        const int localHeight = this->LocalHeight();
        const int localWidth = this->LocalWidth();
        for( int j=0; j<localWidth; ++j )
            for( int i=0; i<localHeight; ++i )
                this->LocalEntry(i,j) = A.LocalEntry(rowShift+j*c,i);
    }
    else
    {
#ifndef RELEASE
        if( grid.VCRank() == 0 )
            cout << "Unaligned [MC,MR]::TransposeFrom." << endl;
#endif
        const int r = grid.Height();
        const int c = grid.Width();
        const int rank = grid.MCRank();
        const int rowShift = this->RowShift();
        const int colAlignment = this->ColAlignment();
        const int rowAlignmentOfA = A.RowAlignment();

        const int sendRank = (rank+r+colAlignment-rowAlignmentOfA) % r;
        const int recvRank = (rank+r+rowAlignmentOfA-colAlignment) % r;

        const int localHeight = this->LocalHeight();
        const int localWidth = this->LocalWidth();
        const int localWidthOfA = A.LocalWidth();

        const int sendSize = localWidthOfA * localWidth;
        const int recvSize = localHeight * localWidth;

        this->_auxMemory.Require( sendSize + recvSize );

        T* buffer = this->_auxMemory.Buffer();
        T* sendBuffer = &buffer[0];
        T* recvBuffer = &buffer[sendSize];

        // Pack
        for( int j=0; j<localWidth; ++j )
            for( int i=0; i<localWidthOfA; ++i )
                sendBuffer[i+j*localWidth] = A.LocalEntry(rowShift+j*c,i);

        // Communicate
        SendRecv
        ( sendBuffer, sendSize, sendRank, 0,
          recvBuffer, recvSize, recvRank, MPI_ANY_TAG, grid.MCComm() );

        // Unpack
        for( int j=0; j<localWidth; ++j )
            for( int i=0; i<localHeight; ++i )
                this->LocalEntry(i,j) = recvBuffer[i+j*localHeight];

        this->_auxMemory.Release();
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
const DistMatrixBase<T,MC,MR>&
elemental::DistMatrixBase<T,MC,MR>::operator=
( const DistMatrixBase<T,MC,MR>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR] = [MC,MR]");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    if( this->Viewing() )
        this->AssertSameSize( A );
#endif
    if( !this->Viewing() )
    {
        if( !this->ConstrainedColAlignment() )
        {
            this->_colAlignment = A.ColAlignment();
            this->_colShift = A.ColShift();
        }
        if( !this->ConstrainedRowAlignment() )
        {
            this->_rowAlignment = A.RowAlignment();
            this->_rowShift = A.RowShift();
        }
        this->ResizeTo( A.Height(), A.Width() );
    }

    if( this->ColAlignment() == A.ColAlignment() &&
        this->RowAlignment() == A.RowAlignment() )
    {
        this->_localMatrix = A.LockedLocalMatrix();
    }
    else
    {
        const Grid& grid = this->GetGrid();
#ifndef RELEASE
        if( grid.VCRank() == 0 )
            cout << "Unaligned [MC,MR] <- [MC,MR]." << endl;
#endif
        const int r = grid.Height();
        const int c = grid.Width();
        const int row = grid.MCRank();
        const int col = grid.MRRank();

        const int colAlignment = this->ColAlignment();
        const int rowAlignment = this->RowAlignment();
        const int colAlignmentOfA = A.ColAlignment();
        const int rowAlignmentOfA = A.RowAlignment();

        const int sendRow = (row+r+colAlignment-colAlignmentOfA) % r;
        const int sendCol = (col+c+rowAlignment-rowAlignmentOfA) % c;
        const int recvRow = (row+r+colAlignmentOfA-colAlignment) % r;
        const int recvCol = (col+c+rowAlignmentOfA-rowAlignment) % c;
        const int sendRank = sendRow + sendCol*r;
        const int recvRank = recvRow + recvCol*r;

        const int localHeight = this->LocalHeight();
        const int localWidth = this->LocalWidth();
        const int localHeightOfA = A.LocalHeight();
        const int localWidthOfA = A.LocalWidth();

        const int sendSize = localHeightOfA * localWidthOfA;
        const int recvSize = localHeight * localWidth;

        this->_auxMemory.Require( sendSize + recvSize );

        T* buffer = this->_auxMemory.Buffer();
        T* sendBuffer = &buffer[0];
        T* recvBuffer = &buffer[sendSize];

        // Pack
        for( int j=0; j<localWidthOfA; ++j )
            for( int i=0; i<localHeightOfA; ++i )
                sendBuffer[i+j*localHeightOfA] = A.LocalEntry(i,j);

        // Communicate
        SendRecv
        ( sendBuffer, sendSize, sendRank, 0,
          recvBuffer, recvSize, recvRank, MPI_ANY_TAG, grid.VCComm() );

        // Unpack
        for( int j=0; j<localWidth; ++j )
            for( int i=0; i<localHeight; ++i )
                this->LocalEntry(i,j) = recvBuffer[i+j*localHeight];

        this->_auxMemory.Release();
    }
#ifndef RELEASE
    PopCallStack();
#endif
    return *this;
}

template<typename T>
const DistMatrixBase<T,MC,MR>&
elemental::DistMatrixBase<T,MC,MR>::operator=
( const DistMatrixBase<T,MC,Star>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR] = [MC,* ]");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    if( this->Viewing() )
        this->AssertSameSize( A );
#endif
    const Grid& grid = this->GetGrid();
    if( !this->Viewing() )
    {
        if( !this->ConstrainedColAlignment() )
        {
            this->_colAlignment = A.ColAlignment();
            this->_colShift = 
                Shift( grid.MCRank(), this->ColAlignment(), grid.Height() );
        }
        this->ResizeTo( A.Height(), A.Width() );
    }

    if( this->ColAlignment() == A.ColAlignment() )
    {
        const int c = grid.Width();
        const int rowShift = this->RowShift();

        const int localHeight = this->LocalHeight();
        const int localWidth = this->LocalWidth();
        for( int j=0; j<localWidth; ++j )
            for( int i=0; i<localHeight; ++i )
                this->LocalEntry(i,j) = A.LocalEntry(i,rowShift+j*c);
    }
    else
    {
#ifndef RELEASE
        if( grid.VCRank() == 0 )
            cout << "Unaligned [MC,MR] <- [MC,* ]." << endl;
#endif
        const int r = grid.Height();
        const int c = grid.Width();
        const int rank = grid.MCRank();
        const int rowShift = this->RowShift();
        const int colAlignment = this->ColAlignment();
        const int colAlignmentOfA = A.ColAlignment();

        const int sendRank = (rank+r+colAlignment-colAlignmentOfA) % r;
        const int recvRank = (rank+r+colAlignmentOfA-colAlignment) % r;

        const int localHeight = this->LocalHeight();
        const int localWidth = this->LocalWidth();
        const int localHeightOfA = A.LocalHeight();

        const int sendSize = localHeightOfA * localWidth;
        const int recvSize = localHeight * localWidth;

        this->_auxMemory.Require( sendSize + recvSize );

        T* buffer = this->_auxMemory.Buffer();
        T* sendBuffer = &buffer[0];
        T* recvBuffer = &buffer[sendSize];

        // Pack
        for( int j=0; j<localWidth; ++j )
            for( int i=0; i<localHeightOfA; ++i )
                sendBuffer[i+j*localWidth] = A.LocalEntry(i,rowShift+j*c);

        // Communicate
        SendRecv
        ( sendBuffer, sendSize, sendRank, 0,
          recvBuffer, recvSize, recvRank, MPI_ANY_TAG, grid.MCComm() );

        // Unpack
        for( int j=0; j<localWidth; ++j )
            for( int i=0; i<localHeight; ++i )
                this->LocalEntry(i,j) = recvBuffer[i+j*localHeight];

        this->_auxMemory.Release();
    }
#ifndef RELEASE
    PopCallStack();
#endif
    return *this;
}

template<typename T>
const DistMatrixBase<T,MC,MR>&
elemental::DistMatrixBase<T,MC,MR>::operator=
( const DistMatrixBase<T,Star,MR>& A )
{ 
#ifndef RELEASE
    PushCallStack("[MC,MR] = [* ,MR]");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    if( this->Viewing() )
        this->AssertSameSize( A );
#endif
    const Grid& grid = this->GetGrid();
    if( !this->Viewing() )
    {
        if( !this->ConstrainedRowAlignment() )
        {
            this->_rowAlignment = A.RowAlignment();
            this->_rowShift = 
                Shift( grid.MRRank(), this->RowAlignment(), grid.Width() );
        }
        this->ResizeTo( A.Height(), A.Width() );
    }

    if( this->RowAlignment() == A.RowAlignment() )
    {
        const int r = grid.Height();
        const int colShift = this->ColShift();

        const int localHeight = this->LocalHeight();
        const int localWidth = this->LocalWidth();
        for( int j=0; j<localWidth; ++j )
            for( int i=0; i<localHeight; ++i )
                this->LocalEntry(i,j) = A.LocalEntry(colShift+i*r,j);
    }
    else
    {
#ifndef RELEASE
        if( grid.VCRank() == 0 )
            cout << "Unaligned [MC,MR] <- [* ,MR]." << endl;
#endif
        const int r = grid.Height(); 
        const int c = grid.Width();
        const int col = grid.MRRank();
        const int colShift = this->ColShift();
        const int rowAlignment = this->RowAlignment();
        const int rowAlignmentOfA = A.RowAlignment();

        const int sendCol = (col+c+rowAlignment-rowAlignmentOfA) % c;
        const int recvCol = (col+c+rowAlignmentOfA-rowAlignment) % c;

        const int localHeight = this->LocalHeight();
        const int localWidth = this->LocalWidth();
        const int localWidthOfA = A.LocalWidth();

        const int sendSize = localHeight * localWidthOfA;
        const int recvSize = localHeight * localWidth;

        this->_auxMemory.Require( sendSize + recvSize );

        T* buffer = this->_auxMemory.Buffer();
        T* sendBuffer = &buffer[0];
        T* recvBuffer = &buffer[sendSize];

        // Pack
        for( int j=0; j<localWidthOfA; ++j )
            for( int i=0; i<localHeight; ++i )
                sendBuffer[i+j*localHeight] = A.LocalEntry(colShift+i*r,j);

        // Communicate
        SendRecv
        ( sendBuffer, sendSize, sendCol, 0,
          recvBuffer, recvSize, recvCol, MPI_ANY_TAG, grid.MRComm() );

        // Unpack
        for( int j=0; j<localWidth; ++j )
            for( int i=0; i<localHeight; ++i )
                this->LocalEntry(i,j) = recvBuffer[i+j*localHeight];

        this->_auxMemory.Release();
    }
#ifndef RELEASE
    PopCallStack();
#endif
    return *this;
}

template<typename T>
const DistMatrixBase<T,MC,MR>&
elemental::DistMatrixBase<T,MC,MR>::operator=
( const DistMatrixBase<T,MD,Star>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR] = [MD,* ]");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    if( this->Viewing() )
        this->AssertSameSize( A );
#endif
    throw logic_error( "[MC,MR] = [MD,* ] not yet implemented." );
#ifndef RELEASE
    PopCallStack();
#endif
    return *this;
}

template<typename T>
const DistMatrixBase<T,MC,MR>&
elemental::DistMatrixBase<T,MC,MR>::operator=
( const DistMatrixBase<T,Star,MD>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR] = [* ,MD]");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    if( this->Viewing() )
        this->AssertSameSize( A );
#endif
    throw logic_error( "[MC,MR] = [* ,MD] not yet implemented." );
#ifndef RELEASE
    PopCallStack();
#endif
    return *this;
}

template<typename T>
const DistMatrixBase<T,MC,MR>&
elemental::DistMatrixBase<T,MC,MR>::operator=
( const DistMatrixBase<T,MR,MC>& A )
{ 
#ifndef RELEASE
    PushCallStack("[MC,MR] = [MR,MC]");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    if( this->Viewing() )
        this->AssertSameSize( A );
#endif
    const Grid& grid = this->GetGrid();
    if( A.Width() == 1 )
    {
        if( !this->Viewing() )
            this->ResizeTo( A.Height(), 1 );

        const int r = grid.Height();
        const int c = grid.Width();
        const int p = grid.Size();
        const int myRow = grid.MCRank();
        const int myCol = grid.MRRank();
        const int rankCM = grid.VCRank();
        const int rankRM = grid.VRRank();
        const int ownerCol = this->RowAlignment();
        const int ownerRow = A.RowAlignment();
        const int colAlignment = this->ColAlignment();
        const int colAlignmentOfA = A.ColAlignment();
        const int colShift = this->ColShift();
        const int colShiftOfA = A.ColShift();

        const int height = A.Height();
        const int maxLocalHeight = MaxLocalLength(height,p);

        const int portionSize = max(maxLocalHeight,MinCollectContrib);

        const int colShiftVC = Shift(rankCM,colAlignment,p);
        const int colShiftVROfA = Shift(rankRM,colAlignmentOfA,p);
        const int sendRankCM = (rankCM+(p+colShiftVROfA-colShiftVC)) % p;
        const int recvRankRM = (rankRM+(p+colShiftVC-colShiftVROfA)) % p;
        const int recvRankCM = (recvRankRM/c)+r*(recvRankRM%c);

        this->_auxMemory.Require( (r+c)*portionSize );
        T* buffer = this->_auxMemory.Buffer();
        T* sendBuf = &buffer[0];
        T* recvBuf = &buffer[c*portionSize];

        if( myRow == ownerRow )
        {
            // Pack
            for( int k=0; k<r; ++k )
            {
                T* data = &recvBuf[k*portionSize];

                const int shift = Shift(myCol+c*k,colAlignmentOfA,p);
                const int offset = (shift-colShiftOfA) / c;
                const int thisLocalHeight = LocalLength(height,shift,p);

                for( int i=0; i<thisLocalHeight; ++i )
                    data[i] = A.LocalEntry(offset+i*r,0);
            }
        }

        // A[VR,* ] <- A[MR,MC]
        Scatter
        ( recvBuf, portionSize, 
          sendBuf, portionSize, ownerRow, grid.MCComm() );

        // A[VC,* ] <- A[VR,* ]
        SendRecv
        ( sendBuf, portionSize, sendRankCM, 0,
          recvBuf, portionSize, recvRankCM, MPI_ANY_TAG, grid.VCComm() );

        // A[MC,MR] <- A[VC,* ]
        Gather
        ( recvBuf, portionSize, 
          sendBuf, portionSize, ownerCol, grid.MRComm() );

        if( myCol == ownerCol )
        {
            // Unpack
            for( int k=0; k<c; ++k )
            {
                const T* data = &sendBuf[k*portionSize];

                const int shift = Shift(myRow+r*k,colAlignment,p);
                const int offset = (shift-colShift) / r;
                const int thisLocalHeight = LocalLength(height,shift,p);

                for( int i=0; i<thisLocalHeight; ++i )
                    this->LocalEntry(offset+i*c,0) = data[i];
            }
        }

        this->_auxMemory.Release();
    }
    else if( A.Height() == 1 )
    {
        if( !this->Viewing() )
            this->ResizeTo( 1, A.Width() );

        const int r = grid.Height();
        const int c = grid.Width();
        const int p = grid.Size();
        const int myRow = grid.MCRank();
        const int myCol = grid.MRRank();
        const int rankCM = grid.VCRank();
        const int rankRM = grid.VRRank();
        const int ownerRow = this->ColAlignment();
        const int ownerCol = A.ColAlignment();
        const int rowAlignment = this->RowAlignment();
        const int rowAlignmentOfA = A.RowAlignment();
        const int rowShift = this->RowShift();
        const int rowShiftOfA = A.RowShift();

        const int width = A.Width();
        const int maxLocalWidth = MaxLocalLength(width,p);

        const int portionSize = max(maxLocalWidth,MinCollectContrib);

        const int rowShiftVR = Shift(rankRM,rowAlignment,p);
        const int rowShiftVCOfA = Shift(rankCM,rowAlignmentOfA,p);
        const int sendRankRM = (rankRM+(p+rowShiftVCOfA-rowShiftVR)) % p;
        const int recvRankCM = (rankCM+(p+rowShiftVR-rowShiftVCOfA)) % p;
        const int recvRankRM = (recvRankCM/r)+c*(recvRankCM%r);

        this->_auxMemory.Require( (r+c)*portionSize );
        T* buffer = this->_auxMemory.Buffer();
        T* sendBuf = &buffer[0];
        T* recvBuf = &buffer[r*portionSize];

        if( myCol == ownerCol )
        {
            // Pack
            for( int k=0; k<c; ++k )
            {
                T* data = &recvBuf[k*portionSize];

                const int shift = Shift(myRow+r*k,rowAlignmentOfA,p);
                const int offset = (shift-rowShiftOfA) / r;
                const int thisLocalWidth = LocalLength(width,shift,p);

                for( int j=0; j<thisLocalWidth; ++j )
                    data[j] = A.LocalEntry(0,offset+j*c);
            }
        }

        // A[* ,VC] <- A[MR,MC]
        Scatter
        ( recvBuf, portionSize, 
          sendBuf, portionSize, ownerCol, grid.MRComm() );

        // A[* ,VR] <- A[* ,VC]
        SendRecv
        ( sendBuf, portionSize, sendRankRM, 0,
          recvBuf, portionSize, recvRankRM, MPI_ANY_TAG, grid.VRComm() );

        // A[MC,MR] <- A[* ,VR]
        Gather
        ( recvBuf, portionSize, 
          sendBuf, portionSize, ownerRow, grid.MCComm() );

        if( myRow == ownerRow )
        {
            // Unpack
            for( int k=0; k<r; ++k )
            {
                const T* data = &sendBuf[k*portionSize];

                const int shift = Shift(myCol+c*k,rowAlignment,p);
                const int offset = (shift-rowShift) / c;
                const int thisLocalWidth = LocalLength(width,shift,p);

                for( int j=0; j<thisLocalWidth; ++j )
                    this->LocalEntry(0,offset+j*r) = data[j];
            }
        }

        this->_auxMemory.Release();
    }
    else
    {
        if( A.Height() >= A.Width() )
        {
            auto_ptr< DistMatrix<T,VR,Star> > A_VR_Star
            ( new DistMatrix<T,VR,Star>(grid) );

            *A_VR_Star = A;

            auto_ptr< DistMatrix<T,VC,Star> > A_VC_Star
            ( new DistMatrix<T,VC,Star>(true,this->ColAlignment(),grid) );
            *A_VC_Star = *A_VR_Star;
            delete A_VR_Star.release(); // lowers memory highwater

            *this = *A_VC_Star;
        }
        else
        {
            auto_ptr< DistMatrix<T,Star,VC> > A_Star_VC
            ( new DistMatrix<T,Star,VC>(grid) );
            *A_Star_VC = A;

            auto_ptr< DistMatrix<T,Star,VR> > A_Star_VR
            ( new DistMatrix<T,Star,VR>(true,this->RowAlignment(),grid) );
            *A_Star_VR = *A_Star_VC;
            delete A_Star_VC.release(); // lowers memory highwater

            *this = *A_Star_VR;
            this->ResizeTo( A_Star_VR->Height(), A_Star_VR->Width() );
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
    return *this;
}

template<typename T>
const DistMatrixBase<T,MC,MR>&
elemental::DistMatrixBase<T,MC,MR>::operator=
( const DistMatrixBase<T,MR,Star>& A )
{ 
#ifndef RELEASE
    PushCallStack("[MC,MR] = [MR,* ]");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    if( this->Viewing() )
        this->AssertSameSize( A );
#endif
    const Grid& grid = this->GetGrid();

    auto_ptr< DistMatrix<T,VR,Star> > A_VR_Star
    ( new DistMatrix<T,VR,Star>(grid) );
    *A_VR_Star = A;

    auto_ptr< DistMatrix<T,VC,Star> > A_VC_Star
    ( new DistMatrix<T,VC,Star>(true,this->ColAlignment(),grid) );
    *A_VC_Star = *A_VR_Star;
    delete A_VR_Star.release(); // lowers memory highwater

    *this = *A_VC_Star;
#ifndef RELEASE
    PopCallStack();
#endif
    return *this;
}

template<typename T>
const DistMatrixBase<T,MC,MR>&
elemental::DistMatrixBase<T,MC,MR>::operator=
( const DistMatrixBase<T,Star,MC>& A )
{ 
#ifndef RELEASE
    PushCallStack("[MC,MR] = [* ,MC]");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    if( this->Viewing() )
        this->AssertSameSize( A );
#endif
    const Grid& grid = this->GetGrid();

    auto_ptr< DistMatrix<T,Star,VC> > A_Star_VC
    ( new DistMatrix<T,Star,VC>(grid) );
    *A_Star_VC = A;

    auto_ptr< DistMatrix<T,Star,VR> > A_Star_VR
    ( new DistMatrix<T,Star,VR>(true,this->RowAlignment(),grid) );
    *A_Star_VR = *A_Star_VC;
    delete A_Star_VC.release(); // lowers memory highwater

    *this = *A_Star_VR;
#ifndef RELEASE
    PopCallStack();
#endif
    return *this;
}

template<typename T>
const DistMatrixBase<T,MC,MR>&
elemental::DistMatrixBase<T,MC,MR>::operator=
( const DistMatrixBase<T,VC,Star>& A )
{ 
#ifndef RELEASE
    PushCallStack("[MC,MR] = [VC,* ]");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    if( this->Viewing() )
        this->AssertSameSize( A );
#endif
    const Grid& grid = this->GetGrid();

    if( !this->Viewing() )
    {
        if( !this->ConstrainedColAlignment() )
        {
            this->_colAlignment = A.ColAlignment() % grid.Height();
            this->_colShift = 
                Shift( grid.MCRank(), this->ColAlignment(), grid.Height() );
        }
        this->ResizeTo( A.Height(), A.Width() );
    }

    if( this->ColAlignment() == A.ColAlignment() % grid.Height() )
    {
        const int r = grid.Height();
        const int c = grid.Width();
        const int p = r * c;
        const int row = grid.MCRank();
        const int colShift = this->ColShift();
        const int rowAlignment = this->RowAlignment();
        const int colAlignmentOfA = A.ColAlignment();

        const int height = this->Height();
        const int width = this->Width();
        const int localWidth = this->LocalWidth();
        const int localHeightOfA = A.LocalHeight();

        const int maxHeight = MaxLocalLength(height,p);
        const int maxWidth = MaxLocalLength(width,c);
        const int portionSize = max(maxHeight*maxWidth,MinCollectContrib);

        this->_auxMemory.Require( 2*c*portionSize );

        T* buffer = this->_auxMemory.Buffer();
        T* sendBuffer = &buffer[0];
        T* recvBuffer = &buffer[c*portionSize];

        // Pack
        for( int k=0; k<c; ++k )
        {
            T* data = &sendBuffer[k*portionSize];

            const int thisRowShift = Shift(k,rowAlignment,c);
            const int thisLocalWidth = LocalLength(width,thisRowShift,c);

            for( int j=0; j<thisLocalWidth; ++j )
                for( int i=0; i<localHeightOfA; ++i )
                    data[i+j*localHeightOfA] = A.LocalEntry(i,thisRowShift+j*c);
        }

        // Communicate
        AllToAll
        ( sendBuffer, portionSize,
          recvBuffer, portionSize, grid.MRComm() );

        // Unpack
        for( int k=0; k<c; ++k )
        {
            const T* data = &recvBuffer[k*portionSize];

            const int thisRank = row+k*r;
            const int thisColShift = Shift(thisRank,colAlignmentOfA,p);
            const int thisColOffset = (thisColShift-colShift) / r;
            const int thisLocalHeight = LocalLength(height,thisColShift,p);

            for( int j=0; j<localWidth; ++j )
                for( int i=0; i<thisLocalHeight; ++i )
                    this->LocalEntry(thisColOffset+i*c,j) = 
                          data[i+j*thisLocalHeight];
        }

        this->_auxMemory.Release();
    }
    else
    {
#ifndef RELEASE
        if( grid.VCRank() == 0 )
            cout << "Unaligned [MC,MR] <- [VC,* ]." << endl;
#endif
        const int r = grid.Height();
        const int c = grid.Width();
        const int p = r * c;
        const int row = grid.MCRank();
        const int colShift = this->ColShift();
        const int colAlignment = this->ColAlignment();
        const int rowAlignment = this->RowAlignment();
        const int colAlignmentOfA = A.ColAlignment();

        const int sendRow = (row+r+colAlignment-(colAlignmentOfA%r)) % r;
        const int recvRow = (row+r+(colAlignmentOfA%r)-colAlignment) % r;

        const int height = this->Height();
        const int width = this->Width();
        const int localWidth = this->LocalWidth();
        const int localHeightOfA = A.LocalHeight();

        const int maxHeight = MaxLocalLength(height,p);
        const int maxWidth = MaxLocalLength(width,c);
        const int portionSize = max(maxHeight*maxWidth,MinCollectContrib);

        this->_auxMemory.Require( 2*c*portionSize );

        T* buffer = this->_auxMemory.Buffer();
        T* firstBuffer = &buffer[0];
        T* secondBuffer = &buffer[c*portionSize];

        // Pack
        for( int k=0; k<c; ++k )
        {
            T* data = &secondBuffer[k*portionSize];

            const int thisRowShift = Shift(k,rowAlignment,c);
            const int thisLocalWidth = LocalLength(width,thisRowShift,c);

            for( int j=0; j<thisLocalWidth; ++j )
                for( int i=0; i<localHeightOfA; ++i )
                    data[i+j*localHeightOfA] = A.LocalEntry(i,thisRowShift+j*c);
        }

        // SendRecv: properly align A[VC,*] via a trade in the column
        SendRecv
        ( secondBuffer, c*portionSize, sendRow, 0,
          firstBuffer,  c*portionSize, recvRow, 0, grid.MCComm() );

        // AllToAll to gather all of the aligned A[VC,*] data into secondBuff.
        AllToAll
        ( firstBuffer,  portionSize,
          secondBuffer, portionSize, grid.MRComm() );

        // Unpack
        for( int k=0; k<c; ++k )
        {
            const T* data = &secondBuffer[k*portionSize];

            const int thisRank = recvRow+k*r;
            const int thisColShift = Shift(thisRank,colAlignmentOfA,p);
            const int thisColOffset = (thisColShift-colShift) / r;
            const int thisLocalHeight = LocalLength(height,thisColShift,p);

            for( int j=0; j<localWidth; ++j )
                for( int i=0; i<thisLocalHeight; ++i )
                    this->LocalEntry(thisColOffset+i*c,j) = 
                          data[i+j*thisLocalHeight];
        }

        this->_auxMemory.Release();
    }
#ifndef RELEASE
    PopCallStack();
#endif
    return *this;
}

template<typename T>
const DistMatrixBase<T,MC,MR>&
elemental::DistMatrixBase<T,MC,MR>::operator=
( const DistMatrixBase<T,Star,VC>& A )
{ 
#ifndef RELEASE
    PushCallStack("[MC,MR] = [* ,VC]");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    if( this->Viewing() )
        this->AssertSameSize( A );
#endif
    const Grid& grid = this->GetGrid();
    DistMatrix<T,Star,VR> A_Star_VR(true,this->RowAlignment(),grid);

    A_Star_VR = A;
    *this = A_Star_VR;
#ifndef RELEASE
    PopCallStack();
#endif
    return *this;
}

template<typename T>
const DistMatrixBase<T,MC,MR>&
elemental::DistMatrixBase<T,MC,MR>::operator=
( const DistMatrixBase<T,VR,Star>& A )
{ 
#ifndef RELEASE
    PushCallStack("[MC,MR] = [VR,* ]");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    if( this->Viewing() )
        this->AssertSameSize( A );
#endif
    const Grid& grid = this->GetGrid();
    DistMatrix<T,VC,Star> A_VC_Star(true,this->ColAlignment(),grid);

    A_VC_Star = A;
    *this = A_VC_Star;
#ifndef RELEASE
    PopCallStack();
#endif
    return *this;
}

template<typename T>
const DistMatrixBase<T,MC,MR>&
elemental::DistMatrixBase<T,MC,MR>::operator=
( const DistMatrixBase<T,Star,VR>& A )
{ 
#ifndef RELEASE
    PushCallStack("[MC,MR] = [* ,VR]");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    if( this->Viewing() )
        this->AssertSameSize( A );
#endif
    const Grid& grid = this->GetGrid();
    if( !this->Viewing() )
    {
        if( !this->ConstrainedRowAlignment() )
        {
            this->_rowAlignment = A.RowAlignment() % grid.Width();
            this->_rowShift = 
                Shift( grid.MRRank(), this->RowAlignment(), grid.Width() );
        }
        this->ResizeTo( A.Height(), A.Width() );
    }

    if( this->RowAlignment() == A.RowAlignment() % grid.Width() )
    {
        const int r = grid.Height();
        const int c = grid.Width();
        const int p = r * c;
        const int col = grid.MRRank();
        const int rowShift = this->RowShift();
        const int colAlignment = this->ColAlignment();
        const int rowAlignmentOfA = A.RowAlignment();

        const int height = this->Height();
        const int width = this->Width();
        const int localHeight = this->LocalHeight();
        const int localWidthOfA = A.LocalWidth();

        const int maxHeight = MaxLocalLength(height,r);
        const int maxWidth = MaxLocalLength(width,p);
        const int portionSize = max(maxHeight*maxWidth,MinCollectContrib);

        this->_auxMemory.Require( 2*r*portionSize );

        T* buffer = this->_auxMemory.Buffer();
        T* sendBuffer = &buffer[0];
        T* recvBuffer = &buffer[r*portionSize];

        // Pack
        for( int k=0; k<r; ++k )
        {
            T* data = &sendBuffer[k*portionSize];

            const int thisColShift = Shift(k,colAlignment,r);
            const int thisLocalHeight = LocalLength(height,thisColShift,r);

            for( int j=0; j<localWidthOfA; ++j )
                for( int i=0; i<thisLocalHeight; ++i )
                    data[i+j*thisLocalHeight] = 
                          A.LocalEntry(thisColShift+i*r,j);
        }

        // Communicate
        AllToAll
        ( sendBuffer, portionSize,
          recvBuffer, portionSize, grid.MCComm() );

        // Unpack
        for( int k=0; k<r; ++k )
        {
            const T* data = &recvBuffer[k*portionSize];

            const int thisRank = col+k*c;
            const int thisRowShift = Shift(thisRank,rowAlignmentOfA,p);
            const int thisRowOffset = (thisRowShift-rowShift) / c;
            const int thisLocalWidth = LocalLength(width,thisRowShift,p);

            for( int j=0; j<thisLocalWidth; ++j )
                for( int i=0; i<localHeight; ++i )
                    this->LocalEntry(i,thisRowOffset+j*r) = 
                        data[i+j*localHeight];
        }

        this->_auxMemory.Release();
    }
    else
    {
#ifndef RELEASE
        if( grid.VCRank() == 0 )
            cout << "Unaligned [MC,MR] <- [* ,VR]." << endl;
#endif
        const int r = grid.Height();
        const int c = grid.Width();
        const int p = r * c;
        const int col = grid.MRRank();
        const int rowShift = this->RowShift();
        const int colAlignment = this->ColAlignment();
        const int rowAlignment = this->RowAlignment();
        const int rowAlignmentOfA = A.RowAlignment();

        const int sendCol = (col+c+rowAlignment-(rowAlignmentOfA%c)) % c;
        const int recvCol = (col+c+(rowAlignmentOfA%c)-rowAlignment) % c;

        const int height = this->Height();
        const int width = this->Width();
        const int localHeight = this->LocalHeight();
        const int localWidthOfA = A.LocalWidth();
        
        const int maxHeight = MaxLocalLength(height,r);
        const int maxWidth = MaxLocalLength(width,p);
        const int portionSize = max(maxHeight*maxWidth,MinCollectContrib);

        this->_auxMemory.Require( 2*r*portionSize );

        T* buffer = this->_auxMemory.Buffer();
        T* firstBuffer = &buffer[0];
        T* secondBuffer = &buffer[r*portionSize];

        // Pack
        for( int k=0; k<r; ++k )
        {
            T* data = &secondBuffer[k*portionSize];

            const int thisColShift = Shift(k,colAlignment,r);
            const int thisLocalHeight = LocalLength(height,thisColShift,r);

            for( int j=0; j<localWidthOfA; ++j )
                for( int i=0; i<thisLocalHeight; ++i )
                    data[i+j*thisLocalHeight] = 
                          A.LocalEntry(thisColShift+i*r,j);
        }

        // SendRecv: properly align A[*,VR] via a trade in the column
        SendRecv
        ( secondBuffer, r*portionSize, sendCol, 0,
          firstBuffer,  r*portionSize, recvCol, 0, grid.MRComm() );

        // AllToAll to gather all of the aligned [*,VR] data into secondBuffer
        AllToAll
        ( firstBuffer,  portionSize,
          secondBuffer, portionSize, grid.MCComm() );

        // Unpack
        for( int k=0; k<r; ++k )
        {
            const T* data = &secondBuffer[k*portionSize];

            const int thisRank = recvCol+k*c;
            const int thisRowShift = Shift(thisRank,rowAlignmentOfA,p);
            const int thisRowOffset = (thisRowShift-rowShift) / c;
            const int thisLocalWidth = LocalLength(width,thisRowShift,p);

            for( int j=0; j<thisLocalWidth; ++j )
                for( int i=0; i<localHeight; ++i )
                    this->LocalEntry(i,thisRowOffset+j*r) = 
                        data[i+j*localHeight];
        }

        this->_auxMemory.Release();
    }
#ifndef RELEASE
    PopCallStack();
#endif
    return *this;
}

template<typename T>
const DistMatrixBase<T,MC,MR>&
elemental::DistMatrixBase<T,MC,MR>::operator=
( const DistMatrixBase<T,Star,Star>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR] = [* ,* ]");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    if( this->Viewing() )
        this->AssertSameSize( A );
#endif
    if( !this->Viewing() )
        this->ResizeTo( A.Height(), A.Width() );

    const int r = this->GetGrid().Height();
    const int c = this->GetGrid().Width();
    const int colShift = this->ColShift();
    const int rowShift = this->RowShift();

    const int localHeight = this->LocalHeight();
    const int localWidth = this->LocalWidth();
    for( int j=0; j<localWidth; ++j )
        for( int i=0; i<localHeight; ++i )
            this->LocalEntry(i,j) = A.LocalEntry(colShift+i*r,rowShift+j*c);
#ifndef RELEASE
    PopCallStack();
#endif
    return *this;
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::SumScatterFrom
( const DistMatrixBase<T,MC,Star>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SumScatterFrom([MC,* ])");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    if( this->Viewing() )
        this->AssertSameSize( A );
#endif
    const Grid& grid = this->GetGrid();
    if( !this->Viewing() )
    {
        if( !this->ConstrainedColAlignment() )
        {
            this->_colAlignment = A.ColAlignment();
            this->_colShift = 
                Shift( grid.MCRank(), this->ColAlignment(), grid.Height() );
        }
        this->ResizeTo( A.Height(), A.Width() );
    }

    if( this->ColAlignment() == A.ColAlignment() )
    {
        if( this->Width() == 1 )
        {
            const int rowAlignment = this->RowAlignment();
            const int myCol = grid.MRRank();

            const int localHeight = this->LocalHeight();

            const int recvSize = max(localHeight,MinCollectContrib);
            const int sendSize = recvSize;

            this->_auxMemory.Require( sendSize + recvSize );

            T* buffer = this->_auxMemory.Buffer();
            T* sendBuffer = &buffer[0];
            T* recvBuffer = &buffer[sendSize];

            // Pack 
            for( int i=0; i<localHeight; ++i )
                sendBuffer[i] = A.LocalEntry(i,0);

            // Reduce to rowAlignment
            Reduce
            ( sendBuffer, recvBuffer, sendSize, 
              MPI_SUM, rowAlignment, grid.MRComm() );

            if( myCol == rowAlignment )
            {
                for( int i=0; i<localHeight; ++i )
                    this->LocalEntry(i,0) = recvBuffer[i];
            }

            this->_auxMemory.Release();
        }
        else
        {
            const int c = grid.Width();
            const int rowAlignment = this->RowAlignment();
        
            const int width = this->Width();
            const int localHeight = this->LocalHeight();
            const int localWidth = this->LocalWidth();
            const int maxLocalWidth = MaxLocalLength(width,c);

            const int recvSize = 
                max(localHeight*maxLocalWidth,MinCollectContrib);
            const int sendSize = c * recvSize;

            this->_auxMemory.Require( sendSize + recvSize );

            T* buffer = this->_auxMemory.Buffer();
            T* sendBuffer = &buffer[0];
            T* recvBuffer = &buffer[sendSize];
        
            // Pack 
            int* recvSizes = new int[c];
            for( int k=0; k<c; ++k )
            {
                T* data = &sendBuffer[k*recvSize];
                recvSizes[k] = recvSize;

                const int thisRowShift = Shift( k, rowAlignment, c );
                const int thisLocalWidth = LocalLength(width,thisRowShift,c);

                for( int j=0; j<thisLocalWidth; ++j )
                    for( int i=0; i<localHeight; ++i )
                        data[i+j*localHeight] = 
                            A.LocalEntry(i,thisRowShift+j*c);
            }

            // Reduce-scatter over each process row
            ReduceScatter
            ( sendBuffer, recvBuffer, recvSizes, MPI_SUM, grid.MRComm() );
            delete[] recvSizes;

            // Unpack our received data
            for( int j=0; j<localWidth; ++j )
                for( int i=0; i<localHeight; ++i )
                    this->LocalEntry(i,j) = recvBuffer[i+j*localHeight];

            this->_auxMemory.Release();
        }
    }
    else
    {
#ifndef RELEASE
        if( grid.VCRank() == 0 )
            cout << "Unaligned SumScatterFrom [MC,MR] <- [MC,* ]." << endl;
#endif
        if( this->Width() == 1 )
        {
            const int r = grid.Height();
            const int rowAlignment = this->RowAlignment();
            const int myRow = grid.MCRank();
            const int myCol = grid.MRRank();

            const int height = this->Height();
            const int localHeight = this->LocalHeight();
            const int localHeightOfA = A.LocalHeight();
            const int maxLocalHeight = MaxLocalLength(height,r);

            const int portionSize = max(maxLocalHeight,MinCollectContrib);

            const int colAlignment = this->ColAlignment();
            const int colAlignmentOfA = A.ColAlignment();
            const int sendRow = (myRow+r+colAlignment-colAlignmentOfA) % r;
            const int recvRow = (myRow+r+colAlignmentOfA-colAlignment) % r;

            this->_auxMemory.Require( 2*portionSize );

            T* buffer = this->_auxMemory.Buffer();
            T* sendBuffer = &buffer[0];
            T* recvBuffer = &buffer[portionSize];

            // Pack 
            for( int i=0; i<localHeightOfA; ++i )
                sendBuffer[i] = A.LocalEntry(i,0);
        
            // Reduce to rowAlignment
            Reduce
            ( sendBuffer, recvBuffer, portionSize, 
              MPI_SUM, rowAlignment, grid.MRComm() );

            if( myCol == rowAlignment )
            {
                // Perform the realignment
                SendRecv
                ( recvBuffer, portionSize, sendRow, 0,
                  sendBuffer, portionSize, recvRow, 0, grid.MCComm() );

                for( int i=0; i<localHeight; ++i )
                    this->LocalEntry(i,0) = sendBuffer[i];
            }

            this->_auxMemory.Release();
        }
        else
        {
            const int r = grid.Height();
            const int c = grid.Width();
            const int row = grid.MCRank();

            const int colAlignment = this->ColAlignment();
            const int rowAlignment = this->RowAlignment();
            const int colAlignmentOfA = A.ColAlignment();
            const int sendRow = (row+r+colAlignment-colAlignmentOfA) % r;
            const int recvRow = (row+r+colAlignmentOfA-colAlignment) % r;

            const int width = this->Width();
            const int localHeight = this->LocalHeight();
            const int localWidth = this->LocalWidth();
            const int localHeightOfA = A.LocalHeight();
            const int maxLocalWidth = MaxLocalLength(width,c);

            const int recvSize_RS = 
                max(localHeightOfA*maxLocalWidth,MinCollectContrib);
            const int sendSize_RS = c * recvSize_RS;
            const int recvSize_SR = localHeight * localWidth;

            this->_auxMemory.Require
            ( recvSize_RS + max(sendSize_RS,recvSize_SR) );

            T* buffer = this->_auxMemory.Buffer();
            T* firstBuffer = &buffer[0];
            T* secondBuffer = &buffer[recvSize_RS];

            // Pack 
            int* recvSizes = new int[c];
            for( int k=0; k<c; ++k )
            {
                T* data = &secondBuffer[k*recvSize_RS];
                recvSizes[k] = recvSize_RS;

                const int thisRowShift = Shift( k, rowAlignment, c );
                const int thisLocalWidth = LocalLength(width,thisRowShift,c);

                for( int j=0; j<thisLocalWidth; ++j )
                    for( int i=0; i<localHeightOfA; ++i )
                        data[i+j*localHeightOfA] = 
                            A.LocalEntry(i,thisRowShift+j*c);
            }

            // Reduce-scatter over each process row
            ReduceScatter
            ( secondBuffer, firstBuffer, recvSizes, MPI_SUM, grid.MRComm() );
            delete[] recvSizes;

            // Trade reduced data with the appropriate process row
            SendRecv
            ( firstBuffer,  localHeightOfA*localWidth, sendRow, 0,
              secondBuffer, localHeight*localWidth,    recvRow, 0, 
              grid.MCComm() );

            // Unpack the received data
            for( int j=0; j<localWidth; ++j )
                for( int i=0; i<localHeight; ++i )
                    this->LocalEntry(i,j) = secondBuffer[i+j*localHeight];

            this->_auxMemory.Release();
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::SumScatterFrom
( const DistMatrixBase<T,Star,MR>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SumScatterFrom([* ,MR])");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    if( this->Viewing() )
        this->AssertSameSize( A );
#endif
    const Grid& grid = this->GetGrid();
    if( !this->Viewing() )
    {
        if( !this->ConstrainedRowAlignment() )
        {
            this->_rowAlignment = A.RowAlignment();
            this->_rowShift = 
                Shift( grid.MRRank(), this->RowAlignment(), grid.Width() );
        }
        this->ResizeTo( A.Height(), A.Width() );
    }

    if( this->RowAlignment() == A.RowAlignment() )
    {
        const int r = grid.Height();
        const int colAlignment = this->ColAlignment();

        const int height = this->Height();
        const int localHeight = this->LocalHeight();
        const int localWidth = this->LocalWidth();
        const int maxLocalHeight = MaxLocalLength(height,r);

        const int recvSize = 
            max(maxLocalHeight*localWidth,MinCollectContrib);
        const int sendSize = r * recvSize;

        this->_auxMemory.Require( sendSize + recvSize );

        T* buffer = this->_auxMemory.Buffer();
        T* sendBuffer = &buffer[0];
        T* recvBuffer = &buffer[sendSize];

        // Pack 
        int* recvSizes = new int[r];
        for( int k=0; k<r; ++k )
        {
            T* data = &sendBuffer[k*recvSize];
            recvSizes[k] = recvSize;

            const int thisColShift = Shift( k, colAlignment, r );
            const int thisLocalHeight = LocalLength( height, thisColShift, r );

            for( int j=0; j<localWidth; ++j )
                for( int i=0; i<thisLocalHeight; ++i )
                    data[i+j*thisLocalHeight] = 
                          A.LocalEntry(thisColShift+i*r,j);
        }

        // Reduce-scatter over each process col
        ReduceScatter
        ( sendBuffer, recvBuffer, recvSizes, MPI_SUM, grid.MCComm() );
        delete[] recvSizes;

        // Unpack our received data
        for( int j=0; j<localWidth; ++j )
            for( int i=0; i<localHeight; ++i )
                this->LocalEntry(i,j) = recvBuffer[i+j*localHeight];

        this->_auxMemory.Release();
    }
    else
    {
#ifndef RELEASE
        if( grid.VCRank() == 0 )
            cout << "Unaligned SumScatterFrom [MC,MR] <- [* ,MR]." << endl;
#endif
        const int r = grid.Height();
        const int c = grid.Width();
        const int col = grid.MRRank();

        const int colAlignment = this->ColAlignment();
        const int rowAlignment = this->RowAlignment();
        const int rowAlignmentOfA = A.RowAlignment();
        const int sendCol = (col+c+rowAlignment-rowAlignmentOfA) % c;
        const int recvCol = (col+c+rowAlignmentOfA-rowAlignment) % c;

        const int height = this->Height();
        const int localHeight = this->LocalHeight();
        const int localWidth = this->LocalWidth();
        const int localWidthOfA = A.LocalWidth();
        const int maxLocalHeight = MaxLocalLength(height,r);

        const int recvSize_RS = 
            max(maxLocalHeight*localWidthOfA,MinCollectContrib);
        const int sendSize_RS = r * recvSize_RS;
        const int recvSize_SR = localHeight * localWidth;

        this->_auxMemory.Require( recvSize_RS + max(sendSize_RS,recvSize_SR) );

        T* buffer = this->_auxMemory.Buffer();
        T* firstBuffer = &buffer[0];
        T* secondBuffer = &buffer[recvSize_RS];

        // Pack 
        int* recvSizes = new int[r];
        for( int k=0; k<r; ++k )
        {
            T* data = &secondBuffer[k*recvSize_RS];
            recvSizes[k] = recvSize_RS;

            const int thisColShift = Shift( k, colAlignment, r );
            const int thisLocalHeight = LocalLength( height, thisColShift, r );

            for( int j=0; j<localWidthOfA; ++j )
                for( int i=0; i<thisLocalHeight; ++i )
                    data[i+j*thisLocalHeight] = 
                          A.LocalEntry(thisColShift+i*r,j);
        }

        // Reduce-scatter over each process col
        ReduceScatter
        ( secondBuffer, firstBuffer, recvSizes, MPI_SUM, grid.MCComm() );
        delete[] recvSizes;

        // Trade reduced data with the appropriate process col
        SendRecv
        ( firstBuffer,  localHeight*localWidthOfA, sendCol, 0,
          secondBuffer, localHeight*localWidth,    recvCol, 0,
          grid.MRComm() );

        // Unpack the received data
        for( int j=0; j<localWidth; ++j )
            for( int i=0; i<localHeight; ++i )
                this->LocalEntry(i,j) = secondBuffer[i+j*localHeight];
        
        this->_auxMemory.Release();
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::SumScatterFrom
( const DistMatrixBase<T,Star,Star>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SumScatterFrom([* ,* ])");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    if( this->Viewing() )
        this->AssertSameSize( A );
#endif
    if( !this->Viewing() )
        this->ResizeTo( A.Height(), A.Width() );

    const Grid& grid = this->GetGrid();
    const int r = grid.Height();
    const int c = grid.Width();
    const int colAlignment = this->ColAlignment();
    const int rowAlignment = this->RowAlignment();

    const int height = this->Height();
    const int width = this->Width();
    const int localHeight = this->LocalHeight();
    const int localWidth = this->LocalWidth();
    const int maxLocalHeight = MaxLocalLength(height,r);
    const int maxLocalWidth = MaxLocalLength(width,c);

    const int recvSize = max(maxLocalHeight*maxLocalWidth,MinCollectContrib);
    const int sendSize = r * c * recvSize;

    this->_auxMemory.Require( sendSize + recvSize );

    T* buffer = this->_auxMemory.Buffer();
    T* sendBuffer = &buffer[0];
    T* recvBuffer = &buffer[sendSize];

    // Pack 
    int* recvSizes = new int[r*c];
    for( int l=0; l<c; ++l )
    {
        const int thisRowShift = Shift( l, rowAlignment, c );
        const int thisLocalWidth = LocalLength( width, thisRowShift, c );

        for( int k=0; k<r; ++k )
        {
            T* data = &sendBuffer[(k+l*r)*recvSize];
            recvSizes[k+l*r] = recvSize;

            const int thisColShift = Shift( k, colAlignment, r );
            const int thisLocalHeight = LocalLength( height, thisColShift, r );

            for( int j=0; j<thisLocalWidth; ++j )
                for( int i=0; i<thisLocalHeight; ++i )
                    data[i+j*thisLocalHeight] = 
                          A.LocalEntry(thisColShift+i*r,thisRowShift+j*c);
        }
    }

    // Reduce-scatter over each process col
    ReduceScatter
    ( sendBuffer, recvBuffer, recvSizes, MPI_SUM, grid.VCComm() );
    delete[] recvSizes;

    // Unpack our received data
    for( int j=0; j<localWidth; ++j )
        for( int i=0; i<localHeight; ++i )
            this->LocalEntry(i,j) = recvBuffer[i+j*localHeight];

    this->_auxMemory.Release();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::SumScatterUpdate
( T alpha, const DistMatrixBase<T,MC,Star>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SumScatterUpdate([MC,* ])");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    this->AssertSameSize( A );
#endif
    const Grid& grid = this->GetGrid();
    if( this->ColAlignment() == A.ColAlignment() )
    {
        if( this->Width() == 1 )
        {
            const int rowAlignment = this->RowAlignment();
            const int myCol = grid.MRRank();

            const int localHeight = this->LocalHeight();

            const int portionSize = max(localHeight,MinCollectContrib);

            this->_auxMemory.Require( 2*portionSize );

            T* buffer = this->_auxMemory.Buffer();
            T* sendBuffer = &buffer[0];
            T* recvBuffer = &buffer[portionSize];

            // Pack 
            for( int i=0; i<localHeight; ++i )
                sendBuffer[i] = A.LocalEntry(i,0);
        
            // Reduce to rowAlignment
            Reduce
            ( sendBuffer, recvBuffer, portionSize, 
              MPI_SUM, rowAlignment, grid.MRComm() );

            if( myCol == rowAlignment )
            {
                for( int i=0; i<localHeight; ++i )
                    this->LocalEntry(i,0) += alpha*recvBuffer[i];
            }

            this->_auxMemory.Release();
        }
        else
        {
            const int c = grid.Width();
            const int rowAlignment = this->RowAlignment();

            const int width = this->Width();
            const int localHeight = this->LocalHeight();
            const int localWidth = this->LocalWidth();
            const int maxLocalWidth = MaxLocalLength(width,c);

            const int portionSize = 
                max(localHeight*maxLocalWidth,MinCollectContrib);

            this->_auxMemory.Require( (c+1)*portionSize );

            T* buffer = this->_auxMemory.Buffer();
            T* sendBuffer = &buffer[0];
            T* recvBuffer = &buffer[c*portionSize];

            // Pack 
            int* recvSizes = new int[c];
            for( int k=0; k<c; ++k )
            {
                T* data = &sendBuffer[k*portionSize];
                recvSizes[k] = portionSize;

                const int thisRowShift = Shift( k, rowAlignment, c );
                const int thisLocalWidth = LocalLength(width,thisRowShift,c);

                for( int j=0; j<thisLocalWidth; ++j )
                    for( int i=0; i<localHeight; ++i )
                        data[i+j*localHeight] = 
                            A.LocalEntry(i,thisRowShift+j*c);
            }
        
            // Reduce-scatter over each process row
            ReduceScatter
            ( sendBuffer, recvBuffer, recvSizes, MPI_SUM, grid.MRComm() );
            delete[] recvSizes;

            // Update with our received data
            for( int j=0; j<localWidth; ++j )
                for( int i=0; i<localHeight; ++i )
                    this->LocalEntry(i,j) += alpha*recvBuffer[i+j*localHeight];

            this->_auxMemory.Release();
        }
    }
    else
    {
#ifndef RELEASE
        if( grid.VCRank() == 0 )
            cout << "Unaligned SumScatterUpdate [MC,MR] <- [MC,* ]." << endl;
#endif
        if( this->Width() == 1 )
        {
            const int r = grid.Height();
            const int rowAlignment = this->RowAlignment();
            const int myRow = grid.MCRank();
            const int myCol = grid.MRRank();

            const int height = this->Height();
            const int localHeight = this->LocalHeight();
            const int localHeightOfA = A.LocalHeight();
            const int maxLocalHeight = MaxLocalLength(height,r);

            const int portionSize = max(maxLocalHeight,MinCollectContrib);

            const int colAlignment = this->ColAlignment();
            const int colAlignmentOfA = A.ColAlignment();
            const int sendRow = (myRow+r+colAlignment-colAlignmentOfA) % r;
            const int recvRow = (myRow+r+colAlignmentOfA-colAlignment) % r;

            this->_auxMemory.Require( 2*portionSize );

            T* buffer = this->_auxMemory.Buffer();
            T* sendBuffer = &buffer[0];
            T* recvBuffer = &buffer[portionSize];

            // Pack 
            for( int i=0; i<localHeightOfA; ++i )
                sendBuffer[i] = A.LocalEntry(i,0);
        
            // Reduce to rowAlignment
            Reduce
            ( sendBuffer, recvBuffer, portionSize, 
              MPI_SUM, rowAlignment, grid.MRComm() );

            if( myCol == rowAlignment )
            {
                // Perform the realignment
                SendRecv
                ( recvBuffer, portionSize, sendRow, 0,
                  sendBuffer, portionSize, recvRow, 0, grid.MCComm() );

                for( int i=0; i<localHeight; ++i )
                    this->LocalEntry(i,0) += alpha*sendBuffer[i];
            }

            this->_auxMemory.Release();
        }
        else
        {
            const int r = grid.Height();
            const int c = grid.Width();
            const int row = grid.MCRank();

            const int colAlignment = this->ColAlignment();
            const int rowAlignment = this->RowAlignment();
            const int colAlignmentOfA = A.ColAlignment();
            const int sendRow = (row+r+colAlignment-colAlignmentOfA) % r;
            const int recvRow = (row+r+colAlignmentOfA-colAlignment) % r;

            const int width = this->Width();
            const int localHeight = this->LocalHeight();
            const int localWidth = this->LocalWidth();
            const int localHeightOfA = A.LocalHeight();
            const int maxLocalWidth = MaxLocalLength(width,c);

            const int recvSize_RS = 
                max(localHeightOfA*maxLocalWidth,MinCollectContrib);
            const int sendSize_RS = c * recvSize_RS;
            const int recvSize_SR = localHeight * localWidth;

            this->_auxMemory.Require
            ( recvSize_RS + max(sendSize_RS,recvSize_SR) );

            T* buffer = this->_auxMemory.Buffer();
            T* firstBuffer = &buffer[0];
            T* secondBuffer = &buffer[recvSize_RS];

            // Pack 
            int* recvSizes = new int[c];
            for( int k=0; k<c; ++k )
            {
                T* data = &secondBuffer[k*recvSize_RS];
                recvSizes[k] = recvSize_RS;

                const int thisRowShift = Shift( k, rowAlignment, c );
                const int thisLocalWidth = LocalLength(width,thisRowShift,c);

                for( int j=0; j<thisLocalWidth; ++j )
                    for( int i=0; i<localHeightOfA; ++i )
                        data[i+j*localHeightOfA] = 
                            A.LocalEntry(i,thisRowShift+j*c);
            }

            // Reduce-scatter over each process row
            ReduceScatter
            ( secondBuffer, firstBuffer, recvSizes, MPI_SUM, grid.MRComm() );
            delete[] recvSizes;

            // Trade reduced data with the appropriate process row
            SendRecv
            ( firstBuffer,  localHeightOfA*localWidth, sendRow, 0,
              secondBuffer, localHeight*localWidth,    recvRow, 0, 
              grid.MCComm() );

            // Update with our received data
            for( int j=0; j<localWidth; ++j )
                for( int i=0; i<localHeight; ++i )
                    this->LocalEntry(i,j) += 
                        alpha*secondBuffer[i+j*localHeight];

            this->_auxMemory.Release();
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::SumScatterUpdate
( T alpha, const DistMatrixBase<T,Star,MR>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SumScatterUpdate([* ,MR])");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    this->AssertSameSize( A );
#endif
    const Grid& grid = this->GetGrid();
    if( this->RowAlignment() == A.RowAlignment() )
    {
        const int r = grid.Height();
        const int colAlignment = this->ColAlignment();

        const int height = this->Height();
        const int localHeight = this->LocalHeight();
        const int localWidth = this->LocalWidth();
        const int maxLocalHeight = MaxLocalLength(height,r);

        const int recvSize = max(maxLocalHeight*localWidth,MinCollectContrib);
        const int sendSize = r * recvSize;

        this->_auxMemory.Require( sendSize + recvSize );

        T* buffer = this->_auxMemory.Buffer();
        T* sendBuffer = &buffer[0];
        T* recvBuffer = &buffer[sendSize];

        // Pack 
        int* recvSizes = new int[r];
        for( int k=0; k<r; ++k )
        {
            T* data = &sendBuffer[k*recvSize];
            recvSizes[k] = recvSize;

            const int thisColShift = Shift( k, colAlignment, r );
            const int thisLocalHeight = LocalLength( height, thisColShift, r );

            for( int j=0; j<localWidth; ++j )
                for( int i=0; i<thisLocalHeight; ++i )
                    data[i+j*thisLocalHeight] = 
                          A.LocalEntry(thisColShift+i*r,j);
        }

        // Reduce-scatter over each process col
        ReduceScatter
        ( sendBuffer, recvBuffer, recvSizes, MPI_SUM, grid.MCComm() );
        delete[] recvSizes;

        // Update with our received data
        for( int j=0; j<localWidth; ++j )
            for( int i=0; i<localHeight; ++i )
                this->LocalEntry(i,j) += alpha*recvBuffer[i+j*localHeight];

        this->_auxMemory.Release();
    }
    else
    {
#ifndef RELEASE
        if( grid.VCRank() == 0 )
            cout << "Unaligned SumScatterUpdate [MC,MR] <- [* ,MR]." << endl;
#endif
        const int r = grid.Height();
        const int c = grid.Width();
        const int col = grid.MRRank();

        const int colAlignment = this->ColAlignment();
        const int rowAlignment = this->RowAlignment();
        const int rowAlignmentOfA = A.RowAlignment();
        const int sendCol = (col+c+rowAlignment-rowAlignmentOfA) % c;
        const int recvCol = (col+c+rowAlignmentOfA-rowAlignment) % c;

        const int height = this->Height();
        const int localHeight = this->LocalHeight();
        const int localWidth = this->LocalWidth();
        const int localWidthOfA = A.LocalWidth();
        const int maxLocalHeight = MaxLocalLength(height,r);

        const int recvSize_RS = 
            max(maxLocalHeight*localWidthOfA,MinCollectContrib);
        const int sendSize_RS = r * recvSize_RS;
        const int recvSize_SR = localHeight * localWidth;

        this->_auxMemory.Require( recvSize_RS + max(sendSize_RS,recvSize_SR) );

        T* buffer = this->_auxMemory.Buffer();
        T* firstBuffer = &buffer[0];
        T* secondBuffer = &buffer[recvSize_RS];

        // Pack
        int* recvSizes = new int[r];
        for( int k=0; k<r; ++k )
        {
            T* data = &secondBuffer[k*recvSize_RS];
            recvSizes[k] = recvSize_RS;

            const int thisColShift = Shift( k, colAlignment, r );
            const int thisLocalHeight = LocalLength( height, thisColShift, r );

            for( int j=0; j<localWidthOfA; ++j )
                for( int i=0; i<thisLocalHeight; ++i )
                    data[i+j*thisLocalHeight] = 
                          A.LocalEntry(thisColShift+i*r,j);
        }

        // Reduce-scatter over each process col
        ReduceScatter
        ( secondBuffer, firstBuffer, recvSizes, MPI_SUM, grid.MCComm() );
        delete[] recvSizes;

        // Trade reduced data with the appropriate process col
        SendRecv
        ( firstBuffer,  localHeight*localWidthOfA, sendCol, 0,
          secondBuffer, localHeight*localWidth,    recvCol, MPI_ANY_TAG,
          grid.MRComm() );

        // Update with our received data
        for( int j=0; j<localWidth; ++j )
            for( int i=0; i<localHeight; ++i )
                this->LocalEntry(i,j) += alpha*secondBuffer[i+j*localHeight];

        this->_auxMemory.Release();
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename T>
void
elemental::DistMatrixBase<T,MC,MR>::SumScatterUpdate
( T alpha, const DistMatrixBase<T,Star,Star>& A )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SumScatterUpdate([* ,* ])");
    this->AssertNotLockedView();
    this->AssertSameGrid( A );
    if( this->Viewing() )
        this->AssertSameSize( A );
#endif
    if( !this->Viewing() )
        this->ResizeTo( A.Height(), A.Width() );

    const Grid& grid = this->GetGrid();
    const int r = grid.Height();
    const int c = grid.Width();
    const int colAlignment = this->ColAlignment();
    const int rowAlignment = this->RowAlignment();

    const int height = this->Height();
    const int width = this->Width();
    const int localHeight = this->LocalHeight();
    const int localWidth = this->LocalWidth();
    const int maxLocalHeight = MaxLocalLength(height,r);
    const int maxLocalWidth = MaxLocalLength(width,c);

    const int recvSize = max(maxLocalHeight*maxLocalWidth,MinCollectContrib);
    const int sendSize = r * c * recvSize;

    this->_auxMemory.Require( sendSize + recvSize );

    T* buffer = this->_auxMemory.Buffer();
    T* sendBuffer = &buffer[0];
    T* recvBuffer = &buffer[sendSize];

    // Pack 
    int* recvSizes = new int[r*c];
    for( int l=0; l<c; ++l )
    {
        const int thisRowShift = Shift( l, rowAlignment, c );
        const int thisLocalWidth = LocalLength( width, thisRowShift, c );

        for( int k=0; k<r; ++k )
        {
            T* data = &sendBuffer[(k+l*r)*recvSize];
            recvSizes[k+l*r] = recvSize;

            const int thisColShift = Shift( k, colAlignment, r );
            const int thisLocalHeight = LocalLength( height, thisColShift, r );

            for( int j=0; j<thisLocalWidth; ++j )
                for( int i=0; i<thisLocalHeight; ++i )
                    data[i+j*thisLocalHeight] = 
                          A.LocalEntry(thisColShift+i*r,thisRowShift+j*c);
        }
    }

    // Reduce-scatter over each process col
    ReduceScatter
    ( sendBuffer, recvBuffer, recvSizes, MPI_SUM, grid.VCComm() );
    delete[] recvSizes;

    // Unpack our received data
    for( int j=0; j<localWidth; ++j )
        for( int i=0; i<localHeight; ++i )
            this->LocalEntry(i,j) += alpha*recvBuffer[i+j*localHeight];

    this->_auxMemory.Release();
#ifndef RELEASE
    PopCallStack();
#endif
}

//----------------------------------------------------------------------------//
// DistMatrix                                                                 //
//----------------------------------------------------------------------------//

template<typename R>
void
elemental::DistMatrix<R,MC,MR>::SetToRandomHPD()
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SetToRandomHPD");
    this->AssertNotLockedView();
    if( this->Height() != this->Width() )
        throw logic_error( "Positive-definite matrices must be square." );
#endif
    const int r = this->GetGrid().Height();
    const int c = this->GetGrid().Width();

    const int localHeight = this->LocalHeight();
    const int localWidth = this->LocalWidth();
    const int colShift = this->ColShift();
    const int rowShift = this->RowShift();

    this->SetToRandom();
    for( int iLoc=0; iLoc<localHeight; ++iLoc )
    {
        const int i = colShift + iLoc*r;                
        if( i % c == rowShift )
        {
            const int jLoc = (i-rowShift) / c;
            if( jLoc < localWidth )
                this->LocalEntry(iLoc,jLoc) += (R)this->Width();
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

#ifndef WITHOUT_COMPLEX
template<typename R>
void
elemental::DistMatrix<complex<R>,MC,MR>::SetToRandomHPD()
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SetToRandomHPD");
    this->AssertNotLockedView();
    if( this->Height() != this->Width() )
        throw logic_error( "Positive-definite matrices must be square." );
#endif
    const int r = this->GetGrid().Height();
    const int c = this->GetGrid().Width();

    const int localHeight = this->LocalHeight();
    const int localWidth = this->LocalWidth();
    const int colShift = this->ColShift();
    const int rowShift = this->RowShift();

    this->SetToRandom();
    for( int iLoc=0; iLoc<localHeight; ++iLoc )
    {
        const int i = colShift + iLoc*r;                
        if( i % c == rowShift )
        {
            const int jLoc = (i-rowShift) / c;
            if( jLoc < localWidth )
            {
                this->LocalEntry(iLoc,jLoc) = 
                    real(this->LocalEntry(iLoc,jLoc)) + (R)this->Width();
            }
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R>
R
elemental::DistMatrix<complex<R>,MC,MR>::GetReal
( int i, int j ) const
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::GetReal");
    this->AssertValidEntry( i, j );
#endif
    // We will determine the owner of the (i,j) entry and have him Broadcast
    // throughout the entire process grid
    const Grid& grid = this->GetGrid();
    const int ownerRow = (i + this->ColAlignment()) % grid.Height();
    const int ownerCol = (j + this->RowAlignment()) % grid.Width();
    const int ownerRank = ownerRow + ownerCol * grid.Height();

    R u;
    if( grid.VCRank() == ownerRank )
    {
        const int iLoc = (i-this->ColShift()) / grid.Height();
        const int jLoc = (j-this->RowShift()) / grid.Width();
        u = real(this->LocalEntry(iLoc,jLoc));
    }
    Broadcast( &u, 1, ownerRank, grid.VCComm() );

#ifndef RELEASE
    PopCallStack();
#endif
    return u;
}

template<typename R>
R
elemental::DistMatrix<complex<R>,MC,MR>::GetImag
( int i, int j ) const
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::GetImag");
    this->AssertValidEntry( i, j );
#endif
    // We will determine the owner of the (i,j) entry and have him Broadcast
    // throughout the entire process grid
    const Grid& grid = this->GetGrid();
    const int ownerRow = (i + this->ColAlignment()) % grid.Height();
    const int ownerCol = (j + this->RowAlignment()) % grid.Width();
    const int ownerRank = ownerRow + ownerCol * grid.Height();

    R u;
    if( grid.VCRank() == ownerRank )
    {
        const int iLoc = (i-this->ColShift()) / grid.Height();
        const int jLoc = (j-this->RowShift()) / grid.Width();
        u = imag(this->LocalEntry(iLoc,jLoc));
    }
    Broadcast( &u, 1, ownerRank, grid.VCComm() );

#ifndef RELEASE
    PopCallStack();
#endif
    return u;
}

template<typename R>
void
elemental::DistMatrix<complex<R>,MC,MR>::SetReal
( int i, int j, R u )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SetReal");
    this->AssertValidEntry( i, j );
#endif
    const Grid& grid = this->GetGrid();
    const int ownerRow = (i + this->ColAlignment()) % grid.Height();
    const int ownerCol = (j + this->RowAlignment()) % grid.Width();
    const int ownerRank = ownerRow + ownerCol * grid.Height();

    if( grid.VCRank() == ownerRank )
    {
        const int iLoc = (i-this->ColShift()) / grid.Height();
        const int jLoc = (j-this->RowShift()) / grid.Width();
        real(this->LocalEntry(iLoc,jLoc)) = u;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R>
void
elemental::DistMatrix<complex<R>,MC,MR>::SetImag
( int i, int j, R u )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SetImag");
    this->AssertValidEntry( i, j );
#endif
    const Grid& grid = this->GetGrid();
    const int ownerRow = (i + this->ColAlignment()) % grid.Height();
    const int ownerCol = (j + this->RowAlignment()) % grid.Width();
    const int ownerRank = ownerRow + ownerCol * grid.Height();

    if( grid.VCRank() == ownerRank )
    {
        const int iLoc = (i-this->ColShift()) / grid.Height();
        const int jLoc = (j-this->RowShift()) / grid.Width();
        imag(this->LocalEntry(iLoc,jLoc)) = u;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R>
void
elemental::DistMatrix<complex<R>,MC,MR>::GetRealDiagonal
( DistMatrix<R,MD,Star>& d, int offset )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::GetRealDiagonal");
    this->AssertNotLockedView();
#endif
    int width = this->Width();
    int height = this->Height();
    int length;
    if( offset > 0 )
    {
        const int remainingWidth = max(width-offset,0);
        length = min(height,remainingWidth);
    }
    else
    {
        const int remainingHeight = max(height+offset,0);
        length = min(remainingHeight,width);
    }
#ifndef RELEASE
    if( d.Viewing() && length != d.Height() )
    {
        ostringstream msg;
        msg << "d is not of the same length as the diagonal:" << endl
            << "  A ~ " << this->Height() << " x " << this->Width() << endl
            << "  d ~ " << d.Height() << " x " << d.Width() << endl
            << "  A diag length: " << length << endl;
        throw logic_error( msg.str() );
    }
#endif

    if( !d.Viewing() )
    {
        if( !d.ConstrainedColAlignment() )
        {
            d.AlignWithDiag( *this, offset );
        }
        d.ResizeTo( length, 1 );
    }

    if( d.InDiagonal() )
    {
        const Grid& grid = this->GetGrid();
        const int r = grid.Height();
        const int c = grid.Width();
        const int lcm = grid.LCM();
        const int colShift = this->ColShift();
        const int rowShift = this->RowShift();
        const int diagShift = d.ColShift();

        int iStart, jStart;
        if( offset >= 0 )
        {
            iStart = diagShift;
            jStart = diagShift+offset;
        }
        else
        {
            iStart = diagShift-offset;
            jStart = diagShift;
        }

        const int iLocStart = (iStart-colShift) / r;
        const int jLocStart = (jStart-rowShift) / c;

        const int localDiagLength = d.LocalHeight();
        for( int k=0; k<localDiagLength; ++k )
            d.LocalEntry(k,0) = 
                real(this->LocalEntry(iLocStart+k*(lcm/r),jLocStart+k*(lcm/c)));
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R>
void
elemental::DistMatrix<complex<R>,MC,MR>::GetImagDiagonal
( DistMatrix<R,MD,Star>& d, int offset )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::GetImagDiagonal");
    this->AssertNotLockedView();
#endif
    int width = this->Width();
    int height = this->Height();
    int length;
    if( offset > 0 )
    {
        const int remainingWidth = max(width-offset,0);
        length = min(height,remainingWidth);
    }
    else
    {
        const int remainingHeight = max(height+offset,0);
        length = min(remainingHeight,width);
    }
#ifndef RELEASE
    if( d.Viewing() && length != d.Height() )
    {
        ostringstream msg;
        msg << "d is not of the same length as the diagonal:" << endl
            << "  A ~ " << this->Height() << " x " << this->Width() << endl
            << "  d ~ " << d.Height() << " x " << d.Width() << endl
            << "  A diag length: " << length << endl;
        throw logic_error( msg.str() );
    }
#endif

    if( !d.Viewing() )
    {
        if( !d.ConstrainedColAlignment() )
        {
            d.AlignWithDiag( *this, offset );
        }
        d.ResizeTo( length, 1 );
    }

    if( d.InDiagonal() )
    {
        const Grid& grid = this->GetGrid();
        const int r = grid.Height();
        const int c = grid.Width();
        const int lcm = grid.LCM();
        const int colShift = this->ColShift();
        const int rowShift = this->RowShift();
        const int diagShift = d.ColShift();

        int iStart, jStart;
        if( offset >= 0 )
        {
            iStart = diagShift;
            jStart = diagShift+offset;
        }
        else
        {
            iStart = diagShift-offset;
            jStart = diagShift;
        }

        const int iLocStart = (iStart-colShift) / r;
        const int jLocStart = (jStart-rowShift) / c;

        const int localDiagLength = d.LocalHeight();
        for( int k=0; k<localDiagLength; ++k )
            d.LocalEntry(k,0) = 
                imag(this->LocalEntry(iLocStart+k*(lcm/r),jLocStart+k*(lcm/c)));
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R>
void
elemental::DistMatrix<complex<R>,MC,MR>::GetRealDiagonal
( DistMatrix<R,Star,MD>& d, int offset )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::GetRealDiagonal");
    this->AssertNotLockedView();
#endif
    int height = this->Height();
    int width = this->Width();
    int length;
    if( offset > 0 )
    {
        const int remainingWidth = max(width-offset,0);
        length = min(height,remainingWidth);
    }
    else
    {
        const int remainingHeight = max(height+offset,0);
        length = min(remainingHeight,width);
    }
#ifndef RELEASE
    if( d.Viewing() && length != d.Width() )
    {
        ostringstream msg;
        msg << "d is not of the same length as the diagonal:" << endl
            << "  A ~ " << this->Height() << " x " << this->Width() << endl
            << "  d ~ " << d.Height() << " x " << d.Width() << endl
            << "  A diag length: " << length << endl;
        throw logic_error( msg.str() );
    }
#endif

    if( !d.Viewing() )
    {
        if( !d.ConstrainedRowAlignment() )
        {
            d.AlignWithDiag( *this, offset );
        }
        d.ResizeTo( 1, length );
    }

    if( d.InDiagonal() )
    {
        const Grid& grid = this->GetGrid();
        const int r = grid.Height();
        const int c = grid.Width();
        const int lcm = grid.LCM();
        const int colShift = this->ColShift();
        const int rowShift = this->RowShift();
        const int diagShift = d.RowShift();

        int iStart, jStart;
        if( offset >= 0 )
        {
            iStart = diagShift;
            jStart = diagShift+offset;
        }
        else
        {
            iStart = diagShift-offset;
            jStart = diagShift;
        }

        const int iLocStart = (iStart-colShift) / r;
        const int jLocStart = (jStart-rowShift) / c;

        const int localDiagLength = d.LocalWidth();
        for( int k=0; k<localDiagLength; ++k )
            d.LocalEntry(0,k) = 
                real(this->LocalEntry(iLocStart+k*(lcm/r),jLocStart+k*(lcm/c)));
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R>
void
elemental::DistMatrix<complex<R>,MC,MR>::GetImagDiagonal
( DistMatrix<R,Star,MD>& d, int offset )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::GetImagDiagonal");
    this->AssertNotLockedView();
#endif
    int height = this->Height();
    int width = this->Width();
    int length;
    if( offset > 0 )
    {
        const int remainingWidth = max(width-offset,0);
        length = min(height,remainingWidth);
    }
    else
    {
        const int remainingHeight = max(height+offset,0);
        length = min(remainingHeight,width);
    }
#ifndef RELEASE
    if( d.Viewing() && length != d.Width() )
    {
        ostringstream msg;
        msg << "d is not of the same length as the diagonal:" << endl
            << "  A ~ " << this->Height() << " x " << this->Width() << endl
            << "  d ~ " << d.Height() << " x " << d.Width() << endl
            << "  A diag length: " << length << endl;
        throw logic_error( msg.str() );
    }
#endif

    if( !d.Viewing() )
    {
        if( !d.ConstrainedRowAlignment() )
        {
            d.AlignWithDiag( *this, offset );
        }
        d.ResizeTo( 1, length );
    }

    if( d.InDiagonal() )
    {
        const Grid& grid = this->GetGrid();
        const int r = grid.Height();
        const int c = grid.Width();
        const int lcm = grid.LCM();
        const int colShift = this->ColShift();
        const int rowShift = this->RowShift();
        const int diagShift = d.RowShift();

        int iStart, jStart;
        if( offset >= 0 )
        {
            iStart = diagShift;
            jStart = diagShift+offset;
        }
        else
        {
            iStart = diagShift-offset;
            jStart = diagShift;
        }

        const int iLocStart = (iStart-colShift) / r;
        const int jLocStart = (jStart-rowShift) / c;

        const int localDiagLength = d.LocalWidth();
        for( int k=0; k<localDiagLength; ++k )
            d.LocalEntry(0,k) = 
                imag(this->LocalEntry(iLocStart+k*(lcm/r),jLocStart+k*(lcm/c)));
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R>
void
elemental::DistMatrix<complex<R>,MC,MR>::SetDiagonal
( const DistMatrixBase<R,MD,Star>& d, int offset )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SetDiagonal");
    if( d.Width() != 1 )
        throw logic_error( "d must be a column vector." );
    {
        int height = this->Height();
        int width = this->Width();
        int length;
        if( offset >= 0 )
        {
            const int remainingWidth = max(width-offset,0);
            length = min(remainingWidth,height);
        }
        else
        {
            const int remainingHeight = max(height+offset,0);
            length = min(remainingHeight,width);
        }
        if( length != d.Height() )
        {
            ostringstream msg;
            msg << "d is not of the same length as the diagonal:" << endl
                << "  A ~ " << this->Height() << " x " << this->Width() << endl
                << "  d ~ " << d.Height() << " x " << d.Width() << endl
                << "  A diag length: " << length << endl;
            throw logic_error( msg.str() );
        }
    }
#endif
    if( d.InDiagonal() )
    {
        const Grid& grid = this->GetGrid();
        const int r = grid.Height();
        const int c = grid.Width();
        const int lcm = grid.LCM();
        const int colShift = this->ColShift();
        const int rowShift = this->RowShift();
        const int diagShift = d.ColShift();

        int iStart,jStart;
        if( offset >= 0 )
        {
            iStart = diagShift;
            jStart = diagShift+offset;
        }
        else
        {
            iStart = diagShift-offset;
            jStart = diagShift;
        }

        const int iLocStart = (iStart-colShift) / r;
        const int jLocStart = (jStart-rowShift) / c;

        const int localDiagLength = d.LocalHeight();
        for( int k=0; k<localDiagLength; ++k )
            this->LocalEntry(iLocStart+k*(lcm/r),jLocStart+k*(lcm/c)) = 
                    d.LocalEntry(k,0);
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R>
void
elemental::DistMatrix<complex<R>,MC,MR>::SetRealDiagonal
( const DistMatrixBase<R,MD,Star>& d, int offset )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SetRealDiagonal");
    if( d.Width() != 1 )
        throw logic_error( "d must be a column vector." );
    {
        int height = this->Height();
        int width = this->Width();
        int length;
        if( offset >= 0 )
        {
            const int remainingWidth = max(width-offset,0);
            length = min(remainingWidth,height);
        }
        else
        {
            const int remainingHeight = max(height+offset,0);
            length = min(remainingHeight,width);
        }
        if( length != d.Height() )
        {
            ostringstream msg;
            msg << "d is not of the same length as the diagonal:" << endl
                << "  A ~ " << this->Height() << " x " << this->Width() << endl
                << "  d ~ " << d.Height() << " x " << d.Width() << endl
                << "  A diag length: " << length << endl;
            throw logic_error( msg.str() );
        }
    }
#endif
    if( d.InDiagonal() )
    {
        const Grid& grid = this->GetGrid();
        const int r = grid.Height();
        const int c = grid.Width();
        const int lcm = grid.LCM();
        const int colShift = this->ColShift();
        const int rowShift = this->RowShift();
        const int diagShift = d.ColShift();

        int iStart,jStart;
        if( offset >= 0 )
        {
            iStart = diagShift;
            jStart = diagShift+offset;
        }
        else
        {
            iStart = diagShift-offset;
            jStart = diagShift;
        }

        const int iLocStart = (iStart-colShift) / r;
        const int jLocStart = (jStart-rowShift) / c;

        const int localDiagLength = d.LocalHeight();
        for( int k=0; k<localDiagLength; ++k )
            real(this->LocalEntry(iLocStart+k*(lcm/r),jLocStart+k*(lcm/c))) = 
                    d.LocalEntry(k,0);
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R>
void
elemental::DistMatrix<complex<R>,MC,MR>::SetImagDiagonal
( const DistMatrixBase<R,MD,Star>& d, int offset )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SetImagDiagonal");
    if( d.Width() != 1 )
        throw logic_error( "d must be a column vector." );
    {
        int height = this->Height();
        int width = this->Width();
        int length;
        if( offset >= 0 )
        {
            const int remainingWidth = max(width-offset,0);
            length = min(remainingWidth,height);
        }
        else
        {
            const int remainingHeight = max(height+offset,0);
            length = min(remainingHeight,width);
        }
        if( length != d.Height() )
        {
            ostringstream msg;
            msg << "d is not of the same length as the diagonal:" << endl
                << "  A ~ " << this->Height() << " x " << this->Width() << endl
                << "  d ~ " << d.Height() << " x " << d.Width() << endl
                << "  A diag length: " << length << endl;
            throw logic_error( msg.str() );
        }
    }
#endif
    if( d.InDiagonal() )
    {
        const Grid& grid = this->GetGrid();
        const int r = grid.Height();
        const int c = grid.Width();
        const int lcm = grid.LCM();
        const int colShift = this->ColShift();
        const int rowShift = this->RowShift();
        const int diagShift = d.ColShift();

        int iStart,jStart;
        if( offset >= 0 )
        {
            iStart = diagShift;
            jStart = diagShift+offset;
        }
        else
        {
            iStart = diagShift-offset;
            jStart = diagShift;
        }

        const int iLocStart = (iStart-colShift) / r;
        const int jLocStart = (jStart-rowShift) / c;

        const int localDiagLength = d.LocalHeight();
        for( int k=0; k<localDiagLength; ++k )
            imag(this->LocalEntry(iLocStart+k*(lcm/r),jLocStart+k*(lcm/c))) = 
                d.LocalEntry(k,0);
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R>
void
elemental::DistMatrix<complex<R>,MC,MR>::SetDiagonal
( const DistMatrixBase<R,Star,MD>& d, int offset )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SetDiagonal");
    if( d.Height() != 1 )
        throw logic_error( "d must be a row vector." );
    {
        const int height = this->Height();
        const int width = this->Width();

        int length;
        if( offset >= 0 )
        {
            const int remainingWidth = max(width-offset,0);
            length = min(remainingWidth,height);
        }
        else
        {
            const int remainingHeight = max(height+offset,0);
            length = min(remainingHeight,width);
        }
        if( length != d.Width() )
        {
            ostringstream msg;
            msg << "d is not of the same length as the diagonal:" << endl
                << "  A ~ " << this->Height() << " x " << this->Width() << endl
                << "  d ~ " << d.Height() << " x " << d.Width() << endl
                << "  A diag length: " << length << endl;
            throw logic_error( msg.str() );
        }
    }
#endif
    if( d.InDiagonal() )
    {
        const Grid& grid = this->GetGrid();
        const int r = grid.Height();
        const int c = grid.Width();
        const int lcm = grid.LCM();
        const int colShift = this->ColShift();
        const int rowShift = this->RowShift();
        const int diagShift = d.RowShift();

        int iStart,jStart;
        if( offset >= 0 )
        {
            iStart = diagShift;
            jStart = diagShift+offset;
        }
        else
        {
            iStart = diagShift-offset;
            jStart = diagShift;
        }

        const int iLocStart = (iStart-colShift) / r;
        const int jLocStart = (jStart-rowShift) / c;

        const int localDiagLength = d.LocalWidth();
        for( int k=0; k<localDiagLength; ++k )
            this->LocalEntry(iLocStart+k*(lcm/r),jLocStart+k*(lcm/c)) = 
                d.LocalEntry(0,k);
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R>
void
elemental::DistMatrix<complex<R>,MC,MR>::SetRealDiagonal
( const DistMatrixBase<R,Star,MD>& d, int offset )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SetRealDiagonal");
    if( d.Height() != 1 )
        throw logic_error( "d must be a row vector." );
    {
        const int height = this->Height();
        const int width = this->Width();

        int length;
        if( offset >= 0 )
        {
            const int remainingWidth = max(width-offset,0);
            length = min(remainingWidth,height);
        }
        else
        {
            const int remainingHeight = max(height+offset,0);
            length = min(remainingHeight,width);
        }
        if( length != d.Width() )
        {
            ostringstream msg;
            msg << "d is not of the same length as the diagonal:" << endl
                << "  A ~ " << this->Height() << " x " << this->Width() << endl
                << "  d ~ " << d.Height() << " x " << d.Width() << endl
                << "  A diag length: " << length << endl;
            throw logic_error( msg.str() );
        }
    }
#endif
    if( d.InDiagonal() )
    {
        const Grid& grid = this->GetGrid();
        const int r = grid.Height();
        const int c = grid.Width();
        const int lcm = grid.LCM();
        const int colShift = this->ColShift();
        const int rowShift = this->RowShift();
        const int diagShift = d.RowShift();

        int iStart,jStart;
        if( offset >= 0 )
        {
            iStart = diagShift;
            jStart = diagShift+offset;
        }
        else
        {
            iStart = diagShift-offset;
            jStart = diagShift;
        }

        const int iLocStart = (iStart-colShift) / r;
        const int jLocStart = (jStart-rowShift) / c;

        const int localDiagLength = d.LocalWidth();
        for( int k=0; k<localDiagLength; ++k )
            real(this->LocalEntry(iLocStart+k*(lcm/r),jLocStart+k*(lcm/c))) = 
                d.LocalEntry(0,k);
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R>
void
elemental::DistMatrix<complex<R>,MC,MR>::SetImagDiagonal
( const DistMatrixBase<R,Star,MD>& d, int offset )
{
#ifndef RELEASE
    PushCallStack("[MC,MR]::SetImagDiagonal");
    if( d.Height() != 1 )
        throw logic_error( "d must be a row vector." );
    {
        const int height = this->Height();
        const int width = this->Width();

        int length;
        if( offset >= 0 )
        {
            const int remainingWidth = max(width-offset,0);
            length = min(remainingWidth,height);
        }
        else
        {
            const int remainingHeight = max(height+offset,0);
            length = min(remainingHeight,width);
        }
        if( length != d.Width() )
        {
            ostringstream msg;
            msg << "d is not of the same length as the diagonal:" << endl
                << "  A ~ " << this->Height() << " x " << this->Width() << endl
                << "  d ~ " << d.Height() << " x " << d.Width() << endl
                << "  A diag length: " << length << endl;
            throw logic_error( msg.str() );
        }
    }
#endif
    if( d.InDiagonal() )
    {
        const Grid& grid = this->GetGrid();
        const int r = grid.Height();
        const int c = grid.Width();
        const int lcm = grid.LCM();
        const int colShift = this->ColShift();
        const int rowShift = this->RowShift();
        const int diagShift = d.RowShift();

        int iStart,jStart;
        if( offset >= 0 )
        {
            iStart = diagShift;
            jStart = diagShift+offset;
        }
        else
        {
            iStart = diagShift-offset;
            jStart = diagShift;
        }

        const int iLocStart = (iStart-colShift) / r;
        const int jLocStart = (jStart-rowShift) / c;

        const int localDiagLength = d.LocalWidth();
        for( int k=0; k<localDiagLength; ++k )
            imag(this->LocalEntry(iLocStart+k*(lcm/r),jLocStart+k*(lcm/c))) = 
                d.LocalEntry(0,k);
    }
#ifndef RELEASE
    PopCallStack();
#endif
}
#endif // WITHOUT_COMPLEX

template class elemental::DistMatrixBase<int,   MC,MR>;
template class elemental::DistMatrixBase<float, MC,MR>;
template class elemental::DistMatrixBase<double,MC,MR>;
#ifndef WITHOUT_COMPLEX
template class elemental::DistMatrixBase<scomplex,MC,MR>;
template class elemental::DistMatrixBase<dcomplex,MC,MR>;
#endif

template class elemental::DistMatrix<int,     MC,MR>;
template class elemental::DistMatrix<float,   MC,MR>;
template class elemental::DistMatrix<double,  MC,MR>;
#ifndef WITHOUT_COMPLEX
template class elemental::DistMatrix<scomplex,MC,MR>;
template class elemental::DistMatrix<dcomplex,MC,MR>;
#endif

