/*
   This file is part of elemental, a library for distributed-memory dense 
   linear algebra.

   Copyright (C) 2009-2010 Jack Poulson <jack.poulson@gmail.com>

   This program is released under the terms of the license contained in the 
   file LICENSE.
*/
#include "elemental/environment.hpp"
#include "elemental/wrappers/mpi.hpp"
using namespace elemental;
using namespace std;

elemental::Grid::Grid
( MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("Grid::Grid(comm)");
#endif

    // Extract the total number of processes
    MPI_Comm_size( comm, &_p );

    // Factor p
    int r = static_cast<int>(sqrt(static_cast<double>(_p)));
    while( _p % r != 0 )
        ++r;
    int c = _p / r;

    if( _p != r*c )
    {
        ostringstream msg;
        msg << "Number of processes must match grid size:" << endl
            << "  p=" << _p << ", (r,c)=(" << r << "," << c << ")" << endl;
        throw logic_error( msg.str() );
    }

    // Create cartesian communicator
    int dimensions[2] = { r,    c    };
    int periods[2]    = { true, true };
    int reorder       = false;
    MPI_Cart_create( comm, 2, dimensions, periods, reorder, &_comm );
    MPI_Comm_rank( _comm, &_rank  );

    Init( r, c );

#ifndef RELEASE
    PopCallStack();
#endif
}

// Currently forces a columnMajor absolute rank on the grid
elemental::Grid::Grid
( MPI_Comm comm, int r, int c )
{
#ifndef RELEASE
    PushCallStack("Grid::Grid( comm, r, c )");
#endif

    // Extract the total number of processes
    MPI_Comm_size( comm, &_p );

    if( _p != r*c )
    {
        ostringstream msg;
        msg << "Number of processes must match grid size:" << endl
            << "  p=" << _p << ", (r,c)=(" << r << "," << c << ")" << endl;
        throw logic_error( msg.str() );
    }

    // Create a cartesian communicator
    int dimensions[2] = { r,   c     };
    int periods[2]    = { true, true };
    int reorder       = false;
    MPI_Cart_create( comm, 2, dimensions, periods, reorder, &_comm );
    MPI_Comm_rank( _comm, &_rank );
    
    Init( r, c );

#ifndef RELEASE
    PopCallStack();
#endif
}

void
elemental::Grid::Init
( int r, int c )
{
#ifndef RELEASE
    PushCallStack("Grid::Init(r,c)");
    if( r <= 0 || c <= 0 )
        throw logic_error( "r and c must be positive." );
#endif
    _r = r;
    _c = c;

    _gcd = utilities::GCD( r, c );
    int lcm = _p / _gcd;

#ifndef RELEASE
    if( _rank == 0 )
    {
        cout << "Building process grid with:" << endl;   
        cout << "  p=" << _p << ", (r,c)=(" << r << "," << c << ")" << endl;
        cout << "  gcd=" << _gcd << endl;
    }
#endif

    // Set up the MatrixCol and MatrixRow communicators
    int remainingDimensions[2];
    remainingDimensions[0] = true;
    remainingDimensions[1] = false;
    MPI_Cart_sub( _comm, remainingDimensions, &_matrixColComm );
    remainingDimensions[0] = false;
    remainingDimensions[1] = true;
    MPI_Cart_sub( _comm, remainingDimensions, &_matrixRowComm );
    MPI_Comm_rank( _matrixColComm, &_matrixColRank );
    MPI_Comm_rank( _matrixRowComm, &_matrixRowRank );

    // Set up the VectorCol and VectorRow communicators
    int dimensions[2];
    int periods[2];
    int coordinates[2];
    MPI_Cart_get( _comm, 2, dimensions, periods, coordinates );
    int colMajorRank = coordinates[0] + dimensions[0]*coordinates[1];
    int rowMajorRank = coordinates[1] + dimensions[1]*coordinates[0];
    MPI_Comm_split( _comm, 0, colMajorRank, &_vectorColComm );
    MPI_Comm_split( _comm, 0, rowMajorRank, &_vectorRowComm );
    MPI_Comm_rank( _vectorColComm, &_vectorColRank );
    MPI_Comm_rank( _vectorRowComm, &_vectorRowRank );

    // Compute which diagonal 'path' we're in, and what our rank is, then
    // perform AllGather world to store everyone's info
    _diagPathsAndRanks = new int[2*_p];
    int* myDiagPathAndRank = new int[2];
    myDiagPathAndRank[0] = (_matrixRowRank+r-_matrixColRank) % _gcd;
    int diagPathRank = 0;
    int row = 0;
    int col = myDiagPathAndRank[0];
    for( int j=0; j<lcm; ++j )
    {
        if( row == _matrixColRank && col == _matrixRowRank )
        {
            myDiagPathAndRank[1] = diagPathRank;
            break;
        }
        else
        {
            row = (row + 1) % r;
            col = (col + 1) % c;
            ++diagPathRank;
        }
    }
    wrappers::mpi::AllGather
    ( myDiagPathAndRank, 2, _diagPathsAndRanks, 2, _vectorColComm );
    delete myDiagPathAndRank;

#ifndef RELEASE
    MPI_Errhandler_set( _matrixColComm, MPI_ERRORS_RETURN );
    MPI_Errhandler_set( _matrixRowComm, MPI_ERRORS_RETURN );
    MPI_Errhandler_set( _vectorColComm, MPI_ERRORS_RETURN );
    MPI_Errhandler_set( _vectorRowComm, MPI_ERRORS_RETURN );
#endif

#ifndef RELEASE
    PopCallStack();
#endif
}

