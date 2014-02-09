/*
   Copyright (c) 2009-2014, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef ELEM_RQ_APPLYQ_HPP
#define ELEM_RQ_APPLYQ_HPP

#include ELEM_APPLYPACKEDREFLECTORS_INC

namespace elem {
namespace rq {

template<typename F>
inline void
ApplyQ
( LeftOrRight side, Orientation orientation, 
  const Matrix<F>& A, const Matrix<F>& t, Matrix<F>& B )
{
    DEBUG_ONLY(CallStackEntry cse("rq::ApplyQ"))
    const bool normal = (orientation==NORMAL);
    const bool onLeft = (side==LEFT);
    const ForwardOrBackward direction = ( normal==onLeft ? BACKWARD : FORWARD );
    const Conjugation conjugation = ( normal ? CONJUGATED : UNCONJUGATED );
    const Int offset = A.Width()-A.Height();
    ApplyPackedReflectors
    ( side, LOWER, HORIZONTAL, direction, conjugation, offset, A, t, B );
}

template<typename F>
inline void
ApplyQ
( LeftOrRight side, Orientation orientation, 
  const DistMatrix<F>& A, const DistMatrix<F,MD,STAR>& t, DistMatrix<F>& B )
{
    DEBUG_ONLY(CallStackEntry cse("rq::ApplyQ"))
    const bool normal = (orientation==NORMAL);
    const bool onLeft = (side==LEFT);
    const ForwardOrBackward direction = ( normal==onLeft ? BACKWARD : FORWARD );
    const Conjugation conjugation = ( normal ? CONJUGATED : UNCONJUGATED );
    const Int offset = A.Width()-A.Height();
    ApplyPackedReflectors
    ( side, LOWER, HORIZONTAL, direction, conjugation, offset, A, t, B );
}

template<typename F>
inline void
ApplyQ
( LeftOrRight side, Orientation orientation, 
  const DistMatrix<F>& A, const DistMatrix<F,STAR,STAR>& t, DistMatrix<F>& B )
{
    DEBUG_ONLY(CallStackEntry cse("rq::ApplyQ"))
    const Int offset = A.Width()-A.Height();
    DistMatrix<F,MD,STAR> tDiag(A.Grid());
    A.ForceDiagonalAlign( tDiag, offset ); 
    tDiag = t;
    ApplyQ( side, orientation, A, tDiag, B );
}

} // namespace rq
} // namespace elem

#endif // ifndef ELEM_RQ_APPLYQ_HPP
