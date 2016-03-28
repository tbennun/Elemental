/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_NUMBER_THEORY_HPP
#define EL_NUMBER_THEORY_HPP

namespace El {

#ifdef EL_HAVE_MPC

BigInt ModSqrt( const BigInt& n, const BigInt& p );
void ModSqrt( const BigInt& n, const BigInt& p, BigInt& nSqrt );

int LegendreSymbol( const BigInt& n, const BigInt& p );
int JacobiSymbol( const BigInt& m, const BigInt& n );

enum Primality
{
  PRIME,
  PROBABLY_PRIME,
  PROBABLY_COMPOSITE,
  COMPOSITE
};

Primality MillerRabin( const BigInt& n, Int numReps=30 );

// Use a combination of trial divisions and Miller-Rabin 
// (with numReps representatives) to test for primality.
Primality PrimalityTest( const BigInt& n, Int numReps=30 );

// Return the first prime greater than n (with high likelihood)
BigInt NextProbablePrime( const BigInt& n, Int numReps=30 );
void NextProbablePrime( const BigInt& n, BigInt& nextPrime, Int numReps=30 );

namespace factor {

struct PollardRhoCtrl
{
    Int a0=1;
    Int a1=-1;
    unsigned long numSteps=1u;
    BigInt x0=BigInt(2);
    Int gcdDelay=100;
    Int numReps=30;
    bool progress=false;
    bool time=false;
};

vector<BigInt> PollardRho
( const BigInt& n,
  const PollardRhoCtrl& ctrl=PollardRhoCtrl() );

namespace pollard_rho {

BigInt FindDivisor
( const BigInt& n,
        Int a=1,
  const PollardRhoCtrl& ctrl=PollardRhoCtrl() );

} // namespace pollard_rho

struct PollardPMinusOneCtrl
{
    BigInt smoothness=BigInt(1000000);
    BigInt smoothnessSecond=BigInt(10000000);
    Int numReps=30;
    bool progress=false;
    bool time=false;
};

vector<BigInt> PollardPMinusOne
( const BigInt& n,
  const PollardPMinusOneCtrl& ctrl=PollardPMinusOneCtrl() );

namespace pollard_pm1 {

BigInt FindFactor
( const BigInt& n,
  const PollardPMinusOneCtrl& ctrl=PollardPMinusOneCtrl() );

} // namespace pollard_pm1

} // namespace factor

bool IsPrimitiveRoot
( const BigInt& primitive,
  const BigInt& p,
  const vector<BigInt>& pm1Factors,
        bool progress=false );
bool IsPrimitiveRoot
( const BigInt& primitive,
  const BigInt& p,
        bool progress=false,
  const factor::PollardRhoCtrl& ctrl=factor::PollardRhoCtrl() );

// Return a primitive root of a prime number p
BigInt PrimitiveRoot( const BigInt& p, Int numReps=30 );
void PrimitiveRoot( const BigInt& p, BigInt& primitive, Int numReps=30 );

namespace dlog {

struct PollardRhoCtrl
{
    BigInt a0=0;
    BigInt b0=0;
    bool multistage=true;
    factor::PollardRhoCtrl factorCtrl;

    bool progress=false;
    bool time=false;
};

// Return k such that r^k = q (mod p)
BigInt PollardRho
( const BigInt& q,
  const BigInt& r,
  const BigInt& p,
  const PollardRhoCtrl& ctrl=PollardRhoCtrl() );

} // namespace dlog

#endif // ifdef EL_HAVE_MPC

} // namespace El

#include <El/number_theory/ModSqrt.hpp>
#include <El/number_theory/LegendreSymbol.hpp>
#include <El/number_theory/JacobiSymbol.hpp>
#include <El/number_theory/MillerRabin.hpp>
#include <El/number_theory/PrimalityTest.hpp>
#include <El/number_theory/NextProbablePrime.hpp>
#include <El/number_theory/factor/PollardRho.hpp>
#include <El/number_theory/factor/PollardPMinusOne.hpp>
#include <El/number_theory/PrimitiveRoot.hpp>
#include <El/number_theory/dlog/PollardRho.hpp>

#endif // ifndef EL_NUMBER_THEORY_HPP