#ifndef EL_NUMERICTYPECONVERSION_HPP
#define EL_NUMERICTYPECONVERSION_HPP

/*!
  @file

  Utilities for converting between numeric types (float, double, int, half).
  This is non-trivial when using the half-precision type.

  Allows conversions from type F to type T using To<T>(F const &), for example
  To<half>(0) returns a half value of zero.
*/

#ifdef HYDROGEN_HAVE_HALF
#include <half.hpp>

template <typename F>
struct Caster
{
  static half_float::half Cast(F const& x) {
    return half_float::half_cast<half_float::half>(x);
  }
};


template <typename T, typename F>
T To(F const& x)
{
  return Caster<F>::Cast(x);
}

#else

template <typename T, typename F>
T To(F const& x)
{
  return static_cast<T>(x);
}

#endif


#endif // ifndef EL_NUMERICTYPECONVERSION_HPP
