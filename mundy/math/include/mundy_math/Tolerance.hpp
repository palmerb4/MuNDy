// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
//                                                 Author: Bryce Palmer
//
// Mundy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
//
// Mundy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with Mundy. If not, see
// <https://www.gnu.org/licenses/>.
//
// **********************************************************************************************************************
// @HEADER

#ifndef MUNDY_MATH_TOLERANCE_HPP_
#define MUNDY_MATH_TOLERANCE_HPP_

// C++ core includes
#include <Kokkos_Core.hpp>  // for KOKKOS_FUNCTION
#include <type_traits>      // for std::is_same_v

namespace mundy {

namespace math {

/// \brief Function to get the zero tolerance for a type. That is, the smallest value that we will consider non-zero.
/// We use approximately 10 * std::numeric_limits<T>::epsilon() as the default tolerance for floats and doubles and 0
/// for integer types. To make this code GPU-compatable, we'll directly evaluate the epsilon instead of using
/// std::numeric_limits.
///
/// \tparam T The type to get the tolerance for.
template <typename T>
KOKKOS_FUNCTION constexpr T get_zero_tolerance() {
  if constexpr (std::is_same_v<T, float>) {
    return 1e-06f;
  } else if constexpr (std::is_same_v<T, double>) {
    return 1e-15;
  } else {
    // For integral types, tolerance doesn't make sense, return 0
    return T(0);
  }
}

/// \brief Function to get the relaxed zero tolerance for a type. That is, the smallest value that we will consider
/// non-zero. We use approximately sqrt(std::numeric_limits<T>::epsilon()) as the default tolerance for floats and
/// doubles and 0 for integer types. To make this code GPU-compatable, we'll directly evaluate the epsilon instead of
/// using std::numeric_limits.
///
/// \tparam T The type to get the tolerance for.
template <typename T>
KOKKOS_FUNCTION constexpr T get_relaxed_zero_tolerance() {
  if constexpr (std::is_same_v<T, float>) {
    return 1e-3f;
  } else if constexpr (std::is_same_v<T, double>) {
    return 1e-8;
  } else {
    // For integral types, tolerance doesn't make sense, return 0
    return T(0);
  }
}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_TOLERANCE_HPP_
