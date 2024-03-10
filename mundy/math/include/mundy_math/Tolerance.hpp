// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2023 Flatiron Institute
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

// External libs
#include <Kokkos_Core.hpp>

// C++ core libs
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <iostream>
#include <type_traits>

// Our libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

namespace math {

/// \brief Function to get the default tolerance for a type.
/// \tparam T The type to get the tolerance for.
template <typename T>
constexpr T get_default_tolerance() {
  if constexpr (std::is_same_v<T, float>) {
    return 1e-6f;
  } else if constexpr (std::is_same_v<T, double>) {
    return 1e-6;
  } else {
    // For integral types, tolerance doesn't make sense, return 0
    return T(0);
  }
}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_TOLERANCE_HPP_
