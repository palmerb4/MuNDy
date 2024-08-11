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

#ifndef MUNDY_MATH_VECTOR3_HPP_
#define MUNDY_MATH_VECTOR3_HPP_

// External
#include <Kokkos_Core.hpp>

// C++ core
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <iostream>
#include <type_traits>
#include <utility>

// Mundy
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_math/Accessor.hpp>      // for mundy::math::ValidAccessor
#include <mundy_math/Array.hpp>         // for mundy::math::Array
#include <mundy_math/Tolerance.hpp>     // for mundy::math::get_zero_tolerance
#include <mundy_math/Vector.hpp>        // for mundy::math::Vector

namespace mundy {

namespace math {

/// \brief Class for a 3x1 vector with arithmetic entries
/// \tparam T The type of the entries.
/// \tparam Accessor The type of the accessor.
template <typename T, ValidAccessor<T> Accessor = Array<T, 3>, typename OwnershipType = Ownership::Owns>
  requires std::is_arithmetic_v<T>
using Vector3 = Vector<T, 3, Accessor, OwnershipType>;

template <typename T, ValidAccessor<T> Accessor = Array<T, 3>, typename OwnershipType = Ownership::Owns>
  requires std::is_arithmetic_v<T>
using Vector3View = Vector<T, 3, Accessor, Ownership::Views>;

//! \name Non-member functions
//@{

//! \name Special vector3 operations
//@{

/// \brief Cross product
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <typename U, typename T>
KOKKOS_INLINE_FUNCTION auto cross(const Vector3<U, auto, auto>& a,
                                  const Vector3<T, auto, auto>& b) -> Vector3<std::common_type_t<T, U>> {
  using CommonType = std::common_type_t<T, U>;
  Vector3<CommonType> result;
  result[0] = static_cast<CommonType>(a[1]) * static_cast<CommonType>(b[2]) -
              static_cast<CommonType>(a[2]) * static_cast<CommonType>(b[1]);
  result[1] = static_cast<CommonType>(a[2]) * static_cast<CommonType>(b[0]) -
              static_cast<CommonType>(a[0]) * static_cast<CommonType>(b[2]);
  result[2] = static_cast<CommonType>(a[0]) * static_cast<CommonType>(b[1]) -
              static_cast<CommonType>(a[1]) * static_cast<CommonType>(b[0]);
  return result;
}

/// \brief Element-wise product
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <typename U, typename T>
KOKKOS_INLINE_FUNCTION auto element_multiply(const Vector3<U, auto, auto>& a,
                                             const Vector3<T, auto, auto>& b) -> Vector3<std::common_type_t<T, U>> {
  using CommonType = std::common_type_t<T, U>;
  Vector3<CommonType> result;
  result[0]= static_cast<CommonType>(a[0]) * static_cast<CommonType>(b[0]);
  result[1]= static_cast<CommonType>(a[1]) * static_cast<CommonType>(b[1]);
  result[2]= static_cast<CommonType>(a[2]) * static_cast<CommonType>(b[2]);
  return result;
}
//@}

//! \name Vector3<T, Accessor> views
//@{

/// \brief A helper function to create a Vector3<T, Accessor> based on a given accessor.
/// \param[in] data The data accessor.
///
/// In practice, this function is syntactic sugar to avoid having to specify the template parameters
/// when creating a Vector3<T, Accessor> from a data accessor.
/// Instead of writing
/// \code
///   Vector3<T, Accessor> vec(data);
/// \endcode
/// you can write
/// \code
///   auto vec = get_vector3_view<T>(data);
/// \endcode
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION auto get_vector3_view(Accessor& data) {
  return Vector3View<T, Accessor>(data);
}

template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION auto get_vector3_view(Accessor&& data) {
  return Vector3View<T, Accessor>(std::forward<Accessor>(data));
}
//@}

//@}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_VECTOR3_HPP_
