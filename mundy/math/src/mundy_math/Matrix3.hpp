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

#ifndef MUNDY_MATH_MATRIX3_HPP_
#define MUNDY_MATH_MATRIX3_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core libs
#include <cmath>
#include <concepts>
#include <iostream>
#include <type_traits>  // for std::decay_t

// Our libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_math/Accessor.hpp>      // for mundy::math::ValidAccessor
#include <mundy_math/Array.hpp>         // for mundy::math::Array
#include <mundy_math/Matrix.hpp>        // for mundy::math::Matrix
#include <mundy_math/Tolerance.hpp>     // for mundy::math::get_zero_tolerance

namespace mundy {

namespace math {

/// \brief Class for a 3x3 matrix with arithmetic entries
/// \tparam T The type of the entries.
/// \tparam Accessor The type of the accessor.
template <typename T, ValidAccessor<T> Accessor = Array<T, 9>, typename OwnershipType = Ownership::Owns>
using Matrix3 = Matrix<T, 3, 3, Accessor, OwnershipType>;

template <typename T, ValidAccessor<T> Accessor = Array<T, 9>>
using Matrix3View = Matrix<T, 3, 3, Accessor, Ownership::Views>;

template <typename T, ValidAccessor<T> Accessor = Array<T, 9>>
using OwningMatrix3 = Matrix<T, 3, 3, Accessor, Ownership::Owns>;

/// \brief (Implementation) Type trait to determine if a type is a Matrix3
template <typename TypeToCheck>
struct is_matrix3_impl : std::false_type {};
//
template <typename T, typename Accessor, typename OwnershipType>
struct is_matrix3_impl<Matrix3<T, Accessor, OwnershipType>> : std::true_type {};

/// \brief Type trait to determine if a type is a Matrix3
template <typename T>
struct is_matrix3 : is_matrix3_impl<std::decay_t<T>> {};
//
template <typename TypeToCheck>
constexpr bool is_matrix3_v = is_matrix3<TypeToCheck>::value;

/// \brief A temporary concept to check if a type is a valid Matrix3 type
/// TODO(palmerb4): Extend this concept to contain all shared setters and getters for our quaternions.
template <typename Matrix3Type>
concept ValidMatrix3Type = 
  is_matrix3_v<std::decay_t<Matrix3Type>> &&
requires(std::decay_t<Matrix3Type> matrix3, const std::decay_t<Matrix3Type> const_matrix3) {
  typename std::decay_t<Matrix3Type>::scalar_t;
  { matrix3[0] } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3[1] } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3[2] } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3[3] } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3[4] } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3[5] } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3[6] } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3[7] } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3[8] } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;

  { matrix3(0) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3(1) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3(2) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3(3) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3(4) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3(5) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3(6) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3(7) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3(8) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;

  { matrix3(0, 0) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3(0, 1) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3(0, 2) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3(1, 0) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3(1, 1) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3(1, 2) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3(2, 0) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3(2, 1) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;
  { matrix3(2, 2) } -> std::convertible_to<typename std::decay_t<Matrix3Type>::scalar_t>;

  { const_matrix3[0] } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3[1] } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3[2] } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3[3] } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3[4] } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3[5] } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3[6] } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3[7] } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3[8] } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;

  { const_matrix3(0) } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3(1) } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3(2) } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3(3) } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3(4) } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3(5) } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3(6) } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3(7) } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3(8) } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;

  { const_matrix3(0, 0) } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3(0, 1) } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3(0, 2) } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3(1, 0) } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3(1, 1) } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3(1, 2) } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3(2, 0) } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3(2, 1) } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
  { const_matrix3(2, 2) } -> std::convertible_to<const typename std::decay_t<Matrix3Type>::scalar_t>;
};  // ValidMatrix3Type

static_assert(is_matrix3_v<Matrix3<int>>, "Odd, default matrix3 is not a matrix3.");
static_assert(is_matrix3_v<Matrix3<int, Array<int, 9>>>, "Odd, default matrix3 with Array accessor is not a matrix3.");
static_assert(is_matrix3_v<Matrix3View<int>>, "Odd, Matrix3View is not a matrix3.");
static_assert(is_matrix3_v<OwningMatrix3<int>>, "Odd, OwningMatrix3 is not a matrix3.");

//! \name Matrix3<T, Accessor> views
//@{

/// \brief A helper function to create a Matrix3<T, Accessor> based on a given (valid) accessor.
/// \param[in] data The data accessor.
///
/// In practice, this function is syntactic sugar to avoid having to specify the template parameters
/// when creating a Matrix3<T, Accessor> from a data accessor.
/// Instead of writing
/// \code
///   Matrix3<T, Accessor> mat(data);
/// \endcode
/// you can write
/// \code
///   auto mat = get_matrix3_view<T>(data);
/// \endcode
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_matrix3_view(Accessor& data) {
  return Matrix3View<T, Accessor>(data);
}

template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_matrix3_view(Accessor&& data) {
  return Matrix3View<T, Accessor>(std::forward<Accessor>(data));
}

template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_matrix3(Accessor& data) {
  return OwningMatrix3<T, Accessor>(data);
}

template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_matrix3(Accessor&& data) {
  return OwningMatrix3<T, Accessor>(std::forward<Accessor>(data));
}
//@}

//@}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_MATRIX3_HPP_
