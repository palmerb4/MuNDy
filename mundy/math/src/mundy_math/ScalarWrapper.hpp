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

#ifndef MUNDY_MATH_SCALARWRAPPER_HPP_
#define MUNDY_MATH_SCALARWRAPPER_HPP_

// External
#include <Kokkos_Core.hpp>

// C++ core
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <iostream>
#include <type_traits>  // for std::decay_t
#include <utility>

// Mundy
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_math/Accessor.hpp>      // for mundy::math::ValidAccessor
#include <mundy_math/Array.hpp>         // for mundy::math::Array
#include <mundy_math/Tolerance.hpp>     // for mundy::math::get_zero_tolerance
#include <mundy_math/Vector.hpp>        // for mundy::math::Vector

namespace mundy {

namespace math {

/// \brief An owning/viewing scalar type
///
/// This scalar type is just a 1D vector with a single entry.
template <typename T, ValidAccessor<T> Accessor = Array<T, 1>, typename OwnershipType = Ownership::Owns>
  requires std::is_arithmetic_v<T>
using ScalarWrapper = Vector<T, 1, Accessor, OwnershipType>;

template <typename T, ValidAccessor<T> Accessor = Array<T, 1>>
  requires std::is_arithmetic_v<T>
using ScalarView = Vector<T, 1, Accessor, Ownership::Views>;

template <typename T, ValidAccessor<T> Accessor = Array<T, 1>>
  requires std::is_arithmetic_v<T>
using OwningScalar = Vector<T, 1, Accessor, Ownership::Owns>;

/// \brief (Implementation) Type trait to determine if a type is a ScalarWrapper
template <typename TypeToCheck>
struct is_scalar_wrapper_impl : std::false_type {};
//
template <typename T, typename Accessor, typename OwnershipType>
struct is_scalar_wrapper_impl<ScalarWrapper<T, Accessor, OwnershipType>> : std::true_type {};

/// \brief Type trait to determine if a type is a ScalarWrapper
template <typename TypeToCheck>
struct is_scalar_wrapper : public is_scalar_wrapper_impl<std::decay_t<TypeToCheck>> {};
//
template <typename TypeToCheck>
constexpr bool is_scalar_wrapper_v = is_scalar_wrapper<TypeToCheck>::value;

/// \brief A temporary concept to check if a type is a valid ScalarWrapper type
/// TODO(palmerb4): Extend this concept to contain all shared setters and getters for our vectors.
template <typename ScalarWrapperType>
concept ValidScalarWrapperType = is_scalar_wrapper_v<std::decay_t<ScalarWrapperType>> &&
                                 requires(std::decay_t<ScalarWrapperType> scalar_wrapper,
                                          const std::decay_t<ScalarWrapperType> const_scalar_wrapper) {
                                   typename std::decay_t<ScalarWrapperType>::scalar_t;
                                   {
                                     scalar_wrapper[0]
                                   } -> std::convertible_to<typename std::decay_t<ScalarWrapperType>::scalar_t>;

                                   {
                                     scalar_wrapper(0)
                                   } -> std::convertible_to<typename std::decay_t<ScalarWrapperType>::scalar_t>;

                                   {
                                     const_scalar_wrapper[0]
                                   } -> std::convertible_to<const typename std::decay_t<ScalarWrapperType>::scalar_t>;

                                   {
                                     const_scalar_wrapper(0)
                                   } -> std::convertible_to<const typename std::decay_t<ScalarWrapperType>::scalar_t>;
                                 };  // ValidScalarWrapperType

//! \name ScalarWrapper<T, Accessor> views
//@{

/// \brief A helper function to create a ScalarWrapper<T, Accessor> based on a given accessor.
/// \param[in] data The data accessor.
///
/// In practice, this function is syntactic sugar to avoid having to specify the template parameters
/// when creating a ScalarWrapper<T, Accessor> from a data accessor.
/// Instead of writing
/// \code
///   ScalarView<T, Accessor> s(data);
/// \endcode
/// you can write
/// \code
///   auto vec = get_scalar_view<T>(data);
/// \endcode
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_scalar_view(Accessor& data) {
  return ScalarView<T, Accessor>(data);
}

template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_scalar_view(Accessor&& data) {
  return ScalarView<T, Accessor>(std::forward<Accessor>(data));
}

template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_scalar(Accessor& data) {
  return OwningScalar<T, Accessor>(data);
}

template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_scalar(Accessor&& data) {
  return OwningScalar<T, Accessor>(std::forward<Accessor>(data));
}
//@}

//@}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_SCALARWRAPPER_HPP_
