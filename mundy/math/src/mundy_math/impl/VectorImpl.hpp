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

#ifndef MUNDY_MATH_IMPL_VECTORIMPL_HPP_
#define MUNDY_MATH_IMPL_VECTORIMPL_HPP_

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

namespace mundy {

namespace math {

template <typename T, size_t N, ValidAccessor<T> Accessor = Array<T, N>, typename OwnershipType = Ownership::Owns>
  requires std::is_arithmetic_v<T>
class Vector;

namespace impl {
//! \name Helper functions for generic vector operators applied to an abstract accessor.
//@{

/// \brief Deep copy assignment operator with (potentially) different accessor
/// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
template <size_t... Is, typename T, size_t N, ValidAccessor<T> Accessor, typename OwnershipType,
          ValidAccessor<T> OtherAccessor, typename OtherOwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION constexpr void deep_copy_impl(std::index_sequence<Is...>,
                                                     Vector<T, N, Accessor, OwnershipType>& vec,
                                                     const Vector<T, N, OtherAccessor, OtherOwnershipType>& other) {
  ((vec[Is] = other[Is]), ...);
}

/// \brief Move assignment operator with same accessor
/// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
template <size_t... Is, typename T, size_t N, ValidAccessor<T> Accessor, typename OwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION constexpr void move_impl(std::index_sequence<Is...>, Vector<T, N, Accessor, OwnershipType>& vec,
                                                Vector<T, N, Accessor, Ownership::Owns>&& other) {
  ((vec[Is] = std::move(other[Is])), ...);
}

/// \brief Set all elements of the vector
template <size_t... Is, typename T, size_t N, ValidAccessor<T> Accessor, typename OwnershipType, typename... Args>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION constexpr void set_from_args_impl(std::index_sequence<Is...>,
                                                         Vector<T, N, Accessor, OwnershipType>& vec, Args&&... args) {
  ((vec[Is] = std::forward<Args>(args)), ...);
}

/// \brief Set all elements of the vector using an accessor
/// \param[in] accessor A valid accessor.
/// \note A Vector is also a valid accessor.
template <size_t... Is, typename T, size_t N, ValidAccessor<T> Accessor, typename OwnershipType,
          ValidAccessor<T> OtherAccessor>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION constexpr void set_from_accessor_impl(std::index_sequence<Is...>,
                                                             Vector<T, N, Accessor, OwnershipType>& vec,
                                                             const OtherAccessor& accessor) {
  ((vec[Is] = accessor[Is]), ...);
}

/// \brief Set all elements of the vector to a single value
/// \param[in] value The value to set all elements to.
template <size_t... Is, typename T, size_t N, ValidAccessor<T> Accessor, typename OwnershipType>
  requires HasNonConstAccessOperator<Accessor, T>
KOKKOS_INLINE_FUNCTION constexpr void fill_impl(std::index_sequence<Is...>, Vector<T, N, Accessor, OwnershipType>& vec,
                                                const T& value) {
  ((vec[Is] = value), ...);
}

/// \brief Unary minus operator
template <size_t... Is, typename T, size_t N, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr Vector<T, N> unary_minus_impl(std::index_sequence<Is...>,
                                                               const Vector<T, N, Accessor, OwnershipType>& vec) {
  Vector<T, N> result;
  ((result[Is] = -vec[Is]), ...);
  return result;
}

/// \brief Vector-vector addition
/// \param[in] other The other vector.
template <size_t... Is, typename U, typename T, size_t N, ValidAccessor<T> Accessor, typename OwnershipType,
          ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto vector_vector_add_impl(
    std::index_sequence<Is...>, const Vector<T, N, Accessor, OwnershipType>& vec,
    const Vector<U, N, OtherAccessor, OtherOwnershipType>& other) -> Vector<std::common_type_t<T, U>, N> {
  using CommonType = std::common_type_t<T, U>;
  Vector<CommonType, N> result;
  ((result[Is] = static_cast<CommonType>(vec[Is]) + static_cast<CommonType>(other[Is])), ...);
  return result;
}

/// \brief Self-vector addition
/// \param[in] other The other vector.
template <size_t... Is, typename T, size_t N, typename U, ValidAccessor<T> Accessor, typename OwnershipType,
          ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION constexpr void self_vector_add_impl(std::index_sequence<Is...>,
                                                           Vector<T, N, Accessor, OwnershipType>& vec,
                                                           const Vector<U, N, OtherAccessor, OtherOwnershipType>& other)
  requires HasNonConstAccessOperator<Accessor, T>
{
  ((vec[Is] += static_cast<T>(other[Is])), ...);
}

/// \brief Vector-vector subtraction
/// \param[in] other The other vector.
template <size_t... Is, typename T, size_t N, typename U, ValidAccessor<T> Accessor, typename OwnershipType,
          ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto vector_vector_subtraction_impl(
    std::index_sequence<Is...>, const Vector<T, N, Accessor, OwnershipType>& vec,
    const Vector<U, N, OtherAccessor, OtherOwnershipType>& other) -> Vector<std::common_type_t<T, U>, N> {
  using CommonType = std::common_type_t<T, U>;
  Vector<CommonType, N> result;
  ((result[Is] = static_cast<CommonType>(vec[Is]) - static_cast<CommonType>(other[Is])), ...);
  return result;
}

/// \brief Vector-vector subtraction
/// \param[in] other The other vector.
template <size_t... Is, typename T, size_t N, typename U, ValidAccessor<T> Accessor, typename OwnershipType,
          ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION constexpr void self_vector_subtraction_impl(
    std::index_sequence<Is...>, Vector<T, N, Accessor, OwnershipType>& vec,
    const Vector<U, N, OtherAccessor, OtherOwnershipType>& other)
  requires HasNonConstAccessOperator<Accessor, T>
{
  ((vec[Is] -= static_cast<T>(other[Is])), ...);
}

/// \brief Vector-scalar addition
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, typename U, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto vector_scalar_add_impl(std::index_sequence<Is...>,
                                                             const Vector<T, N, Accessor, OwnershipType>& vec,
                                                             const U& scalar) -> Vector<std::common_type_t<T, U>, N> {
  using CommonType = std::common_type_t<T, U>;
  Vector<CommonType, N> result;
  ((result[Is] = static_cast<CommonType>(vec[Is]) + static_cast<CommonType>(scalar)), ...);
  return result;
}

/// \brief Vector-scalar addition
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, typename U, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr void self_scalar_add_impl(std::index_sequence<Is...>,
                                                           Vector<T, N, Accessor, OwnershipType>& vec, const U& scalar)
  requires HasNonConstAccessOperator<Accessor, T>
{
  ((vec[Is] += static_cast<T>(scalar)), ...);
}

/// \brief Vector-scalar subtraction
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, typename U, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto vector_scalar_subtraction_impl(std::index_sequence<Is...>,
                                                                     const Vector<T, N, Accessor, OwnershipType>& vec,
                                                                     const U& scalar)
    -> Vector<std::common_type_t<T, U>, N> {
  using CommonType = std::common_type_t<T, U>;
  Vector<CommonType, N> result;
  ((result[Is] = static_cast<CommonType>(vec[Is]) - static_cast<CommonType>(scalar)), ...);
  return result;
}

/// \brief Self-scalar subtraction
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, typename U, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr void self_scalar_subtraction_impl(std::index_sequence<Is...>,
                                                                   Vector<T, N, Accessor, OwnershipType>& vec,
                                                                   const U& scalar)
  requires HasNonConstAccessOperator<Accessor, T>
{
  ((vec[Is] -= static_cast<T>(scalar)), ...);
}

/// \brief Vector-scalar multiplication
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, typename U, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto vector_scalar_multiplication_impl(
    std::index_sequence<Is...>, const Vector<T, N, Accessor, OwnershipType>& vec, const U& scalar)
    -> Vector<std::common_type_t<T, U>, N> {
  using CommonType = std::common_type_t<T, U>;
  Vector<CommonType, N> result;
  ((result[Is] = static_cast<CommonType>(vec[Is]) * static_cast<CommonType>(scalar)), ...);
  return result;
}

/// \brief Self-scalar multiplication
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, typename U, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr void self_scalar_multiplication_impl(std::index_sequence<Is...>,
                                                                      Vector<T, N, Accessor, OwnershipType>& vec,
                                                                      const U& scalar)
  requires HasNonConstAccessOperator<Accessor, T>
{
  ((vec[Is] *= static_cast<T>(scalar)), ...);
}

/// \brief Vector-scalar division (with type promotion)
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, typename U, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto vector_scalar_division_impl(std::index_sequence<Is...>,
                                                                  const Vector<T, N, Accessor, OwnershipType>& vec,
                                                                  const U& scalar) {
  if constexpr (std::is_integral_v<T> && std::is_integral_v<U>) {
    using CommonType = double;
    Vector<CommonType, N> result;
    const CommonType scalar_inv = static_cast<CommonType>(1) / static_cast<CommonType>(scalar);
    ((result[Is] = static_cast<CommonType>(vec[Is]) * scalar_inv), ...);
    return result;
  } else {
    using CommonType = std::common_type_t<T, U>;
    Vector<CommonType, N> result;
    const CommonType scalar_inv = static_cast<CommonType>(1) / static_cast<CommonType>(scalar);
    ((result[Is] = static_cast<CommonType>(vec[Is]) * scalar_inv), ...);
    return result;
  }
}

/// \brief Self-scalar division (no type promotion!!!)
/// \param[in] scalar The scalar.
template <size_t... Is, typename T, size_t N, typename U, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr void self_scalar_division_impl(std::index_sequence<Is...>,
                                                                Vector<T, N, Accessor, OwnershipType>& vec,
                                                                const U& scalar)
  requires HasNonConstAccessOperator<decltype(vec), T>
{
  ((vec[Is] /= static_cast<T>(scalar)), ...);
}

/// \brief Vector-vector equality (element-wise within a tolerance)
/// \param[in] vec1 The first vector.
/// \param[in] vec2 The second vector.
/// \param[in] tol The tolerance.
template <size_t... Is, size_t N, typename U, typename T, typename V, ValidAccessor<U> Accessor, typename OwnershipType,
          ValidAccessor<T> OtherAccessor, typename OtherOwnershipType>
  requires std::is_arithmetic_v<V>
KOKKOS_INLINE_FUNCTION constexpr bool is_close_impl(std::index_sequence<Is...>,
                                                    const Vector<U, N, Accessor, OwnershipType>& vec1,
                                                    const Vector<T, N, OtherAccessor, OtherOwnershipType>& vec2,
                                                    const V& tol) {
  // Use the type of the tolerance to determine the comparison type
  return ((Kokkos::abs(static_cast<V>(vec1[Is]) - static_cast<V>(vec2[Is])) <= tol) && ...);
}

/// \brief Sum of all elements
template <size_t... Is, size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION T constexpr sum_impl(std::index_sequence<Is...>,
                                            const Vector<T, N, Accessor, OwnershipType>& vec) {
  return (vec[Is] + ...);
}

/// \brief Product of all elements
template <size_t... Is, size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION T constexpr product_impl(std::index_sequence<Is...>,
                                                const Vector<T, N, Accessor, OwnershipType>& vec) {
  return (vec[Is] * ...);
}

/// \brief Min of all elements
template <size_t... Is, size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION T constexpr min_impl(std::index_sequence<Is...>,
                                            const Vector<T, N, Accessor, OwnershipType>& vec) {
  // Initialize min_value with the first element
  T min_value = vec[0];
  ((min_value = (vec[Is] < min_value ? vec[Is] : min_value)), ...);
  return min_value;
}

/// \brief Max of all elements
template <size_t... Is, size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION T constexpr max_impl(std::index_sequence<Is...>,
                                            const Vector<T, N, Accessor, OwnershipType>& vec) {
  // Initialize max_value with the first element
  T max_value = vec[0];
  ((max_value = (vec[Is] > max_value ? vec[Is] : max_value)), ...);
  return max_value;
}

/// \brief Variance of all elements
template <size_t... Is, size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType variance_impl(std::index_sequence<Is...>,
                                                          const Vector<T, N, Accessor, OwnershipType>& vec) {
  OutputType inv_N = static_cast<OutputType>(1.0) / static_cast<OutputType>(N);
  OutputType vec_mean = inv_N * sum_impl(std::make_index_sequence<N>{}, vec);
  return (((static_cast<OutputType>(vec[Is]) - vec_mean) * (static_cast<OutputType>(vec[Is]) - vec_mean)) + ...) *
         inv_N;
}

/// \brief Standard deviation of all elements
template <size_t... Is, size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto standard_deviation_impl(std::index_sequence<Is...>,
                                                              const Vector<T, N, Accessor, OwnershipType>& vec) {
  return Kokkos::sqrt(variance_impl(std::make_index_sequence<N>{}, vec));
}

/// \brief Dot product of two vectors
template <size_t... Is, size_t N, typename U, typename T, ValidAccessor<U> Accessor, typename OwnershipType,
          ValidAccessor<T> OtherAccessor, typename OtherOwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto dot_product_impl(std::index_sequence<Is...>,
                                                       const Vector<U, N, Accessor, OwnershipType>& vec1,
                                                       const Vector<T, N, OtherAccessor, OtherOwnershipType>& vec2) {
  using CommonType = std::common_type_t<U, T>;
  return ((static_cast<CommonType>(vec1[Is]) * static_cast<CommonType>(vec2[Is])) + ...);
}
//@}
}  // namespace impl

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_VECTORIMPL_HPP_
