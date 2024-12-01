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

#ifndef MUNDY_MATH_ARRAY_HPP_
#define MUNDY_MATH_ARRAY_HPP_

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
#include <mundy_math/Tolerance.hpp>     // for mundy::math::get_zero_tolerance

namespace mundy {

namespace math {

/// \brief A simplistic array type with a fixed size and type
template <typename T, size_t N>
class Array;

namespace impl {

/// \brief Deep copy implementation for Array
template <size_t... Is, typename T, size_t N>
KOKKOS_INLINE_FUNCTION void deep_copy_impl(std::index_sequence<Is...>, Array<T, N>& array, const Array<T, N>& other) {
  ((array[Is] = other[Is]), ...);
}

/// \brief Fill implementation for Array
template <size_t... Is, typename T, size_t N>
KOKKOS_INLINE_FUNCTION void fill_impl(std::index_sequence<Is...>, Array<T, N>& array, const T& value) {
  ((array[Is] = value), ...);
}

}  // namespace impl

/// \brief A simplistic array type with a fixed size and type
template <typename T, size_t N>
class Array {
 public:
  //! \name Type aliases
  //@{

  /// \brief The type of the entries
  using value_type = T;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor. Elements are uninitialized.
  KOKKOS_INLINE_FUNCTION
  Array() : data_{} {
  }

  /// \brief Constructor to initialize all elements explicitly.
  /// Requires the number of arguments to be N and the type of each to be T.
  template <typename... Args>
    requires(sizeof...(Args) == N) && (N != 1) &&
            (std::is_same_v<std::remove_cv_t<std::remove_reference_t<Args>>, T> && ...)
  KOKKOS_INLINE_FUNCTION explicit Array(Args&&... args) : data_{std::forward<Args>(args)...} {
  }

  /// \brief Constructor to initialize all elements via initializer list
  KOKKOS_INLINE_FUNCTION
  Array(const std::initializer_list<T>& list)
    requires(!std::is_const_v<T>)
  {
    if (list.size() == N) {
      size_t i = 0;
      for (auto it = list.begin(); it != list.end(); ++it) {
        data_[i] = *it;
        ++i;
      }
    } else if (list.size() == 1) {
      impl::fill_impl(std::make_index_sequence<N>{}, *this, *list.begin());
    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Array: Initializer list must have either 1 or N elements.");
    }
  }

  /// \brief Constructor to initialize all elements to a single value
  KOKKOS_INLINE_FUNCTION
  Array(const T& value) : Array(value, std::make_index_sequence<N>{}) {
  }

  /// \brief Destructor
  KOKKOS_INLINE_FUNCTION ~Array() {
  }

  /// \brief Deep copy constructor
  // Deep copy constructor
  KOKKOS_INLINE_FUNCTION
  Array(const Array<T, N>& other) : Array(other, std::make_index_sequence<N>{}) {
  }

  /// \brief Deep move constructor
  KOKKOS_INLINE_FUNCTION
  Array(Array<T, N>&& other) : Array(other, std::make_index_sequence<N>{}) {
  }

  /// \brief Deep copy assignment operator
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  KOKKOS_INLINE_FUNCTION
  Array<T, N>& operator=(const Array<T, N>& other)
    requires(!std::is_const_v<T>)
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }

  /// \brief Move assignment operator
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  KOKKOS_INLINE_FUNCTION
  Array<T, N>& operator=(Array<T, N>&& other)
    requires(!std::is_const_v<T>)
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_INLINE_FUNCTION
  T& operator[](size_t idx) {
    return data_[idx];
  }

  /// \brief Const element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_INLINE_FUNCTION
  const T& operator[](size_t idx) const {
    return data_[idx];
  }

  /// \brief Get our size
  KOKKOS_INLINE_FUNCTION
  constexpr int size() const {
    return N;
  }

  /// \brief Get a pointer to our data
  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<T, N>& data() {
    return data_;
  }

  /// \brief Get a pointer to our data
  KOKKOS_INLINE_FUNCTION
  const Kokkos::Array<T, N>& data() const {
    return data_;
  }
  //@}

 private:
  //! \name Private constructors
  //@{

  /// \brief Constructor to initialize all elements to a single value using index_sequence
  template <size_t... I>
  KOKKOS_INLINE_FUNCTION Array(const T& value, std::index_sequence<I...>) : data_{((void)I, value)...} {
  }

  /// \brief Deep copy constructor using index_sequence
  template <size_t... I>
  KOKKOS_INLINE_FUNCTION Array(const Array<T, N>& other, std::index_sequence<I...>) : data_{other.data_[I]...} {
  }

  /// \brief Deep move constructor using index_sequence
  template <size_t... I>
  KOKKOS_INLINE_FUNCTION Array(Array<T, N>&& other, std::index_sequence<I...>) : data_{other.data_[I]...} {
  }
  //@}

  //! \name Internal data
  //@{

  /// \brief Our data
  Kokkos::Array<T, N> data_;
  //@}
};  // Array

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_ARRAY_HPP_
