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
#include <mundy_core/throw_assert.hpp>    // for MUNDY_THROW_ASSERT
#include <mundy_math/Tolerance.hpp>  // for mundy::math::get_default_tolerance

namespace mundy {

namespace math {

/// \brief A simplistic array type with a fixed size and type
template <typename T, int N>
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
  KOKKOS_FUNCTION
  Array() {
  }

  // Constructor to initialize all elements explicitly. Requires the number of arguments to be N and the type of each to be T.
  template <typename... Args>
    requires(sizeof...(Args) == N) && (std::is_same_v<std::remove_cv_t<std::remove_reference_t<Args>>, T> && ...)
  KOKKOS_FUNCTION explicit Array(Args&&... args) : data_{std::forward<Args>(args)...} {
  }

  /// \brief Constructor to initialize all elements via initializer list
  KOKKOS_FUNCTION
  Array(const std::initializer_list<T>& list)
    requires(!std::is_const_v<T>)
  {
    MUNDY_THROW_ASSERT(list.size() == N, std::invalid_argument, "Array: Initializer list must have N elements.");
    std::copy(list.begin(), list.end(), data_);
  }

  /// \brief Constructor to initialize all elements to a single value
  Array(const T& value) : Array(value, std::make_index_sequence<N>{}) {
  }

  /// \brief Destructor
  KOKKOS_FUNCTION ~Array() {
  }

  /// \brief Deep copy constructor
  // Deep copy constructor
  KOKKOS_FUNCTION
  Array(const Array<T, N>& other) : Array(other, std::make_index_sequence<N>{}) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  Array(Array<T, N>&& other) : Array(other, std::make_index_sequence<N>{}) {
  }

  /// \brief Deep copy assignment operator
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  KOKKOS_FUNCTION
  Array<T, N>& operator=(const Array<T, N>& other)
    requires(!std::is_const_v<T>)
  {
    std::copy(other.data_, other.data_ + N, data_);
    return *this;
  }

  /// \brief Move assignment operator
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  KOKKOS_FUNCTION
  Array<T, N>& operator=(Array<T, N>&& other)
    requires(!std::is_const_v<T>)
  {
    std::copy(other.data_, other.data_ + N, data_);
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_FUNCTION
  T& operator[](unsigned idx) {
    return data_[idx];
  }

  /// \brief Const element access operator
  /// \param[in] idx The index of the element.
  KOKKOS_FUNCTION
  const T& operator[](unsigned idx) const {
    return data_[idx];
  }

  /// \brief Get our size
  KOKKOS_FUNCTION
  constexpr int size() const {
    return N;
  }

  /// \brief Get a pointer to our data
  KOKKOS_FUNCTION
  T* data() {
    return data_;
  }

  /// \brief Get a pointer to our data
  KOKKOS_FUNCTION
  const T* data() const {
    return data_;
  }
  //@}

 private:
  //! \name Private constructors
  //@{

  /// \brief Constructor to initialize all elements to a single value using index_sequence
  template <std::size_t... I>
  Array(const T& value, std::index_sequence<I...>) : data_{(I, value)...} {
  }

  /// \brief Deep copy constructor using index_sequence
  template <std::size_t... I>
  KOKKOS_FUNCTION Array(const Array<T, N>& other, std::index_sequence<I...>) : data_{other.data_[I]...} {
  }

  /// \brief Deep move constructor using index_sequence
  template <std::size_t... I>
  KOKKOS_FUNCTION Array(Array<T, N>&& other, std::index_sequence<I...>) : data_{other.data_[I]...} {
  }
  //@}

  //! \name Internal data
  //@{

  /// \brief Our data
  T data_[N];
  //@}
};  // Array

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_ARRAY_HPP_
