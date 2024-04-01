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
#include <mundy_math/Accessor.hpp>      // for mundy::math::ValidAccessor
#include <mundy_math/Array.hpp>         // for mundy::math::Array
#include <mundy_math/Tolerance.hpp>     // for mundy::math::get_zero_tolerance

namespace mundy {

namespace math {

/// \brief Class for a 3x1 vector with arithmetic entries
/// \tparam T The type of the entries.
/// \tparam Accessor The type of the accessor.
///
/// This class is designed to be used with Kokkos. It is a simple 3x1 vector with arithmetic entries. It is templated
/// on the type of the entries and Accessor type. See Accessor.hpp for more details on the Accessor type requirements.
///
/// The goal of Vector3 is to be a lightweight class that can be used with Kokkos to perform mathematical operations on
/// vectors in R3. It does not own the data, but rather it is templated on an Accessor type that provides access to the
/// underlying data. This allows us to use Vector3 with Kokkos Views, raw pointers, or any other type that meets the
/// ValidAccessor requirements without copying the data. This is especially important for GPU-compatable code.
///
/// Vector3s can be constructed by passing an accessor to the constructor. However, if the accessor has a 3-argument
/// constructor, then the Vector3 can also be constructed by passing the elements directly to the constructor.
/// Similarly, if the accessor has an initializer list constructor, then the Vector3 can be constructed by passing an
/// initializer list to the constructor. This is a convenience feature which makes working with the default accessor
/// (Array<T, 3>) easier. For example, the following are all valid ways to construct a Vector3:
///
/// \code{.cpp}
///   // Constructs a Vector3 with the default accessor (Array<int, 3>)
///   Vector3<int> vec1({1, 2, 3});
///   Vector3<int> vec2(1, 2, 3);
///   Vector3<int> vec3(Array<int, 3>({1, 2, 3}));
///   Vector3<int> vec4;
///   vec4.set(1, 2, 3);
///
///   // Construct a Vector3 from a double array
///   double data[3] = {1.0, 2.0, 3.0};
///   Vector3<double, double*> vec5(data);
///   Vector3<double, double*> vec6{1.0, 2.0, 3.0};
///   // Not allowed as double* doesn't have a 3-argument constructor
///   // Vector3<double, double*> vec7(1.0, 2.0, 3.0);
/// \endcode
///
/// \note Accessors may be owning or non-owning, that is irrelevant to the Vector3 class; however, these accessors
/// should be lightweight such that they can be copied around without much overhead. Furthermore, the lifetime of the
/// data underlying the accessor should be as long as the Vector3 that use it.
template <typename T, ValidAccessor<T> Accessor = Array<T, 3>>
  requires std::is_arithmetic_v<T>
class Vector3 {
 private:
  //! \name Internal data
  //@{

  /// \brief Our data accessor
  Accessor data_;
  //@}

 public:
  //! \name Type aliases
  //@{

  /// \brief The type of the entries
  using value_type = T;

  /// \brief The non-const type of the entries
  using non_const_value_type = std::remove_const_t<T>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor. Assume elements are uninitialized.
  /// \note This constructor is only enabled if the Accessor has a default constructor.
  KOKKOS_FUNCTION Vector3()
    requires HasDefaultConstructor<Accessor>
      : data_() {
  }

  /// \brief Constructor from a given accessor
  /// \param[in] data The accessor.
  KOKKOS_FUNCTION
  Vector3(const Accessor& data) : data_(data) {
  }

  /// \brief Constructor to initialize all elements
  /// \param[in] x The x element.
  /// \param[in] y The y element.
  /// \param[in] z The z element.
  /// \note This constructor is only enabled if the Accessor has a 3-argument constructor.
  KOKKOS_FUNCTION Vector3(const T& x, const T& y, const T& z)
    requires Has3ArgConstructor<Accessor, T>
      : data_(x, y, z) {
  }

  /// \brief Constructor to initialize all elements via initializer list
  /// \param[in] list The initializer list.
  KOKKOS_FUNCTION Vector3(const std::initializer_list<T>& list)
    requires HasInitializerListConstructor<Accessor, T>
      : data_{list.begin()[0], list.begin()[1], list.begin()[2]} {
    MUNDY_THROW_ASSERT(list.size() == 3, std::invalid_argument, "Vector3: Initializer list must have 3 elements.");
  }

  /// \brief Destructor
  KOKKOS_FUNCTION
  ~Vector3() {
  }

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION Vector3(const Vector3<T, Accessor>& other) : data_(other.data()) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION Vector3(Vector3<T, Accessor>&& other) : data_(std::move(other.data())) {
  }

  /// \brief Deep copy assignment operator with different accessor
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  template <typename OtherAccessor>
  KOKKOS_FUNCTION Vector3<T, Accessor>& operator=(const Vector3<T, OtherAccessor>& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = other[0];
    data_[1] = other[1];
    data_[2] = other[2];
    return *this;
  }

  /// \brief Deep copy assignment operator with same accessor
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  KOKKOS_FUNCTION Vector3<T, Accessor>& operator=(const Vector3<T, Accessor>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = other[0];
    data_[1] = other[1];
    data_[2] = other[2];
    return *this;
  }

  /// \brief Deep copy assignment operator from a single value
  /// \param[in] value The value to set all elements to.
  KOKKOS_FUNCTION Vector3<T, Accessor>& operator=(const T value)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = value;
    data_[1] = value;
    data_[2] = value;
    return *this;
  }

  /// \brief Move assignment operator with different accessor
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  template <typename OtherAccessor>
  KOKKOS_FUNCTION Vector3<T, Accessor>& operator=(Vector3<T, OtherAccessor>&& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = other[0];
    data_[1] = other[1];
    data_[2] = other[2];
    return *this;
  }

  /// \brief Move assignment operator with same accessor
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  KOKKOS_FUNCTION Vector3<T, Accessor>& operator=(Vector3<T, Accessor>&& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = other[0];
    data_[1] = other[1];
    data_[2] = other[2];
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_FUNCTION
  T& operator[](unsigned index) {
    return data_[index];
  }

  /// \brief Const element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_FUNCTION
  const T& operator[](unsigned index) const {
    return data_[index];
  }

  /// \brief Element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_FUNCTION
  T& operator()(unsigned index) {
    return data_[index];
  }

  /// \brief Const element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_FUNCTION
  const T& operator()(unsigned index) const {
    return data_[index];
  }

  /// \brief Get the internal data accessor
  KOKKOS_FUNCTION
  Accessor data() {
    return data_;
  }

  /// \brief Get the internal data accessor
  KOKKOS_FUNCTION
  const Accessor data() const {
    return data_;
  }

  /// \brief Get the x element
  KOKKOS_FUNCTION
  T& x() {
    return data_[0];
  }

  /// \brief Get the x element
  KOKKOS_FUNCTION
  const T& x() const {
    return data_[0];
  }

  /// \brief Get the y element
  KOKKOS_FUNCTION
  T& y() {
    return data_[1];
  }

  /// \brief Get the y element
  KOKKOS_FUNCTION
  const T& y() const {
    return data_[1];
  }

  /// \brief Get the z element
  KOKKOS_FUNCTION
  T& z() {
    return data_[2];
  }

  /// \brief Get the z element
  KOKKOS_FUNCTION
  const T& z() const {
    return data_[2];
  }
  //@}

  //! \name Setters and modifiers
  //@{

  /// \brief Set all elements of the vector
  /// \param[in] x The x element.
  /// \param[in] y The y element.
  /// \param[in] z The z element.
  KOKKOS_FUNCTION void set(const T& x, const T& y, const T& z)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = x;
    data_[1] = y;
    data_[2] = z;
  }

  /// \brief Set all elements of the vector using an accessor
  /// \param[in] accessor A valid accessor.
  /// \note A Vector3 is also a valid accessor.
  KOKKOS_FUNCTION
  template <ValidAccessor<T> OtherAccessor>
  void set(const OtherAccessor& accessor)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = accessor[0];
    data_[1] = accessor[1];
    data_[2] = accessor[2];
  }

  /// \brief Set all elements of the vector to a single value
  /// \param[in] value The value to set all elements to.
  KOKKOS_FUNCTION
  void fill(const T& value)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = value;
    data_[1] = value;
    data_[2] = value;
  }
  //@}

  //! \name Unary operators
  //@{

  /// \brief Unary plus operator
  KOKKOS_FUNCTION
  Vector3<T> operator+() const {
    return *this;
  }

  /// \brief Unary minus operator
  KOKKOS_FUNCTION
  Vector3<T> operator-() const {
    Vector3<T> result;
    result[0] = -data_[0];
    result[1] = -data_[1];
    result[2] = -data_[2];
    return result;
  }
  //@}

  //! \name Addition and subtraction
  //@{

  /// \brief Vector-vector addition
  /// \param[in] other The other vector.
  template <typename U, typename OtherAccessor>
  KOKKOS_FUNCTION auto operator+(const Vector3<U, OtherAccessor>& other) const -> Vector3<std::common_type_t<T, U>> {
    using CommonType = std::common_type_t<T, U>;
    Vector3<CommonType> result;
    result[0] = static_cast<CommonType>(data_[0]) + static_cast<CommonType>(other[0]);
    result[1] = static_cast<CommonType>(data_[1]) + static_cast<CommonType>(other[1]);
    result[2] = static_cast<CommonType>(data_[2]) + static_cast<CommonType>(other[2]);
    return result;
  }

  /// \brief Vector-vector addition
  /// \param[in] other The other vector.
  template <typename U, typename OtherAccessor>
  KOKKOS_FUNCTION Vector3<T, Accessor>& operator+=(const Vector3<U, OtherAccessor>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] += static_cast<T>(other[0]);
    data_[1] += static_cast<T>(other[1]);
    data_[2] += static_cast<T>(other[2]);
    return *this;
  }

  /// \brief Vector-vector subtraction
  /// \param[in] other The other vector.
  template <typename U, typename OtherAccessor>
  KOKKOS_FUNCTION auto operator-(const Vector3<U, OtherAccessor>& other) const -> Vector3<std::common_type_t<T, U>> {
    using CommonType = std::common_type_t<T, U>;
    Vector3<CommonType> result;
    result[0] = static_cast<CommonType>(data_[0]) - static_cast<CommonType>(other[0]);
    result[1] = static_cast<CommonType>(data_[1]) - static_cast<CommonType>(other[1]);
    result[2] = static_cast<CommonType>(data_[2]) - static_cast<CommonType>(other[2]);
    return result;
  }

  /// \brief Vector-vector subtraction
  /// \param[in] other The other vector.
  template <typename U, typename OtherAccessor>
  KOKKOS_FUNCTION Vector3<T, Accessor>& operator-=(const Vector3<U, OtherAccessor>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] -= static_cast<T>(other[0]);
    data_[1] -= static_cast<T>(other[1]);
    data_[2] -= static_cast<T>(other[2]);
    return *this;
  }

  /// \brief Vector-scalar addition
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION auto operator+(const U& scalar) const -> Vector3<std::common_type_t<T, U>> {
    using CommonType = std::common_type_t<T, U>;
    Vector3<CommonType> result;
    result[0] = static_cast<CommonType>(data_[0]) + static_cast<CommonType>(scalar);
    result[1] = static_cast<CommonType>(data_[1]) + static_cast<CommonType>(scalar);
    result[2] = static_cast<CommonType>(data_[2]) + static_cast<CommonType>(scalar);
    return result;
  }

  /// \brief Vector-scalar addition
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION Vector3<T, Accessor>& operator+=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] += static_cast<T>(scalar);
    data_[1] += static_cast<T>(scalar);
    data_[2] += static_cast<T>(scalar);
    return *this;
  }

  /// \brief Vector-scalar subtraction
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION auto operator-(const U& scalar) const -> Vector3<std::common_type_t<T, U>> {
    using CommonType = std::common_type_t<T, U>;
    Vector3<CommonType> result;
    result[0] = static_cast<CommonType>(data_[0]) - static_cast<CommonType>(scalar);
    result[1] = static_cast<CommonType>(data_[1]) - static_cast<CommonType>(scalar);
    result[2] = static_cast<CommonType>(data_[2]) - static_cast<CommonType>(scalar);
    return result;
  }

  /// \brief Vector-scalar subtraction
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION Vector3<T, Accessor>& operator-=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] -= static_cast<T>(scalar);
    data_[1] -= static_cast<T>(scalar);
    data_[2] -= static_cast<T>(scalar);
    return *this;
  }
  //@}

  //! \name Multiplication and division
  //@{

  /// \brief Vector-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION auto operator*(const U& scalar) const -> Vector3<std::common_type_t<T, U>> {
    using CommonType = std::common_type_t<T, U>;
    Vector3<CommonType> result;
    result[0] = static_cast<CommonType>(data_[0]) * static_cast<CommonType>(scalar);
    result[1] = static_cast<CommonType>(data_[1]) * static_cast<CommonType>(scalar);
    result[2] = static_cast<CommonType>(data_[2]) * static_cast<CommonType>(scalar);
    return result;
  }

  /// \brief Vector-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION Vector3<T, Accessor>& operator*=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] *= static_cast<T>(scalar);
    data_[1] *= static_cast<T>(scalar);
    data_[2] *= static_cast<T>(scalar);
    return *this;
  }

  /// \brief Vector-scalar division
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION auto operator/(const U& scalar) const -> Vector3<std::common_type_t<T, U>> {
    using CommonType = std::common_type_t<T, U>;
    Vector3<CommonType> result;
    result[0] = static_cast<CommonType>(data_[0]) / static_cast<CommonType>(scalar);
    result[1] = static_cast<CommonType>(data_[1]) / static_cast<CommonType>(scalar);
    result[2] = static_cast<CommonType>(data_[2]) / static_cast<CommonType>(scalar);
    return result;
  }

  /// \brief Vector-scalar division
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION Vector3<T, Accessor>& operator/=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] /= static_cast<T>(scalar);
    data_[1] /= static_cast<T>(scalar);
    data_[2] /= static_cast<T>(scalar);
    return *this;
  }
  //@}

  //! \name Static methods
  //@{

  /// \brief Get a vector of ones
  KOKKOS_FUNCTION static Vector3<T> ones() {
    return Vector3<T>(T(1), T(1), T(1));
  }

  /// \brief Get the zero vector
  KOKKOS_FUNCTION static Vector3<T> zeros() {
    return Vector3<T>(T(0), T(0), T(0));
  }
  //@}

  //! \name Friends <3
  //@{

  // Declare the << operator as a friend
  template <typename U, typename OtherAccessor>
  friend std::ostream& operator<<(std::ostream& os, const Vector3<U, OtherAccessor>& vec);

  // We are friends with all Vector3s regardless of their Accessor or type
  template <typename U, ValidAccessor<U> OtherAccessor>
  friend class Vector3;
  //@}
};  // class Vector3

//! \name Non-member functions
//@{

//! \name Write to output stream
//@{

/// \brief Write the quaternion to an output stream
/// \param[in] os The output stream.
/// \param[in] quat The quaternion.
template <typename T, typename Accessor>
KOKKOS_FUNCTION std::ostream& operator<<(std::ostream& os, const Vector3<T, Accessor>& vec) {
  os << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ")";
  return os;
}
//@}

//! \name Non-member comparison functions
//@{

/// TODO(palmerb4): These really shouldn't be in the vector class. They should be in a separate file.
/// \brief Scalar-scalar equality (within a tolerance)
/// \param[in] scalar1 The first scalar.
/// \param[in] scalar2 The second scalar.
/// \param[in] tol The tolerance (default is determined by the given type).
template <typename U, typename T>
  requires std::is_arithmetic_v<U> && std::is_arithmetic_v<T>
KOKKOS_FUNCTION bool is_close(
    const U& scalar1, const T& scalar2,
    const decltype(get_comparison_tolerance<U, T>())& tol = get_comparison_tolerance<U, T>()) {
  // Our tolerance type is using the the smallest type of U and T
  using ComparisonType = std::remove_reference_t<decltype(tol)>;
  if constexpr (std::is_floating_point_v<ComparisonType>) {
    // For floating-point types, compare with a tolerance
    return std::abs(static_cast<ComparisonType>(scalar1) - static_cast<ComparisonType>(scalar2)) < tol;
  } else {
    // For integral types, compare with exact equality using the common type to avoid type promotion warnings.
    return static_cast<ComparisonType>(scalar1) == static_cast<ComparisonType>(scalar2);
  }
}

/// \brief Scalar-scalar equality (within a relaxed tolerance)
/// \param[in] scalar1 The first scalar.
/// \param[in] scalar2 The second scalar.
/// \param[in] tol The tolerance (default is determined by the given type).
template <typename U, typename T>
  requires std::is_arithmetic_v<U> && std::is_arithmetic_v<T>
KOKKOS_FUNCTION bool is_approx_close(
    const U& scalar1, const T& scalar2,
    const decltype(get_relaxed_comparison_tolerance<T, U>())& tol = get_relaxed_comparison_tolerance<T, U>()) {
  return is_close(scalar1, scalar2, tol);
}

/// \brief Vector-vector equality (element-wise within a tolerance)
/// \param[in] vec1 The first vector.
/// \param[in] vec2 The second vector.
/// \param[in] tol The tolerance (default is determined by the given type).
template <typename U, typename OtherAccessor, typename T, typename Accessor>
KOKKOS_FUNCTION bool is_close(
    const Vector3<U, OtherAccessor>& vec1, const Vector3<T, Accessor>& vec2,
    const decltype(get_comparison_tolerance<U, T>())& tol = get_comparison_tolerance<U, T>()) {
  // Our tolerance type is using the the smallest type of U and T
  using ComparisonType = std::remove_reference_t<decltype(tol)>;
  if constexpr (std::is_floating_point_v<ComparisonType>) {
    // For floating-point types, compare with a tolerance
    return (std::abs(static_cast<ComparisonType>(vec1[0]) - static_cast<ComparisonType>(vec2[0])) < tol) &&
           std::abs(static_cast<ComparisonType>(vec1[1]) - static_cast<ComparisonType>(vec2[1])) < tol &&
           std::abs(static_cast<ComparisonType>(vec1[2]) - static_cast<ComparisonType>(vec2[2])) < tol;
  } else {
    // For integral types, compare with exact equality using the common type to avoid type promotion warnings.
    return (static_cast<ComparisonType>(vec1[0]) == static_cast<ComparisonType>(vec2[0])) &&
           (static_cast<ComparisonType>(vec1[1]) == static_cast<ComparisonType>(vec2[1])) &&
           (static_cast<ComparisonType>(vec1[2]) == static_cast<ComparisonType>(vec2[2]));
  }
}

/// \brief Vector-vector equality (element-wise within a relaxed tolerance)
/// \param[in] vec1 The first vector.
/// \param[in] vec2 The second vector.
/// \param[in] tol The tolerance (default is determined by the given type).
template <typename U, typename OtherAccessor, typename T, typename Accessor>
KOKKOS_FUNCTION bool is_approx_close(
    const Vector3<U, OtherAccessor>& vec1, const Vector3<T, Accessor>& vec2,
    const decltype(get_relaxed_comparison_tolerance<T, U>())& tol = get_relaxed_comparison_tolerance<T, U>()) {
  return is_close(vec1, vec2, tol);
}
//@}

//! \name Non-member addition and subtraction operators
//@{

/// \brief Scalar-vector addition
/// \param[in] scalar The scalar.
/// \param[in] vec The vector.
template <typename U, typename T, typename Accessor>
KOKKOS_FUNCTION auto operator+(const U& scalar, const Vector3<T, Accessor>& vec) -> Vector3<std::common_type_t<T, U>> {
  return vec + scalar;
}

/// \brief Scalar-vector subtraction
/// \param[in] scalar The scalar.
/// \param[in] vec The vector.
template <typename U, typename T, typename Accessor>
KOKKOS_FUNCTION auto operator-(const U& scalar, const Vector3<T, Accessor>& vec) -> Vector3<std::common_type_t<T, U>> {
  return -vec + scalar;
}
//@}

//! \name Non-member multiplication and division operators
//@{

/// \brief Scalar-vector multiplication
/// \param[in] scalar The scalar.
/// \param[in] vec The vector.
template <typename U, typename T, typename Accessor>
KOKKOS_FUNCTION auto operator*(const U& scalar, const Vector3<T, Accessor>& vec) -> Vector3<std::common_type_t<T, U>> {
  return vec * scalar;
}
//@}

//! \name Basic arithmetic reduction operations
//@{

/// \brief Sum of all elements
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto sum(const Vector3<T, Accessor>& vec) {
  return vec[0] + vec[1] + vec[2];
}

/// \brief Product of all elements
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto product(const Vector3<T, Accessor>& vec) {
  return vec[0] * vec[1] * vec[2];
}

/// \brief Minimum element
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto min(const Vector3<T, Accessor>& vec) {
  return vec[0] < vec[1] ? (vec[0] < vec[2] ? vec[0] : vec[2]) : (vec[1] < vec[2] ? vec[1] : vec[2]);
}

/// \brief Maximum element
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto max(const Vector3<T, Accessor>& vec) {
  return vec[0] > vec[1] ? (vec[0] > vec[2] ? vec[0] : vec[2]) : (vec[1] > vec[2] ? vec[1] : vec[2]);
}

/// \brief Mean of all elements (returns a double if T is an integral type, otherwise returns T)
template <typename T, typename Accessor, typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_FUNCTION OutputType mean(const Vector3<T, Accessor>& vec) {
  auto vec_sum = sum(vec);
  return static_cast<OutputType>(vec_sum) / OutputType(3);
}

/// \brief Mean of all elements (returns a float if T is an integral type, otherwise returns T)
template <typename T, typename Accessor, typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_FUNCTION OutputType mean_f(const Vector3<T, Accessor>& vec) {
  return mean<T, Accessor, OutputType>(vec);
}

/// \brief Variance of all elements (returns a double if T is an integral type, otherwise returns T)
template <typename T, typename Accessor, typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_FUNCTION OutputType variance(const Vector3<T, Accessor>& vec) {
  OutputType vec_mean = mean<T, Accessor, OutputType>(vec);
  return ((static_cast<OutputType>(vec[0]) - vec_mean) * (static_cast<OutputType>(vec[0]) - vec_mean) +
          (static_cast<OutputType>(vec[1]) - vec_mean) * (static_cast<OutputType>(vec[1]) - vec_mean) +
          (static_cast<OutputType>(vec[2]) - vec_mean) * (static_cast<OutputType>(vec[2]) - vec_mean)) /
         OutputType(3);
}

/// \brief Variance of all elements (returns a float if T is an integral type, otherwise returns T)
template <typename T, typename Accessor, typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_FUNCTION OutputType variance_f(const Vector3<T, Accessor>& vec) {
  return variance<T, Accessor, OutputType>(vec);
}

/// \brief Standard deviation of all elements (returns a double if T is an integral type, otherwise returns T)
template <typename T, typename Accessor, typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_FUNCTION OutputType stddev(const Vector3<T, Accessor>& vec) {
  return std::sqrt(variance<T, Accessor, OutputType>(vec));
}

/// \brief Standard deviation of all elements (returns a float if T is an integral type, otherwise returns T)
template <typename T, typename Accessor, typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_FUNCTION OutputType stddev_f(const Vector3<T, Accessor>& vec) {
  return stddev<T, Accessor, OutputType>(vec);
}
//@}

//! \name Special vector operations
//@{

/// \brief Dot product of two vectors
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <typename U, typename OtherAccessor, typename T, typename Accessor>
KOKKOS_FUNCTION auto dot(const Vector3<U, OtherAccessor>& a, const Vector3<T, Accessor>& b)
    -> std::common_type_t<T, U> {
  using CommonType = std::common_type_t<T, U>;
  return static_cast<CommonType>(a[0]) * static_cast<CommonType>(b[0]) +
         static_cast<CommonType>(a[1]) * static_cast<CommonType>(b[1]) +
         static_cast<CommonType>(a[2]) * static_cast<CommonType>(b[2]);
}

/// \brief Cross product
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <typename U, typename OtherAccessor, typename T, typename Accessor>
KOKKOS_FUNCTION auto cross(const Vector3<U, OtherAccessor>& a, const Vector3<T, Accessor>& b)
    -> Vector3<std::common_type_t<T, U>> {
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

//@}

//! \name Vector norms
//@{

/// \brief Vector infinity norm
/// \param[in] vec The vector.
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto infinity_norm(const Vector3<T, Accessor>& vec) {
  return max(vec);
}

/// \brief Vector 1-norm
/// \param[in] vec The vector.
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto one_norm(const Vector3<T, Accessor>& vec) {
  return sum(vec);
}

/// \brief Vector 2-norm (Returns a double if T is an integral type, otherwise returns T)
/// \param[in] vec The vector.
template <typename T, typename Accessor, typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_FUNCTION OutputType two_norm(const Vector3<T, Accessor>& vec) {
  return std::sqrt(static_cast<OutputType>(dot(vec, vec)));
}

/// \brief Vector 2-norm (Returns a float if T is an integral type, otherwise returns T)
/// \param[in] vec The vector.
template <typename T, typename Accessor, typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_FUNCTION OutputType two_norm_f(const Vector3<T, Accessor>& vec) {
  return two_norm<T, Accessor, OutputType>(vec);
}

/// \brief Vector squared 2-norm
/// \param[in] vec The vector.
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto two_norm_squared(const Vector3<T, Accessor>& vec) {
  return dot(vec, vec);
}

/// \brief Default vector norm (2-norm, returns a double if T is an integral type, otherwise returns T)
/// \param[in] vec The vector.
template <typename T, typename Accessor, typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_FUNCTION OutputType norm(const Vector3<T, Accessor>& vec) {
  return two_norm<T, Accessor, OutputType>(vec);
}

/// \brief Default vector norm (2-norm, returns a float if T is an integral type, otherwise returns T)
/// \param[in] vec The vector.
template <typename T, typename Accessor, typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_FUNCTION OutputType norm_f(const Vector3<T, Accessor>& vec) {
  return norm<T, Accessor, OutputType>(vec);
}

/// \brief Default vector norm squared (2-norm)
/// \param[in] vec The vector.
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto norm_squared(const Vector3<T, Accessor>& vec) {
  return two_norm_squared(vec);
}

/// \brief Minor angle between two vectors (returns a double if T is an integral type, otherwise returns T)
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <typename U, typename OtherAccessor, typename T, typename Accessor,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_FUNCTION OutputType minor_angle(const Vector3<U, OtherAccessor>& a, const Vector3<T, Accessor>& b) {
  return std::acos(static_cast<OutputType>(dot(a, b)) /
                   (static_cast<OutputType>(two_norm(a)) * static_cast<OutputType>(two_norm(b))));
}

/// \brief Minor angle between two vectors (returns a float if T is an integral type, otherwise returns T)
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <typename U, typename OtherAccessor, typename T, typename Accessor,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_FUNCTION OutputType minor_angle_f(const Vector3<U, OtherAccessor>& a, const Vector3<T, Accessor>& b) {
  return minor_angle<U, OtherAccessor, T, Accessor, OutputType>(a, b);
}

/// \brief Major angle between two vectors (returns a double if T is an integral type, otherwise returns T)
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <typename U, typename OtherAccessor, typename T, typename Accessor,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_FUNCTION OutputType major_angle(const Vector3<U, OtherAccessor>& a, const Vector3<T, Accessor>& b) {
  return OutputType(M_PI) - minor_angle<U, OtherAccessor, T, Accessor, OutputType>(a, b);
}

/// \brief Major angle between two vectors (returns a float if T is an integral type, otherwise returns T)
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <typename U, typename OtherAccessor, typename T, typename Accessor,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_FUNCTION OutputType major_angle_f(const Vector3<U, OtherAccessor>& a, const Vector3<T, Accessor>& b) {
  return major_angle<U, OtherAccessor, T, Accessor, OutputType>(a, b);
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
template <typename T, typename Accessor>
KOKKOS_FUNCTION Vector3<T, Accessor> get_vector3_view(const Accessor& data) {
  return Vector3<T, Accessor>(data);
}
//@}

//@}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_VECTOR3_HPP_
