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
#include <initializer_list>
#include <iostream>
#include <type_traits>

// Our libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_math/Accessor.hpp>      // for mundy::math::ValidAccessor
#include <mundy_math/Array.hpp>         // for mundy::math::Array
#include <mundy_math/Tolerance.hpp>     // for mundy::math::get_zero_tolerance
#include <mundy_math/Vector3.hpp>       // for mundy::math::Vector3

namespace mundy {

namespace math {

/// \brief Class for a 3x3 matrix with arithmetic entries
/// \tparam T The type of the entries.
/// \tparam Accessor The type of the accessor.
///
/// This class is designed to be used with Kokkos. It is a simple 3x3 matrix with arithmetic entries. It is templated
/// on the type of the entries and Accessor type. See Accessor.hpp for more details on the Accessor type requirements.
///
/// The goal of Matrix3 is to be a lightweight class that can be used with Kokkos to perform mathematical operations on
/// matrices in R3x3. It does not own or manage the underlying data, but rather it is templated on an Accessor type that
/// provides access to the underlying data. This allows us to use Matrix3 with Kokkos Views, raw pointers, or any other
/// type that meets the ValidAccessor requirements without copying the data. This is especially important for
/// GPU-compatable code.
///
/// Matrix3s can be constructed by passing an accessor to the constructor. However, if the accessor has a 9-argument
/// constructor, then the Matrix3 can also be constructed by passing the elements directly to the constructor.
/// Similarly, if the accessor has an initializer list constructor, then the Matrix3 can be constructed by passing an
/// initializer list to the constructor. This is a convenience feature which makes working with the default accessor
/// (Array<T, 9>) easier. For example, the following are all valid ways to construct a Matrix3:
///
/// \code{.cpp}
///   // Constructs a Matrix3 with the default accessor (Array<int, 9>)
///   Matrix3<int> mat1({1, 2, 3, 4, 5, 6, 7, 8, 9});
///   Matrix3<int> mat2(1, 2, 3, 4, 5, 6, 7, 8, 9);
///   Matrix3<int> mat3(Array<int, 9>{1, 2, 3, 4, 5, 6, 7, 8, 9});
///   Matrix3<int> mat4;
///   mat4.set(1, 2, 3, 4, 5, 6, 7, 8, 9);
///
///   // Construct a Matrix3 from a double array
///   double data[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
///   Matrix3<double, double*> mat5(data);
///   Matrix3<double, double*> mat6{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
///   // Not allowed as double* doesn't have a 9-argument constructor
///   // Matrix3<double, double*> mat7(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
/// \endcode
///
/// \note Accessors may be owning or non-owning, that is irrelevant to the Matrix3 class; however, these accessors
/// should be lightweight such that they can be copied around without much overhead. Furthermore, the lifetime of the
/// data underlying the accessor should be as long as the Matrix3 that use it.
template <typename T, ValidAccessor<T> Accessor = Array<T, 9>>
  requires std::is_arithmetic_v<T>
class Matrix3 {
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
  KOKKOS_FUNCTION Matrix3()
    requires HasDefaultConstructor<Accessor>
      : data_() {
  }

  /// \brief Constructor from a given accessor
  /// \param[in] data The accessor.
  KOKKOS_FUNCTION
  Matrix3(const Accessor& data) : data_(data) {
  }

  /// \brief Constructor to initialize all elements
  /// \param[in] a11 The element at row 1, column 1.
  /// \param[in] a12 The element at row 1, column 2.
  /// \param[in] a13 The element at row 1, column 3.
  /// \param[in] a21 The element at row 2, column 1.
  /// \param[in] a22 The element at row 2, column 2.
  /// \param[in] a23 The element at row 2, column 3.
  /// \param[in] a31 The element at row 3, column 1.
  /// \param[in] a32 The element at row 3, column 2.
  /// \param[in] a33 The element at row 3, column 3.
  /// \note This constructor is only enabled if the Accessor has a 9-argument constructor.
  KOKKOS_FUNCTION Matrix3(const T& a11, const T& a12, const T& a13, const T& a21, const T& a22, const T& a23,
                          const T& a31, const T& a32, const T& a33)
    requires Has9ArgConstructor<Accessor, T>
      : data_(a11, a12, a13, a21, a22, a23, a31, a32, a33) {
  }

  /// \brief Constructor to initialize all elements via initializer list
  KOKKOS_FUNCTION Matrix3(const std::initializer_list<T>& list)
    requires HasInitializerListConstructor<Accessor, T>
      : data_{list.begin()[0], list.begin()[1], list.begin()[2], list.begin()[3], list.begin()[4],
              list.begin()[5], list.begin()[6], list.begin()[7], list.begin()[8]} {
    MUNDY_THROW_ASSERT(list.size() == 9, std::invalid_argument, "Matrix3: Initializer list must have 9 elements.");
  }

  /// \brief Destructor
  KOKKOS_FUNCTION
  ~Matrix3() {
  }

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION Matrix3(const Matrix3<T, Accessor>& other) : data_(other.data()) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION Matrix3(Matrix3<T, Accessor>&& other) : data_(std::move(other.data())) {
  }

  /// \brief Deep copy assignment operator with different accessor
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  template <typename OtherAccessor>
  KOKKOS_FUNCTION Matrix3<T, Accessor>& operator=(const Matrix3<T, OtherAccessor>& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = other[0];
    data_[1] = other[1];
    data_[2] = other[2];
    data_[3] = other[3];
    data_[4] = other[4];
    data_[5] = other[5];
    data_[6] = other[6];
    data_[7] = other[7];
    data_[8] = other[8];
    return *this;
  }

  /// \brief Deep copy assignment operator with same accessor
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  /// This operator exists to avoid issues with template deduction.
  KOKKOS_FUNCTION Matrix3<T, Accessor>& operator=(const Matrix3<T, Accessor>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = other[0];
    data_[1] = other[1];
    data_[2] = other[2];
    data_[3] = other[3];
    data_[4] = other[4];
    data_[5] = other[5];
    data_[6] = other[6];
    data_[7] = other[7];
    data_[8] = other[8];
    return *this;
  }

  /// \brief Deep copy assignment operator from a single value
  /// \param[in] value The value to set all elements to.
  KOKKOS_FUNCTION Matrix3<T, Accessor>& operator=(const T value)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = value;
    data_[1] = value;
    data_[2] = value;
    data_[3] = value;
    data_[4] = value;
    data_[5] = value;
    data_[6] = value;
    data_[7] = value;
    data_[8] = value;
    return *this;
  }

  /// \brief Move assignment operator with different accessor
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  template <typename OtherAccessor>
  KOKKOS_FUNCTION Matrix3<T, Accessor>& operator=(Matrix3<T, OtherAccessor>&& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = other[0];
    data_[1] = other[1];
    data_[2] = other[2];
    data_[3] = other[3];
    data_[4] = other[4];
    data_[5] = other[5];
    data_[6] = other[6];
    data_[7] = other[7];
    data_[8] = other[8];
    return *this;
  }

  /// \brief Move assignment operator with same accessor
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  /// This operator exists to avoid issues with template deduction.
  KOKKOS_FUNCTION Matrix3<T, Accessor>& operator=(Matrix3<T, Accessor>&& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = other[0];
    data_[1] = other[1];
    data_[2] = other[2];
    data_[3] = other[3];
    data_[4] = other[4];
    data_[5] = other[5];
    data_[6] = other[6];
    data_[7] = other[7];
    data_[8] = other[8];
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Element access operator via flat index
  /// \param[in] row The row index.
  KOKKOS_FUNCTION
  T& operator[](int index) {
    return data_[index];
  }

  /// \brief Const element access operator via flat index
  /// \param[in] row The row index.
  KOKKOS_FUNCTION
  const T& operator[](int index) const {
    return data_[index];
  }

  /// \brief Element access operator via flat index
  /// \param[in] index The flat index.
  KOKKOS_FUNCTION
  T& operator()(int index) {
    return data_[index];
  }

  /// \brief Const element access operator via flat index
  /// \param[in] index The flat index.
  KOKKOS_FUNCTION
  const T& operator()(int index) const {
    return data_[index];
  }

  /// \brief Element access operator via row and column indices
  /// \note This operator is preferred over using m[row][col]
  /// \param[in] row The row index.
  /// \param[in] col The column index.
  KOKKOS_FUNCTION
  T& operator()(int row, int col) {
    return data_[row * 3 + col];
  }

  /// \brief Const element access operators
  /// \note This operator is preferred over using m[row][col]
  /// \param[in] row The row index.
  /// \param[in] col The column index.
  KOKKOS_FUNCTION
  const T& operator()(int row, int col) const {
    return data_[row * 3 + col];
  }

  /// \brief Get the internal data accessor
  KOKKOS_FUNCTION
  Accessor& data() {
    return data_;
  }

  /// \brief Get the internal data accessor
  KOKKOS_FUNCTION
  const Accessor& data() const {
    return data_;
  }

  /// \brief Get a copy of a certain row of the matrix
  /// \param[in] row The row index.
  KOKKOS_FUNCTION
  Vector3<non_const_value_type> get_row(int row) const {
    return Vector3<non_const_value_type>(data_[row * 3], data_[row * 3 + 1], data_[row * 3 + 2]);
  }

  /// \brief Get a copy of a certain column of the matrix
  /// \param[in] col The column index.
  KOKKOS_FUNCTION
  Vector3<non_const_value_type> get_column(int col) const {
    return Vector3<non_const_value_type>(data_[col], data_[col + 3], data_[col + 6]);
  }
  //@}

  //! \name Setters and modifiers
  //@{

  /// \brief Set all elements of the matrix
  /// \param[in] a11 The element at row 1, column 1.
  /// \param[in] a12 The element at row 1, column 2.
  /// \param[in] a13 The element at row 1, column 3.
  /// \param[in] a21 The element at row 2, column 1.
  /// \param[in] a22 The element at row 2, column 2.
  /// \param[in] a23 The element at row 2, column 3.
  /// \param[in] a31 The element at row 3, column 1.
  /// \param[in] a32 The element at row 3, column 2.
  /// \param[in] a33 The element at row 3, column 3.
  KOKKOS_FUNCTION
  void set(const T& a11, const T& a12, const T& a13, const T& a21, const T& a22, const T& a23, const T& a31,
           const T& a32, const T& a33)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = a11;
    data_[1] = a12;
    data_[2] = a13;
    data_[3] = a21;
    data_[4] = a22;
    data_[5] = a23;
    data_[6] = a31;
    data_[7] = a32;
    data_[8] = a33;
  }

  /// \brief Set all elements of the matrix using an accessor
  /// \param[in] accessor A valid accessor.
  /// \note A Matrix3 is also a valid accessor.
  KOKKOS_FUNCTION
  template <ValidAccessor<T> OtherAccessor>
  void set(const OtherAccessor& accessor)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = accessor[0];
    data_[1] = accessor[1];
    data_[2] = accessor[2];
    data_[3] = accessor[3];
    data_[4] = accessor[4];
    data_[5] = accessor[5];
    data_[6] = accessor[6];
    data_[7] = accessor[7];
    data_[8] = accessor[8];
  }

  /// \brief Set a certain row of the matrix
  /// \param[in] i The row index.
  /// \param[in] ai1 The element at row i, column 1.
  /// \param[in] ai2 The element at row i, column 2.
  /// \param[in] ai3 The element at row i, column 3.
  KOKKOS_FUNCTION
  void set_row(const int& i, const T& ai1, const T& ai2, const T& ai3)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[i * 3] = ai1;
    data_[i * 3 + 1] = ai2;
    data_[i * 3 + 2] = ai3;
  }

  /// \brief Set a certain row of the matrix
  /// \param[in] i The row index.
  /// \param[in] row The row vector.
  template <typename OtherAccessor>
  KOKKOS_FUNCTION void set_row(const int& i, const Vector3<T, OtherAccessor>& row)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[i * 3] = row[0];
    data_[i * 3 + 1] = row[1];
    data_[i * 3 + 2] = row[2];
  }

  /// \brief Set a certain column of the matrix
  /// \param[in] j The column index.
  /// \param[in] a1j The element at row 1, column j.
  /// \param[in] a2j The element at row 2, column j.
  /// \param[in] a3j The element at row 3, column j.
  KOKKOS_FUNCTION
  void set_column(const int& j, const T& a1j, const T& a2j, const T& a3j)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[j] = a1j;
    data_[j + 3] = a2j;
    data_[j + 6] = a3j;
  }

  /// \brief Set a certain column of the matrix
  /// \param[in] j The column index.
  /// \param[in] col The column vector.
  template <typename OtherAccessor>
  KOKKOS_FUNCTION void set_column(const int& j, const Vector3<T, OtherAccessor>& col)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[j] = col[0];
    data_[j + 3] = col[1];
    data_[j + 6] = col[2];
  }

  /// \brief Fill all elements of the matrix with a single value
  /// \param[in] value The value to set all elements to.
  KOKKOS_FUNCTION void fill(const T& value)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = value;
    data_[1] = value;
    data_[2] = value;
    data_[3] = value;
    data_[4] = value;
    data_[5] = value;
    data_[6] = value;
    data_[7] = value;
    data_[8] = value;
  }
  //@}

  //! \name Unary operators
  //@{

  /// \brief Unary plus operator
  KOKKOS_FUNCTION
  Matrix3<T> operator+() const {
    return *this;
  }

  /// \brief Unary minus operator
  KOKKOS_FUNCTION
  Matrix3<T> operator-() const {
    Matrix3<T> result;
    result(0) = -data_[0];
    result(1) = -data_[1];
    result(2) = -data_[2];
    result(3) = -data_[3];
    result(4) = -data_[4];
    result(5) = -data_[5];
    result(6) = -data_[6];
    result(7) = -data_[7];
    result(8) = -data_[8];
    return result;
  }
  //@}

  //! \name Addition and subtraction
  //@{

  /// \brief Matrix-matrix addition
  /// \param[in] other The other matrix.
  template <typename U, typename OtherAccessor>
  KOKKOS_FUNCTION auto operator+(const Matrix3<U, OtherAccessor>& other) const -> Matrix3<std::common_type_t<T, U>> {
    using CommonType = std::common_type_t<T, U>;
    Matrix3<CommonType> result;
    result(0) = static_cast<CommonType>(data_[0]) + static_cast<CommonType>(other(0));
    result(1) = static_cast<CommonType>(data_[1]) + static_cast<CommonType>(other(1));
    result(2) = static_cast<CommonType>(data_[2]) + static_cast<CommonType>(other(2));
    result(3) = static_cast<CommonType>(data_[3]) + static_cast<CommonType>(other(3));
    result(4) = static_cast<CommonType>(data_[4]) + static_cast<CommonType>(other(4));
    result(5) = static_cast<CommonType>(data_[5]) + static_cast<CommonType>(other(5));
    result(6) = static_cast<CommonType>(data_[6]) + static_cast<CommonType>(other(6));
    result(7) = static_cast<CommonType>(data_[7]) + static_cast<CommonType>(other(7));
    result(8) = static_cast<CommonType>(data_[8]) + static_cast<CommonType>(other(8));
    return result;
  }

  /// \brief Matrix-matrix addition
  /// \param[in] other The other matrix.
  template <typename U, typename OtherAccessor>
  KOKKOS_FUNCTION Matrix3<T, Accessor>& operator+=(const Matrix3<U, OtherAccessor>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] += static_cast<T>(other(0));
    data_[1] += static_cast<T>(other(1));
    data_[2] += static_cast<T>(other(2));
    data_[3] += static_cast<T>(other(3));
    data_[4] += static_cast<T>(other(4));
    data_[5] += static_cast<T>(other(5));
    data_[6] += static_cast<T>(other(6));
    data_[7] += static_cast<T>(other(7));
    data_[8] += static_cast<T>(other(8));
    return *this;
  }

  /// \brief Matrix-matrix subtraction
  /// \param[in] other The other matrix.
  template <typename U, typename OtherAccessor>
  KOKKOS_FUNCTION auto operator-(const Matrix3<U, OtherAccessor>& other) const -> Matrix3<std::common_type_t<T, U>> {
    using CommonType = std::common_type_t<T, U>;
    Matrix3<CommonType> result;
    result(0) = static_cast<CommonType>(data_[0]) - static_cast<CommonType>(other(0));
    result(1) = static_cast<CommonType>(data_[1]) - static_cast<CommonType>(other(1));
    result(2) = static_cast<CommonType>(data_[2]) - static_cast<CommonType>(other(2));
    result(3) = static_cast<CommonType>(data_[3]) - static_cast<CommonType>(other(3));
    result(4) = static_cast<CommonType>(data_[4]) - static_cast<CommonType>(other(4));
    result(5) = static_cast<CommonType>(data_[5]) - static_cast<CommonType>(other(5));
    result(6) = static_cast<CommonType>(data_[6]) - static_cast<CommonType>(other(6));
    result(7) = static_cast<CommonType>(data_[7]) - static_cast<CommonType>(other(7));
    result(8) = static_cast<CommonType>(data_[8]) - static_cast<CommonType>(other(8));
    return result;
  }

  /// \brief Matrix-matrix subtraction
  /// \param[in] other The other matrix.
  template <typename U, typename OtherAccessor>
  KOKKOS_FUNCTION Matrix3<T, Accessor>& operator-=(const Matrix3<U, OtherAccessor>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] -= static_cast<T>(other(0));
    data_[1] -= static_cast<T>(other(1));
    data_[2] -= static_cast<T>(other(2));
    data_[3] -= static_cast<T>(other(3));
    data_[4] -= static_cast<T>(other(4));
    data_[5] -= static_cast<T>(other(5));
    data_[6] -= static_cast<T>(other(6));
    data_[7] -= static_cast<T>(other(7));
    data_[8] -= static_cast<T>(other(8));
    return *this;
  }

  /// \brief Matrix-scalar addition
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION auto operator+(const U& scalar) const -> Matrix3<std::common_type_t<T, U>> {
    using CommonType = std::common_type_t<T, U>;
    Matrix3<CommonType> result;
    result(0) = static_cast<CommonType>(data_[0]) + static_cast<CommonType>(scalar);
    result(1) = static_cast<CommonType>(data_[1]) + static_cast<CommonType>(scalar);
    result(2) = static_cast<CommonType>(data_[2]) + static_cast<CommonType>(scalar);
    result(3) = static_cast<CommonType>(data_[3]) + static_cast<CommonType>(scalar);
    result(4) = static_cast<CommonType>(data_[4]) + static_cast<CommonType>(scalar);
    result(5) = static_cast<CommonType>(data_[5]) + static_cast<CommonType>(scalar);
    result(6) = static_cast<CommonType>(data_[6]) + static_cast<CommonType>(scalar);
    result(7) = static_cast<CommonType>(data_[7]) + static_cast<CommonType>(scalar);
    result(8) = static_cast<CommonType>(data_[8]) + static_cast<CommonType>(scalar);
    return result;
  }

  /// \brief Matrix-scalar addition
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION Matrix3<T, Accessor>& operator+=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] += static_cast<T>(scalar);
    data_[1] += static_cast<T>(scalar);
    data_[2] += static_cast<T>(scalar);
    data_[3] += static_cast<T>(scalar);
    data_[4] += static_cast<T>(scalar);
    data_[5] += static_cast<T>(scalar);
    data_[6] += static_cast<T>(scalar);
    data_[7] += static_cast<T>(scalar);
    data_[8] += static_cast<T>(scalar);
    return *this;
  }

  /// \brief Matrix-scalar subtraction
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION auto operator-(const U& scalar) const -> Matrix3<std::common_type_t<T, U>> {
    using CommonType = std::common_type_t<T, U>;
    Matrix3<CommonType> result;
    result(0) = static_cast<CommonType>(data_[0]) - static_cast<CommonType>(scalar);
    result(1) = static_cast<CommonType>(data_[1]) - static_cast<CommonType>(scalar);
    result(2) = static_cast<CommonType>(data_[2]) - static_cast<CommonType>(scalar);
    result(3) = static_cast<CommonType>(data_[3]) - static_cast<CommonType>(scalar);
    result(4) = static_cast<CommonType>(data_[4]) - static_cast<CommonType>(scalar);
    result(5) = static_cast<CommonType>(data_[5]) - static_cast<CommonType>(scalar);
    result(6) = static_cast<CommonType>(data_[6]) - static_cast<CommonType>(scalar);
    result(7) = static_cast<CommonType>(data_[7]) - static_cast<CommonType>(scalar);
    result(8) = static_cast<CommonType>(data_[8]) - static_cast<CommonType>(scalar);
    return result;
  }

  /// \brief Matrix-scalar subtraction
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION Matrix3<T, Accessor>& operator-=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] -= static_cast<T>(scalar);
    data_[1] -= static_cast<T>(scalar);
    data_[2] -= static_cast<T>(scalar);
    data_[3] -= static_cast<T>(scalar);
    data_[4] -= static_cast<T>(scalar);
    data_[5] -= static_cast<T>(scalar);
    data_[6] -= static_cast<T>(scalar);
    data_[7] -= static_cast<T>(scalar);
    data_[8] -= static_cast<T>(scalar);
    return *this;
  }
  //@}

  //! \name Multiplication and division
  //@{

  /// \brief Matrix-matrix multiplication
  /// \param[in] other The other matrix.
  template <typename U, typename OtherAccessor>
  KOKKOS_FUNCTION auto operator*(const Matrix3<U, OtherAccessor>& other) const -> Matrix3<std::common_type_t<T, U>> {
    using CommonType = std::common_type_t<T, U>;
    Matrix3<CommonType> result;
    result(0, 0) = static_cast<CommonType>(data_[0]) * static_cast<CommonType>(other(0, 0)) +
                   static_cast<CommonType>(data_[1]) * static_cast<CommonType>(other(1, 0)) +
                   static_cast<CommonType>(data_[2]) * static_cast<CommonType>(other(2, 0));
    result(0, 1) = static_cast<CommonType>(data_[0]) * static_cast<CommonType>(other(0, 1)) +
                   static_cast<CommonType>(data_[1]) * static_cast<CommonType>(other(1, 1)) +
                   static_cast<CommonType>(data_[2]) * static_cast<CommonType>(other(2, 1));
    result(0, 2) = static_cast<CommonType>(data_[0]) * static_cast<CommonType>(other(0, 2)) +
                   static_cast<CommonType>(data_[1]) * static_cast<CommonType>(other(1, 2)) +
                   static_cast<CommonType>(data_[2]) * static_cast<CommonType>(other(2, 2));
    result(1, 0) = static_cast<CommonType>(data_[3]) * static_cast<CommonType>(other(0, 0)) +
                   static_cast<CommonType>(data_[4]) * static_cast<CommonType>(other(1, 0)) +
                   static_cast<CommonType>(data_[5]) * static_cast<CommonType>(other(2, 0));
    result(1, 1) = static_cast<CommonType>(data_[3]) * static_cast<CommonType>(other(0, 1)) +
                   static_cast<CommonType>(data_[4]) * static_cast<CommonType>(other(1, 1)) +
                   static_cast<CommonType>(data_[5]) * static_cast<CommonType>(other(2, 1));
    result(1, 2) = static_cast<CommonType>(data_[3]) * static_cast<CommonType>(other(0, 2)) +
                   static_cast<CommonType>(data_[4]) * static_cast<CommonType>(other(1, 2)) +
                   static_cast<CommonType>(data_[5]) * static_cast<CommonType>(other(2, 2));
    result(2, 0) = static_cast<CommonType>(data_[6]) * static_cast<CommonType>(other(0, 0)) +
                   static_cast<CommonType>(data_[7]) * static_cast<CommonType>(other(1, 0)) +
                   static_cast<CommonType>(data_[8]) * static_cast<CommonType>(other(2, 0));
    result(2, 1) = static_cast<CommonType>(data_[6]) * static_cast<CommonType>(other(0, 1)) +
                   static_cast<CommonType>(data_[7]) * static_cast<CommonType>(other(1, 1)) +
                   static_cast<CommonType>(data_[8]) * static_cast<CommonType>(other(2, 1));
    result(2, 2) = static_cast<CommonType>(data_[6]) * static_cast<CommonType>(other(0, 2)) +
                   static_cast<CommonType>(data_[7]) * static_cast<CommonType>(other(1, 2)) +
                   static_cast<CommonType>(data_[8]) * static_cast<CommonType>(other(2, 2));
    return result;
  }

  /// \brief Matrix-matrix multiplication
  /// \param[in] other The other matrix.
  template <typename U, typename OtherAccessor>
  KOKKOS_FUNCTION Matrix3<T, Accessor>& operator*=(const Matrix3<U, OtherAccessor>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    std::remove_const_t<T> temp[9];
    temp[0] = data_[0] * static_cast<T>(other(0, 0)) + data_[1] * static_cast<T>(other(1, 0)) +
              data_[2] * static_cast<T>(other(2, 0));
    temp[1] = data_[0] * static_cast<T>(other(0, 1)) + data_[1] * static_cast<T>(other(1, 1)) +
              data_[2] * static_cast<T>(other(2, 1));
    temp[2] = data_[0] * static_cast<T>(other(0, 2)) + data_[1] * static_cast<T>(other(1, 2)) +
              data_[2] * static_cast<T>(other(2, 2));
    temp[3] = data_[3] * static_cast<T>(other(0, 0)) + data_[4] * static_cast<T>(other(1, 0)) +
              data_[5] * static_cast<T>(other(2, 0));
    temp[4] = data_[3] * static_cast<T>(other(0, 1)) + data_[4] * static_cast<T>(other(1, 1)) +
              data_[5] * static_cast<T>(other(2, 1));
    temp[5] = data_[3] * static_cast<T>(other(0, 2)) + data_[4] * static_cast<T>(other(1, 2)) +
              data_[5] * static_cast<T>(other(2, 2));
    temp[6] = data_[6] * static_cast<T>(other(0, 0)) + data_[7] * static_cast<T>(other(1, 0)) +
              data_[8] * static_cast<T>(other(2, 0));
    temp[7] = data_[6] * static_cast<T>(other(0, 1)) + data_[7] * static_cast<T>(other(1, 1)) +
              data_[8] * static_cast<T>(other(2, 1));
    temp[8] = data_[6] * static_cast<T>(other(0, 2)) + data_[7] * static_cast<T>(other(1, 2)) +
              data_[8] * static_cast<T>(other(2, 2));

    data_[0] = temp[0];
    data_[1] = temp[1];
    data_[2] = temp[2];
    data_[3] = temp[3];
    data_[4] = temp[4];
    data_[5] = temp[5];
    data_[6] = temp[6];
    data_[7] = temp[7];
    data_[8] = temp[8];

    return *this;
  }

  /// \brief Matrix-vector multiplication
  /// \param[in] other The other vector.
  template <typename U, typename OtherAccessor>
  KOKKOS_FUNCTION auto operator*(const Vector3<U, OtherAccessor>& other) const -> Vector3<std::common_type_t<T, U>> {
    using CommonType = std::common_type_t<T, U>;
    Vector3<CommonType> result;
    result[0] = static_cast<CommonType>(data_[0]) * static_cast<CommonType>(other[0]) +
                static_cast<CommonType>(data_[1]) * static_cast<CommonType>(other[1]) +
                static_cast<CommonType>(data_[2]) * static_cast<CommonType>(other[2]);
    result[1] = static_cast<CommonType>(data_[3]) * static_cast<CommonType>(other[0]) +
                static_cast<CommonType>(data_[4]) * static_cast<CommonType>(other[1]) +
                static_cast<CommonType>(data_[5]) * static_cast<CommonType>(other[2]);
    result[2] = static_cast<CommonType>(data_[6]) * static_cast<CommonType>(other[0]) +
                static_cast<CommonType>(data_[7]) * static_cast<CommonType>(other[1]) +
                static_cast<CommonType>(data_[8]) * static_cast<CommonType>(other[2]);
    return result;
  }

  /// \brief Matrix-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION auto operator*(const U& scalar) const -> Matrix3<std::common_type_t<T, U>> {
    using CommonType = std::common_type_t<T, U>;
    Matrix3<CommonType> result;
    result(0) = static_cast<CommonType>(data_[0]) * static_cast<CommonType>(scalar);
    result(1) = static_cast<CommonType>(data_[1]) * static_cast<CommonType>(scalar);
    result(2) = static_cast<CommonType>(data_[2]) * static_cast<CommonType>(scalar);
    result(3) = static_cast<CommonType>(data_[3]) * static_cast<CommonType>(scalar);
    result(4) = static_cast<CommonType>(data_[4]) * static_cast<CommonType>(scalar);
    result(5) = static_cast<CommonType>(data_[5]) * static_cast<CommonType>(scalar);
    result(6) = static_cast<CommonType>(data_[6]) * static_cast<CommonType>(scalar);
    result(7) = static_cast<CommonType>(data_[7]) * static_cast<CommonType>(scalar);
    result(8) = static_cast<CommonType>(data_[8]) * static_cast<CommonType>(scalar);
    return result;
  }

  /// \brief Matrix-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION Matrix3<T, Accessor>& operator*=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] *= static_cast<T>(scalar);
    data_[1] *= static_cast<T>(scalar);
    data_[2] *= static_cast<T>(scalar);
    data_[3] *= static_cast<T>(scalar);
    data_[4] *= static_cast<T>(scalar);
    data_[5] *= static_cast<T>(scalar);
    data_[6] *= static_cast<T>(scalar);
    data_[7] *= static_cast<T>(scalar);
    data_[8] *= static_cast<T>(scalar);
    return *this;
  }

  /// \brief Matrix-scalar division
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION auto operator/(const U& scalar) const -> Matrix3<std::common_type_t<T, U>> {
    using CommonType = std::common_type_t<T, U>;
    Matrix3<CommonType> result;
    result(0) = static_cast<CommonType>(data_[0]) / static_cast<CommonType>(scalar);
    result(1) = static_cast<CommonType>(data_[1]) / static_cast<CommonType>(scalar);
    result(2) = static_cast<CommonType>(data_[2]) / static_cast<CommonType>(scalar);
    result(3) = static_cast<CommonType>(data_[3]) / static_cast<CommonType>(scalar);
    result(4) = static_cast<CommonType>(data_[4]) / static_cast<CommonType>(scalar);
    result(5) = static_cast<CommonType>(data_[5]) / static_cast<CommonType>(scalar);
    result(6) = static_cast<CommonType>(data_[6]) / static_cast<CommonType>(scalar);
    result(7) = static_cast<CommonType>(data_[7]) / static_cast<CommonType>(scalar);
    result(8) = static_cast<CommonType>(data_[8]) / static_cast<CommonType>(scalar);
    return result;
  }

  /// \brief Matrix-scalar division
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION Matrix3<T, Accessor>& operator/=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] /= static_cast<T>(scalar);
    data_[1] /= static_cast<T>(scalar);
    data_[2] /= static_cast<T>(scalar);
    data_[3] /= static_cast<T>(scalar);
    data_[4] /= static_cast<T>(scalar);
    data_[5] /= static_cast<T>(scalar);
    data_[6] /= static_cast<T>(scalar);
    data_[7] /= static_cast<T>(scalar);
    data_[8] /= static_cast<T>(scalar);
    return *this;
  }
  //@}

  //! \name Static methods
  //@{

  /// \brief Get the identity matrix
  KOKKOS_FUNCTION static Matrix3<T> identity() {
    return Matrix3<T>(T(1), T(0), T(0), T(0), T(1), T(0), T(0), T(0), T(1));
  }

  /// \brief Get the ones matrix
  KOKKOS_FUNCTION static Matrix3<T> ones() {
    return Matrix3<T>(T(1), T(1), T(1), T(1), T(1), T(1), T(1), T(1), T(1));
  }

  /// \brief Get the zero matrix
  KOKKOS_FUNCTION static Matrix3<T> zeros() {
    return Matrix3<T>(T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0));
  }
  //@}

  //! \name Friends <3
  //@{

  // Declare the << operator as a friend
  template <typename U, typename OtherAccessor>
  friend std::ostream& operator<<(std::ostream& os, const Matrix3<U, OtherAccessor>& mat);

  // We are friends with all Matrix3s  regardless of their Accessor or type
  template <typename U, ValidAccessor<U> OtherAccessor>
  friend class Matrix3;
  //@}
};  // class Matrix3

//! \name Non-member functions
//@{

//! \name Write to output stream
//@{

/// \brief Write the quaternion to an output stream
/// \param[in] os The output stream.
/// \param[in] quat The quaternion.
template <typename T, typename Accessor>
KOKKOS_FUNCTION std::ostream& operator<<(std::ostream& os, const Matrix3<T, Accessor>& mat) {
  os << "(" << mat(0, 0) << ", " << mat(0, 1) << ", " << mat(0, 2) << ")" << std::endl;
  os << "(" << mat(1, 0) << ", " << mat(1, 1) << ", " << mat(1, 2) << ")" << std::endl;
  os << "(" << mat(2, 0) << ", " << mat(2, 1) << ", " << mat(2, 2) << ")" << std::endl;
  return os;
}
//@}

//! \name Non-member comparison functions
//@{

/// \brief Matrix-matrix equality (element-wise within a tolerance)
/// \param[in] mat1 The first matrix.
/// \param[in] mat2 The second matrix.
/// \param[in] tol The tolerance (default is determined by the given type).
template <typename U, typename OtherAccessor, typename T, typename Accessor>
KOKKOS_FUNCTION bool is_close(const Matrix3<U, OtherAccessor>& mat1, const Matrix3<T, Accessor>& mat2,
                              const std::common_type_t<T, U>& tol = get_zero_tolerance<std::common_type_t<T, U>>()) {
  using CommonType = std::common_type_t<T, U>;
  if constexpr (std::is_floating_point_v<CommonType>) {
    // For floating-point types, compare with a tolerance
    return (std::abs(static_cast<CommonType>(mat1(0)) - static_cast<CommonType>(mat2(0))) < tol) &&
           (std::abs(static_cast<CommonType>(mat1(1)) - static_cast<CommonType>(mat2(1))) < tol) &&
           (std::abs(static_cast<CommonType>(mat1(2)) - static_cast<CommonType>(mat2(2))) < tol) &&
           (std::abs(static_cast<CommonType>(mat1(3)) - static_cast<CommonType>(mat2(3))) < tol) &&
           (std::abs(static_cast<CommonType>(mat1(4)) - static_cast<CommonType>(mat2(4))) < tol) &&
           (std::abs(static_cast<CommonType>(mat1(5)) - static_cast<CommonType>(mat2(5))) < tol) &&
           (std::abs(static_cast<CommonType>(mat1(6)) - static_cast<CommonType>(mat2(6))) < tol) &&
           (std::abs(static_cast<CommonType>(mat1(7)) - static_cast<CommonType>(mat2(7))) < tol) &&
           (std::abs(static_cast<CommonType>(mat1(8)) - static_cast<CommonType>(mat2(8))) < tol);
  } else {
    // For integral types, compare with exact equality
    return (mat1(0) == mat2(0)) && (mat1(1) == mat2(1)) && (mat1(2) == mat2(2)) && (mat1(3) == mat2(3)) &&
           (mat1(4) == mat2(4)) && (mat1(5) == mat2(5)) && (mat1(6) == mat2(6)) && (mat1(7) == mat2(7)) &&
           (mat1(8) == mat2(8));
  }
}

/// \brief Matrix-matrix equality (element-wise within a relaxed tolerance)
/// \param[in] mat1 The first matrix.
/// \param[in] mat2 The second matrix.
/// \param[in] tol The tolerance (default is determined by the given type).
template <typename U, typename OtherAccessor, typename T, typename Accessor>
KOKKOS_FUNCTION bool is_approx_close(
    const Matrix3<U, OtherAccessor>& mat1, const Matrix3<T, Accessor>& mat2,
    const std::common_type_t<T, U>& tol = get_relaxed_zero_tolerance<std::common_type_t<T, U>>()) {
  return is_close(mat1, mat2, tol);
}
//@}

//! \name Non-member addition and subtraction operators
//@{

/// \brief Scalar-matrix addition
/// \param[in] scalar The scalar.
/// \param[in] mat The matrix.
template <typename U, typename T, typename Accessor>
KOKKOS_FUNCTION auto operator+(const U& scalar, const Matrix3<T, Accessor>& mat) -> Matrix3<std::common_type_t<T, U>> {
  return mat + scalar;
}

/// \brief Scalar-matrix subtraction
/// \param[in] scalar The scalar.
/// \param[in] mat The matrix.
template <typename U, typename T, typename Accessor>
KOKKOS_FUNCTION auto operator-(const U& scalar, const Matrix3<T, Accessor>& mat) -> Matrix3<std::common_type_t<T, U>> {
  return -mat + scalar;
}
//@}

//! \name Non-member multiplication and division operators
//@{

/// \brief Scalar-matrix multiplication
/// \param[in] scalar The scalar.
/// \param[in] mat The matrix.
template <typename U, typename T, typename Accessor>
KOKKOS_FUNCTION auto operator*(const U& scalar, const Matrix3<T, Accessor>& mat) -> Matrix3<std::common_type_t<T, U>> {
  return mat * scalar;
}

/// \brief Vector matrix multiplication (v^T M)
/// \param[in] vec The vector.
/// \param[in] mat The matrix.
template <typename U, typename OtherAccessor, typename T, typename Accessor>
KOKKOS_FUNCTION auto operator*(const Vector3<U, OtherAccessor>& vec, const Matrix3<T, Accessor>& mat)
    -> Vector3<std::common_type_t<T, U>> {
  using CommonType = std::common_type_t<T, U>;
  Vector3<CommonType> result;
  result[0] = static_cast<CommonType>(vec[0]) * static_cast<CommonType>(mat(0, 0)) +
              static_cast<CommonType>(vec[1]) * static_cast<CommonType>(mat(1, 0)) +
              static_cast<CommonType>(vec[2]) * static_cast<CommonType>(mat(2, 0));
  result[1] = static_cast<CommonType>(vec[0]) * static_cast<CommonType>(mat(0, 1)) +
              static_cast<CommonType>(vec[1]) * static_cast<CommonType>(mat(1, 1)) +
              static_cast<CommonType>(vec[2]) * static_cast<CommonType>(mat(2, 1));
  result[2] = static_cast<CommonType>(vec[0]) * static_cast<CommonType>(mat(0, 2)) +
              static_cast<CommonType>(vec[1]) * static_cast<CommonType>(mat(1, 2)) +
              static_cast<CommonType>(vec[2]) * static_cast<CommonType>(mat(2, 2));
  return result;
}
//@}

//! \name Basic arithmetic reduction operations
//@{

/// \brief Matrix determinant
/// \param[in] mat The matrix.
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto determinant(const Matrix3<T, Accessor>& mat) {
  return mat(0, 0) * (mat(1, 1) * mat(2, 2) - mat(1, 2) * mat(2, 1)) -
         mat(0, 1) * (mat(1, 0) * mat(2, 2) - mat(1, 2) * mat(2, 0)) +
         mat(0, 2) * (mat(1, 0) * mat(2, 1) - mat(1, 1) * mat(2, 0));
}

/// \brief Matrix trace
/// \param[in] mat The matrix.
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto trace(const Matrix3<T, Accessor>& mat) {
  return mat(0, 0) + mat(1, 1) + mat(2, 2);
}

/// \brief Sum of all elements
/// \param[in] mat The matrix.
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto sum(const Matrix3<T, Accessor>& mat) {
  return mat(0, 0) + mat(0, 1) + mat(0, 2) + mat(1, 0) + mat(1, 1) + mat(1, 2) + mat(2, 0) + mat(2, 1) + mat(2, 2);
}

/// \brief Product of all elements
/// \param[in] mat The matrix.
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto product(const Matrix3<T, Accessor>& mat) {
  return mat(0, 0) * mat(0, 1) * mat(0, 2) * mat(1, 0) * mat(1, 1) * mat(1, 2) * mat(2, 0) * mat(2, 1) * mat(2, 2);
}

/// \brief Minimum element of the matrix
/// \param[in] mat The matrix.
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto min(const Matrix3<T, Accessor>& mat) {
  // min of row 1 using ternary operator
  const T min_row1 = (mat(0, 0) < mat(0, 1)) ? ((mat(0, 0) < mat(0, 2)) ? mat(0, 0) : mat(0, 2))
                                             : ((mat(0, 1) < mat(0, 2)) ? mat(0, 1) : mat(0, 2));
  // min of row 2 using ternary operator
  const T min_row2 = (mat(1, 0) < mat(1, 1)) ? ((mat(1, 0) < mat(1, 2)) ? mat(1, 0) : mat(1, 2))
                                             : ((mat(1, 1) < mat(1, 2)) ? mat(1, 1) : mat(1, 2));
  // min of row 3 using ternary operator
  const T min_row3 = (mat(2, 0) < mat(2, 1)) ? ((mat(2, 0) < mat(2, 2)) ? mat(2, 0) : mat(2, 2))
                                             : ((mat(2, 1) < mat(2, 2)) ? mat(2, 1) : mat(2, 2));
  return (min_row1 < min_row2) ? ((min_row1 < min_row3) ? min_row1 : min_row3)
                               : ((min_row2 < min_row3) ? min_row2 : min_row3);
}

/// \brief Maximum element of the matrix
/// \param[in] mat The matrix.
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto max(const Matrix3<T, Accessor>& mat) {
  // max of row 1 using ternary operator
  const T max_row1 = (mat(0, 0) > mat(0, 1)) ? ((mat(0, 0) > mat(0, 2)) ? mat(0, 0) : mat(0, 2))
                                             : ((mat(0, 1) > mat(0, 2)) ? mat(0, 1) : mat(0, 2));
  // max of row 2 using ternary operator
  const T max_row2 = (mat(1, 0) > mat(1, 1)) ? ((mat(1, 0) > mat(1, 2)) ? mat(1, 0) : mat(1, 2))
                                             : ((mat(1, 1) > mat(1, 2)) ? mat(1, 1) : mat(1, 2));
  // max of row 3 using ternary operator
  const T max_row3 = (mat(2, 0) > mat(2, 1)) ? ((mat(2, 0) > mat(2, 2)) ? mat(2, 0) : mat(2, 2))
                                             : ((mat(2, 1) > mat(2, 2)) ? mat(2, 1) : mat(2, 2));
  return (max_row1 > max_row2) ? ((max_row1 > max_row3) ? max_row1 : max_row3)
                               : ((max_row2 > max_row3) ? max_row2 : max_row3);
}

/// \brief Mean of all elements (returns a double if T is an integral type, otherwise returns T)
/// \param[in] mat The matrix.
template <typename T, typename Accessor, typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_FUNCTION OutputType mean(const Matrix3<T, Accessor>& mat) {
  auto sum = mundy::math::sum(mat);
  return static_cast<OutputType>(sum) / OutputType(9);
}

/// \brief Mean of all elements (returns a float if T is an integral type, otherwise returns T)
/// \param[in] mat The matrix.
template <typename T, typename Accessor, typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_FUNCTION OutputType mean_f(const Matrix3<T, Accessor>& mat) {
  return mean<T, Accessor, OutputType>(mat);
}

/// \brief Variance of all elements (returns a double if T is an integral type, otherwise returns T)
/// \param[in] mat The matrix.
template <typename T, typename Accessor, typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_FUNCTION OutputType variance(const Matrix3<T, Accessor>& mat) {
  auto mat_mean = mean<T, Accessor, OutputType>(mat);
  return ((static_cast<OutputType>(mat(0)) - static_cast<OutputType>(mat_mean)) *
              (static_cast<OutputType>(mat(0)) - static_cast<OutputType>(mat_mean)) +
          (static_cast<OutputType>(mat(1)) - static_cast<OutputType>(mat_mean)) *
              (static_cast<OutputType>(mat(1)) - static_cast<OutputType>(mat_mean)) +
          (static_cast<OutputType>(mat(2)) - static_cast<OutputType>(mat_mean)) *
              (static_cast<OutputType>(mat(2)) - static_cast<OutputType>(mat_mean)) +
          (static_cast<OutputType>(mat(3)) - static_cast<OutputType>(mat_mean)) *
              (static_cast<OutputType>(mat(3)) - static_cast<OutputType>(mat_mean)) +
          (static_cast<OutputType>(mat(4)) - static_cast<OutputType>(mat_mean)) *
              (static_cast<OutputType>(mat(4)) - static_cast<OutputType>(mat_mean)) +
          (static_cast<OutputType>(mat(5)) - static_cast<OutputType>(mat_mean)) *
              (static_cast<OutputType>(mat(5)) - static_cast<OutputType>(mat_mean)) +
          (static_cast<OutputType>(mat(6)) - static_cast<OutputType>(mat_mean)) *
              (static_cast<OutputType>(mat(6)) - static_cast<OutputType>(mat_mean)) +
          (static_cast<OutputType>(mat(7)) - static_cast<OutputType>(mat_mean)) *
              (static_cast<OutputType>(mat(7)) - static_cast<OutputType>(mat_mean)) +
          (static_cast<OutputType>(mat(8)) - static_cast<OutputType>(mat_mean)) *
              (static_cast<OutputType>(mat(8)) - static_cast<OutputType>(mat_mean))) /
         OutputType(9);
}

/// \brief Variance of all elements (returns a float if T is an integral type, otherwise returns T)
/// \param[in] mat The matrix.
template <typename T, typename Accessor, typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_FUNCTION OutputType variance_f(const Matrix3<T, Accessor>& mat) {
  return variance<T, Accessor, OutputType>(mat);
}

/// \brief Standard deviation of all elements (returns a double if T is an integral type, otherwise returns T)
/// \param[in] mat The matrix.
template <typename T, typename Accessor, typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_FUNCTION OutputType stddev(const Matrix3<T, Accessor>& mat) {
  return std::sqrt(variance<T, Accessor, OutputType>(mat));
}

/// \brief Standard deviation of all elements (returns a float if T is an integral type, otherwise returns T)
/// \param[in] mat The matrix.
template <typename T, typename Accessor, typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_FUNCTION OutputType stddev_f(const Matrix3<T, Accessor>& mat) {
  return stddev<T, Accessor, OutputType>(mat);
}
//@}

//! \name Special matrix operations
//@{

/// \brief Matrix transpose
/// \param[in] mat The matrix.
template <typename T, typename Accessor>
KOKKOS_FUNCTION Matrix3<std::remove_const_t<T>> transpose(const Matrix3<T, Accessor>& mat) {
  Matrix3<std::remove_const_t<T>> result;
  result(0, 0) = mat(0, 0);
  result(0, 1) = mat(1, 0);
  result(0, 2) = mat(2, 0);
  result(1, 0) = mat(0, 1);
  result(1, 1) = mat(1, 1);
  result(1, 2) = mat(2, 1);
  result(2, 0) = mat(0, 2);
  result(2, 1) = mat(1, 2);
  result(2, 2) = mat(2, 2);
  return result;
}

/// \brief Matrix inverse (returns a double if T is an integral type, otherwise returns T)
/// \tparam T The input matrix element type.
/// \tparam Accessor The accessor for the Matrix3, assuming this is part of your implementation.
/// \tparam OutputElementType The output matrix element type, defaults T if T is an integral type (e.g., float or
/// double) and double otherwise.
template <typename T, typename Accessor,
          typename OutputElementType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_FUNCTION Matrix3<OutputElementType> inverse(const Matrix3<T, Accessor>& mat) {
  const auto det = determinant(mat);
  MUNDY_THROW_ASSERT(det != T(0), std::runtime_error, "Matrix3<T>: matrix is singular.");

  Matrix3<OutputElementType> result;
  result(0, 0) = static_cast<OutputElementType>(mat(1, 1)) * static_cast<OutputElementType>(mat(2, 2)) -
                 static_cast<OutputElementType>(mat(1, 2)) * static_cast<OutputElementType>(mat(2, 1));
  result(0, 1) = static_cast<OutputElementType>(mat(0, 2)) * static_cast<OutputElementType>(mat(2, 1)) -
                 static_cast<OutputElementType>(mat(0, 1)) * static_cast<OutputElementType>(mat(2, 2));
  result(0, 2) = static_cast<OutputElementType>(mat(0, 1)) * static_cast<OutputElementType>(mat(1, 2)) -
                 static_cast<OutputElementType>(mat(0, 2)) * static_cast<OutputElementType>(mat(1, 1));
  result(1, 0) = static_cast<OutputElementType>(mat(1, 2)) * static_cast<OutputElementType>(mat(2, 0)) -
                 static_cast<OutputElementType>(mat(1, 0)) * static_cast<OutputElementType>(mat(2, 2));
  result(1, 1) = static_cast<OutputElementType>(mat(0, 0)) * static_cast<OutputElementType>(mat(2, 2)) -
                 static_cast<OutputElementType>(mat(0, 2)) * static_cast<OutputElementType>(mat(2, 0));
  result(1, 2) = static_cast<OutputElementType>(mat(0, 2)) * static_cast<OutputElementType>(mat(1, 0)) -
                 static_cast<OutputElementType>(mat(0, 0)) * static_cast<OutputElementType>(mat(1, 2));
  result(2, 0) = static_cast<OutputElementType>(mat(1, 0)) * static_cast<OutputElementType>(mat(2, 1)) -
                 static_cast<OutputElementType>(mat(1, 1)) * static_cast<OutputElementType>(mat(2, 0));
  result(2, 1) = static_cast<OutputElementType>(mat(0, 1)) * static_cast<OutputElementType>(mat(2, 0)) -
                 static_cast<OutputElementType>(mat(0, 0)) * static_cast<OutputElementType>(mat(2, 1));
  result(2, 2) = static_cast<OutputElementType>(mat(0, 0)) * static_cast<OutputElementType>(mat(1, 1)) -
                 static_cast<OutputElementType>(mat(0, 1)) * static_cast<OutputElementType>(mat(1, 0));
  return result / det;
}

/// \brief Matrix inverse (returns a float if T is an integral type, otherwise returns T)
/// \tparam T The input matrix element type.
/// \tparam Accessor The accessor for the Matrix3, assuming this is part of your implementation.
/// \tparam OutputElementType The output matrix element type, defaults T if T is an integral type (e.g., float or
/// double) and float otherwise.
template <typename T, typename Accessor,
          typename OutputElementType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_FUNCTION auto inverse_f(const Matrix3<T, Accessor>& mat) {
  return inverse<T, Accessor, OutputElementType>(mat);
}

/// \brief Matrix Frobenius inner product
/// \param[in] a The left matrix.
/// \param[in] b The right matrix.
template <typename U, typename OtherAccessor, typename T, typename Accessor>
KOKKOS_FUNCTION auto frobenius_inner_product(const Matrix3<U, OtherAccessor>& a, const Matrix3<T, Accessor>& b) {
  using CommonType = std::common_type_t<T, U>;
  return static_cast<CommonType>(a(0, 0)) * static_cast<CommonType>(b(0, 0)) +
         static_cast<CommonType>(a(0, 1)) * static_cast<CommonType>(b(0, 1)) +
         static_cast<CommonType>(a(0, 2)) * static_cast<CommonType>(b(0, 2)) +
         static_cast<CommonType>(a(1, 0)) * static_cast<CommonType>(b(1, 0)) +
         static_cast<CommonType>(a(1, 1)) * static_cast<CommonType>(b(1, 1)) +
         static_cast<CommonType>(a(1, 2)) * static_cast<CommonType>(b(1, 2)) +
         static_cast<CommonType>(a(2, 0)) * static_cast<CommonType>(b(2, 0)) +
         static_cast<CommonType>(a(2, 1)) * static_cast<CommonType>(b(2, 1)) +
         static_cast<CommonType>(a(2, 2)) * static_cast<CommonType>(b(2, 2));
}
//@}

//! \name Special vector operations with matrices
//@{

/// \brief Outer product of two vectors
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <typename U, typename OtherAccessor, typename T, typename Accessor>
KOKKOS_FUNCTION auto outer_product(const Vector3<U, OtherAccessor>& a, const Vector3<T, Accessor>& b)
    -> Matrix3<std::common_type_t<T, U>> {
  using CommonType = std::common_type_t<T, U>;
  Matrix3<CommonType> result;
  result(0, 0) = static_cast<CommonType>(a[0]) * static_cast<CommonType>(b[0]);
  result(0, 1) = static_cast<CommonType>(a[0]) * static_cast<CommonType>(b[1]);
  result(0, 2) = static_cast<CommonType>(a[0]) * static_cast<CommonType>(b[2]);
  result(1, 0) = static_cast<CommonType>(a[1]) * static_cast<CommonType>(b[0]);
  result(1, 1) = static_cast<CommonType>(a[1]) * static_cast<CommonType>(b[1]);
  result(1, 2) = static_cast<CommonType>(a[1]) * static_cast<CommonType>(b[2]);
  result(2, 0) = static_cast<CommonType>(a[2]) * static_cast<CommonType>(b[0]);
  result(2, 1) = static_cast<CommonType>(a[2]) * static_cast<CommonType>(b[1]);
  result(2, 2) = static_cast<CommonType>(a[2]) * static_cast<CommonType>(b[2]);
  return result;
}
//@}

//! \name Matrix norms
//@{

/// \brief Matrix Frobenius norm
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto frobenius_norm(const Matrix3<T, Accessor>& mat) {
  return std::sqrt(frobenius_inner_product(mat, mat));
}

/// \brief Matrix infinity norm
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto infinity_norm(const Matrix3<T, Accessor>& mat) {
  // Max absolute row sum
  const T row0_sum = std::abs(mat(0)) + std::abs(mat(1)) + std::abs(mat(2));
  const T row1_sum = std::abs(mat(3)) + std::abs(mat(4)) + std::abs(mat(5));
  const T row2_sum = std::abs(mat(6)) + std::abs(mat(7)) + std::abs(mat(8));

  return (row0_sum > row1_sum) ? ((row0_sum > row2_sum) ? row0_sum : row2_sum)
                               : ((row1_sum > row2_sum) ? row1_sum : row2_sum);
}

/// \brief Matrix 1-norm
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto one_norm(const Matrix3<T, Accessor>& mat) {
  // Max absolute column sum
  const T col0_sum = std::abs(mat(0)) + std::abs(mat(3)) + std::abs(mat(6));
  const T col1_sum = std::abs(mat(1)) + std::abs(mat(4)) + std::abs(mat(7));
  const T col2_sum = std::abs(mat(2)) + std::abs(mat(5)) + std::abs(mat(8));

  return (col0_sum > col1_sum) ? ((col0_sum > col2_sum) ? col0_sum : col2_sum)
                               : ((col1_sum > col2_sum) ? col1_sum : col2_sum);
}

/// \brief Matrix 2-norm
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto two_norm(const Matrix3<T, Accessor>& mat) {
  return std::sqrt(frobenius_inner_product(mat, mat));
}
//@}

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
template <typename T, typename Accessor>
KOKKOS_FUNCTION Matrix3<T, Accessor> get_matrix3_view(const Accessor& data) {
  return Matrix3<T, Accessor>(data);
}
//@}

//@}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_MATRIX3_HPP_
