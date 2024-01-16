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

#ifndef MUNDY_MATH_QUATERNION_HPP_
#define MUNDY_MATH_QUATERNION_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core libs
#include <initializer_list>  // for std::initializer_list

// Our libs
#include <mundy/throw_assert.hpp>    // for MUNDY_THROW_ASSERT
#include <mundy_math/Accessor.hpp>   // for mundy::math::ValidAccessor
#include <mundy_math/Array.hpp>      // for mundy::math::Array
#include <mundy_math/Matrix3.hpp>    // for mundy::math::Matrix3
#include <mundy_math/Tolerance.hpp>  // for mundy::math::get_default_tolerance
#include <mundy_math/Vector3.hpp>    // for mundy::math::Vector3

namespace mundy {

namespace math {

//! \name Forward declare functions required by Quaternion that also require Quaternion to be defined
//@{
template <typename T, ValidAccessor<T> Accessor = Array<T, 4>>
  requires std::is_floating_point_v<T>
class Quaternion;

/// \brief Get the inverse of a quaternion
/// \param[in] quat The quaternion.
template <typename T, typename Accessor>
KOKKOS_FUNCTION Quaternion<std::remove_const_t<T>> inverse(const Quaternion<T, Accessor> &quat);

/// \brief Get the norm of a quaternion
/// \param[in] quat The quaternion.
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto norm(const Quaternion<T, Accessor> &quat);
//@}

/// \brief Quaternion class with floating point entries (an integer-valued quaternion doesn't make much sense)
/// \tparam T The type of the entries.
/// \tparam Accessor The type of the accessor.
///
/// This class is designed to be used with Kokkos. It is a simple quaternion with arithmetic entries. It is templated
/// on the type of the entries and Accessor type. See Accessor.hpp for more details on the Accessor type requirements.
///
/// The goal of Quaternion is to be a lightweight class that can be used with Kokkos to perform mathematical operations
/// on vectors in R3. It does not own the data, but rather it is templated on an Accessor type that provides access to
/// the underlying data. This allows us to use Quaternion with Kokkos Views, raw pointers, or any other type that meets
/// the ValidAccessor requirements without copying the data. This is especially important for GPU-compatable code.
///
/// Quaternions can be constructed by passing an accessor to the constructor. However, if the accessor has a 4-argument
/// constructor, then the Quaternion can also be constructed by passing the elements directly to the constructor.
/// Similarly, if the accessor has an initializer list constructor, then the Quaternion can be constructed by passing an
/// initializer list to the constructor. This is a convenience feature which makes working with the default accessor
/// (Array<T, 4>) easier. For example, the following are all valid ways to construct a Quaternion:
///
/// \code{.cpp}
///   // Constructs a Quaternion with the default accessor (Array<int, 4>)
///   Quaternion<double> quat1({1.0, 2.0, 3.0, 4.0});
///   Quaternion<double> quat2(1.0, 2.0, 3.0, 4.0);
///   Quaternion<double> quat3(Array<int, 3>({1.0, 2.0, 3.0, 4.0}));
///   Quaternion<double> quat4;
///   quat4.set(1.0, 2.0, 3.0, 4.0);
///
///   // Construct a Quaternion from a double array
///   double data[4] = {1.0, 2.0, 3.0, 4.0};
///   Quaternion<double, double*> quat5(data);
///   Quaternion<double, double*> quat6{1.0, 2.0, 3.0, 4.0};
///   // Not allowed as double* doesn't have a 4-argument constructor
///   // Quaternion<double, double*> quat7(1.0, 2.0, 3.0, 4.0);
/// \endcode
///
/// \note Accessors may be owning or non-owning, that is irrelevant to the Quaternion class; however, these accessors
/// should be lightweight such that they can be copied around without much overhead. Furthermore, the lifetime of the
/// data underlying the accessor should be as long as the Quaternion that use it.
template <typename T, ValidAccessor<T> Accessor>  // the forward declaration sets the default Accessor
  requires std::is_floating_point_v<T>
class Quaternion {
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
  KOKKOS_FUNCTION Quaternion()
    requires HasDefaultConstructor<Accessor>
      : data_() {
  }

  /// \brief Constructor from a given accessor
  /// \param[in] data The accessor.
  KOKKOS_FUNCTION
  Quaternion(const Accessor &data) : data_(data) {
  }

  /// \brief Constructor to initialize all elements
  /// \param[in] w The scalar component.
  /// \param[in] x The x component.
  /// \param[in] y The y component.
  /// \param[in] z The z component.
  /// \note This constructor is only enabled if the Accessor has a 3-argument constructor.
  KOKKOS_FUNCTION Quaternion(const T &w, const T &x, const T &y, const T &z)
    requires Has4ArgConstructor<Accessor, T>
      : data_(w, x, y, z) {
  }

  /// \brief Constructor to initialize all elements via initializer list
  /// \param[in] list The initializer list.
  KOKKOS_FUNCTION Quaternion(const std::initializer_list<T> &list)
    requires HasInitializerListConstructor<Accessor, T>
      : data_{list.begin()[0], list.begin()[1], list.begin()[2], list.begin()[3]} {
    MUNDY_THROW_ASSERT(list.size() == 4, std::invalid_argument, "Quaternion: Initializer list must have 4 elements.");
  }

  /// \brief Destructor
  KOKKOS_FUNCTION
  ~Quaternion() {
  }

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION Quaternion(const Quaternion<T, Accessor> &other) : data_(other.data()) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION Quaternion(Quaternion<T, Accessor> &&other) : data_(std::move(other.data())) {
  }

  /// \brief Deep copy assignment operator with different accessor
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  template <typename OtherAccessor>
  KOKKOS_FUNCTION Quaternion<T, Accessor> &operator=(const Quaternion<T, OtherAccessor> &other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = other[0];
    data_[1] = other[1];
    data_[2] = other[2];
    data_[3] = other[3];
    return *this;
  }

  /// \brief Deep copy assignment operator with same accessor
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  KOKKOS_FUNCTION Quaternion<T, Accessor> &operator=(const Quaternion<T, Accessor> &other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = other[0];
    data_[1] = other[1];
    data_[2] = other[2];
    data_[3] = other[3];
    return *this;
  }

  /// \brief Move assignment operator with different accessor
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  template <typename OtherAccessor>
  KOKKOS_FUNCTION Quaternion<T, Accessor> &operator=(Quaternion<T, OtherAccessor> &&other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = other[0];
    data_[1] = other[1];
    data_[2] = other[2];
    data_[3] = other[3];
    return *this;
  }

  /// \brief Move assignment operator with same accessor
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  KOKKOS_FUNCTION Quaternion<T, Accessor> &operator=(Quaternion<T, Accessor> &&other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = other[0];
    data_[1] = other[1];
    data_[2] = other[2];
    data_[3] = other[3];
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_FUNCTION
  T &operator[](int index) {
    return data_[index];
  }

  /// \brief Const element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_FUNCTION
  const T &operator[](int index) const {
    return data_[index];
  }

  /// \brief Element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_FUNCTION
  T &operator()(int index) {
    return data_[index];
  }

  /// \brief Const element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_FUNCTION
  const T &operator()(int index) const {
    return data_[index];
  }

  /// \brief Get a reference to the scalar component
  KOKKOS_FUNCTION
  T &w() {
    return data_[0];
  }

  /// \brief Get a reference to the scalar component
  KOKKOS_FUNCTION
  const T &w() const {
    return data_[0];
  }

  /// \brief Get a reference to the x component
  KOKKOS_FUNCTION
  T &x() {
    return data_[1];
  }

  /// \brief Get a reference to the x component
  KOKKOS_FUNCTION
  const T &x() const {
    return data_[1];
  }

  /// \brief Get a reference to the y component
  KOKKOS_FUNCTION
  T &y() {
    return data_[2];
  }

  /// \brief Get a reference to the y component
  KOKKOS_FUNCTION
  const T &y() const {
    return data_[2];
  }

  /// \brief Get a reference to the z component
  KOKKOS_FUNCTION
  T &z() {
    return data_[3];
  }

  /// \brief Get a reference to the z component
  KOKKOS_FUNCTION
  const T &z() const {
    return data_[3];
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

  /// \brief Get a copy of the quaternion vector component
  KOKKOS_FUNCTION
  Vector3<T> vector() const {
    return Vector3<T>(data_[1], data_[2], data_[3]);
  }
  //@}

  //! \name Setters and modifiers
  //@{

  /// \brief Set all elements of the quaternion
  /// \param[in] w The scalar component.
  /// \param[in] x The x component.
  /// \param[in] y The y component.
  /// \param[in] z The z component.
  KOKKOS_FUNCTION
  void set(const T &w, const T &x, const T &y, const T &z)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = w;
    data_[1] = x;
    data_[2] = y;
    data_[3] = z;
  }

  /// \brief Set all elements of the quaternion
  /// \param[in] w The scalar component.
  /// \param[in] vec The vector component.
  KOKKOS_FUNCTION
  void set(const T &w, const Vector3<T> &vec)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = w;
    data_[1] = vec[0];
    data_[2] = vec[1];
    data_[3] = vec[2];
  }

  /// \brief Set all elements of the vector using an accessor
  /// \param[in] accessor A valid accessor.
  /// \note A Quaternion is also a valid accessor.
  KOKKOS_FUNCTION
  template <ValidAccessor<T> OtherAccessor>
  void set(const OtherAccessor &accessor)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] = accessor[0];
    data_[1] = accessor[1];
    data_[2] = accessor[2];
    data_[3] = accessor[3];
  }

  /// \brief Set the quaternion vector component
  /// \param[in] vec The vector.
  KOKKOS_FUNCTION
  void set_vector(const Vector3<T> &vec)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[1] = vec[0];
    data_[2] = vec[1];
    data_[3] = vec[2];
  }

  /// \brief Normalize the quaternion in place
  KOKKOS_FUNCTION
  void normalize()
    requires HasNonConstAccessOperator<Accessor, T>
  {
    const T inv_norm = T(1) / norm(*this);
    data_[0] *= inv_norm;
    data_[1] *= inv_norm;
    data_[2] *= inv_norm;
    data_[3] *= inv_norm;
  }

  /// \brief Conjugate the quaternion in place
  KOKKOS_FUNCTION
  void conjugate()
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[1] = -data_[1];
    data_[2] = -data_[2];
    data_[3] = -data_[3];
  }

  /// \brief Invert the quaternion in place
  KOKKOS_FUNCTION
  void invert()
    requires HasNonConstAccessOperator<Accessor, T>
  {
    const T inv_norm_squared =
        T(1) / (data_[0] * data_[0] + data_[1] * data_[1] + data_[2] * data_[2] + data_[3] * data_[3]);
    conjugate();
    data_[0] *= inv_norm_squared;
    data_[1] *= inv_norm_squared;
    data_[2] *= inv_norm_squared;
    data_[3] *= inv_norm_squared;
  }
  //@}

  //! \name Unary operators
  //@{

  /// \brief Unary plus operator
  KOKKOS_FUNCTION
  Quaternion<T> operator+() const {
    return Quaternion<T>(+data_[0], +data_[1], +data_[2], +data_[3]);
  }

  /// \brief Unary minus operator
  KOKKOS_FUNCTION
  Quaternion<T> operator-() const {
    return Quaternion<T>(-data_[0], -data_[1], -data_[2], -data_[3]);
  }
  //@}

  //! \name Addition and subtraction
  //@{

  /// \brief Quaternion-quaternion addition
  /// \param[in] other The other quaternion.
  template <typename U, typename OtherAccessor>
  KOKKOS_FUNCTION auto operator+(const Quaternion<U, OtherAccessor> &other) const -> Quaternion<decltype(T() + U())> {
    using ReturnType = decltype(T() + U());
    Quaternion<ReturnType> result;
    result[0] = data_[0] + other[0];
    result[1] = data_[1] + other[1];
    result[2] = data_[2] + other[2];
    result[3] = data_[3] + other[3];
    return result;
  }

  /// \brief Quaternion-quaternion addition
  /// \param[in] other The other quaternion.
  template <typename U, typename OtherAccessor>
  KOKKOS_FUNCTION Quaternion<T, Accessor> &operator+=(const Quaternion<U, OtherAccessor> &other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] += other[0];
    data_[1] += other[1];
    data_[2] += other[2];
    data_[3] += other[3];
    return *this;
  }

  /// \brief Quaternion-quaternion subtraction
  /// \param[in] other The other quaternion.
  template <typename U, typename OtherAccessor>
  KOKKOS_FUNCTION auto operator-(const Quaternion<U, OtherAccessor> &other) const -> Quaternion<decltype(T() - U())> {
    using ReturnType = decltype(T() - U());
    Quaternion<ReturnType> result;
    result[0] = data_[0] - other[0];
    result[1] = data_[1] - other[1];
    result[2] = data_[2] - other[2];
    result[3] = data_[3] - other[3];
    return result;
  }

  /// \brief Quaternion-quaternion subtraction
  /// \param[in] other The other quaternion.
  template <typename U, typename OtherAccessor>
  KOKKOS_FUNCTION Quaternion<T, Accessor> &operator-=(const Quaternion<U, OtherAccessor> &other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] -= other[0];
    data_[1] -= other[1];
    data_[2] -= other[2];
    data_[3] -= other[3];
    return *this;
  }
  //@}

  //! \name Multiplication and division
  //@{

  /// \brief Quaternion-quaternion multiplication
  /// \param[in] other The other quaternion.
  template <typename U, typename OtherAccessor>
  KOKKOS_FUNCTION auto operator*(const Quaternion<U, OtherAccessor> &other) const -> Quaternion<decltype(T() * U())> {
    using ReturnType = decltype(T() * U());
    Quaternion<ReturnType> result;
    result[0] = data_[0] * other[0] - data_[1] * other[1] - data_[2] * other[2] - data_[3] * other[3];
    result[1] = data_[0] * other[1] + data_[1] * other[0] + data_[2] * other[3] - data_[3] * other[2];
    result[2] = data_[0] * other[2] - data_[1] * other[3] + data_[2] * other[0] + data_[3] * other[1];
    result[3] = data_[0] * other[3] + data_[1] * other[2] - data_[2] * other[1] + data_[3] * other[0];
    return result;
  }

  /// \brief Quaternion-quaternion multiplication
  /// \param[in] other The other quaternion.
  template <typename U, typename OtherAccessor>
  KOKKOS_FUNCTION Quaternion<T, Accessor> &operator*=(const Quaternion<U, OtherAccessor> &other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    const T w = data_[0] * other[0] - data_[1] * other[1] - data_[2] * other[2] - data_[3] * other[3];
    const T x = data_[0] * other[1] + data_[1] * other[0] + data_[2] * other[3] - data_[3] * other[2];
    const T y = data_[0] * other[2] - data_[1] * other[3] + data_[2] * other[0] + data_[3] * other[1];
    const T z = data_[0] * other[3] + data_[1] * other[2] - data_[2] * other[1] + data_[3] * other[0];
    data_[0] = w;
    data_[1] = x;
    data_[2] = y;
    data_[3] = z;
    return *this;
  }

  /// \brief Quaternion-vector multiplication (same as R * v)
  /// \param[in] vec The vector.
  template <typename U, typename OtherAccessor>
  KOKKOS_FUNCTION auto operator*(const Vector3<U, OtherAccessor> &vec) const -> Vector3<decltype(T() * U())> {
    // Quaternion-vector multiplication consists of three parts:
    // 1. The vector is converted to a quaternion with a scalar component of 0
    // 2. The quaternion-quaternion multiplication is performed
    // 3. The quaternion is converted back to a vector
    using ReturnType = decltype(T() * U());
    const Quaternion<U> vec_quat(0.0, vec[0], vec[1], vec[2]);
    const auto quat_inv = inverse(*this);
    const Quaternion<ReturnType> quat_result = (*this) * vec_quat * quat_inv;
    return quat_result.vector();
  }

  /// \brief Quaternion-matrix multiplication
  /// \param[in] other The other matrix.
  template <typename U, typename OtherAccessor>
  KOKKOS_FUNCTION auto operator*(const Matrix3<U, OtherAccessor> &mat) const -> Matrix3<decltype(T() * U())> {
    // Quaternion-vector multiplication consists of applying the quaternion to each column of the matrix
    using ReturnType = decltype(T() * U());
    Matrix3<ReturnType> result;
    result.set_column(0, (*this) * mat.get_column(0));
    result.set_column(1, (*this) * mat.get_column(1));
    result.set_column(2, (*this) * mat.get_column(2));
    return result;
  }

  /// \brief Quaternion-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION auto operator*(const U &scalar) const -> Quaternion<decltype(T() * U())> {
    using ReturnType = decltype(T() * U());
    Quaternion<ReturnType> result;
    result[0] = data_[0] * scalar;
    result[1] = data_[1] * scalar;
    result[2] = data_[2] * scalar;
    result[3] = data_[3] * scalar;
    return result;
  }

  /// \brief Matrix-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION Quaternion<T, Accessor> &operator*=(const U &scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] *= scalar;
    data_[1] *= scalar;
    data_[2] *= scalar;
    data_[3] *= scalar;
    return *this;
  }

  /// \brief Matrix-scalar division
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION auto operator/(const U &scalar) const -> Quaternion<decltype(T() / U())> {
    using ReturnType = decltype(T() / U());
    Quaternion<ReturnType> result;
    result[0] = data_[0] / scalar;
    result[1] = data_[1] / scalar;
    result[2] = data_[2] / scalar;
    result[3] = data_[3] / scalar;
    return result;
  }

  /// \brief Matrix-scalar division
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION Quaternion<T, Accessor> &operator/=(const U &scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    data_[0] /= scalar;
    data_[1] /= scalar;
    data_[2] /= scalar;
    data_[3] /= scalar;
    return *this;
  }
  //@}

  //! \name Static methods
  //@{

  /// \brief Get the identity quaternion
  KOKKOS_FUNCTION
  static Quaternion<T> identity() {
    return Quaternion<T>(T(1), T(0), T(0), T(0));
  }
  //@}

  //! \name Friends <3
  //@{

  // Declare the << operator as a friend
  template <typename U, typename OtherAccessor>
  friend std::ostream &operator<<(std::ostream &os, const Quaternion<U, OtherAccessor> &quat);

  // We are friends with all Quaternions regardless of their Accessor or type
  template <typename U, ValidAccessor<U> OtherAccessor>
  friend class Quaternion;
  //@}
};  // Quaternion

//! \name Non-member functions
//@{

//! \name Write to output stream
//@{

/// \brief Write the quaternion to an output stream
/// \param[in] os The output stream.
/// \param[in] quat The quaternion.
template <typename T, typename Accessor>
KOKKOS_FUNCTION std::ostream &operator<<(std::ostream &os, const Quaternion<T, Accessor> &quat) {
  os << "(" << quat[0] << ", " << quat[1] << ", " << quat[2] << ", " << quat[3] << ")";
  return os;
}
//@}

//! \name Non-member comparison functions
//@{

/// \brief Quaternion-quaternion equality (element-wise within a tolerance)
/// \param[in] quat1 The first quaternion.
/// \param[in] quat2 The second quaternion.
/// \param[in] tol The tolerance.
template <typename U, typename OtherAccessor, typename T, typename Accessor>
KOKKOS_FUNCTION bool is_close(const Quaternion<U, OtherAccessor> &quat1, const Quaternion<T, Accessor> &quat2,
                              const decltype(U() - T()) &tol = get_default_tolerance<decltype(U() - T())>()) {
  return std::abs(quat1[0] - quat2[0]) < tol && std::abs(quat1[1] - quat2[1]) < tol &&
         std::abs(quat1[2] - quat2[2]) < tol && std::abs(quat1[3] - quat2[3]) < tol;
}
//@}

//! \name Non-member multiplication and division operators
//@{

/// \brief Scalar-quaternion multiplication
/// \param[in] scalar The scalar.
/// \param[in] quat The quaternion.
template <typename U, typename T, typename Accessor>
KOKKOS_FUNCTION auto operator*(const U &scalar, const Quaternion<T, Accessor> &quat)
    -> Quaternion<decltype(U() * T())> {
  return quat * scalar;
}

/// \brief Vector-quaternion multiplication (same as v^T * R = transpose(R^T * v))
/// \param[in] vec The vector.
/// \param[in] quat The quaternion.
template <typename U, typename OtherAccessor, typename T, typename Accessor>
KOKKOS_FUNCTION auto operator*(const Vector3<U, OtherAccessor> &vec, const Quaternion<T, Accessor> &quat)
    -> Vector3<decltype(U() * T())> {
  // Vector-quaternion multiplication consists of three parts:
  // 1. The vector is converted to a quaternion with a scalar component of 0
  // 2. The quaternion-quaternion multiplication is performed
  // 3. The quaternion is converted back to a vector
  using ReturnType = decltype(U() * T());
  const Quaternion<U> vec_quat(0.0, vec[0], vec[1], vec[2]);
  const auto quat_inv = inverse(quat);
  const Quaternion<ReturnType> quat_result = quat_inv * vec_quat * quat;
  return quat_result.vector();
}

/// \brief Matrix-quaternion multiplication (same as R * M)
/// \param[in] mat The matrix.
/// \param[in] quat The quaternion.
template <typename U, typename OtherAccessor, typename T, typename Accessor>
KOKKOS_FUNCTION auto operator*(const Matrix3<U, OtherAccessor> &mat, const Quaternion<T, Accessor> &quat)
    -> Matrix3<decltype(U() * T())> {
  // Quaternion-vector multiplication consists of applying the quaternion to each row of the matrix
  using ReturnType = decltype(U() * T());
  Matrix3<ReturnType> result;
  result.set_row(0, mat.get_row(0) * quat);
  result.set_row(1, mat.get_row(1) * quat);
  result.set_row(2, mat.get_row(2) * quat);
  return result;
}
//@}

//! \name Special quaternion operations
//@{

/// \brief Get the dot product of two quaternions
/// \param[in] q1 The first quaternion.
/// \param[in] q2 The second quaternion.
template <typename U, typename OtherAccessor, typename T, typename Accessor>
KOKKOS_FUNCTION auto dot(const Quaternion<U, OtherAccessor> &q1, const Quaternion<T, Accessor> &q2) {
  return q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3];
}

/// \brief Get the conjugate of a quaternion
/// \param[in] quat The quaternion.
template <typename T, typename Accessor>
KOKKOS_FUNCTION Quaternion<std::remove_const_t<T>> conjugate(const Quaternion<T, Accessor> &quat) {
  Quaternion<std::remove_const_t<T>> result;
  result[0] = quat[0];
  result[1] = -quat[1];
  result[2] = -quat[2];
  result[3] = -quat[3];
  return result;
}

/// \brief Get the inverse of a quaternion
/// \param[in] quat The quaternion.
template <typename T, typename Accessor>
KOKKOS_FUNCTION Quaternion<std::remove_const_t<T>> inverse(const Quaternion<T, Accessor> &quat) {
  const T inv_norm_squared = T(1) / (quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);
  return conjugate(quat) * inv_norm_squared;
}

/// \brief Get the norm of a quaternion
/// \param[in] quat The quaternion.
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto norm(const Quaternion<T, Accessor> &quat) {
  return std::sqrt(quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);
}

/// \brief Get the squared norm of a quaternion
/// \param[in] quat The quaternion.
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto norm_squared(const Quaternion<T, Accessor> &quat) {
  return quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3];
}

/// \brief Get the normalized quaternion
/// \param[in] quat The quaternion.
template <typename T, typename Accessor>
KOKKOS_FUNCTION Quaternion<std::remove_const_t<T>> normalize(const Quaternion<T, Accessor> &quat) {
  const T inv_norm = T(1) / norm(quat);
  return quat * inv_norm;
}

/// \brief Perform spherical linear interpolation between two quaternions
/// \param[in] q1 The first quaternion.
/// \param[in] q2 The second quaternion.
/// \param[in] t The interpolation parameter.
template <typename U, typename OtherAccessor, typename T, typename Accessor, typename V>
  requires std::is_arithmetic_v<V>
KOKKOS_FUNCTION auto slerp(const Quaternion<U, OtherAccessor> &q1, const Quaternion<T, Accessor> &q2, const V t)
    -> Quaternion<decltype(U() * T() * V())> {
  using ReturnType = decltype(U() * T() * V());
  const ReturnType epsilon = ReturnType(1e-6);  // Threshold for linear interpolation

  // Compute the dot product
  ReturnType dot_q12 = dot(q1, q2);

  // Adjust second quaternion for negative dot product
  // Note, we cannot directly copy from q2 to q2_adjusted because the Accessor type may be different.
  Quaternion<std::remove_const_t<T>> q2_adjusted;
  q2_adjusted.set(q2);
  if (dot_q12 < 0) {
    dot_q12 = -dot_q12;
    q2_adjusted *= -1;
  }

  // Check for near-parallel case
  if (1 - dot_q12 < epsilon) {
    // Linear Interpolation as fallback
    return Quaternion<ReturnType>{q1[0] + t * (q2_adjusted[0] - q1[0]), q1[1] + t * (q2_adjusted[1] - q1[1]),
                                  q1[2] + t * (q2_adjusted[2] - q1[2]), q1[3] + t * (q2_adjusted[3] - q1[3])};
  } else {
    // Spherical Interpolation
    const ReturnType theta = std::acos(dot_q12);
    const ReturnType sin_theta = std::sin(theta);
    const ReturnType s1 = std::sin((ReturnType(1) - t) * theta) / sin_theta;
    const ReturnType s2 = std::sin(t * theta) / sin_theta;

    return Quaternion<ReturnType>{(s1 * q1[0]) + (s2 * q2_adjusted[0]), (s1 * q1[1]) + (s2 * q2_adjusted[1]),
                                  (s1 * q1[2]) + (s2 * q2_adjusted[2]), (s1 * q1[3]) + (s2 * q2_adjusted[3])};
  }
}

// /// \brief Perform spherical linear interpolation between two quaternions
// /// Source: https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/index.htm
// /// \param[in] q1 The first quaternion.
// /// \param[in] q2 The second quaternion.
// /// \param[in] t The interpolation parameter.
// template <typename U, typename OtherAccessor, typename T, typename Accessor, typename V>
//   requires std::is_arithmetic_v<V>
// KOKKOS_FUNCTION auto slerp(const Quaternion<U, OtherAccessor> &q1, const Quaternion<T, Accessor> &q2, const V t)
//     -> Quaternion<decltype(U() * T() * V())> {
//   using ReturnType = decltype(U() * T() * V());

//   // quaternion to return
//   quat qm = new quat();
//   // Calculate angle between them.
//   double cosHalfTheta = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z;
//   // if qa=qb or qa=-qb then theta = 0 and we can return qa
//   if (abs(cosHalfTheta) >= 1.0) {
//     qm.w = qa.w;
//     qm.x = qa.x;
//     qm.y = qa.y;
//     qm.z = qa.z;
//     return qm;
//   }
//   // Calculate temporary values.
//   double halfTheta = acos(cosHalfTheta);
//   double sinHalfTheta = sqrt(1.0 - cosHalfTheta * cosHalfTheta);
//   // if theta = 180 degrees then result is not fully defined
//   // we could rotate around any axis normal to qa or qb
//   if (fabs(sinHalfTheta) < 0.001) {  // fabs is floating point absolute
//     qm.w = (qa.w * 0.5 + qb.w * 0.5);
//     qm.x = (qa.x * 0.5 + qb.x * 0.5);
//     qm.y = (qa.y * 0.5 + qb.y * 0.5);
//     qm.z = (qa.z * 0.5 + qb.z * 0.5);
//     return qm;
//   }
//   double ratioA = sin((1 - t) * halfTheta) / sinHalfTheta;
//   double ratioB = sin(t * halfTheta) / sinHalfTheta;
//   // calculate Quaternion.
//   qm.w = (qa.w * ratioA + qb.w * ratioB);
//   qm.x = (qa.x * ratioA + qb.x * ratioB);
//   qm.y = (qa.y * ratioA + qb.y * ratioB);
//   qm.z = (qa.z * ratioA + qb.z * ratioB);
//   return qm;
// }

//@}

//! \name Non-member constructors and converters
//@{

/// \brief Get the quaternion from an axis-angle representation
/// \param[in] axis The axis.
/// \param[in] angle The angle.
template <typename T, typename Accessor, typename U>
  requires std::is_arithmetic_v<U>
KOKKOS_FUNCTION auto axis_angle_to_quaternion(const Vector3<T, Accessor> &axis, const T &angle)
    -> Quaternion<decltype(U() * std::sin(T()))> {
  using ReturnType = decltype(U() * std::sin(T()));
  const auto half_angle = T(0.5) * angle;
  const auto sin_half_angle = std::sin(half_angle);
  const auto cos_half_angle = std::cos(half_angle);
  return Quaternion<ReturnType>(cos_half_angle, sin_half_angle * axis[0], sin_half_angle * axis[1],
                                sin_half_angle * axis[2]);
}

/// \brief Get the quaternion from a rotation matrix
/// \param[in] rot_mat The rotation matrix.
template <typename T, typename Accessor>
KOKKOS_FUNCTION auto rotation_matrix_to_quaternion(const Matrix3<T, Accessor> &rot_mat)
    -> Quaternion<decltype(std::sqrt(T()))> {
  // Source: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion<T>/
  using ReturnType = decltype(std::sqrt(T()));
  Quaternion<ReturnType> quat;

  // Computing the quaternion components
  quat.w() = std::sqrt(std::max(0.0, 1.0 + rot_mat(0, 0) + rot_mat(1, 1) + rot_mat(2, 2))) / 2.0;
  quat.x() = std::sqrt(std::max(0.0, 1.0 + rot_mat(0, 0) - rot_mat(1, 1) - rot_mat(2, 2))) / 2.0;
  quat.y() = std::sqrt(std::max(0.0, 1.0 - rot_mat(0, 0) + rot_mat(1, 1) - rot_mat(2, 2))) / 2.0;
  quat.z() = std::sqrt(std::max(0.0, 1.0 - rot_mat(0, 0) - rot_mat(1, 1) + rot_mat(2, 2))) / 2.0;

  // Correcting the signs
  quat.x() = std::copysign(quat[1], rot_mat(2, 1) - rot_mat(1, 2));
  quat.y() = std::copysign(quat[2], rot_mat(0, 2) - rot_mat(2, 0));
  quat.z() = std::copysign(quat[3], rot_mat(1, 0) - rot_mat(0, 1));

  return quat;
}

/// \brief Get the rotation matrix from a quaternion
/// \param[in] quat The quaternion.
template <typename T, typename Accessor>
KOKKOS_FUNCTION Matrix3<std::remove_const_t<T>> quaternion_to_rotation_matrix(const Quaternion<T, Accessor> &quat) {
  Matrix3<std::remove_const_t<T>> rot_mat;
  rot_mat(0, 0) = T(1) - T(2) * quat.y() * quat.y() - T(2) * quat.z() * quat.z();
  rot_mat(0, 1) = T(2) * quat.x() * quat.y() - T(2) * quat.w() * quat.z();
  rot_mat(0, 2) = T(2) * quat.x() * quat.z() + T(2) * quat.w() * quat.y();
  rot_mat(1, 0) = T(2) * quat.x() * quat.y() + T(2) * quat.w() * quat.z();
  rot_mat(1, 1) = T(1) - T(2) * quat.x() * quat.x() - T(2) * quat.z() * quat.z();
  rot_mat(1, 2) = T(2) * quat.y() * quat.z() - T(2) * quat.w() * quat.x();
  rot_mat(2, 0) = T(2) * quat.x() * quat.z() - T(2) * quat.w() * quat.y();
  rot_mat(2, 1) = T(2) * quat.y() * quat.z() + T(2) * quat.w() * quat.x();
  rot_mat(2, 2) = T(1) - T(2) * quat.x() * quat.x() - T(2) * quat.y() * quat.y();

  return rot_mat;
}

/// \brief Get the quaternion from Euler angles
template <typename T>
  requires std::is_arithmetic_v<T>
KOKKOS_FUNCTION Quaternion<std::remove_const_t<T>> euler_to_quat(const T phi, const T theta, const T psi) {
  // Convert Euler angles to quaternion
  Quaternion<std::remove_const_t<T>> quat;
  const T cha1 = std::cos(T(0.5) * phi);
  const T cha2 = std::cos(T(0.5) * theta);
  const T cha3 = std::cos(T(0.5) * psi);
  const T sha1 = std::sin(T(0.5) * phi);
  const T sha2 = std::sin(T(0.5) * theta);
  const T sha3 = std::sin(T(0.5) * psi);
  quat.w() = cha1 * cha2 * cha3 + sha1 * sha2 * sha3;
  quat.x() = sha1 * cha2 * cha3 - cha1 * sha2 * sha3;
  quat.y() = cha1 * sha2 * cha3 + sha1 * cha2 * sha3;
  quat.z() = cha1 * cha2 * sha3 - sha1 * sha2 * cha3;
  return quat;
}
//@}

//! \name Quaternion<T, Accessor> views
//@{

/// \brief A helper function to create a Quaternion<T, Accessor> based on a given accessor.
/// \param[in] data The data accessor.
///
/// In practice, this function is syntactic sugar to avoid having to specify the template parameters
/// when creating a Quaternion<T, Accessor> from a data accessor.
/// Instead of writing
/// \code
///   Quaternion<T, Accessor> quat(data);
/// \endcode
/// you can write
/// \code
///   auto quat = get_quaternion_view<T>(data);
/// \endcode
template <typename T, typename Accessor>
KOKKOS_FUNCTION Quaternion<T, Accessor> get_quaternion_view(const Accessor &data) {
  return Quaternion<T, Accessor>(data);
}
//@}

//@}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_QUATERNION_HPP_
