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

#ifndef MUNDY_GEOM_PRIMITIVES_ELLIPSOID_HPP_
#define MUNDY_GEOM_PRIMITIVES_ELLIPSOID_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <iostream>
#include <stdexcept>
#include <utility>

// Our libs
#include <mundy_core/throw_assert.hpp>      // for MUNDY_THROW_ASSERT
#include <mundy_geom/primitives/Point.hpp>  // for mundy::geom::Point
#include <mundy_math/Quaternion.hpp>        // for mundy::math::Quaternion
#include <mundy_math/Tolerance.hpp>         // for mundy::math::get_zero_tolerance

namespace mundy {

namespace geom {

template <typename Scalar>
class Ellipsoid {
 public:
  //! \name Type aliases
  //@{

  /// \brief The Ellipsoid's scalar type
  using scalar_t = Scalar;
  using point_t = Point<Scalar>;
  using array3_t = mundy::math::Array<Scalar, 3>;
  using orientation_t = mundy::math::Quaternion<Scalar>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor. Default initializes the center and sets the axis lengths to an invalid value of -1
  KOKKOS_FUNCTION
  Ellipsoid()
      : center_(scalar_t(), scalar_t(), scalar_t()),
        axis_lengths_{static_cast<Scalar>(-1), static_cast<Scalar>(-1), static_cast<Scalar>(-1)} {
  }

  /// \brief Constructor to initialize the center and axis lengths.
  /// \param[in] center The center of the Ellipsoid.
  /// \param[in] axis_length_1 The first axis length of the Ellipsoid.
  /// \param[in] axis_length_2 The second axis length of the Ellipsoid.
  /// \param[in] axis_length_3 The third axis length of the Ellipsoid.
  KOKKOS_FUNCTION
  Ellipsoid(const point_t& center, const Scalar& axis_length_1, const Scalar& axis_length_2,
            const Scalar& axis_length_3)
      : center_(center), axis_lengths_{axis_length_1, axis_length_2, axis_length_3} {
  }

  /// \brief Constructor to initialize the center and axis lengths.
  /// \param[in] x The x-coordinate of the center.
  /// \param[in] y The y-coordinate of the center.
  /// \param[in] z The z-coordinate of the center.
  /// \param[in] qw The scalar-component of the orientation quaternion.
  /// \param[in] qx The x-component of the orientation quaternion.
  /// \param[in] qy The y-component of the orientation quaternion.
  /// \param[in] qz The z-component of the orientation quaternion.
  /// \param[in] axis_length_1 The first axis length of the Ellipsoid.
  /// \param[in] axis_length_2 The second axis length of the Ellipsoid.
  /// \param[in] axis_length_3 The third axis length of the Ellipsoid.
  KOKKOS_FUNCTION
  Ellipsoid(const Scalar& x, const Scalar& y, const Scalar& z, const Scalar& qw, const Scalar& qx, const Scalar& qy,
            const Scalar& qz, const Scalar& axis_length_1, const Scalar& axis_length_2, const Scalar& axis_length_3)
      : center_(x, y, z), orientation_(qw, qx, qy, qz), axis_lengths_{axis_length_1, axis_length_2, axis_length_3} {
  }

  /// \brief Constructor to initialize the center and axis lengths.
  /// \param[in] center The center of the Ellipsoid.
  /// \param[in] axis_lengths The axis lengths of the Ellipsoid.
  KOKKOS_FUNCTION
  Ellipsoid(const point_t& center, const orientation_t& orientation, const array3_t& axis_lengths)
      : center_(center), orientation_(orientation), axis_lengths_(axis_lengths) {
  }

  /// \brief Constructor to initialize the center and axis lengths.
  /// \param[in] center The center of the Ellipsoid.
  /// \param[in] axis_lengths The axis lengths of the Ellipsoid.
  KOKKOS_FUNCTION
  Ellipsoid(const point_t& center, const orientation_t& orientation, const Scalar& axis_length_1,
            const Scalar& axis_length_2, const Scalar& axis_length_3)
      : center_(center), orientation_(orientation), axis_lengths_(axis_length_1, axis_length_2, axis_length_3) {
  }

  /// \brief Destructor
  KOKKOS_FUNCTION
  ~Ellipsoid() {
  }

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  Ellipsoid(const Ellipsoid<Scalar>& other)
      : center_(other.center_), orientation_{other.orientation_}, axis_lengths_{other.axis_lengths_} {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  Ellipsoid(Ellipsoid<Scalar>&& other)
      : center_(std::move(other.center_)),
        orientation_{std::move(other.orientation_)},
        axis_lengths_{std::move(other.axis_lengths_)} {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  Ellipsoid<Scalar>& operator=(const Ellipsoid<Scalar>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = other.center_;
    orientation_ = other.orientation_;
    axis_lengths_ = other.axis_lengths_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  Ellipsoid<Scalar>& operator=(Ellipsoid<Scalar>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = std::move(other.center_);
    orientation_ = std::move(other.orientation_);
    axis_lengths_ = std::move(other.axis_lengths_);
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Accessor for the center
  KOKKOS_FUNCTION
  const point_t& center() const {
    return center_;
  }

  /// \brief Accessor for the center
  KOKKOS_FUNCTION
  point_t& center() {
    return center_;
  }

  /// \brief Accessor for the orientation
  KOKKOS_FUNCTION
  const orientation_t& orientation() const {
    return orientation_;
  }

  /// \brief Accessor for the orientation
  KOKKOS_FUNCTION
  orientation_t& orientation() {
    return orientation_;
  }

  /// \brief Accessor for the axis lengths
  KOKKOS_FUNCTION
  const array3_t& axis_lengths() const {
    return axis_lengths_;
  }

  /// \brief Accessor for the axis lengths
  KOKKOS_FUNCTION
  array3_t& axis_lengths() {
    return axis_lengths_;
  }

  /// \brief Accessor for the first axis length
  KOKKOS_FUNCTION
  const Scalar& axis_length_1() const {
    return axis_lengths_[0];
  }

  /// \brief Accessor for the first axis length
  KOKKOS_FUNCTION
  Scalar& axis_length_1() {
    return axis_lengths_[0];
  }

  /// \brief Accessor for the second axis length
  KOKKOS_FUNCTION
  const Scalar& axis_length_2() const {
    return axis_lengths_[1];
  }

  /// \brief Accessor for the second axis length
  KOKKOS_FUNCTION
  Scalar& axis_length_2() {
    return axis_lengths_[1];
  }

  /// \brief Accessor for the third axis length
  KOKKOS_FUNCTION
  const Scalar& axis_length_3() const {
    return axis_lengths_[2];
  }

  /// \brief Accessor for the third axis length
  KOKKOS_FUNCTION
  Scalar& axis_length_3() {
    return axis_lengths_[2];
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the center
  /// \param[in] center The new center.
  KOKKOS_FUNCTION
  void set_center(const point_t& center) {
    center_ = center;
  }

  /// \brief Set the center
  /// \param[in] x The x-coordinate.
  /// \param[in] y The y-coordinate.
  /// \param[in] z The z-coordinate.
  KOKKOS_FUNCTION
  void set_center(const Scalar& x, const Scalar& y, const Scalar& z) {
    center_[0] = x;
    center_[1] = y;
    center_[2] = z;
  }

  /// \brief Set the orientation
  /// \param[in] orientation The new orientation.
  KOKKOS_FUNCTION
  void set_orientation(const orientation_t& orientation) {
    orientation_ = orientation;
  }

  /// \brief Set the orientation
  /// \param[in] qw The scalar-component of the orientation quaternion.
  /// \param[in] qx The x-component of the orientation quaternion.
  /// \param[in] qy The y-component of the orientation quaternion.
  /// \param[in] qz The z-component of the orientation quaternion.
  KOKKOS_FUNCTION
  void set_orientation(const Scalar& qw, const Scalar& qx, const Scalar& qy, const Scalar& qz) {
    orientation_[0] = qw;
    orientation_[1] = qx;
    orientation_[2] = qy;
    orientation_[3] = qz;
  }

  /// \brief Set the axis lengths
  /// \param[in] axis_lengths The new axis lengths.
  KOKKOS_FUNCTION
  void set_axis_lengths(const array3_t& axis_lengths) {
    axis_lengths_ = axis_lengths;
  }

  /// \brief Set the axis lengths
  /// \param[in] axis_length_1 The new first axis length.
  /// \param[in] axis_length_2 The new second axis length.
  /// \param[in] axis_length_3 The new third axis length.
  KOKKOS_FUNCTION
  void set_axis_lengths(const Scalar& axis_length_1, const Scalar& axis_length_2, const Scalar& axis_length_3) {
    axis_lengths_[0] = axis_length_1;
    axis_lengths_[1] = axis_length_2;
    axis_lengths_[2] = axis_length_3;
  }
  //@}

 private:
  point_t center_;
  orientation_t orientation_;
  array3_t axis_lengths_;
};

/// @brief Type trait to determine if a type is am Ellipsoid
template <typename T>
struct is_ellipsoid : std::false_type {};
//
template <typename Scalar>
struct is_ellipsoid<Ellipsoid<Scalar>> : std::true_type {};
//
template <typename Scalar>
struct is_ellipsoid<const Ellipsoid<Scalar>> : std::true_type {};
//
template <typename T>
inline constexpr bool is_ellipsoid_v = is_ellipsoid<T>::value;

/// @brief Concept to check if a type is an Ellipsoid
template <typename EllipsoidType>
concept ValidEllipsoidType =
    requires(std::remove_cv_t<EllipsoidType> ellipsoid, const std::remove_cv_t<EllipsoidType> const_ellipsoid) {
      is_ellipsoid_v<std::remove_cv_t<EllipsoidType>>;
      typename std::remove_cv_t<EllipsoidType>::scalar_t;
      {
        ellipsoid.center()
      } -> std::convertible_to<mundy::geom::Point<typename std::remove_cv_t<EllipsoidType>::scalar_t>>;
      {
        ellipsoid.orientation()
      } -> std::convertible_to<mundy::math::Quaternion<typename std::remove_cv_t<EllipsoidType>::scalar_t>>;
      {
        ellipsoid.axis_lengths()
      } -> std::convertible_to<mundy::math::Array<typename std::remove_cv_t<EllipsoidType>::scalar_t, 3>>;
      { ellipsoid.axis_length_1() } -> std::convertible_to<typename std::remove_cv_t<EllipsoidType>::scalar_t>;
      { ellipsoid.axis_length_2() } -> std::convertible_to<typename std::remove_cv_t<EllipsoidType>::scalar_t>;
      { ellipsoid.axis_length_3() } -> std::convertible_to<typename std::remove_cv_t<EllipsoidType>::scalar_t>;

      {
        const_ellipsoid.center()
      } -> std::convertible_to<const mundy::geom::Point<typename std::remove_cv_t<EllipsoidType>::scalar_t>>;
      {
        const_ellipsoid.orientation()
      } -> std::convertible_to<const mundy::math::Quaternion<typename std::remove_cv_t<EllipsoidType>::scalar_t>>;
      {
        const_ellipsoid.axis_lengths()
      } -> std::convertible_to<const mundy::math::Array<typename std::remove_cv_t<EllipsoidType>::scalar_t, 3>>;
      {
        const_ellipsoid.axis_length_1()
      } -> std::convertible_to<const typename std::remove_cv_t<EllipsoidType>::scalar_t>;
      {
        const_ellipsoid.axis_length_2()
      } -> std::convertible_to<const typename std::remove_cv_t<EllipsoidType>::scalar_t>;
      {
        const_ellipsoid.axis_length_3()
      } -> std::convertible_to<const typename std::remove_cv_t<EllipsoidType>::scalar_t>;
    };  // ValidEllipsoidType

static_assert(ValidEllipsoidType<Ellipsoid<float>> && ValidEllipsoidType<const Ellipsoid<float>> &&
                  ValidEllipsoidType<Ellipsoid<double>> && ValidEllipsoidType<const Ellipsoid<double>>,
              "Ellipsoid should satisfy the ValidEllipsoidType concept.");

//! \name Non-member functions for ValidSphereType objects
//@{

/// \brief Equality operator
KOKKOS_FUNCTION
template <ValidEllipsoidType EllipsoidType1, ValidEllipsoidType EllipsoidType2>
bool operator==(const EllipsoidType1& ellipsoid1, const EllipsoidType2& ellipsoid2) {
  return (ellipsoid1.center() == ellipsoid2.center()) && (ellipsoid1.orientation() == ellipsoid2.orientation()) &&
         (ellipsoid1.axis_lengths() == ellipsoid2.axis_lengths());
}

/// \brief Inequality operator
KOKKOS_FUNCTION
template <ValidEllipsoidType EllipsoidType1, ValidEllipsoidType EllipsoidType2>
bool operator!=(const EllipsoidType1& ellipsoid1, const EllipsoidType2& ellipsoid2) {
  return (ellipsoid1.center() != ellipsoid2.center()) || (ellipsoid1.orientation() != ellipsoid2.orientation()) ||
         (ellipsoid1.axis_lengths() != ellipsoid2.axis_lengths());
}

/// \brief OStream operator
template <ValidEllipsoidType EllipsoidType>
std::ostream& operator<<(std::ostream& os, const EllipsoidType& ellipsoid) {
  os << "{" << ellipsoid.center() << ":" << ellipsoid.orientation() << ":" << ellipsoid.axis_lengths() << "}";
  return os;
}

template <ValidEllipsoidType EllipsoidType>
KOKKOS_FUNCTION mundy::geom::Point<typename EllipsoidType::scalar_t> map_body_frame_normal_to_ellipsoid(
    const mundy::math::Vector3<typename EllipsoidType::scalar_t>& body_frame_nhat, const EllipsoidType& ellipsoid) {
  using Scalar = typename EllipsoidType::scalar_t;
  constexpr Scalar half = static_cast<Scalar>(0.5);
  constexpr Scalar one = static_cast<Scalar>(1.0);
  constexpr Scalar zero = static_cast<Scalar>(0.0);

  const Scalar r1 = half * ellipsoid.axis_length_1();
  const Scalar r2 = half * ellipsoid.axis_length_2();
  const Scalar r3 = half * ellipsoid.axis_length_3();

  const Scalar sign0 = Kokkos::copysign(one, body_frame_nhat[0]);
  const Scalar sign1 = Kokkos::copysign(one, body_frame_nhat[1]);
  const Scalar sign2 = Kokkos::copysign(one, body_frame_nhat[2]);

  Scalar alpha1, alpha2;
  if (sign0 * body_frame_nhat[0] > mundy::math::get_zero_tolerance<Scalar>()) {
    const Scalar tmp0 = one / (r1 * body_frame_nhat[0]);
    const Scalar tmp1 = tmp0 * r2 * body_frame_nhat[1];
    const Scalar tmp2 = tmp0 * r3 * body_frame_nhat[2];
    alpha1 = one / (one + tmp1 * tmp1);
    alpha2 = one / (one + tmp2 * tmp2 * alpha1);
  } else if (sign1 * body_frame_nhat[1] > mundy::math::get_zero_tolerance<Scalar>()) {
    const Scalar tmp = r3 * body_frame_nhat[2] / (r2 * body_frame_nhat[1]);
    alpha1 = zero;
    alpha2 = one / (one + tmp * tmp);
  } else {
    alpha1 = zero;
    alpha2 = zero;
  }

  const Scalar sqrt_alpha1 = Kokkos::sqrt(alpha1);
  const Scalar sqrt_alpha2 = Kokkos::sqrt(alpha2);

  const Scalar x = half * sign0 * ((one + sign0) * r1 + (one - sign0) * r1) * sqrt_alpha1 * sqrt_alpha2;
  const Scalar y = half * sign1 * ((one + sign1) * r2 + (one - sign1) * r2) * Kokkos::sqrt(one - alpha1) * sqrt_alpha2;
  const Scalar z = half * sign2 * ((one + sign2) * r3 + (one - sign2) * r3) * Kokkos::sqrt(one - alpha2);

  return mundy::geom::Point<Scalar>(x, y, z);
}

template <typename Scalar, typename Accessor1, typename OwnershipType1, ValidEllipsoidType EllipsoidType>
KOKKOS_FUNCTION mundy::math::Vector3<Scalar> map_surface_normal_to_foot_point_on_ellipsoid(
    const mundy::math::Vector3<Scalar, Accessor1, OwnershipType1>& lab_frame_ellipsoid_nhat,
    const EllipsoidType& ellipsoid) {
  const auto body_frame_nhat = conjugate(ellipsoid.orientation()) * lab_frame_ellipsoid_nhat;
  const auto body_frame_foot_point = map_body_frame_normal_to_ellipsoid(body_frame_nhat, ellipsoid);
  return ellipsoid.orientation() * body_frame_foot_point + ellipsoid.center();
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_ELLIPSOID_HPP_
