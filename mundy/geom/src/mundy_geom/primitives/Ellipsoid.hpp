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
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_math/Point.hpp>         // for mundy::math::Point

namespace mundy {

namespace geom {

template <typename Scalar>
class Ellipsoid {
 public:
  //! \name Type aliases
  //@{

  /// \brief The Ellipsoid's scalar type
  using scalar_type = Scalar;
  using point_type = mundy::math::Point<Scalar>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor. Default initializes the center and sets the axis lengths to an invalid value of -1
  KOKKOS_FUNCTION
  Ellipsoid()
      : center_(scalar_type(), scalar_type(), scalar_type()),
        axis_lengths_{static_cast<Scalar>(-1), static_cast<Scalar>(-1), static_cast<Scalar>(-1)} {
  }

  /// \brief Constructor to initialize the center and axis lengths.
  /// \param[in] center The center of the Ellipsoid.
  /// \param[in] axis_length_1 The first axis length of the Ellipsoid.
  /// \param[in] axis_length_2 The second axis length of the Ellipsoid.
  /// \param[in] axis_length_3 The third axis length of the Ellipsoid.
  KOKKOS_FUNCTION
  Ellipsoid(const point_type& center, const Scalar& axis_length_1, const Scalar& axis_length_2,
            const Scalar& axis_length_3)
      : center_(center), axis_lengths_{axis_length_1, axis_length_2, axis_length_3} {
  }

  /// \brief Constructor to initialize the center and axis lengths.
  /// \param[in] x The x-coordinate of the center.
  /// \param[in] y The y-coordinate of the center.
  /// \param[in] z The z-coordinate of the center.
  /// \param[in] axis_length_1 The first axis length of the Ellipsoid.
  /// \param[in] axis_length_2 The second axis length of the Ellipsoid.
  /// \param[in] axis_length_3 The third axis length of the Ellipsoid.
  KOKKOS_FUNCTION
  Ellipsoid(const Scalar& x, const Scalar& y, const Scalar& z, const Scalar& axis_length_1, const Scalar& axis_length_2,
            const Scalar& axis_length_3)
      : center_(x, y, z), axis_lengths_{axis_length_1, axis_length_2, axis_length_3} {
  }

  /// \brief Constructor to initialize the center and axis lengths.
  /// \param[in] center The center of the Ellipsoid.
  /// \param[in] axis_lengths The axis lengths of the Ellipsoid.
  KOKKOS_FUNCTION
  Ellipsoid(const point_type& center, const Array<Scalar, 3>& axis_lengths)
      : center_(center), axis_lengths_(axis_lengths) {
  }

  /// \brief Destructor
  KOKKOS_FUNCTION
  ~Ellipsoid() {
  }

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  Ellipsoid(const Ellipsoid<Scalar>& other) : center_(other.center_), axis_lengths_{other.axis_lengths_} {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  Ellipsoid(Ellipsoid<Scalar>&& other)
      : center_(std::move(other.center_)), axis_lengths_{std::move(other.axis_lengths_)} {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  Ellipsoid<Scalar>& operator=(const Ellipsoid<Scalar>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = other.center_;
    axis_lengths_ = other.axis_lengths_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  Ellipsoid<Scalar>& operator=(Ellipsoid<Scalar>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = std::move(other.center_);
    axis_lengths_ = std::move(other.axis_lengths_);
    return *this;
  }

  /// \brief Equality operator
  KOKKOS_FUNCTION
  bool operator==(const Ellipsoid<Scalar>& other) const {
    return (axis_lengths_ == other.axis_lengths_) && (center_ == other.center_);
  }

  /// \brief Inequality operator
  KOKKOS_FUNCTION
  bool operator!=(const Ellipsoid<Scalar>& other) const {
    return (axis_lengths_ != other.axis_lengths_) || (center_ != other.center_);
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Accessor for the center
  KOKKOS_FUNCTION
  const point_type& center() const {
    return center_;
  }

  /// \brief Accessor for the center
  KOKKOS_FUNCTION
  point_type& center() {
    return center_;
  }

  /// \brief Accessor for the axis lengths
  KOKKOS_FUNCTION
  const Array<Scalar, 3>& axis_lengths() const {
    return axis_lengths_;
  }

  /// \brief Accessor for the axis lengths
  KOKKOS_FUNCTION
  Array<Scalar, 3>& axis_lengths() {
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
  void set_center(const point_type& center) {
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

  /// \brief Set the axis lengths
  /// \param[in] axis_lengths The new axis lengths.
  KOKKOS_FUNCTION
  void set_axis_lengths(const Array<Scalar, 3>& axis_lengths) {
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
  point_type center_;
  Array<Scalar, 3> axis_lengths_;
};

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const Ellipsoid<Scalar>& ellipsoid) {
  os << "{" << ellipsoid.center() << ":" << ellipsoid.axis_lengths() << "}";
  return os;
}

template <typename Scalar>
KOKKOS_FUNCTION Vector3<Scalar> map_body_frame_normal_to_ellipsoid(const Vector3<Scalar>& body_frame_nhat, Scalar r1,
                                                                   Scalar r2, Scalar r3) {
  Scalar alpha1, alpha2;
  if (Kokkos::fabs(body_frame_nhat[0]) < get_zero_tolerance<Scalar>()) {
    const Scalar tmp1 = r2 * body_frame_nhat[1] / (r1 * body_frame_nhat[0]);
    const Scalar tmp2 = r3 * body_frame_nhat[2] / (r1 * body_frame_nhat[0]);
    alpha1 = static_cast<Scalar>(1.0) / (static_cast<Scalar>(1.0) + tmp1 * tmp1);
    alpha2 = static_cast<Scalar>(1.0) / (static_cast<Scalar>(1.0) + tmp2 * tmp2 * alpha1);
  } else if (body_frame_nhat[1] != 0) {
    const Scalar tmp = r3 * body_frame_nhat[2] / (r2 * body_frame_nhat[1]);
    alpha1 = static_cast<Scalar>(0.0);
    alpha2 = static_cast<Scalar>(1.0) / (static_cast<Scalar>(1.0) + tmp * tmp);
  } else {
    alpha1 = static_cast<Scalar>(0.0);
    alpha2 = static_cast<Scalar>(0.0);
  }

  const Scalar x = static_cast<Scalar>(0.5) * impl::sign(body_frame_nhat[0]) *
                   ((static_cast<Scalar>(1.0) + impl::sign(body_frame_nhat[0])) * r1 +
                    (static_cast<Scalar>(1.0) - impl::sign(body_frame_nhat[0])) * r1) *
                   Kokkos::sqrt(alpha1) * Kokkos::sqrt(alpha2);
  const Scalar y = static_cast<Scalar>(0.5) * impl::sign(body_frame_nhat[1]) *
                   ((static_cast<Scalar>(1.0) + impl::sign(body_frame_nhat[1])) * r2 +
                    (static_cast<Scalar>(1.0) - impl::sign(body_frame_nhat[1])) * r2) *
                   Kokkos::sqrt(1.0 - alpha1) * Kokkos::sqrt(alpha2);
  const Scalar z = static_cast<Scalar>(0.5) * impl::sign(body_frame_nhat[2]) *
                   ((static_cast<Scalar>(1.0) + impl::sign(body_frame_nhat[2])) * r3 +
                    (static_cast<Scalar>(1.0) - impl::sign(body_frame_nhat[2])) * r3) *
                   Kokkos::sqrt(1.0 - alpha2);

  return Vector3<Scalar>(x, y, z);
}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_ELLIPSOID_HPP_
