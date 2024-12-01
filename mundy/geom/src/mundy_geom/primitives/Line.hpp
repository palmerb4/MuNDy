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

#ifndef MUNDY_GEOM_PRIMITIVES_LINE_HPP_
#define MUNDY_GEOM_PRIMITIVES_LINE_HPP_

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
class Line {
 public:
  //! \name Type aliases
  //@{

  /// \brief The Line's scalar type
  using scalar_type = Scalar;
  using point_type = mundy::math::Point<Scalar>;
  using vector_type = mundy::math::Vector3<Scalar>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor. Default initialize the 
  KOKKOS_FUNCTION
  Line() : center_(scalar_type(), scalar_type(), scalar_type()), direction_(scalar_type(), scalar_type(), scalar_type()) {
  }

  /// \brief Constructor to initialize the center and radius.
  /// \param[in] center The center of the Line.
  /// \param[in] direction The direction of the Line.
  KOKKOS_FUNCTION
  Line(const point_type& center, const vector_type& direction) : center_(center), direction_(direction) {
  }

  /// \brief Destructor
  KOKKOS_FUNCTION
  ~Line() {
  }

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  Line(const Line<Scalar>& other) : center_(other.center_), direction_(other.direction_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  Line(Line<Scalar>&& other) : center_(std::move(other.center_)), direction_(std::move(other.direction_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  Line<Scalar>& operator=(const Line<Scalar>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = other.center_;
    direction_ = other.direction_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  Line<Scalar>& operator=(Line<Scalar>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = std::move(other.center_);
    direction_ = std::move(other.direction_);
    return *this;
  }

  /// \brief Equality operator
  KOKKOS_FUNCTION
  bool operator==(const Line<Scalar>& other) const {
    return (direction_ == other.direction_) && (center_ == other.center_);
  }

  /// \brief Inequality operator
  KOKKOS_FUNCTION
  bool operator!=(const Line<Scalar>& other) const {
    return (direction_ != other.direction_) || (center_ != other.center_);
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

  /// \brief Accessor for the radius
  KOKKOS_FUNCTION
  const Scalar& radius() const {
    return radius_;
  }

  /// \brief Accessor for the radius
  KOKKOS_FUNCTION
  Scalar& radius() {
    return radius_;
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

  /// \brief Set the radius
  /// \param[in] radius The new radius.
  KOKKOS_FUNCTION
  void set_radius(const Scalar& radius) {
    radius_ = radius;
  }
  //@}

 private:
  point_type center_;    ///< The center of the line.
  vector_type direction_;  ///< The direction of the line.
};

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const Line<Scalar>& sphere) {
  os << "{" << sphere.center() << ":" << sphere.radius() << "}";
  return os;
}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_LINE_HPP_
