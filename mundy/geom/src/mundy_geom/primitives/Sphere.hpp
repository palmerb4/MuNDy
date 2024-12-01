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

#ifndef MUNDY_GEOM_PRIMITIVES_SPHERE_HPP_
#define MUNDY_GEOM_PRIMITIVES_SPHERE_HPP_

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
class Sphere {
 public:
  //! \name Type aliases
  //@{

  /// \brief The Sphere's scalar type
  using scalar_type = Scalar;
  using point_type = mundy::math::Point<Scalar>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor. Default initializes the center and sets the radius to an invalid value of -1
  KOKKOS_FUNCTION
  Sphere() : center_(scalar_type(), scalar_type(), scalar_type()), radius_(static_cast<Scalar>(-1)) {
  }

  /// \brief Constructor to initialize the center and radius.
  /// \param[in] center The center of the Sphere.
  /// \param[in] radius The radius of the Sphere.
  KOKKOS_FUNCTION
  Sphere(const point_type& center, const Scalar& radius) : center_(center), radius_(radius) {
  }

  /// \brief Destructor
  KOKKOS_FUNCTION
  ~Sphere() {
  }

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  Sphere(const Sphere<Scalar>& other) : center_(other.center_), radius_(other.radius_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  Sphere(Sphere<Scalar>&& other) : center_(std::move(other.center_)), radius_(std::move(other.radius_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  Sphere<Scalar>& operator=(const Sphere<Scalar>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = other.center_;
    radius_ = other.radius_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  Sphere<Scalar>& operator=(Sphere<Scalar>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = std::move(other.center_);
    radius_ = std::move(other.radius_);
    return *this;
  }

  /// \brief Equality operator
  KOKKOS_FUNCTION
  bool operator==(const Sphere<Scalar>& other) const {
    return (radius_ == other.radius_) && (center_ == other.center_);
  }

  /// \brief Inequality operator
  KOKKOS_FUNCTION
  bool operator!=(const Sphere<Scalar>& other) const {
    return (radius_ != other.radius_) || (center_ != other.center_);
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
  point_type center_;
  Scalar radius_;
};

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const Sphere<Scalar>& sphere) {
  os << "{" << sphere.center() << ":" << sphere.radius() << "}";
  return os;
}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_SPHERE_HPP_
