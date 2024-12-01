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

#ifndef MUNDY_GEOM_PRIMITIVES_AABB_HPP_
#define MUNDY_GEOM_PRIMITIVES_AABB_HPP_

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
class AABB {
 public:
  //! \name Type aliases
  //@{

  /// \brief The AABB's scalar type
  using scalar_type = Scalar;

  /// \brief The AABB's point type
  using point_type = mundy::math::Point<Scalar>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor. Initializes the box inside out and as large as possible.
  /// Nothing can be inside this box.
  KOKKOS_FUNCTION
  AABB()
      : min_corner_(scalar_max(), scalar_max(), scalar_max()), max_corner_(scalar_min(), scalar_min(), scalar_min()) {
  }

  /// \brief Constructor to directly set the min and max corners.
  /// \param[in] min_corner The minimum corner of the box.
  /// \param[in] max_corner The maximum corner of the box.
  KOKKOS_FUNCTION
  AABB(const point_type& min_corner, const point_type& max_corner) : min_corner_(min_corner), max_corner_(max_corner) {
  }

  /// \brief Constructor to directly set the min and max corners.
  /// \param[in] x_min The minimum x-coordinate.
  /// \param[in] y_min The minimum y-coordinate.
  /// \param[in] z_min The minimum z-coordinate.
  /// \param[in] x_max The maximum x-coordinate.
  /// \param[in] y_max The maximum y-coordinate.
  /// \param[in] z_max The maximum z-coordinate.
  KOKKOS_FUNCTION
  AABB(Scalar x_min, Scalar y_min, Scalar z_min, Scalar x_max, Scalar y_max, Scalar z_max)
      : min_corner_(x_min, y_min, z_min), max_corner_(x_max, y_max, z_max) {
  }

  /// \brief Destructor
  KOKKOS_FUNCTION
  ~AABB() {
  }

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  AABB(const AABB<Scalar>& other) : min_corner_(other.min_corner_), max_corner_(other.max_corner_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  AABB(AABB<Scalar>&& other) : min_corner_(std::move(other.min_corner_)), max_corner_(std::move(other.max_corner_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  AABB<Scalar>& operator=(const AABB<Scalar>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    min_corner_ = other.min_corner_;
    max_corner_ = other.max_corner_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  AABB<Scalar>& operator=(AABB<Scalar>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    min_corner_ = std::move(other.min_corner_);
    max_corner_ = std::move(other.max_corner_);
    return *this;
  }

  /// \brief Equality operator
  KOKKOS_FUNCTION
  bool operator==(const AABB<Scalar>& other) const {
    return (min_corner_ == other.min_corner_) && (max_corner_ == other.max_corner_);
  }

  /// \brief Inequality operator
  KOKKOS_FUNCTION
  bool operator!=(const AABB<Scalar>& other) const {
    return (min_corner_ != other.min_corner_) || (max_corner_ != other.max_corner_);
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Accessor for the minimum corner
  KOKKOS_FUNCTION
  const point_type& min_corner() const {
    return min_corner_;
  }

  /// \brief Accessor for the minimum corner
  KOKKOS_FUNCTION
  point_type& min_corner() {
    return min_corner_;
  }

  /// \brief Accessor for the maximum corner
  KOKKOS_FUNCTION
  const point_type& max_corner() const {
    return max_corner_;
  }

  /// \brief Accessor for the maximum corner
  KOKKOS_FUNCTION
  point_type& max_corner() {
    return max_corner_;
  }

  /// \brief Accessor for x_min
  KOKKOS_FUNCTION
  const& Scalar x_min() const {
    return min_corner_[0];
  }

  /// \brief Accessor for x_min
  KOKKOS_FUNCTION
  Scalar& x_min() {
    return min_corner_[0];
  }

  /// \brief Accessor for y_min
  KOKKOS_FUNCTION
  const& Scalar y_min() const {
    return min_corner_[1];
  }

  /// \brief Accessor for y_min
  KOKKOS_FUNCTION
  Scalar& y_min() {
    return min_corner_[1];
  }

  /// \brief Accessor for z_min
  KOKKOS_FUNCTION
  const& Scalar z_min() const {
    return min_corner_[2];
  }

  /// \brief Accessor for z_min
  KOKKOS_FUNCTION
  Scalar& z_min() {
    return min_corner_[2];
  }

  /// \brief Accessor for x_max
  KOKKOS_FUNCTION
  const& Scalar x_max() const {
    return max_corner_[0];
  }

  /// \brief Accessor for x_max
  KOKKOS_FUNCTION
  Scalar& x_max() {
    return max_corner_[0];
  }

  /// \brief Accessor for y_max
  KOKKOS_FUNCTION
  const& Scalar y_max() const {
    return max_corner_[1];
  }

  /// \brief Accessor for y_max
  KOKKOS_FUNCTION
  Scalar& y_max() {
    return max_corner_[1];
  }

  /// \brief Accessor for z_max
  KOKKOS_FUNCTION
  const& Scalar z_max() const {
    return max_corner_[2];
  }

  /// \brief Accessor for z_max
  KOKKOS_FUNCTION
  Scalar& z_max() {
    return max_corner_[2];
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the minimum corner
  /// \param[in] min_corner The new minimum corner.
  KOKKOS_FUNCTION
  void set_min_corner(const point_type& min_corner) {
    min_corner_ = min_corner;
  }

  /// \brief Set the minimum corner
  /// \param[in] x The new x-coordinate.
  /// \param[in] y The new y-coordinate.
  /// \param[in] z The new z-coordinate.
  KOKKOS_FUNCTION
  void set_min_corner(const Scalar& x, const Scalar& y, const Scalar& z) {
    min_corner_[0] = x;
    min_corner_[1] = y;
    min_corner_[2] = z;
  }

  /// \brief Set the maximum corner
  /// \param[in] max_corner The new maximum corner.
  KOKKOS_FUNCTION
  void set_max_corner(const point_type& max_corner) {
    max_corner_ = max_corner;
  }

  /// \brief Set the maximum corner
  /// \param[in] x The new x-coordinate.
  /// \param[in] y The new y-coordinate.
  /// \param[in] z The new z-coordinate.
  KOKKOS_FUNCTION
  void set_max_corner(const Scalar& x, const Scalar& y, const Scalar& z) {
    max_corner_[0] = x;
    max_corner_[1] = y;
    max_corner_[2] = z;
  }
  //@}

 private:
  //! \name Private helpers
  //@{

  /// \brief Get the maximum possible scalar value
  static KOKKOS_FUNCTION constexpr value_type scalar_max() {
    return Kokkos::Experimental::finite_max_v<T>;
  }

  /// \brief Get the minimum possible scalar value
  static KOKKOS_FUNCTION constexpr value_type scalar_min() {
    // finite_min_v<T> returns the most negative real value (equivalent to numeric_limits<T>::lowest).
    // it is the 'lowest' value that we want here.
    return Kokkos::Experimental::finite_min_v<T>;
  }
  //@}

  point_type min_corner_;
  point_type max_corner_;
};

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const AABB<Scalar>& aabb) {
  os << "{" << aabb.min_corner() << "->" << aabb.max_corner() << "}";
  return os;
}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_AABB_HPP_
