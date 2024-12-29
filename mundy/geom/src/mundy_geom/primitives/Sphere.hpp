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
#include <mundy_core/throw_assert.hpp>      // for MUNDY_THROW_ASSERT
#include <mundy_geom/primitives/Point.hpp>  // for mundy::geom::Point

namespace mundy {

namespace geom {

template <typename Scalar>
class Sphere {
 public:
  //! \name Type aliases
  //@{

  /// \brief The Sphere's scalar type
  using scalar_t = Scalar;
  using point_t = Point<Scalar>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor. Default initializes the center and sets the radius to an invalid value of -1
  KOKKOS_FUNCTION
  Sphere() : center_(scalar_t(), scalar_t(), scalar_t()), radius_(static_cast<Scalar>(-1)) {
  }

  /// \brief Constructor to initialize the center and radius.
  /// \param[in] center The center of the Sphere.
  /// \param[in] radius The radius of the Sphere.
  KOKKOS_FUNCTION
  Sphere(const point_t& center, const Scalar& radius) : center_(center), radius_(radius) {
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

  /// \brief Set the radius
  /// \param[in] radius The new radius.
  KOKKOS_FUNCTION
  void set_radius(const Scalar& radius) {
    radius_ = radius;
  }
  //@}

 private:
  point_t center_;
  Scalar radius_;
};

/// @brief Type trait to determine if a type is a Sphere
template <typename T>
struct is_sphere : std::false_type {};
//
template <typename Scalar>
struct is_sphere<Sphere<Scalar>> : std::true_type {};
//
template <typename Scalar>
struct is_sphere<const Sphere<Scalar>> : std::true_type {};
//
template <typename T>
constexpr bool is_sphere_v = is_sphere<T>::value;

/// @brief Concept to check if a type is a valid Sphere type
template <typename SphereType>
concept ValidSphereType =
    requires(std::decay_t<SphereType> sphere, const std::decay_t<SphereType> const_sphere) {
      is_sphere_v<std::decay_t<SphereType>>;
      typename std::decay_t<SphereType>::scalar_t;
      typename std::decay_t<SphereType>::point_t;
      is_point_v<typename std::decay_t<SphereType>::point_t>;
      is_point_v<decltype(sphere.center())>;
      is_point_v<decltype(const_sphere.center())>;
      { sphere.radius() } -> std::convertible_to<typename std::decay_t<SphereType>::scalar_t&>;
      { const_sphere.radius() } -> std::convertible_to<const typename std::decay_t<SphereType>::scalar_t&>;
    };  // ValidSphereType

static_assert(ValidSphereType<Sphere<float>> && ValidSphereType<const Sphere<float>> &&
                  ValidSphereType<Sphere<double>> && ValidSphereType<const Sphere<double>>,
              "Sphere should satisfy the ValidSphereType concept.");

//! \name Non-member functions for ValidSphereType objects
//@{

/// \brief Equality operator
template <ValidSphereType SphereType1, ValidSphereType SphereType2>
KOKKOS_FUNCTION bool operator==(const SphereType1& sphere1, const SphereType2& sphere2) {
  return (sphere1.radius() == sphere2.radius()) && (sphere1.center() == sphere2.center());
}

/// \brief Inequality operator
template <ValidSphereType SphereType1, ValidSphereType SphereType2>
KOKKOS_FUNCTION bool operator!=(const SphereType1& sphere1, const SphereType2& sphere2) {
  return (sphere1.radius() != sphere2.radius()) || (sphere1.center() != sphere2.center());
}

/// \brief OStream operator
template <ValidSphereType SphereType>
std::ostream& operator<<(std::ostream& os, const SphereType& sphere) {
  os << "{" << sphere.center() << ":" << sphere.radius() << "}";
  return os;
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_SPHERE_HPP_
