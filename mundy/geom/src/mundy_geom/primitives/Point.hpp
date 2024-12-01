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

#ifndef MUNDY_GEOM_PRIMITIVES_POINT_HPP_
#define MUNDY_GEOM_PRIMITIVES_POINT_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <iostream>
#include <stdexcept>
#include <utility>

// Our libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_math/Vector3.hpp>       // for mundy::math::Vector3

namespace mundy {

namespace geom {

/// @brief A point in 3D space
/// @tparam Scalar
///
/// The following is a methodological choice to use the Vector3 class as the underlying data structure for the Point
/// class. This is done to allow points to access the same mathematical operations as vectors (dot product, cross
/// product, etc.). Had we created our own interface, we would have hidden the mathematical operations from the user.
template <typename Scalar>
using Point = mundy::math::Vector3<Scalar>;

// template <typename Scalar>
// class Point {
//  public:
//   //! \name Type aliases
//   //@{

//   /// \brief The point's scalar type
//   using scalar_type = Scalar;
//   using vector_type = mundy::math::Vector3<Scalar>;
//   //@}

//   //! \name Constructors and destructor
//   //@{

//   /// \brief Default constructor. Initializes the position to the origin.
//   KOKKOS_FUNCTION
//   Point() : position_(0.0, 0.0, 0.0) {
//   }

//   /// \brief Constructor to initialize the position.
//   /// \param[in] position The position of the point.
//   KOKKOS_FUNCTION
//   Point(const mundy::math::Vector3<Scalar>& position) : position_(position) {
//   }

//   /// \brief Destructor
//   KOKKOS_FUNCTION
//   ~Point() {
//   }

//   /// \brief Deep copy constructor
//   KOKKOS_FUNCTION
//   Point(const Point<Scalar>& other) : position_(other.position_) {
//   }

//   /// \brief Deep move constructor
//   KOKKOS_FUNCTION
//   Point(Point<Scalar>&& other) : position_(std::move(other.position_)) {
//   }
//   //@}

//   //! \name Operators
//   //@{

//   /// \brief Copy assignment operator
//   KOKKOS_FUNCTION
//   Point<Scalar>& operator=(const Point<Scalar>& other) {
//     MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
//     position_ = other.position_;
//     return *this;
//   }

//   /// \brief Move assignment operator
//   KOKKOS_FUNCTION
//   Point<Scalar>& operator=(Point<Scalar>&& other) {
//     MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
//     position_ = std::move(other.position_);
//     return *this;
//   }

//   /// \brief Equality operator
//   KOKKOS_FUNCTION
//   bool operator==(const Point<Scalar>& other) const {
//     return position_ == other.position_;
//   }

//   /// \brief Inequality operator
//   KOKKOS_FUNCTION
//   bool operator!=(const Point<Scalar>& other) const {
//     return position_ != other.position_;
//   }
//   //@}

//   //! \name Accessors
//   //@{

//   /// \brief Accessor for the position
//   KOKKOS_FUNCTION
//   const mundy::math::Vector3<Scalar>& position() const {
//     return position_;
//   }

//   /// \brief Accessor for the position
//   KOKKOS_FUNCTION
//   mundy::math::Vector3<Scalar>& position() {
//     return position_;
//   }

//   /// \brief Accessor for the coordinates
//   KOKKOS_FUNCTION
//   Scalar& operator[](size_t idx) {
//     MUNDY_THROW_ASSERT(idx < 3, std::out_of_range, "Index out of range");
//     return position_[idx];
//   }

//   /// \brief Const accessor for the coordinates
//   KOKKOS_FUNCTION
//   const Scalar& operator[](size_t idx) const {
//     MUNDY_THROW_ASSERT(idx < 3, std::out_of_range, "Index out of range");
//     return position_[idx];
//   }
//   //@}

//   //! \name Setters
//   //@{

//   /// \brief Set the position
//   /// \param[in] position The new position.
//   KOKKOS_FUNCTION
//   void set_position(const mundy::math::Vector3<Scalar>& position) {
//     position_ = position;
//   }

//   /// \brief Set the position
//   /// \param[in] x The x-coordinate.
//   /// \param[in] y The y-coordinate.
//   /// \param[in] z The z-coordinate.
//   KOKKOS_FUNCTION
//   void set_position(Scalar x, Scalar y, Scalar z) {
//     position_[0] = x;
//     position_[1] = y;
//     position_[2] = z;
//   }
//   //@}

//  private:
//   mundy::math::Vector3<Scalar> position_;
// };

// template <typename Scalar>
// std::ostream& operator<<(std::ostream& os, const Point<Scalar>& point) {
//   os << "(" << point[0] << ", " << point[1] << ", " << point[2] << ")";
//   return os;
// }

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_POINT_HPP_
