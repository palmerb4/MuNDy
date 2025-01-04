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
#include <mundy_geom/primitives/Point.hpp>         // for mundy::geom::Point
#include <mundy_math/Vector3.hpp>       // for mundy::math::Vector3

namespace mundy {

namespace geom {

template <typename Scalar>
class Line {
 public:
  //! \name Type aliases
  //@{

  /// \brief Our scalar type
  using scalar_t = Scalar;

  /// \brief Our point type
  using point_t = Point<Scalar>;

  /// \brief Our vector type
  using vector_type = mundy::math::Vector3<Scalar>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor. Default initialize the
  KOKKOS_FUNCTION
  Line()
      : center_(scalar_t(), scalar_t(), scalar_t()), direction_(scalar_t(), scalar_t(), scalar_t()) {
  }

  /// \brief Constructor to initialize the center and radius.
  /// \param[in] center The center of the Line.
  /// \param[in] direction The direction of the Line.
  KOKKOS_FUNCTION
  Line(const point_t& center, const vector_type& direction) : center_(center), direction_(direction) {
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

  /// \brief Accessor for the direction
  KOKKOS_FUNCTION
  const vector_type& direction() const {
    return direction_;
  }

  /// \brief Accessor for the direction
  KOKKOS_FUNCTION
  vector_type& direction() {
    return direction_;
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

  /// \brief Set the direction
  /// \param[in] direction The new direction.
  KOKKOS_FUNCTION
  void set_direction(const vector_type& direction) {
    direction_ = direction;
  }

  /// \brief Set the direction
  /// \param[in] x The x-component.
  /// \param[in] y The y-component.
  /// \param[in] z The z-component.
  KOKKOS_FUNCTION
  void set_direction(const Scalar& x, const Scalar& y, const Scalar& z) {
    direction_[0] = x;
    direction_[1] = y;
    direction_[2] = z;
  }
  //@}

 private:
  point_t center_;      ///< The center of the line.
  vector_type direction_;  ///< The direction of the line.
};

/// @brief Type trait to determine if a type is a Line
template <typename T>
struct is_line : std::false_type {};
//
template <typename Scalar>
struct is_line<Line<Scalar>> : std::true_type {};
//
template <typename Scalar>
struct is_line<const Line<Scalar>> : std::true_type {};
//
template <typename T>
inline constexpr bool is_line_v = is_line<T>::value;

/// @brief Concept to check if a type is a valid Line type
template <typename LineType>
concept ValidLineType = 
    requires(std::remove_cv_t<LineType> line, const std::remove_cv_t<LineType> const_line) {
      is_line_v<LineType>;
      typename std::remove_cv_t<LineType>::scalar_t;
      { line.center() } -> std::convertible_to<mundy::geom::Point<typename std::remove_cv_t<LineType>::scalar_t>>;
      { line.direction() } -> std::convertible_to<mundy::math::Vector3<typename std::remove_cv_t<LineType>::scalar_t>>;

      { const_line.center() } -> std::convertible_to<const mundy::geom::Point<typename std::remove_cv_t<LineType>::scalar_t>>;
      {
        const_line.direction()
      } -> std::convertible_to<const mundy::math::Vector3<typename std::remove_cv_t<LineType>::scalar_t>>;
    };  // ValidLineType

static_assert(ValidLineType<Line<float>> && ValidLineType<const Line<float>> &&
              ValidLineType<Line<double>> && ValidLineType<const Line<double>>,
              "Line should satisfy the ValidLineType concept");

//! \name Non-member functions for ValidLineType objects
//@{

/// \brief Equality operator
template <ValidLineType LineType1, ValidLineType LineType2>
KOKKOS_FUNCTION
bool operator==(const LineType1& line1, const LineType2& line2) {
  return (line1.center() == line2.center()) && (line1.direction() == line2.direction());
}

/// \brief Inequality operator
template <ValidLineType LineType1, ValidLineType LineType2>
KOKKOS_FUNCTION
bool operator!=(const LineType1& line1, const LineType2& line2) {
  return (line1.center() != line2.center()) || (line1.direction() != line2.direction());
}

/// \brief Output stream operator
template <ValidLineType LineType>
std::ostream& operator<<(std::ostream& os, const LineType& line) {
  os << "{" << line.center() << ":" << line.direction() << "}";
  return os;
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_LINE_HPP_
