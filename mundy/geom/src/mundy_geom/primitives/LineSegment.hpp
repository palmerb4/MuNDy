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

#ifndef MUNDY_GEOM_PRIMITIVES_LINESEGMENT_HPP_
#define MUNDY_GEOM_PRIMITIVES_LINESEGMENT_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <iostream>
#include <stdexcept>
#include <utility>

// Our libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_geom/primitives/Point.hpp>         // for mundy::geom::Point

namespace mundy {

namespace geom {

template <typename Scalar>
class LineSegment {
 public:
  //! \name Type aliases
  //@{

  /// \brief The LineSegment's scalar type
  using scalar_type = Scalar;
  using point_type = Point<Scalar>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor. Default initialize the start and end points.
  KOKKOS_FUNCTION
  LineSegment()
      : start_(scalar_type(), scalar_type(), scalar_type()), end_(scalar_type(), scalar_type(), scalar_type()) {
  }

  /// \brief Constructor to initialize the start and end points.
  /// \param[in] start The start of the LineSegment.
  /// \param[in] end The end of the LineSegment.
  KOKKOS_FUNCTION
  LineSegment(const point_type& start, const point_type& end) : start_(start), end_(end) {
  }

  /// \brief Destructor
  KOKKOS_FUNCTION
  ~LineSegment() {
  }

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  LineSegment(const LineSegment<Scalar>& other) : start_(other.start_), end_(other.end_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  LineSegment(LineSegment<Scalar>&& other) : start_(std::move(other.start_)), end_(std::move(other.end_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  LineSegment<Scalar>& operator=(const LineSegment<Scalar>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    start_ = other.start_;
    end_ = other.end_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  LineSegment<Scalar>& operator=(LineSegment<Scalar>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    start_ = std::move(other.start_);
    end_ = std::move(other.end_);
    return *this;
  }

  /// \brief Equality operator
  KOKKOS_FUNCTION
  bool operator==(const LineSegment<Scalar>& other) const {
    return (start_ == other.start_) && (end_ == other.end_);
  }

  /// \brief Inequality operator
  KOKKOS_FUNCTION
  bool operator!=(const LineSegment<Scalar>& other) const {
    return (start_ != other.start_) || (end_ != other.end_);
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Accessor for the start
  KOKKOS_FUNCTION
  const point_type& start() const {
    return start_;
  }

  /// \brief Accessor for the start
  KOKKOS_FUNCTION
  point_type& start() {
    return start_;
  }

  /// \brief Accessor for the end
  KOKKOS_FUNCTION
  const point_type& end() const {
    return end_;
  }

  /// \brief Accessor for the end
  KOKKOS_FUNCTION
  point_type& end() {
    return end_;
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the start point
  /// \param[in] start The new start point.
  KOKKOS_FUNCTION
  void set_start(const point_type& start) {
    start_ = start;
  }

  /// \brief Set the start point
  /// \param[in] x The x-coordinate.
  /// \param[in] y The y-coordinate.
  /// \param[in] z The z-coordinate.
  KOKKOS_FUNCTION
  void set_start(const Scalar& x, const Scalar& y, const Scalar& z) {
    start_[0] = x;
    start_[1] = y;
    start_[2] = z;
  }

  /// \brief Set the end point
  /// \param[in] end The new end point.
  KOKKOS_FUNCTION
  void set_end(const point_type& end) {
    end_ = end;
  }

  /// \brief Set the end point
  /// \param[in] x The x-coordinate.
  /// \param[in] y The y-coordinate.
  /// \param[in] z The z-coordinate.
  KOKKOS_FUNCTION
  void set_end(const Scalar& x, const Scalar& y, const Scalar& z) {
    end_[0] = x;
    end_[1] = y;
    end_[2] = z;
  }
  //@}

 private:
  point_type start_;
  point_type end_;
};

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const LineSegment<Scalar>& line_segment) {
  os << "{" << line_segment.start() << "->" << line_segment.end() << "}";
  return os;
}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_LINESEGMENT_HPP_
