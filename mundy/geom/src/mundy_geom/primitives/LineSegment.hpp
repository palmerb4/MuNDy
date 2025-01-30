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
#include <mundy_core/throw_assert.hpp>      // for MUNDY_THROW_ASSERT
#include <mundy_geom/primitives/Point.hpp>  // for mundy::geom::Point

namespace mundy {

namespace geom {

template <typename Scalar, ValidPointType PointType = Point<Scalar>,
          typename OwnershipType = mundy::math::Ownership::Owns>
class LineSegment {
  static_assert(std::is_same_v<typename PointType::scalar_t, Scalar>,
                "The scalar_t of the PointType must match the Scalar type.");
  static_assert(std::is_same_v<typename PointType::ownership_t, OwnershipType>,
                "The ownership type of the PointType must match the OwnershipType.\n"
                "This is somewhat restrictive, and we may want to relax this constraint in the future.\n"
                "If you need to use a different ownership type, please let us know and we'll remove this restriction.");

 public:
  //! \name Type aliases
  //@{

  /// \brief Our scalar type
  using scalar_t = Scalar;

  /// \brief Our point type
  using point_t = PointType;

  /// \brief Our ownership type
  using ownership_t = OwnershipType;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor for owning LineSegments. Default initialize the start and end points.
  KOKKOS_FUNCTION
  LineSegment()
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Owns>
      : start_(scalar_t(), scalar_t(), scalar_t()), end_(scalar_t(), scalar_t(), scalar_t()) {
  }

  /// \brief No default constructor for viewing LineSegmentss.
  KOKKOS_FUNCTION
  LineSegment()
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Views>
  = delete;

  /// \brief Constructor to initialize the start and end points.
  /// \param[in] start The start of the LineSegment.
  /// \param[in] end The end of the LineSegment.
  KOKKOS_FUNCTION
  LineSegment(const point_t& start, const point_t& end) : start_(start), end_(end) {
  }

  /// \brief Constructor to initialize the start and end points.
  /// \param[in] start The start of the LineSegment.
  /// \param[in] end The end of the LineSegment.
  template <ValidPointType OtherPointType>
  KOKKOS_FUNCTION LineSegment(const OtherPointType& start, const OtherPointType& end) : start_(start), end_(end) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  ~LineSegment() = default;

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  LineSegment(const LineSegment<scalar_t, point_t, ownership_t>& other) : start_(other.start_), end_(other.end_) {
  }

  /// \brief Deep copy constructor
  template <typename OtherLineSegmentType>
  KOKKOS_FUNCTION LineSegment(const OtherLineSegmentType& other)
    requires(!std::is_same_v<OtherLineSegmentType, LineSegment<scalar_t, point_t, ownership_t>>)
      : start_(other.start_), end_(other.end_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  LineSegment(LineSegment<scalar_t, point_t, ownership_t>&& other)
      : start_(std::move(other.start_)), end_(std::move(other.end_)) {
  }

  /// \brief Deep move constructor
  template <typename OtherLineSegmentType>
  KOKKOS_FUNCTION LineSegment(OtherLineSegmentType&& other)
    requires(!std::is_same_v<OtherLineSegmentType, LineSegment<scalar_t, point_t, ownership_t>>)
      : start_(std::move(other.start_)), end_(std::move(other.end_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  LineSegment<scalar_t, point_t, ownership_t>& operator=(const LineSegment<scalar_t, point_t, ownership_t>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    start_ = other.start_;
    end_ = other.end_;
    return *this;
  }

  /// \brief Copy assignment operator
  template <typename OtherLineSegmentType>
  KOKKOS_FUNCTION LineSegment<scalar_t, point_t, ownership_t>& operator=(const OtherLineSegmentType& other)
    requires(!std::is_same_v<OtherLineSegmentType, LineSegment<scalar_t, point_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    start_ = other.start_;
    end_ = other.end_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  LineSegment<scalar_t, point_t, ownership_t>& operator=(LineSegment<scalar_t, point_t, ownership_t>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    start_ = std::move(other.start_);
    end_ = std::move(other.end_);
    return *this;
  }

  /// \brief Move assignment operator
  template <typename OtherLineSegmentType>
  KOKKOS_FUNCTION LineSegment<scalar_t, point_t, ownership_t>& operator=(OtherLineSegmentType&& other)
    requires(!std::is_same_v<OtherLineSegmentType, LineSegment<scalar_t, point_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    start_ = std::move(other.start_);
    end_ = std::move(other.end_);
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Accessor for the start
  KOKKOS_FUNCTION
  const point_t& start() const {
    return start_;
  }

  /// \brief Accessor for the start
  KOKKOS_FUNCTION
  point_t& start() {
    return start_;
  }

  /// \brief Accessor for the end
  KOKKOS_FUNCTION
  const point_t& end() const {
    return end_;
  }

  /// \brief Accessor for the end
  KOKKOS_FUNCTION
  point_t& end() {
    return end_;
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the start point
  /// \param[in] start The new start point.
  template <ValidPointType OtherPointType>
  KOKKOS_FUNCTION void set_start(const OtherPointType& start) {
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
  template <ValidPointType OtherPointType>
  KOKKOS_FUNCTION void set_end(const OtherPointType& end) {
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
  point_t start_;
  point_t end_;
};

/// @brief Type trait to determine if a type is a LineSegment
template <typename T>
struct is_line_segment : std::false_type {};
//
template <typename Scalar, ValidPointType PointType, typename OwnershipType>
struct is_line_segment<LineSegment<Scalar, PointType, OwnershipType>> : std::true_type {};
//
template <typename Scalar, ValidPointType PointType, typename OwnershipType>
struct is_line_segment<const LineSegment<Scalar, PointType, OwnershipType>> : std::true_type {};
//
template <typename T>
inline constexpr bool is_line_segment_v = is_line_segment<T>::value;

/// @brief Concept to check if a type is a valid LineSegment type
template <typename LineSegmentType>
concept ValidLineSegmentType = is_line_segment_v<std::remove_cv_t<LineSegmentType>> &&
                               is_point_v<decltype(std::declval<std::remove_cv_t<LineSegmentType>>().start())> &&
                               is_point_v<decltype(std::declval<std::remove_cv_t<LineSegmentType>>().end())> &&
                               is_point_v<decltype(std::declval<const std::remove_cv_t<LineSegmentType>>().start())> &&
                               is_point_v<decltype(std::declval<const std::remove_cv_t<LineSegmentType>>().end())> &&
                               requires(std::remove_cv_t<LineSegmentType> line) {
                                 typename std::remove_cv_t<LineSegmentType>::scalar_t;
                               };  // ValidLineSegmentType

static_assert(ValidLineSegmentType<LineSegment<float>> && ValidLineSegmentType<const LineSegment<float>> &&
                  ValidLineSegmentType<LineSegment<double>> && ValidLineSegmentType<const LineSegment<double>>,
              "LineSegment should satisfy the ValidLineSegmentType concept");

//! \name Non-member functions for ValidLineSegmentType objects
//@{

/// \brief Equality operator
template <ValidLineSegmentType LineSegmentType1, ValidLineSegmentType LineSegmentType2>
KOKKOS_FUNCTION bool operator==(const LineSegmentType1& line_segment1, const LineSegmentType2& line_segment2) {
  return (line_segment1.start() == line_segment2.start()) && (line_segment1.end() == line_segment2.end());
}

/// \brief Inequality operator
template <ValidLineSegmentType LineSegmentType1, ValidLineSegmentType LineSegmentType2>
KOKKOS_FUNCTION bool operator!=(const LineSegmentType1& line_segment1, const LineSegmentType2& line_segment2) {
  return (line_segment1.start() != line_segment2.start()) || (line_segment1.end() != line_segment2.end());
}

/// \brief OStream operator
template <ValidLineSegmentType LineSegmentType>
std::ostream& operator<<(std::ostream& os, const LineSegmentType& line_segment) {
  os << "{" << line_segment.start() << "->" << line_segment.end() << "}";
  return os;
}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_LINESEGMENT_HPP_
