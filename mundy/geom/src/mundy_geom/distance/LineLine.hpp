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

#ifndef MUNDY_MATH_DISTANCE_LINELINE_HPP_
#define MUNDY_MATH_DISTANCE_LINELINE_HPP_

// External libs
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_math/Line.hpp>                 // for mundy::math::Line
#include <mundy_math/Point.hpp>                // for mundy::math::Point
#include <mundy_math/distance/PointPoint.hpp>  // for distance(Point, Point)
#include <mundy_math/distance/Types.hpp>       // for SharedNormalSigned

namespace mundy {

namespace math {

/// \brief Compute the distance between two lines (defaults to SharedNormalSigned distance)
/// \tparam Scalar The scalar type
/// \param[in] line1 One line
/// \param[in] line2 The other line
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const Line<Scalar>& line1, const Line<Scalar>& line2) {
  return distance(SharedNormalSigned{}, line1, line2);
}

/// \brief Compute the distance between two lines
/// \tparam Scalar The scalar type
/// \param[in] distance_type The distance type
/// \param[in] line1 One line
/// \param[in] line2 The other line
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const SharedNormalSigned distance_type, const Line<Scalar>& line1,
                                const Line<Scalar>& line2) {
  // Part of this function was adapted from VTK, which, in turn adapted part of it from "GeometryAlgorithms.com"
  const auto center_center = line1.center() - line2.center();
  const Scalar a = dot(line1.direction(), line1.direction());
  const Scalar b = dot(line1.direction(), line2.direction());
  const Scalar c = dot(line2.direction(), line2.direction());
  const Scalar d = dot(line1.direction(), center_center);
  const Scalar e = dot(line2.direction(), center_center);
  const Scalar D = a * c - b * b;  // always >= 0

  // Check if the lines are colinear
  // Two infinite colinear lines intersect at all points.
  if (D < get_zero_tolerance<Scalar>()) {
    return static_cast<Scalar>(0.0);
  }

  // Compute the closest points on the lines
  const Scalar inv_D = static_cast<Scalar>(1.0) / D;
  const Point<double> closest_point1 = line1.center() + (b * e - c * d) * line1.direction() * inv_D;
  const Point<double> closest_point2 = line2.center() + (a * e - b * d) * line2.direction() * inv_D;
  return distance(closest_point1, closest_point2);
}

/// \brief Compute the distance between two lines
/// \tparam Scalar The scalar type
/// \param[in] distance_type The distance type
/// \param[in] line1 One line
/// \param[in] line2 The other line
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const Euclidean distance_type, const Line<Scalar>& line1,
                                const Line<Scalar>& line2) {
  return distance(SharedNormalSigned{}, line1, line2);  // no difference between distance types for lines
}

/// \brief Compute the distance between two lines (defaults to SharedNormalSigned distance)
/// \tparam Scalar The scalar type
/// \param[in] line1 One line
/// \param[in] line2 The other line
/// \param[out] sep The separation vector (from line1 to line2)
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const Line<Scalar>& line1, const Line<Scalar>& line2, Vector3<Scalar>& sep) {
  return distance(SharedNormalSigned{}, line1, line2, sep);
}

/// \brief Compute the distance between two lines
/// \tparam Scalar The scalar type
/// \param[in] distance_type The distance type
/// \param[in] line1 One line
/// \param[in] line2 The other line
/// \param[out] closest_point1 The closest point on line1
/// \param[out] closest_point2 The closest point on line2
/// \param[out] arch_length1 The arch-length parameter of the closest point on line1
/// \param[out] arch_length2 The arch-length parameter of the closest point on line2
/// \param[out] sep The separation vector (from line1 to line2)
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const SharedNormalSigned distance_type, const Line<Scalar>& line1,
                                const Line<Scalar>& line2, Point<Scalar>& closest_point1, Point<Scalar>& closest_point2,
                                Scalar& arch_length1, Scalar& arch_length2, Vector3<Scalar>& sep) {
  // Part of this function was adapted from VTK, which, in turn adapted part of it from "GeometryAlgorithms.com"
  const auto center_center = line1.center() - line2.center();
  const Scalar a = dot(line1.direction(), line1.direction());
  const Scalar b = dot(line1.direction(), line2.direction());
  const Scalar c = dot(line2.direction(), line2.direction());
  const Scalar d = dot(line1.direction(), center_center);
  const Scalar e = dot(line2.direction(), center_center);
  const Scalar D = a * c - b * b;  // always >= 0

  // Check if the lines are colinear
  // Two infinite colinear lines intersect at all points.
  if (D < get_zero_tolerance<Scalar>()) {
    sep = {0.0, 0.0, 0.0};
    return static_cast<Scalar>(0.0);
  }

  // Compute the closest points on the lines
  const Scalar inv_D = static_cast<Scalar>(1.0) / D;
  arch_length1 = (b * e - c * d) * inv_D;
  arch_length2 = (a * e - b * d) * inv_D;
  closest_point1 = line1.center() + arch_length1 * line1.direction();
  closest_point2 = line2.center() + arch_length2 * line2.direction();
  sep = closest_point1 - closest_point2;
  return norm(extras.sep);
}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_DISTANCE_LINELINE_HPP_
