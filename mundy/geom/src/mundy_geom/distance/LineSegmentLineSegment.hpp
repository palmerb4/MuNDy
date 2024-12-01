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

#ifndef MUNDY_MATH_DISTANCE_LINESEGMENTLINESEGMENT_HPP_
#define MUNDY_MATH_DISTANCE_LINESEGMENTLINESEGMENT_HPP_

// C++ core
#include <utility>  // for std::move

// External libs
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_math/LineSegment.hpp>                // for mundy::math::LineSegment
#include <mundy_math/Point.hpp>                      // for mundy::math::Point
#include <mundy_math/distance/PointLineSegment.hpp>  // for mundy::math::distance(Point, LineSegment)
#include <mundy_math/distance/Types.hpp>             // for SharedNormalSigned

namespace mundy {

namespace math {

/// \brief Compute the distance between two line segments
/// \tparam Scalar The scalar type
/// \param[in] line_segment1 The first line segment
/// \param[in] line_segment2 The second line segment
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const LineSegment<Scalar>& line_segment1, const LineSegment<Scalar>& line_segment2) {
  return distance(SharedNormalSigned{}, line_segment1, line_segment2);
}

/// \brief Compute the shared normal signed separation distance between two line segments
/// \tparam Scalar The scalar type
/// \param[in] line_segment1 The first line segment
/// \param[in] line_segment2 The second line segment
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const SharedNormalSigned distance_type,
                                const LineSegment<Scalar>& line_segment1, const LineSegment<Scalar>& line_segment2) {
  // Part of this function was adapted from VTK, which, in turn adapted part of it from "GeometryAlgorithms.com"
  const auto& l0 = line_segment1.start();
  const auto& l1 = line_segment1.end();
  const auto& m0 = line_segment2.start();
  const auto& m1 = line_segment2.end();

  const auto u = l1 - l0;
  const auto v = m1 - m0;
  const auto w = l0 - m0;
  const Scalar a = dot(u, u);
  const Scalar b = dot(u, v);
  const Scalar c = dot(v, v);
  const Scalar d = dot(u, w);
  const Scalar e = dot(v, w);
  const Scalar D = a * c - b * b;  // always >= 0

  // Compute the line parameters of the two closest points
  if (D < get_zero_tolerance<Scalar>()) {
    // CASE 1: The lines are colinear. Therefore, one of the four endpoints is the
    // point of closest approach. We'll directly compute the 4 distances and the closest point on the line.
    const Scalar dist1 = distance(l0, line_segment2);
    const Scalar dist2 = distance(l1, line_segment2);
    const Scalar dist3 = distance(m0, line_segment1);
    const Scalar dist4 = distance(m1, line_segment1);

    // Determine which of the 4 distances is the minimum using exact Scalar comparison.
    Scalar min_distance = Kokkos::min(Kokkos::min(dist1, dist2), Kokkos::min(dist3, dist4));
    return min_distance;
  }

  // CASE 2: The lines aren't parallel. Get the closest points on the infinite lines.
  // TODO(palmerb4): Comment the various if-else cases in terms of their physical meaning.
  Scalar sN = b * e - c * d;
  Scalar tN = a * e - b * d;
  Scalar sD = D;                        // sc = sN / sD, default sD = D >= 0
  Scalar tD = D;                        // tc = tN / tD, default tD = D >= 0
  if (sN < static_cast<Scalar>(0.0)) {  // sc < 0 => the s=0 edge is visible
    sN = static_cast<Scalar>(0.0);
    tN = e;
    tD = c;
  } else if (sN > sD) {  // sc > 1 => the s=1 edge is visible
    sN = sD;
    tN = e + b;
    tD = c;
  }

  if (tN < static_cast<Scalar>(0.0)) {  // tc < 0 => the t=0 edge is visible
    tN = static_cast<Scalar>(0.0);

    // Recompute sc for this edge
    if (-d < static_cast<Scalar>(0.0)) {
      sN = static_cast<Scalar>(0.0);
    } else if (-d > a) {
      sN = sD;
    } else {
      sN = -d;
      sD = a;
    }
  } else if (tN > tD) {  // tc > 1 => the t=1 edge is visible
    tN = tD;

    // Recompute sc for this edge
    if ((-d + b) < static_cast<Scalar>(0.0)) {
      sN = static_cast<Scalar>(0.0);
    } else if ((-d + b) > a) {
      sN = sD;
    } else {
      sN = (-d + b);
      sD = a;
    }
  }

  // Finally, get the arch-length parameters, the corresponding closest points, and their distance squared.
  const Scalar t1_tmp = (Kokkos::fabs(sN) < get_zero_tolerance<Scalar>()) ? static_cast<Scalar>(0.0) : sN / sD;
  const Scalar t2_tmp = (Kokkos::fabs(tN) < get_zero_tolerance<Scalar>()) ? static_cast<Scalar>(0.0) : tN / tD;
  const auto closest_point1_tmp = l0 + t1_tmp * u;
  const auto closest_point2_tmp = m0 + t2_tmp * v;
  const Scalar dist = distance(closest_point1_tmp - closest_point2_tmp);

  return dist;
}

/// \brief Compute the euclidean separation distance between two line segments
/// \tparam Scalar The scalar type
/// \param[in] line_segment1 The first line segment
/// \param[in] line_segment2 The second line segment
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const Euclidean distance_type,
                                const LineSegment<Scalar>& line_segment1, const LineSegment<Scalar>& line_segment2) {
  return distance(SharedNormalSigned{}, line_segment1,
                  line_segment2);  // no difference between distance types for line segments
}

/// \brief Compute the distance between two line segments
/// \tparam Scalar The scalar type
/// \param[in] line_segment1 The first line segment
/// \param[in] line_segment2 The second line segment
/// \param[out] closest_point1 The closest point on the first line segment
/// \param[out] closest_point2 The closest point on the second line segment
/// \param[out] arch_length1 The arch-length parameter of the closest point on the first line segment
/// \param[out] arch_length2 The arch-length parameter of the closest point on the second line segment
/// \param[out] sep The separation vector (from line_segment1 to line_segment2)
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const LineSegment<Scalar>& line_segment1, const LineSegment<Scalar>& line_segment2,
                                Point<Scalar>& closest_point1, Point<Scalar>& closest_point2, Scalar& arch_length1,
                                Scalar& arch_length2, Vector3<Scalar>& sep) {
  return distance(SharedNormalSigned{}, line_segment1, line_segment2, closest_point1, closest_point2, arch_length1,
                  arch_length2, sep);
}

/// \brief Compute the shared normal signed separation distance between two line segments
/// \tparam Scalar The scalar type
/// \param[in] line_segment1 The first line segment
/// \param[in] line_segment2 The second line segment
/// \param[out] closest_point1 The closest point on the first line segment
/// \param[out] closest_point2 The closest point on the second line segment
/// \param[out] arch_length1 The arch-length parameter of the closest point on the first line segment
/// \param[out] arch_length2 The arch-length parameter of the closest point on the second line segment
/// \param[out] sep The separation vector (from line_segment1 to line_segment2)
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const SharedNormalSigned distance_type,
                                const LineSegment<Scalar>& line_segment1, const LineSegment<Scalar>& line_segment2,
                                Point<Scalar>& closest_point1, Point<Scalar>& closest_point2, Scalar& arch_length1,
                                Scalar& arch_length2, Vector3<Scalar>& sep) {
  // Part of this function was adapted from VTK, which, in turn adapted part of it from "GeometryAlgorithms.com"
  const auto& l0 = line_segment1.start();
  const auto& l1 = line_segment1.end();
  const auto& m0 = line_segment2.start();
  const auto& m1 = line_segment2.end();

  const auto u = l1 - l0;
  const auto v = m1 - m0;
  const auto w = l0 - m0;
  const Scalar a = dot(u, u);
  const Scalar b = dot(u, v);
  const Scalar c = dot(v, v);
  const Scalar d = dot(u, w);
  const Scalar e = dot(v, w);
  const Scalar D = a * c - b * b;  // always >= 0

  // Compute the line parameters of the two closest points
  if (D < get_zero_tolerance<Scalar>()) {
    // CASE 1: The lines are colinear. Therefore, one of the four endpoints is the
    // point of closest approach. We'll directly compute the 4 distances and the closest point on the line.
    Point<Scalar> closest_point_tmp1;
    Point<Scalar> closest_point_tmp2;
    Point<Scalar> closest_point_tmp3;
    Point<Scalar> closest_point_tmp4;
    Scalar arch_length_tmp1;
    Scalar arch_length_tmp2;
    Scalar arch_length_tmp3;
    Scalar arch_length_tmp4;
    const Scalar dist1 = distance(l0, line_segment2, extras1, closest_point_tmp1, arch_length_tmp1);
    const Scalar dist2 = distance(l1, line_segment2, extras2, closest_point_tmp2, arch_length_tmp2);
    const Scalar dist3 = distance(m0, line_segment1, extras3, closest_point_tmp3, arch_length_tmp3);
    const Scalar dist4 = distance(m1, line_segment1, extras4, closest_point_tmp4, arch_length_tmp4);

    // Determine which of the 4 distances is the minimum using exact Scalar comparison.
    Scalar min_distance = Kokkos::min(Kokkos::min(dist1, dist2), Kokkos::min(dist3, dist4));
    if (min_distance == dist1) {
      // CASE 1.1: l0 is closest to the line segment m0-m1.
      arch_length1 = std::move(arch_length_tmp1);
      arch_length2 = static_cast<Scalar>(0.0);
      closest_point1 = l0;
      closest_point2 = std::move(closest_point_tmp1);
      sep = std::move(extras1.sep);
    } else if (min_distance == dist2) {
      // CASE 1.2: l1 is closest to the line segment m0-m1.
      arch_length1 = std::move(arch_length_tmp2);
      arch_length2 = static_cast<Scalar>(1.0);
      closest_point1 = l1;
      closest_point2 = std::move(closest_point_tmp2);
      sep = std::move(extras2.sep);
    } else if (min_distance == dist3) {
      // CASE 1.3: m0 is closest to the line segment l0-l1.
      arch_length1 = static_cast<Scalar>(0.0);
      arch_length2 = std::move(arch_length_tmp3);
      closest_point1 = std::move(closest_point_tmp3);
      closest_point2 = m0;
      sep = std::move(extras3.sep);
    } else {
      // CASE 1.4: m1 is closest to the line segment l0-l1.
      arch_length1 = static_cast<Scalar>(1.0);
      arch_length2 = std::move(arch_length_tmp4);
      closest_point1 = std::move(closest_point_tmp4);
      closest_point2 = m1;
      sep = std::move(extras4.sep);
    }

    return min_distance;
  }

  // CASE 2: The lines aren't parallel. Get the closest points on the infinite lines.
  // TODO(palmerb4): Comment the various if-else cases in terms of their physical meaning.
  Scalar sN = b * e - c * d;
  Scalar tN = a * e - b * d;
  Scalar sD = D;                        // sc = sN / sD, default sD = D >= 0
  Scalar tD = D;                        // tc = tN / tD, default tD = D >= 0
  if (sN < static_cast<Scalar>(0.0)) {  // sc < 0 => the s=0 edge is visible
    sN = static_cast<Scalar>(0.0);
    tN = e;
    tD = c;
  } else if (sN > sD) {  // sc > 1 => the s=1 edge is visible
    sN = sD;
    tN = e + b;
    tD = c;
  }

  if (tN < static_cast<Scalar>(0.0)) {  // tc < 0 => the t=0 edge is visible
    tN = static_cast<Scalar>(0.0);

    // Recompute sc for this edge
    if (-d < static_cast<Scalar>(0.0)) {
      sN = static_cast<Scalar>(0.0);
    } else if (-d > a) {
      sN = sD;
    } else {
      sN = -d;
      sD = a;
    }
  } else if (tN > tD) {  // tc > 1 => the t=1 edge is visible
    tN = tD;

    // Recompute sc for this edge
    if ((-d + b) < static_cast<Scalar>(0.0)) {
      sN = static_cast<Scalar>(0.0);
    } else if ((-d + b) > a) {
      sN = sD;
    } else {
      sN = (-d + b);
      sD = a;
    }
  }

  // Finally, get the arch-length parameters, the corresponding closest points, and their distance.
  arch_length1 = (Kokkos::fabs(sN) < get_zero_tolerance<Scalar>()) ? static_cast<Scalar>(0.0) : sN / sD;
  arch_length2 = (Kokkos::fabs(tN) < get_zero_tolerance<Scalar>()) ? static_cast<Scalar>(0.0) : tN / tD;
  closest_point1 = l0 + extras.arch_length1 * u;
  closest_point2 = m0 + extras.arch_length2 * v;
  sep = extras.closest_point1 - extras.closest_point2;
  const Scalar dist = distance(extras.sep);

  return dist;
}

/// \brief Compute the euclidean separation distance between two line segments
/// \tparam Scalar The scalar type
/// \param[in] line_segment1 The first line segment
/// \param[in] line_segment2 The second line segment
/// \param[out] closest_point1 The closest point on the first line segment
/// \param[out] closest_point2 The closest point on the second line segment
/// \param[out] arch_length1 The arch-length parameter of the closest point on the first line segment
/// \param[out] arch_length2 The arch-length parameter of the closest point on the second line segment
/// \param[out] sep The separation vector (from line_segment1 to line_segment2)
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const LineSegment<Scalar>& line_segment1, const LineSegment<Scalar>& line_segment2,
                                LineSegmentLineSegmentExtras<Scalar>& extras, Point<Scalar>& closest_point1,
                                Point<Scalar>& closest_point2, Scalar& arch_length1, Scalar& arch_length2,
                                Vector3<Scalar>& sep) {
  return distance(SharedNormalSigned{}, line_segment1, line_segment2, extras, closest_point1, closest_point2,
                  arch_length1, arch_length2, sep);
}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_DISTANCE_LINESEGMENTLINESEGMENT_HPP_
