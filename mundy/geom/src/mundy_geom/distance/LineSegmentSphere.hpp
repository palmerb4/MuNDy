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

#ifndef MUNDY_GEOM_DISTANCE_LINESEGMENTSPHERE_HPP_
#define MUNDY_GEOM_DISTANCE_LINESEGMENTSPHERE_HPP_

// External libs
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_geom/primitives/LineSegment.hpp>                // for mundy::geom::LineSegment
#include <mundy_geom/primitives/Point.hpp>                      // for mundy::geom::Point
#include <mundy_geom/primitives/Sphere.hpp>                     // for mundy::geom::Sphere
#include <mundy_geom/distance/PointLineSegment.hpp>  // for mundy::geom::distance(Point, LineSegment)
#include <mundy_geom/distance/Types.hpp>             // for mundy::geom::SharedNormalSigned

namespace mundy {

namespace geom {

/// \brief Compute the distance between a line segment and a sphere
/// \tparam Scalar The scalar type
/// \param[in] line_segment The line segment
/// \param[in] sphere The sphere
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const LineSegment<Scalar>& line_segment, const Sphere<Scalar>& sphere) {
  return distance(SharedNormalSigned{}, line_segment, sphere);
}

/// \brief Compute the shared normal signed separation distance between a line segment and a sphere
/// \tparam Scalar The scalar type
/// \param[in] line_segment The line segment
/// \param[in] sphere The sphere
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const SharedNormalSigned distance_type,
                                const LineSegment<Scalar>& line_segment, const Sphere<Scalar>& sphere) {
  return distance(sphere.center(), line_segment) - sphere.radius();
}

/// \brief Compute the distance between a line segment and a sphere
/// \tparam Scalar The scalar type
/// \param[in] line_segment The line segment
/// \param[in] sphere The sphere
/// \param[out] closest_point The closest point on the line segment
/// \param[out] arch_length The arch-length parameter of the closest point on the line segment
/// \param[out] sep The separation vector (from line_segment to sphere)
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const LineSegment<Scalar>& line_segment, const Sphere<Scalar>& sphere,
                                Point<Scalar>& closest_point, Scalar& arch_length, mundy::math::Vector3<Scalar>& sep) {
  return distance(SharedNormalSigned{}, line_segment, sphere, closest_point, arch_length, sep);
}

/// \brief Compute the shared normal signed separation distance between a line segment and a sphere
/// \tparam Scalar The scalar type
/// \param[in] line_segment The line segment
/// \param[in] sphere The sphere
/// \param[out] closest_point The closest point on the line segment
/// \param[out] arch_length The arch-length parameter of the closest point on the line segment
/// \param[out] sep The separation vector (from line_segment to sphere)
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const SharedNormalSigned distance_type,
                                const LineSegment<Scalar>& line_segment, const Sphere<Scalar>& sphere,
                                Point<Scalar>& closest_point, Scalar& arch_length, mundy::math::Vector3<Scalar>& sep) {
  const Scalar line_center_distance = distance(sphere.center(), line_segment, closest_point, arch_length, sep);

  // Rescale the separation vector to the surface of the sphere
  const Scalar surface_distance = line_center_distance - sphere.radius();
  sep *= surface_distance / line_center_distance;
  return surface_distance;
}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_LINESEGMENTSPHERE_HPP_
