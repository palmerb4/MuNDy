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

#ifndef MUNDY_GEOM_DISTANCE_LINESPHERE_HPP_
#define MUNDY_GEOM_DISTANCE_LINESPHERE_HPP_

// External libs
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_geom/distance/PointLine.hpp>  // for distance(Point, Line)
#include <mundy_geom/distance/Types.hpp>      // for mundy::geom::SharedNormalSigned
#include <mundy_geom/primitives/Line.hpp>     // for mundy::geom::Line
#include <mundy_geom/primitives/Point.hpp>    // for mundy::geom::Point
#include <mundy_geom/primitives/Sphere.hpp>   // for mundy::geom::Sphere

namespace mundy {

namespace geom {

/// \brief Compute the distance between a line and a sphere
/// \tparam Scalar The scalar type
/// \param[in] line The line
/// \param[in] sphere The sphere
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const Line<Scalar>& line, const Sphere<Scalar>& sphere) {
  return distance(SharedNormalSigned{}, line, sphere);
}

/// \brief Compute the shared normal signed separation distance between a line and a sphere
/// \tparam Scalar The scalar type
/// \param[in] line The line
/// \param[in] sphere The sphere
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const SharedNormalSigned distance_type, const Line<Scalar>& line,
                                const Sphere<Scalar>& sphere) {
  return distance(sphere.center(), line) - sphere.radius();
}

/// \brief Compute the distance between a line and a sphere
/// \tparam Scalar The scalar type
/// \param[in] line The line
/// \param[in] sphere The sphere
/// \param[out] closest_point The closest point on the line
/// \param[out] arch_length The arch-length parameter of the closest point on the line
/// \param[out] sep The separation vector (from line to sphere)
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const Line<Scalar>& line, const Sphere<Scalar>& sphere, Point<Scalar>& closest_point,
                                Scalar& arch_length, mundy::math::Vector3<Scalar>& sep) {
  return distance(SharedNormalSigned{}, line, sphere, closest_point, arch_length, sep);
}

/// \brief Compute the shared normal signed separation distance between a line and a sphere
/// \tparam Scalar The scalar type
/// \param[in] line The line
/// \param[in] sphere The sphere
/// \param[out] closest_point The closest point on the line
/// \param[out] arch_length The arch-length parameter of the closest point on the line
/// \param[out] sep The separation vector (from line to sphere)
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const SharedNormalSigned distance_type, const Line<Scalar>& line,
                                const Sphere<Scalar>& sphere, Point<Scalar>& closest_point, Scalar& arch_length,
                                mundy::math::Vector3<Scalar>& sep) {
  const Scalar line_point_distance = distance(sphere.center(), line, closest_point, arch_length, sep);

  // Rescale the separation vector to the surface of the sphere
  const Scalar surface_distance = line_point_distance - sphere.radius();
  sep *= surface_distance / line_point_distance;
  return surface_distance;
}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_LINESPHERE_HPP_
