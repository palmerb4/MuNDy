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

#ifndef MUNDY_MATH_DISTANCE_POINTSPHERE_HPP_
#define MUNDY_MATH_DISTANCE_POINTSPHERE_HPP_

// External libs
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_math/Point.hpp>                // for mundy::math::Point
#include <mundy_math/Sphere.hpp>               // for mundy::math::Sphere
#include <mundy_math/distance/PointPoint.hpp>  // for distance(Point, Point)
#include <mundy_math/distance/Types.hpp>       // for SharedNormalSigned

namespace mundy {

namespace math {

/// \brief Compute the shared normal signed separation distance between a point and a sphere
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] sphere The sphere
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const Point<Scalar>& point, const Sphere<Scalar>& sphere) {
  return distance(SharedNormalSigned{}, point, sphere);
}

/// \brief Compute the shared normal signed separation distance between a point and a sphere
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] sphere The sphere
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const SharedNormalSigned distance_type, const Point<Scalar>& point,
                                const Sphere<Scalar>& sphere) {
  return distance(point, sphere.center()) - sphere.radius();
}

/// \brief Compute the distance between a point and a sphere
/// \tparam Scalar The scalar type
/// \param[in] point The point
/// \param[in] sphere The sphere
/// \param[out] sep The separation vector (from point to sphere)
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const Point<Scalar>& point, const Sphere<Scalar>& sphere,
                                Vector3<Scalar>& sep) {
  const Scalar center_point_distance = distance(point, sphere.center(), closest_point, sep);

  // Rescale the separation vector to the surface of the sphere
  const Scalar surface_distance = center_point_distance - sphere.radius();
  sep *= surface_distance / center_point_distance;
  return surface_distance;
}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_DISTANCE_POINTSPHERE_HPP_
