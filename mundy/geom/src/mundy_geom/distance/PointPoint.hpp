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

#ifndef MUNDY_MATH_DISTANCE_POINTPOINT_HPP_
#define MUNDY_MATH_DISTANCE_POINTPOINT_HPP_

// External libs
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_math/Point.hpp>           // for mundy::math::Point
#include <mundy_math/distance/Types.hpp>  // for SharedNormalSigned, Euclidean

namespace mundy {

namespace math {

/// \brief Compute the shared normal signed separation distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const Point<Scalar>& point1, const Point<Scalar>& point2) {
  return distance(SharedNormalSigned{}, point1, point2);
}

/// \brief Compute the shared normal signed separation distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const SharedNormalSigned distance_type, const Point<Scalar>& point1,
                                const Point<Scalar>& point2) {
  Scalar dx = point1[0] - point2[0];
  Scalar dy = point1[1] - point2[1];
  Scalar dz = point1[2] - point2[2];
  return Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
}

/// \brief Compute the euclidean distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const Euclidean distance_type, const Point<Scalar>& point1,
                                const Point<Scalar>& point2) {
  return distance(SharedNormalSigned{}, point1, point2);
}

/// \brief Compute the euclidean distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
/// \param[out] sep The separation vector (from point1 to point2)
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance(const Point<Scalar>& point1, const Point<Scalar>& point2, Vector3<Scalar>& sep) {
  return distance(SharedNormalSigned{}, point1, point2, sep);
}

/// \brief Compute the shared normal signed separation distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
/// \param[out] sep The separation vector (from point1 to point2)
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const SharedNormalSigned distance_type, const Point<Scalar>& point1,
                                const Point<Scalar>& point2, Vector3<Scalar>& sep) {
  sep = point1 - point2;
  return norm(sep);
}

/// \brief Compute the euclidean distance between two points
/// \tparam Scalar The scalar type
/// \param[in] point1 The first point
/// \param[in] point2 The second point
/// \param[out] sep The separation vector (from point1 to point2)
template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const Euclidean distance_type, const Point<Scalar>& point1,
                                const Point<Scalar>& point2, Vector3<Scalar>& sep) {
  return distance(SharedNormalSigned{}, point1, point2, sep);
}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_DISTANCE_POINTPOINT_HPP_
