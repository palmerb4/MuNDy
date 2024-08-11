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

#ifndef MUNDY_MATH_DISTANCE_ELLIPSOIDELLIPSOID_HPP_
#define MUNDY_MATH_DISTANCE_ELLIPSOIDELLIPSOID_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core libs
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <iostream>
#include <type_traits>
#include <utility>

// Our libs
#include <mundy_math/Quaternion.hpp>  // for mundy::math::Quaternion
#include <mundy_math/Vector.hpp>      // for mundy::math::Vector
#include <mundy_math/Vector3.hpp>     // for mundy::math::Vector3
#include <mundy_math/minimize.hpp>    // for mundy::math::find_min_using_approximate_derivatives

namespace mundy {

namespace math {

namespace distance {

/* Notes:

The purpose of the shared normal ellipsoid-ellipsoid distance function is to compute the minimum distance between two
ellipsoids in 3D space under the constraint that they share a normal at the corresponding points of closest approach.
The benefit of the shared normal approach is that it takes the optimization problem from 4D to 2D. These two dimensions
are the spherical polar coordinates for the angle of the shared normal vector. We convert these coordinates to the
normal vector in R3 and then use a special map which takes the normal vector and maps it its foot point on the
ellipsoid.
*/

namespace impl {

KOKKOS_INLINE_FUNCTION void if_not_nullptr_then_set(auto const ptr, const auto& value) {
  if (ptr) {
    *ptr = value;
  }
}

KOKKOS_INLINE_FUNCTION double sign(double x) {
  return (x > 0) - (x < 0);
}

}  // namespace impl

KOKKOS_INLINE_FUNCTION Vector3<double> map_spherical_to_unit_vector(double theta, double phi) {
  const double sin_theta = std::sin(theta);
  const double cos_theta = std::cos(theta);
  const double sin_phi = std::sin(phi);
  const double cos_phi = std::cos(phi);
  return Vector3<double>(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
}

KOKKOS_INLINE_FUNCTION Vector3<double> map_body_frame_normal_to_superellipsoid(
    const Vector3<double, auto, auto>& body_frame_nhat, double r1, double r2, double r3, double e1, double e2) {
  double alpha1, alpha2;
  if (body_frame_nhat[0] != 0) {
    alpha1 = 1.0 / (1.0 + std::pow(std::abs(r2 * body_frame_nhat[1] / (r1 * body_frame_nhat[0])), 2.0 / (2.0 - e1)));
    alpha2 = 1.0 / (1.0 + std::pow(std::abs(r3 * body_frame_nhat[2] / (r1 * body_frame_nhat[0])), 2.0 / (2.0 - e2)) *
                              std::pow(alpha1, (2.0 - e1) / (2.0 - e2)));
  } else if (body_frame_nhat[1] != 0) {
    alpha1 = 0.0;
    alpha2 = 1.0 / (1.0 + std::pow(std::abs(r3 * body_frame_nhat[2] / (r2 * body_frame_nhat[1])), 2.0 / (2.0 - e2)));
  } else {
    alpha1 = 0.0;
    alpha2 = 0.0;
  }

  const double x = 0.5 * impl::sign(body_frame_nhat[0]) *
                   ((1.0 + impl::sign(body_frame_nhat[0])) * r1 + (1.0 - impl::sign(body_frame_nhat[0])) * r1) *
                   std::pow(alpha1, 0.5 * e1) * std::pow(alpha2, 0.5 * e2);
  const double y = 0.5 * impl::sign(body_frame_nhat[1]) *
                   ((1.0 + impl::sign(body_frame_nhat[1])) * r2 + (1.0 - impl::sign(body_frame_nhat[1])) * r2) *
                   std::pow(1.0 - alpha1, 0.5 * e1) * std::pow(alpha2, 0.5 * e2);
  const double z = 0.5 * impl::sign(body_frame_nhat[2]) *
                   ((1.0 + impl::sign(body_frame_nhat[2])) * r3 + (1.0 - impl::sign(body_frame_nhat[2])) * r3) *
                   std::pow(1.0 - alpha2, 0.5 * e2);

  return Vector3<double>(x, y, z);
}

KOKKOS_INLINE_FUNCTION double centerline_projection_ssd_point_to_ellipsoid(
    const Vector3<double, auto, auto>& center, const Quaternion<double, auto, auto>& orientation, const double r1,
    const double r2, const double r3,
    const Vector3<double, auto, auto>& point,
    Vector3<double>* const closest_point = nullptr) { 
  // Step 1: Map the point to the ellipsoid body frame
  const Vector3<double> body_frame_point = inverse(orientation) * (point - center);

  // Step 2: Map the body frame point to the coordinate system where the ellipsoid is a unit sphere
  const Vector3<double> body_frame_point_unstretched = Vector3<double>(body_frame_point[0] / r1, body_frame_point[1] / r2, body_frame_point[2] / r3);
  const double body_frame_point_unstretched_norm = two_norm(body_frame_point_unstretched);
  const double is_inside = body_frame_point_unstretched_norm < 1.0;

  // Step 3: Compute the closest point on the unit sphere
  const Vector3<double> contact_point_unstretched = body_frame_point_unstretched / body_frame_point_unstretched_norm;

  // Step 4: Map the closest point back to the ellipsoid lab frame
  const Vector3<double> contact_point(r1 * contact_point_unstretched[0], r2 * contact_point_unstretched[1], r3 * contact_point_unstretched[2]);
  const Vector3<double> lab_frame_contact_point = orientation * contact_point + center;

  // Step 5: Compute the centerline projection distance
  impl::if_not_nullptr_then_set(closest_point, lab_frame_contact_point);

  return mundy::math::norm(point - lab_frame_contact_point) * (is_inside ? -1.0 : 1.0);
}

KOKKOS_INLINE_FUNCTION double shared_normal_ssd_between_superellipsoid_and_point(
    const Vector3<double, auto, auto>& center, const Quaternion<double, auto, auto>& orientation, const double r1,
    const double r2, const double r3, const double e1, const double e2,
    const Vector3<double, auto, auto>& point,
    Vector3<double>* const closest_point = nullptr) {
  // Setup our cost function
  auto theta_phi_to_shared_normal_contact_points_and_ssd = KOKKOS_LAMBDA(const Vector<double, 2>& theta_phi) {
    // Step 1: Map theta and phi to the lab frame normal vector
    const Vector3<double> lab_frame_nhat = map_spherical_to_unit_vector(theta_phi[0], theta_phi[1]);

    // Step 2: Map each normal vector to their corresponding superellipsoid body frame
    const Vector3<double> body_frame_nhat = inverse(orientation) * lab_frame_nhat;

    // Step 3: Map the body frame normal to its body frame foot point on the superellipsoid
    const Vector3<double> body_frame_foot_point =
        map_body_frame_normal_to_superellipsoid(body_frame_nhat, r1, r2, r3, e1, e2);

    // Step 4: Compute the lab frame foot point
    const Vector3<double> lab_frame_foot_point = orientation * body_frame_foot_point + center;
    const double signed_separation_distance =
        mundy::math::dot(point - lab_frame_foot_point, lab_frame_nhat);

    return std::make_tuple(lab_frame_foot_point, signed_separation_distance);
  };

  auto shared_normal_objective_function = KOKKOS_LAMBDA(const Vector<double, 2>& theta_phi) {
    // Step 1: Map theta and phi to the lab frame foot points and signed separation distance
    [[maybe_unused]] const auto [foot_point, signed_separation_distance] =
        theta_phi_to_shared_normal_contact_points_and_ssd(theta_phi);
    return two_norm_squared(point - foot_point);
  };

  // Setup the minimization
  // Note, the actual error is not guaranteed to be less than min_objective_delta due to the use of approximate
  // derivatives. Instead, we saw that the error was typically less than the square root of min_objective_delta.
  const double min_objective_delta = 1e-7;
  constexpr size_t lbfgs_max_memory_size = 10;

  // N-dimensional Rosenbrock function
  double global_separation_distance = Kokkos::Experimental::infinity_v<double>;
  Vector<double, 2> global_theta_phi_guess_and_solution;
  constexpr double pi = Kokkos::numbers::pi_v<double>;
  Kokkos::Array<double, 3> theta_guesses = {0, 0.5 * pi, pi};
  Kokkos::Array<double, 3> phi_guesses = {pi / 3.0, pi, 5.0 * pi / 3.0};

  for (size_t t_idx = 0; t_idx < 3; ++t_idx) {
    for (size_t p_idx = 0; p_idx < 3; ++p_idx) {
      Vector<double, 2> theta_phi_guess_and_solution = {theta_guesses[t_idx], phi_guesses[p_idx]};
      const double separation_distance = find_min_using_approximate_derivatives<lbfgs_max_memory_size>(
          shared_normal_objective_function, theta_phi_guess_and_solution, min_objective_delta);
      if (separation_distance < global_separation_distance) {
        global_separation_distance = separation_distance;
        global_theta_phi_guess_and_solution = theta_phi_guess_and_solution;
      }
    }
  }

  // Write out the results
  const auto [foot_point, global_signed_separation_distance] =
      theta_phi_to_shared_normal_contact_points_and_ssd(global_theta_phi_guess_and_solution);
  impl::if_not_nullptr_then_set(closest_point, foot_point);
  return global_signed_separation_distance;
}

KOKKOS_INLINE_FUNCTION double shared_normal_ssd_between_ellipsoid_and_point(
    const Vector3<double, auto, auto>& center, const Quaternion<double, auto, auto>& orientation, const double r1,
    const double r2, const double r3, 
    const Vector3<double, auto, auto>& point,
    Vector3<double>* const closest_point = nullptr) {
  return shared_normal_ssd_between_superellipsoid_and_point(center, orientation, r1, r2, r3, 1.0, 1.0, point, closest_point);
}

KOKKOS_INLINE_FUNCTION double shared_normal_ssd_between_superellipsoids(
    const Vector3<double, auto, auto>& center0, const Quaternion<double, auto, auto>& orientation0, const double r1_0,
    const double r2_0, const double r3_0, const double e1_0, const double e2_0,
    const Vector3<double, auto, auto>& center1, const Quaternion<double, auto, auto>& orientation1, const double r1_1,
    const double r2_1, const double r3_1, const double e1_1, const double e2_1,
    Vector3<double>* const closest_point0 = nullptr, Vector3<double>* const closest_point1 = nullptr) {
  // Setup our cost function
  auto theta_phi_to_shared_normal_contact_points_and_ssd = KOKKOS_LAMBDA(const Vector<double, 2>& theta_phi) {
    // Step 1: Map theta and phi to the lab frame normal vector
    const Vector3<double> lab_frame_nhat0 = map_spherical_to_unit_vector(theta_phi[0], theta_phi[1]);
    const Vector3<double> lab_frame_nhat1 = -lab_frame_nhat0;

    // Step 2: Map each normal vector to their corresponding ellipsoid body frame
    const Vector3<double> body_frame_nhat0 = inverse(orientation0) * lab_frame_nhat0;
    const Vector3<double> body_frame_nhat1 = inverse(orientation1) * lab_frame_nhat1;

    // Step 3: Map each body frame normal to their body frame foot point on the ellipsoid
    const Vector3<double> body_frame_foot_point0 =
        map_body_frame_normal_to_superellipsoid(body_frame_nhat0, r1_0, r2_0, r3_0, e1_0, e2_0);
    const Vector3<double> body_frame_foot_point1 =
        map_body_frame_normal_to_superellipsoid(body_frame_nhat1, r1_1, r2_1, r3_1, e1_1, e2_1);

    // Step 4: Compute the lab frame foot points
    const Vector3<double> lab_frame_foot_point0 = orientation0 * body_frame_foot_point0 + center0;
    const Vector3<double> lab_frame_foot_point1 = orientation1 * body_frame_foot_point1 + center1;
    const double signed_separation_distance =
        mundy::math::dot(lab_frame_foot_point1 - lab_frame_foot_point0, lab_frame_nhat0);

    return std::make_tuple(lab_frame_foot_point0, lab_frame_foot_point1, signed_separation_distance);
  };

  auto shared_normal_objective_function = KOKKOS_LAMBDA(const Vector<double, 2>& theta_phi) {
    // Step 1: Map theta and phi to the lab frame foot points and signed separation distance
    [[maybe_unused]] const auto [foot_point0, foot_point1, signed_separation_distance] =
        theta_phi_to_shared_normal_contact_points_and_ssd(theta_phi);
    return two_norm_squared(foot_point1 - foot_point0);
  };

  // Setup the minimization
  // Note, the actual error is not guaranteed to be less than min_objective_delta due to the use of approximate
  // derivatives. Instead, we saw that the error was typically less than the square root of min_objective_delta.
  const double min_objective_delta = 1e-7;
  constexpr size_t lbfgs_max_memory_size = 10;

  // N-dimensional Rosenbrock function
  double global_separation_distance = Kokkos::Experimental::infinity_v<double>;
  Vector<double, 2> global_theta_phi_guess_and_solution;
  constexpr double pi = Kokkos::numbers::pi_v<double>;
  Kokkos::Array<double, 3> theta_guesses = {0, 0.5 * pi, pi};
  Kokkos::Array<double, 3> phi_guesses = {pi / 3.0, pi, 5.0 * pi / 3.0};

  for (size_t t_idx = 0; t_idx < 3; ++t_idx) {
    for (size_t p_idx = 0; p_idx < 3; ++p_idx) {
      Vector<double, 2> theta_phi_guess_and_solution = {theta_guesses[t_idx], phi_guesses[p_idx]};
      const double separation_distance = find_min_using_approximate_derivatives<lbfgs_max_memory_size>(
          shared_normal_objective_function, theta_phi_guess_and_solution, min_objective_delta);
      if (separation_distance < global_separation_distance) {
        global_separation_distance = separation_distance;
        global_theta_phi_guess_and_solution = theta_phi_guess_and_solution;
      }
    }
  }

  // Write out the results
  const auto [foot_point0, foot_point1, global_signed_separation_distance] =
      theta_phi_to_shared_normal_contact_points_and_ssd(global_theta_phi_guess_and_solution);
  impl::if_not_nullptr_then_set(closest_point0, foot_point0);
  impl::if_not_nullptr_then_set(closest_point1, foot_point1);
  return global_signed_separation_distance;
}

}  // namespace distance

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_DISTANCE_ELLIPSOIDELLIPSOID_HPP_
