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

#ifndef MUNDY_MATH_DISTANCE_POINTELLIPSOID_HPP_
#define MUNDY_MATH_DISTANCE_POINTELLIPSOID_HPP_

// External libs
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_geom/Point.hpp>           // for mundy::geom::Point
#include <mundy_math/Quaternion.hpp>      // for mundy::math::Quaternion
#include <mundy_math/Tolerance.hpp>       // for mundy::math::get_zero_tolerance
#include <mundy_math/Vector3.hpp>         // for mundy::math::Vector3
#include <mundy_math/distance/Types.hpp>  // for SharedNormalSigned

namespace mundy {

namespace math {

template <typename Scalar>
KOKKOS_FUNCTION Vector3<Scalar> map_spherical_to_unit_vector(const Scalar theta, const Scalar phi) {
  const Scalar sin_theta = Kokkos::sin(theta);
  const Scalar cos_theta = Kokkos::cos(theta);
  const Scalar sin_phi = Kokkos::sin(phi);
  const Scalar cos_phi = Kokkos::cos(phi);
  return Vector3<Scalar>(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
}

template <typename Scalar>
KOKKOS_FUNCTION Scalar distance([[maybe_unused]] const SharedNormalSigned distance_type,
                                const Ellipsoid<Scalar>& ellipsoid, const Point<Scalar>& point,
                                Vector3<Scalar>& closest_point, Vector3<Scalar>& ellipsoid_nhat) {
  // Setup our cost function
  auto theta_phi_to_shared_normal_contact_points_and_ssd = KOKKOS_LAMBDA(const Vector<Scalar, 2>& theta_phi) {
    // Step 1: Map theta and phi to the lab frame normal vector
    const Vector3<Scalar> lab_frame_ellipsoid_nhat = map_spherical_to_unit_vector(theta_phi[0], theta_phi[1]);

    // Step 2: Map each normal vector to their corresponding ellipsoid body frame
    const Vector3<Scalar> body_frame_nhat = conjugate(ellipsoid.orientation()) * lab_frame_ellipsoid_nhat;

    // Step 3: Map the body frame normal to its body frame foot point on the ellipsoid
    const Vector3<Scalar> body_frame_foot_point = map_body_frame_normal_to_ellipsoid(
        body_frame_nhat, static_cast<Scalar>(0.5) * ellipsoid.axis_length_1(),
        static_cast<Scalar>(0.5) * ellipsoid.axis_length_2(), static_cast<Scalar>(0.5) * ellipsoid.axis_length_3());

    // Step 4: Compute the lab frame foot point
    const Vector3<Scalar> lab_frame_foot_point = ellipsoid.orientation() * body_frame_foot_point + ellipsoid.center();
    const Scalar signed_separation_dist = mundy::math::dot(point - lab_frame_foot_point, lab_frame_ellipsoid_nhat);

    return Kokkos::make_tuple(lab_frame_ellipsoid_nhat, lab_frame_foot_point, signed_separation_dist);
  };

  auto shared_normal_objective_function = KOKKOS_LAMBDA(const Vector<Scalar, 2>& theta_phi) {
    // Step 1: Map theta and phi to the lab frame foot points and signed separation distance
    [[maybe_unused]] const auto [nhat, foot_point, signed_separation_dist] =
        theta_phi_to_shared_normal_contact_points_and_ssd(theta_phi);
    return two_norm_squared(point - foot_point);
  };

  // Setup the minimization
  // Note, the actual error is not guaranteed to be less than min_objective_delta due to the use of approximate
  // derivatives. Instead, we saw that the error was typically less than the square root of min_objective_delta.
  constexpr Scalar min_objective_delta = get_relaxed_zero_tolerance<Scalar>();
  constexpr size_t lbfgs_max_memory_size = 10;

  Scalar global_separation_dist = Kokkos::Experimental::infinity_v<Scalar>;
  Vector<Scalar, 2> global_theta_phi_guess_and_solution;
  constexpr Scalar pi = Kokkos::numbers::pi_v<Scalar>;
  Kokkos::Array<Scalar, 3> theta_guesses = {static_cast<Scalar>(0.0), static_cast<Scalar>(0.5) * pi, pi};
  Kokkos::Array<Scalar, 3> phi_guesses = {pi / static_cast<Scalar>(3.0), pi,
                                          static_cast<Scalar>(5.0) * pi / static_cast<Scalar>(3.0)};

  Vector<Scalar, 2> theta_phi_guess_and_solution;
  for (size_t t_idx = 0; t_idx < 3; ++t_idx) {
    for (size_t p_idx = 0; p_idx < 3; ++p_idx) {
      theta_phi_guess_and_solution = {theta_guesses[t_idx], phi_guesses[p_idx]};
      const Scalar separation_dist = find_min_using_approximate_derivatives<lbfgs_max_memory_size>(
          shared_normal_objective_function, theta_phi_guess_and_solution, min_objective_delta);
      if (separation_dist < global_separation_dist) {
        global_separation_dist = separation_dist;
        global_theta_phi_guess_and_solution = theta_phi_guess_and_solution;
      }
    }
  }

  // Write out the results
  const auto [lab_frame_ellipsoid_nhat, foot_point, global_signed_separation_dist] =
      theta_phi_to_shared_normal_contact_points_and_ssd(global_theta_phi_guess_and_solution);
  closest_point = foot_point;
  ellipsoid_nhat = lab_frame_ellipsoid_nhat;
  return global_signed_separation_dist;
}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_DISTANCE_POINTELLIPSOID_HPP_
