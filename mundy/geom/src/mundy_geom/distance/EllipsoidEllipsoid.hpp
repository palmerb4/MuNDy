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

// Mundy
#include <mundy_geom/Point.hpp>           // for mundy::geom::Point
#include <mundy_math/Quaternion.hpp>      // for mundy::math::Quaternion
#include <mundy_math/Tolerance.hpp>       // for mundy::math::get_zero_tolerance
#include <mundy_math/Vector3.hpp>         // for mundy::math::Vector3
#include <mundy_math/distance/Types.hpp>  // for SharedNormalSigned

namespace mundy {

namespace math {

/* Notes:

The purpose of the shared normal ellipsoid-ellipsoid distance function is to compute the minimum distance between two
ellipsoids in 3D space under the constraint that they share a normal at the corresponding points of closest approach.
The benefit of the shared normal approach is that it takes the optimization problem from 4D to 2D. These two dimensions
are the spherical polar coordinates for the angle of the shared normal vector. We convert these coordinates to the
normal vector in R3 and then use a special map which takes the normal vector and maps it its foot point on the
ellipsoid.
*/

namespace impl {

template <typename Scalar>
KOKKOS_FUNCTION Scalar sign(Scalar x) {
  return (x > 0) - (x < 0);
}

}  // namespace impl

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
                                const Ellipsoid<Scalar>& ellipsoid1, const Ellipsoid<Scalar>& ellipsoid2,
                                Vector3<Scalar>& closest_point1, Vector3<Scalar>& closest_point2,
                                Vector3<Scalar>& shared_normal1, Vector3<Scalar>& shared_normal2) {
  // Setup our cost function
  auto theta_phi_to_shared_normal_contact_points_and_ssd = KOKKOS_LAMBDA(const Vector<Scalar, 2>& theta_phi) {
    // Step 1: Map theta and phi to the lab frame normal vector
    const auto lab_frame_nhat0 = map_spherical_to_unit_vector(theta_phi[0], theta_phi[1]);
    const auto lab_frame_nhat1 = -lab_frame_nhat0;

    // Step 2: Map each normal vector to their corresponding ellipsoid body frame
    const auto body_frame_nhat0 = conjugate(ellipsoid1.orientation()) * lab_frame_nhat0;
    const auto body_frame_nhat1 = conjugate(ellipsoid2.orientation()) * lab_frame_nhat1;

    // Step 3: Map each body frame normal to their body frame foot point on the ellipsoid
    const auto body_frame_foot_point0 = map_body_frame_normal_to_ellipsoid(
        body_frame_nhat0, static_cast<Scalar>(0.5) * ellipsoid1.axis_length_1(),
        static_cast<Scalar>(0.5) * ellipsoid1.axis_length_2(), static_cast<Scalar>(0.5) * ellipsoid1.axis_length_3());
    const auto body_frame_foot_point1 = map_body_frame_normal_to_ellipsoid(
        body_frame_nhat1, static_cast<Scalar>(0.5) * ellipsoid2.axis_length_1(),
        static_cast<Scalar>(0.5) * ellipsoid2.axis_length_2(), static_cast<Scalar>(0.5) * ellipsoid2.axis_length_3());

    // Step 4: Compute the lab frame foot points
    const auto lab_frame_foot_point0 = ellipsoid1.orientation() * body_frame_foot_point0 + ellipsoid1.center();
    const auto lab_frame_foot_point1 = ellipsoid2.orientation() * body_frame_foot_point1 + ellipsoid2.center();
    const Scalar signed_separation_dist =
        mundy::math::dot(lab_frame_foot_point1 - lab_frame_foot_point0, lab_frame_nhat0);

    return Kokkos::make_tuple(lab_frame_nhat0, lab_frame_nhat1, lab_frame_foot_point0, lab_frame_foot_point1,
                              signed_separation_dist);
  };

  auto shared_normal_objective_function = KOKKOS_LAMBDA(const Vector<Scalar, 2>& theta_phi) {
    // Step 1: Map theta and phi to the lab frame foot points and signed separation distance
    [[maybe_unused]] const auto [nhat1, nhat2, foot_point0, foot_point1, signed_separation_dist] =
        theta_phi_to_shared_normal_contact_points_and_ssd(theta_phi);
    return two_norm_squared(foot_point1 - foot_point0);
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
  const auto [lab_frame_shared_normal0, lab_frame_shared_normal1, foot_point0, foot_point1,
              global_signed_separation_dist] =
      theta_phi_to_shared_normal_contact_points_and_ssd(global_theta_phi_guess_and_solution);
  closest_point0 = foot_point0;
  closest_point1 = foot_point1;
  shared_normal0 = lab_frame_shared_normal0;
  shared_normal1 = lab_frame_shared_normal1;
  return global_signed_separation_dist;
}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_DISTANCE_ELLIPSOIDELLIPSOID_HPP_
