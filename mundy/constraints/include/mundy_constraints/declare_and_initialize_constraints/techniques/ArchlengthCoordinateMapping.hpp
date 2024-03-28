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

#ifndef MUNDY_CONSTRAINTS_DECLARE_AND_INITIALIZE_CONSTRAINTS_TECHNIQUES_ARCHLENGTHCOORDINATEMAPPING_HPP_
#define MUNDY_CONSTRAINTS_DECLARE_AND_INITIALIZE_CONSTRAINTS_TECHNIQUES_ARCHLENGTHCOORDINATEMAPPING_HPP_

/// \file ArchlengthCoordinateMapping.hpp
/// \brief An interface and some concrete implementations for mapping archlength to coordinates.

// C++ core libs
#include <array>  // for std::array
#include <cmath>  // for std::sin, std::cos, std::sqrt

// Mundy includes
#include <mundy_math/Matrix3.hpp>     // for mundy::math::Matrix3
#include <mundy_math/Quaternion.hpp>  // for mundy::math::Quaternion
#include <mundy_math/Vector3.hpp>     // for mundy::math::Vector3

namespace mundy {

namespace constraints {

namespace declare_and_initialize_constraints {

namespace techniques {

/// @brief An interface for mapping grid indices to coordinates.
class ArchlengthCoordinateMapping {
 public:
  /// \brief Virtual destructor.
  virtual ~ArchlengthCoordinateMapping() = default;

  /// \brief Get the coordinate corresponding to a given archlength index in [0, num_nodes-1]
  /// \param archlength_index The archlength index.
  /// \return The corresponding coordinate.
  virtual std::array<double, 3> get_grid_coordinate(const size_t &archlength_index) const = 0;
};  // class ArchlengthCoordinateMapping

/// @brief Straight line with given center, length, and orientation.
class StraightLine : public ArchlengthCoordinateMapping {
 public:
  /// Constructor
  StraightLine(const size_t num_nodes, const double center_x, const double center_y, const double center_z,
               const double length, const double orientation_x, const double orientation_y, const double orientation_z)
      : num_nodes_(num_nodes),
        center_x_(center_x),
        center_y_(center_y),
        center_z_(center_z),
        length_(length),
        orientation_x_(orientation_x),
        orientation_y_(orientation_y),
        orientation_z_(orientation_z) {
  }

  /// \brief Get the grid coordinate corresponding to a given grid index.
  /// \param archlength_index The archlength index.
  /// \return The corresponding coordinate.
  std::array<double, 3> get_grid_coordinate(const size_t &archlength_index) const override {
    const double x = center_x_ + (static_cast<double>(archlength_index) / static_cast<double>(num_nodes_ - 1) - 0.5) *
                                     length_ * orientation_x_;
    const double y = center_y_ + (static_cast<double>(archlength_index) / static_cast<double>(num_nodes_ - 1) - 0.5) *
                                     length_ * orientation_y_;
    const double z = center_z_ + (static_cast<double>(archlength_index) / static_cast<double>(num_nodes_ - 1) - 0.5) *
                                     length_ * orientation_z_;
    return {x, y, z};
  }

 private:
  size_t num_nodes_;
  double center_x_;
  double center_y_;
  double center_z_;
  double length_;
  double orientation_x_;
  double orientation_y_;
  double orientation_z_;
};  // class StraightLine

/// @brief Helix with given center, radius, length and the axis to the helix's center.
class Helix : public ArchlengthCoordinateMapping {
 public:
  /// Constructor
  Helix(const size_t &num_nodes, const double &radius, const double &length, const double &center_x,
        const double &center_y, const double &center_z, const double &axis_x, const double &axis_y,
        const double &axis_z)
      : num_nodes_(num_nodes),
        radius_(radius),
        length_(length),
        center_{center_x, center_y, center_z},
        axis_{axis_x, axis_y, axis_z} {
    // Not that we don't trust you or anything, but we need to make sure the axis is normalized.
    axis_ /= mundy::math::norm(axis_);

    // We need to find two orthonormal vectors to the axis of the helix.
    // We can do this by finding an arbitrary vector that is not parallel to the axis, taking the cross product
    // with the normal, and normalizing the result. This gives us a vector that is orthogonal to the axis.
    // By taking the cross product of the axis and this vector, we get a second vector that is orthogonal to both.
    const mundy::math::Vector3<double> ihat(1.0, 0.0, 0.0);
    const mundy::math::Vector3<double> jhat(0.0, 1.0, 0.0);
    basis_vector0_ = mundy::math::norm(mundy::math::cross(axis_, ihat)) > 1.0e-12 ? ihat : jhat;
    basis_vector0_ /= mundy::math::norm(basis_vector0_);
    basis_vector1_ = mundy::math::norm(mundy::math::cross(axis_, basis_vector0_));
    basis_vector1_ /= mundy::math::norm(basis_vector1_);
  }

  /// \brief Get the grid coordinate corresponding to a given grid index.
  /// \param archlength_index The archlength index.
  /// \return The corresponding coordinate.
  std::array<double, 3> get_grid_coordinate(const size_t &archlength_index) const override {
    const double angle = 2.0 * M_PI * static_cast<double>(archlength_index) / static_cast<double>(num_nodes_);
    const auto pos_circle = center_ + radius_ * (basis_vector0_ * std::cos(angle) + basis_vector1_ * std::sin(angle));
    const double shift = (static_cast<double>(archlength_index) / static_cast<double>(num_nodes_ - 1) - 0.5) * length_;
    const auto pos = pos_circle + shift * axis_;
    return {pos[0], pos[1], pos[2]};
  }

 private:
  size_t num_nodes_;
  double radius_;
  double length_;
  mundy::math::Vector3<double> center_;
  mundy::math::Vector3<double> axis_;
  mundy::math::Vector3<double> basis_vector0_;
  mundy::math::Vector3<double> basis_vector1_;
};  // class Helix

}  // namespace techniques

}  // namespace declare_and_initialize_constraints

}  // namespace constraints

}  // namespace mundy

#endif  // MUNDY_CONSTRAINTS_DECLARE_AND_INITIALIZE_CONSTRAINTS_TECHNIQUES_ARCHLENGTHCOORDINATEMAPPING_HPP_