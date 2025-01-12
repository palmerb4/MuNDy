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

// External libs
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::numbers::pi

// STK mesh
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh

// Mundy geom
#include <mundy_geom/aggregates.hpp>  // for mundy::geom::SphereData
#include <mundy_geom/primitives.hpp>  // for mundy::geom::Sphere

namespace mundy {

namespace geom {

namespace {

template <bool use_first>
decltype(auto) get_first_or_second(auto& first, auto& second) {
  if constexpr (use_first) {
    return first;
  } else {
    return second;
  }
}

template <stk::topology::topology_t OurTopology, bool is_radius_shared>
void test_sphere_data(stk::mesh::BulkData& bulk_data,          //
                      stk::mesh::Entity sphere,                //
                      stk::mesh::Field<double>& center_field,  //
                      stk::mesh::Field<double>& radius_field) {
  stk::topology our_topology = OurTopology;
  stk::topology::rank_t sphere_rank = our_topology.rank();
  ASSERT_TRUE(bulk_data.is_valid(sphere));
  ASSERT_TRUE(bulk_data.bucket(sphere).topology().rank() == sphere_rank);
  ASSERT_TRUE(sphere_rank == stk::topology::ELEMENT_RANK || sphere_rank == stk::topology::NODE_RANK);

  // The shared data for the sphere
  double radius = 1.0;

  // Test the regular sphere data to ensure that the stored shared data/fields are as expected
  auto sphere_data = create_sphere_data<double, OurTopology>(
      bulk_data, center_field, get_first_or_second<is_radius_shared>(radius, radius_field));
  ASSERT_EQ(&sphere_data.center_data(), &center_field);
  if constexpr (is_radius_shared) {
    ASSERT_EQ(&sphere_data.radius_data(), &radius);
  } else {
    ASSERT_EQ(&sphere_data.radius_data(), &radius_field);
  }

  // Same test for the NGP sphere data
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
  stk::mesh::NgpField<double>& ngp_center_field = stk::mesh::get_updated_ngp_field<double>(center_field);
  stk::mesh::NgpField<double>& ngp_radius_field = stk::mesh::get_updated_ngp_field<double>(radius_field);
  auto ngp_sphere_data = create_ngp_sphere_data<double, OurTopology>(
      ngp_mesh, ngp_center_field, get_first_or_second<is_radius_shared>(radius, ngp_radius_field));

  ASSERT_EQ(&ngp_sphere_data.center_data(), &ngp_center_field);
  if constexpr (is_radius_shared) {
    ASSERT_EQ(&ngp_sphere_data.radius_data(), &radius);
  } else {
    ASSERT_EQ(&ngp_sphere_data.radius_data(), &ngp_radius_field);
  }

  // Set the center and radius data for the sphere directly via their fields
  const double non_shared_radius = 2.0;
  const Point<double> non_shared_center{1.0, 2.0, 3.0};
  double old_non_shared_radius = non_shared_radius;
  Point<double> old_non_shared_center = non_shared_center;
  if (sphere_rank == stk::topology::NODE_RANK) {
    stk::mesh::field_data(center_field, sphere)[0] = non_shared_center[0];
    stk::mesh::field_data(center_field, sphere)[1] = non_shared_center[1];
    stk::mesh::field_data(center_field, sphere)[2] = non_shared_center[2];
    if constexpr (!is_radius_shared) {
      stk::mesh::field_data(radius_field, sphere)[0] = non_shared_radius;
    }
  } else {
    ASSERT_EQ(bulk_data.num_nodes(sphere), 1);
    stk::mesh::Entity node = bulk_data.begin_nodes(sphere)[0];
    ASSERT_TRUE(bulk_data.is_valid(node));
    stk::mesh::field_data(center_field, node)[0] = non_shared_center[0];
    stk::mesh::field_data(center_field, node)[1] = non_shared_center[1];
    stk::mesh::field_data(center_field, node)[2] = non_shared_center[2];
    if constexpr (!is_radius_shared) {
      stk::mesh::field_data(radius_field, sphere)[0] = non_shared_radius;
    }
  }

  // Test that the sphere data properly views the updated fields

  // Test that the data is modifiable
  // Add and then remove a constant value to the center and radius
  const double add_value = 1.1;
  if (sphere_rank == stk::topology::NODE_RANK) {
    auto sphere_view = create_sphere_entity_view(sphere_data, sphere);
    ASSERT_DOUBLE_EQ(sphere_view.center()[0], non_shared_center[0]);
    ASSERT_DOUBLE_EQ(sphere_view.center()[1], non_shared_center[1]);
    ASSERT_DOUBLE_EQ(sphere_view.center()[2], non_shared_center[2]);
    if constexpr (is_radius_shared) {
      ASSERT_DOUBLE_EQ(sphere_view.radius(), radius);
    } else {
      ASSERT_DOUBLE_EQ(sphere_view.radius(), non_shared_radius);
    }
    sphere_view.center()[0] += add_value;
    sphere_view.center()[1] -= add_value;
    sphere_view.center()[2] *= add_value;
    if (!~is_radius_shared) {
      // Shared values are not modifiable
      sphere_view.radius() += 2 * add_value;
    }

    ASSERT_DOUBLE_EQ(stk::mesh::field_data(center_field, sphere)[0], old_non_shared_center[0] + add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(center_field, sphere)[1], old_non_shared_center[1] - add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(center_field, sphere)[2], old_non_shared_center[2] * add_value);
    if constexpr (!is_radius_shared) {
      ASSERT_DOUBLE_EQ(stk::mesh::field_data(radius_field, sphere)[0], old_non_shared_radius + 2 * add_value);
    }

    // Remove the added value
    sphere_view.center()[0] -= add_value;
    sphere_view.center()[1] += add_value;
    sphere_view.center()[2] /= add_value;
    if (!~is_radius_shared) {
      sphere_view.radius() -= 2 * add_value;
    }
  } else {
    auto sphere_view = create_sphere_entity_view(sphere_data, sphere);
    ASSERT_DOUBLE_EQ(sphere_view.center()[0], non_shared_center[0]);
    ASSERT_DOUBLE_EQ(sphere_view.center()[1], non_shared_center[1]);
    ASSERT_DOUBLE_EQ(sphere_view.center()[2], non_shared_center[2]);
    if constexpr (is_radius_shared) {
      ASSERT_DOUBLE_EQ(sphere_view.radius(), radius);
    } else {
      ASSERT_DOUBLE_EQ(sphere_view.radius(), non_shared_radius);
    }
    sphere_view.center()[0] += add_value;
    sphere_view.center()[1] -= add_value;
    sphere_view.center()[2] *= add_value;
    if (!~is_radius_shared) {
      sphere_view.radius() += 2 * add_value;
    }

    ASSERT_EQ(bulk_data.num_nodes(sphere), 1);
    stk::mesh::Entity node = bulk_data.begin_nodes(sphere)[0];
    ASSERT_TRUE(bulk_data.is_valid(node));
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(center_field, node)[0], old_non_shared_center[0] + add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(center_field, node)[1], old_non_shared_center[1] - add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(center_field, node)[2], old_non_shared_center[2] * add_value);
    if constexpr (!is_radius_shared) {
      ASSERT_DOUBLE_EQ(stk::mesh::field_data(radius_field, sphere)[0], old_non_shared_radius + 2 * add_value);
    }

    // Remove the added value
    sphere_view.center()[0] -= add_value;
    sphere_view.center()[1] += add_value;
    sphere_view.center()[2] /= add_value;
    if (!~is_radius_shared) {
      sphere_view.radius() -= 2 * add_value;
    }
  }

  // Test that the NGP sphere data properly views the updated fields
  stk::mesh::FastMeshIndex sphere_index = ngp_mesh.fast_mesh_index(sphere);
  if (sphere_rank == stk::topology::NODE_RANK) {
    auto ngp_sphere_view = create_ngp_sphere_entity_view(ngp_sphere_data, sphere_index);
    ASSERT_DOUBLE_EQ(ngp_sphere_view.center()[0], non_shared_center[0]);
    ASSERT_DOUBLE_EQ(ngp_sphere_view.center()[1], non_shared_center[1]);
    ASSERT_DOUBLE_EQ(ngp_sphere_view.center()[2], non_shared_center[2]);
    if constexpr (is_radius_shared) {
      ASSERT_DOUBLE_EQ(ngp_sphere_view.radius(), radius);
    } else {
      ASSERT_DOUBLE_EQ(ngp_sphere_view.radius(), non_shared_radius);
    }

    // Test that the data is modifiable
    ngp_sphere_view.center()[0] += add_value;
    ngp_sphere_view.center()[1] -= add_value;
    ngp_sphere_view.center()[2] *= add_value;
    if (!~is_radius_shared) {
      ngp_sphere_view.radius() += 2 * add_value;
    }

    ASSERT_DOUBLE_EQ(ngp_center_field(sphere_index, 0), old_non_shared_center[0] + add_value);
    ASSERT_DOUBLE_EQ(ngp_center_field(sphere_index, 1), old_non_shared_center[1] - add_value);
    ASSERT_DOUBLE_EQ(ngp_center_field(sphere_index, 2), old_non_shared_center[2] * add_value);
    if constexpr (is_radius_shared) {
      ASSERT_DOUBLE_EQ(radius, old_radius + 2 * add_value);
    } else {
      ASSERT_DOUBLE_EQ(ngp_radius_field(sphere_index, 0), old_non_shared_radius + 2 * add_value);
    }
  } else {
    auto ngp_sphere_view = create_ngp_sphere_entity_view(ngp_sphere_data, sphere_index);
    ASSERT_DOUBLE_EQ(ngp_sphere_view.center()[0], non_shared_center[0]);
    ASSERT_DOUBLE_EQ(ngp_sphere_view.center()[1], non_shared_center[1]);
    ASSERT_DOUBLE_EQ(ngp_sphere_view.center()[2], non_shared_center[2]);
    if constexpr (!is_radius_shared) {
      ASSERT_DOUBLE_EQ(ngp_sphere_view.radius(), non_shared_radius);
    }

    // Test that the data is modifiable
    ngp_sphere_view.center()[0] += add_value;
    ngp_sphere_view.center()[1] -= add_value;
    ngp_sphere_view.center()[2] *= add_value;
    if (!~is_radius_shared) {
      ngp_sphere_view.radius() += 2 * add_value;
    }
    stk::mesh::Entity node = bulk_data.begin_nodes(sphere)[0];
    stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(node);

    ASSERT_EQ(ngp_center_field(node_index, 0), old_non_shared_center[0] + add_value);
    ASSERT_EQ(ngp_center_field(node_index, 1), old_non_shared_center[1] - add_value);
    ASSERT_EQ(ngp_center_field(node_index, 2), old_non_shared_center[2] * add_value);
    if constexpr (!is_radius_shared) {
      ASSERT_EQ(ngp_radius_field(sphere_index, 0), old_non_shared_radius + 2 * add_value);
    }
  }
}

template <stk::topology::topology_t OurTopology, bool is_axis_lengths_shared>
void test_ellipsoid_data(stk::mesh::BulkData& bulk_data,               //
                         stk::mesh::Entity ellipsoid,                  //
                         stk::mesh::Field<double>& center_field,       //
                         stk::mesh::Field<double>& orientation_field,  //
                         stk::mesh::Field<double>& axis_lengths_field) {
  stk::topology our_topology = OurTopology;
  stk::topology::rank_t ellipsoid_rank = our_topology.rank();
  ASSERT_TRUE(bulk_data.is_valid(ellipsoid));
  ASSERT_TRUE(bulk_data.bucket(ellipsoid).topology().rank() == ellipsoid_rank);
  ASSERT_TRUE(ellipsoid_rank == stk::topology::ELEMENT_RANK || ellipsoid_rank == stk::topology::NODE_RANK);

  // The shared data for the ellipsoid
  mundy::math::Quaternion<double> orientation{0.1, 0.2, 0.3,
                                              0.4};  // Not a valid unit quaternion but that's fine for this test.
  mundy::math::Vector3<double> axis_lengths{1.01, 2.02, 3.03};

  // Test the regular ellipsoid data to ensure that the stored shared data/fields are as expected
  auto ellipsoid_data = create_ellipsoid_data<double, OurTopology>(
      bulk_data, center_field, orientation_field,
      get_first_or_second<is_axis_lengths_shared>(axis_lengths, axis_lengths_field));
  ASSERT_EQ(&ellipsoid_data.center_data(), &center_field);
  ASSERT_EQ(&ellipsoid_data.orientation_data(), &orientation_field);
  if constexpr (is_axis_lengths_shared) {
    ASSERT_EQ(&ellipsoid_data.axis_lengths_data(), &axis_lengths);
  } else {
    ASSERT_EQ(&ellipsoid_data.axis_lengths_data(), &axis_lengths_field);
  }

  // Same test for the NGP ellipsoid data
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
  stk::mesh::NgpField<double>& ngp_center_field = stk::mesh::get_updated_ngp_field<double>(center_field);
  stk::mesh::NgpField<double>& ngp_orientation_field = stk::mesh::get_updated_ngp_field<double>(orientation_field);
  stk::mesh::NgpField<double>& ngp_axis_lengths_field = stk::mesh::get_updated_ngp_field<double>(axis_lengths_field);
  auto ngp_ellipsoid_data = create_ngp_ellipsoid_data<double, OurTopology>(
      ngp_mesh, ngp_center_field, ngp_orientation_field,
      get_first_or_second<is_axis_lengths_shared>(axis_lengths, ngp_axis_lengths_field));

  ASSERT_EQ(&ngp_ellipsoid_data.center_data(), &ngp_center_field);
  ASSERT_EQ(&ngp_ellipsoid_data.orientation_data(), &ngp_orientation_field);
  if constexpr (is_axis_lengths_shared) {
    ASSERT_EQ(&ngp_ellipsoid_data.axis_lengths_data(), &axis_lengths);
  } else {
    ASSERT_EQ(&ngp_ellipsoid_data.axis_lengths_data(), &ngp_axis_lengths_field);
  }

  // Set the center and radius data for the ellipsoid directly via their fields
  const Point<double> non_shared_center{7.7, 8.8, 9.9};
  const mundy::math::Quaternion<double> non_shared_orientation{0.5, 0.6, 0.7, 0.8};
  const mundy::math::Vector3<double> non_shared_axis_lengths{4.04, 5.05, 6.06};

  Point<double> old_non_shared_center = non_shared_center;
  mundy::math::Quaternion<double> old_orientation = orientation;
  mundy::math::Quaternion<double> old_non_shared_orientation = non_shared_orientation;
  mundy::math::Vector3<double> old_axis_lengths = axis_lengths;
  mundy::math::Vector3<double> old_non_shared_axis_lengths = non_shared_axis_lengths;

  if (ellipsoid_rank == stk::topology::NODE_RANK) {
    stk::mesh::field_data(center_field, ellipsoid)[0] = non_shared_center[0];
    stk::mesh::field_data(center_field, ellipsoid)[1] = non_shared_center[1];
    stk::mesh::field_data(center_field, ellipsoid)[2] = non_shared_center[2];
    stk::mesh::field_data(orientation_field, ellipsoid)[0] = non_shared_orientation[0];
    stk::mesh::field_data(orientation_field, ellipsoid)[1] = non_shared_orientation[1];
    stk::mesh::field_data(orientation_field, ellipsoid)[2] = non_shared_orientation[2];
    stk::mesh::field_data(orientation_field, ellipsoid)[3] = non_shared_orientation[3];
    if constexpr (!is_axis_lengths_shared) {
      stk::mesh::field_data(axis_lengths_field, ellipsoid)[0] = non_shared_axis_lengths[0];
      stk::mesh::field_data(axis_lengths_field, ellipsoid)[1] = non_shared_axis_lengths[1];
      stk::mesh::field_data(axis_lengths_field, ellipsoid)[2] = non_shared_axis_lengths[2];
    }
  } else {
    ASSERT_EQ(bulk_data.num_nodes(ellipsoid), 1);
    stk::mesh::Entity node = bulk_data.begin_nodes(ellipsoid)[0];
    ASSERT_TRUE(bulk_data.is_valid(node));
    stk::mesh::field_data(center_field, node)[0] = non_shared_center[0];
    stk::mesh::field_data(center_field, node)[1] = non_shared_center[1];
    stk::mesh::field_data(center_field, node)[2] = non_shared_center[2];
    stk::mesh::field_data(orientation_field, ellipsoid)[0] = non_shared_orientation[0];
    stk::mesh::field_data(orientation_field, ellipsoid)[1] = non_shared_orientation[1];
    stk::mesh::field_data(orientation_field, ellipsoid)[2] = non_shared_orientation[2];
    stk::mesh::field_data(orientation_field, ellipsoid)[3] = non_shared_orientation[3];
    if constexpr (!is_axis_lengths_shared) {
      stk::mesh::field_data(axis_lengths_field, ellipsoid)[0] = non_shared_axis_lengths[0];
      stk::mesh::field_data(axis_lengths_field, ellipsoid)[1] = non_shared_axis_lengths[1];
      stk::mesh::field_data(axis_lengths_field, ellipsoid)[2] = non_shared_axis_lengths[2];
    }
  }

  // Test that the ellipsoid data properly views the updated fields

  // Test that the data is modifiable
  const double add_value = 1.1;
  if (ellipsoid_rank == stk::topology::NODE_RANK) {
    auto ellipsoid_view = create_ellipsoid_entity_view(ellipsoid_data, ellipsoid);
    ASSERT_DOUBLE_EQ(ellipsoid_view.center()[0], non_shared_center[0]);
    ASSERT_DOUBLE_EQ(ellipsoid_view.center()[1], non_shared_center[1]);
    ASSERT_DOUBLE_EQ(ellipsoid_view.center()[2], non_shared_center[2]);
    ASSERT_DOUBLE_EQ(ellipsoid_view.orientation()[0], non_shared_orientation[0]);
    ASSERT_DOUBLE_EQ(ellipsoid_view.orientation()[1], non_shared_orientation[1]);
    ASSERT_DOUBLE_EQ(ellipsoid_view.orientation()[2], non_shared_orientation[2]);
    ASSERT_DOUBLE_EQ(ellipsoid_view.orientation()[3], non_shared_orientation[3]);
    if constexpr (!is_axis_lengths_shared) {
      ASSERT_DOUBLE_EQ(ellipsoid_view.axis_lengths()[0], non_shared_axis_lengths[0]);
      ASSERT_DOUBLE_EQ(ellipsoid_view.axis_lengths()[1], non_shared_axis_lengths[1]);
      ASSERT_DOUBLE_EQ(ellipsoid_view.axis_lengths()[2], non_shared_axis_lengths[2]);
    }
    ellipsoid_view.center()[0] += add_value;
    ellipsoid_view.center()[1] -= add_value;
    ellipsoid_view.center()[2] *= add_value;
    ellipsoid_view.orientation()[0] += add_value;
    ellipsoid_view.orientation()[1] -= add_value;
    ellipsoid_view.orientation()[2] *= add_value;
    ellipsoid_view.orientation()[3] += 2 * add_value;
    if constexpr (!is_axis_lengths_shared) {
      ellipsoid_view.axis_lengths()[0] += 3 * add_value;
      ellipsoid_view.axis_lengths()[1] -= 4 * add_value;
      ellipsoid_view.axis_lengths()[2] *= 5 * add_value;
    }

    ASSERT_DOUBLE_EQ(stk::mesh::field_data(center_field, ellipsoid)[0], old_non_shared_center[0] + add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(center_field, ellipsoid)[1], old_non_shared_center[1] - add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(center_field, ellipsoid)[2], old_non_shared_center[2] * add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(orientation_field, ellipsoid)[0], old_non_shared_orientation[0] + add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(orientation_field, ellipsoid)[1], old_non_shared_orientation[1] - add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(orientation_field, ellipsoid)[2], old_non_shared_orientation[2] * add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(orientation_field, ellipsoid)[3],
                     old_non_shared_orientation[3] + 2 * add_value);
    if constexpr (!is_axis_lengths_shared) {
      ASSERT_DOUBLE_EQ(stk::mesh::field_data(axis_lengths_field, ellipsoid)[0],
                       old_non_shared_axis_lengths[0] + 3 * add_value);
      ASSERT_DOUBLE_EQ(stk::mesh::field_data(axis_lengths_field, ellipsoid)[1],
                       old_non_shared_axis_lengths[1] - 4 * add_value);
      ASSERT_DOUBLE_EQ(stk::mesh::field_data(axis_lengths_field, ellipsoid)[2],
                       old_non_shared_axis_lengths[2] * 5 * add_value);
    }

    // Remove the added value
    ellipsoid_view.center()[0] -= add_value;
    ellipsoid_view.center()[1] += add_value;
    ellipsoid_view.center()[2] /= add_value;
    ellipsoid_view.orientation()[0] -= add_value;
    ellipsoid_view.orientation()[1] += add_value;
    ellipsoid_view.orientation()[2] /= add_value;
    ellipsoid_view.orientation()[3] -= 2 * add_value;
    if constexpr (!is_axis_lengths_shared) {
      ellipsoid_view.axis_lengths()[0] -= 3 * add_value;
      ellipsoid_view.axis_lengths()[1] += 4 * add_value;
      ellipsoid_view.axis_lengths()[2] /= 5 * add_value;
    }
  } else {
    auto ellipsoid_view = create_ellipsoid_entity_view(ellipsoid_data, ellipsoid);
    ASSERT_DOUBLE_EQ(ellipsoid_view.center()[0], non_shared_center[0]);
    ASSERT_DOUBLE_EQ(ellipsoid_view.center()[1], non_shared_center[1]);
    ASSERT_DOUBLE_EQ(ellipsoid_view.center()[2], non_shared_center[2]);
    ASSERT_DOUBLE_EQ(ellipsoid_view.orientation()[0], non_shared_orientation[0]);
    ASSERT_DOUBLE_EQ(ellipsoid_view.orientation()[1], non_shared_orientation[1]);
    ASSERT_DOUBLE_EQ(ellipsoid_view.orientation()[2], non_shared_orientation[2]);
    ASSERT_DOUBLE_EQ(ellipsoid_view.orientation()[3], non_shared_orientation[3]);
    if constexpr (!is_axis_lengths_shared) {
      ASSERT_DOUBLE_EQ(ellipsoid_view.axis_lengths()[0], non_shared_axis_lengths[0]);
      ASSERT_DOUBLE_EQ(ellipsoid_view.axis_lengths()[1], non_shared_axis_lengths[1]);
      ASSERT_DOUBLE_EQ(ellipsoid_view.axis_lengths()[2], non_shared_axis_lengths[2]);
    }
    ellipsoid_view.center()[0] += add_value;
    ellipsoid_view.center()[1] -= add_value;
    ellipsoid_view.center()[2] *= add_value;
    ellipsoid_view.orientation()[0] += add_value;
    ellipsoid_view.orientation()[1] -= add_value;
    ellipsoid_view.orientation()[2] *= add_value;
    ellipsoid_view.orientation()[3] += 2 * add_value;
    if constexpr (!is_axis_lengths_shared) {
      ellipsoid_view.axis_lengths()[0] += 3 * add_value;
      ellipsoid_view.axis_lengths()[1] -= 4 * add_value;
      ellipsoid_view.axis_lengths()[2] *= 5 * add_value;
    }

    ASSERT_EQ(bulk_data.num_nodes(ellipsoid), 1);
    stk::mesh::Entity node = bulk_data.begin_nodes(ellipsoid)[0];
    ASSERT_TRUE(bulk_data.is_valid(node));
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(center_field, node)[0], old_non_shared_center[0] + add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(center_field, node)[1], old_non_shared_center[1] - add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(center_field, node)[2], old_non_shared_center[2] * add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(orientation_field, ellipsoid)[0], old_non_shared_orientation[0] + add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(orientation_field, ellipsoid)[1], old_non_shared_orientation[1] - add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(orientation_field, ellipsoid)[2], old_non_shared_orientation[2] * add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(orientation_field, ellipsoid)[3],
                     old_non_shared_orientation[3] + 2 * add_value);
    if constexpr (!is_axis_lengths_shared) {
      ASSERT_DOUBLE_EQ(stk::mesh::field_data(axis_lengths_field, ellipsoid)[0],
                       old_non_shared_axis_lengths[0] + 3 * add_value);
      ASSERT_DOUBLE_EQ(stk::mesh::field_data(axis_lengths_field, ellipsoid)[1],
                       old_non_shared_axis_lengths[1] - 4 * add_value);
      ASSERT_DOUBLE_EQ(stk::mesh::field_data(axis_lengths_field, ellipsoid)[2],
                       old_non_shared_axis_lengths[2] * 5 * add_value);
    }

    // Remove the added value
    ellipsoid_view.center()[0] -= add_value;
    ellipsoid_view.center()[1] += add_value;
    ellipsoid_view.center()[2] /= add_value;
    ellipsoid_view.orientation()[0] -= add_value;
    ellipsoid_view.orientation()[1] += add_value;
    ellipsoid_view.orientation()[2] /= add_value;
    ellipsoid_view.orientation()[3] -= 2 * add_value;
    if constexpr (!is_axis_lengths_shared) {
      ellipsoid_view.axis_lengths()[0] -= 3 * add_value;
      ellipsoid_view.axis_lengths()[1] += 4 * add_value;
      ellipsoid_view.axis_lengths()[2] /= 5 * add_value;
    }
  }

  // Test that the NGP ellipsoid data properly views the updated fields
  stk::mesh::FastMeshIndex ellipsoid_index = ngp_mesh.fast_mesh_index(ellipsoid);
  if (ellipsoid_rank == stk::topology::NODE_RANK) {
    auto ngp_ellipsoid_view = create_ngp_ellipsoid_entity_view(ngp_ellipsoid_data, ellipsoid_index);
    ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.center()[0], non_shared_center[0]);
    ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.center()[1], non_shared_center[1]);
    ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.center()[2], non_shared_center[2]);
    ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.orientation()[0], non_shared_orientation[0]);
    ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.orientation()[1], non_shared_orientation[1]);
    ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.orientation()[2], non_shared_orientation[2]);
    ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.orientation()[3], non_shared_orientation[3]);
    if constexpr (!is_axis_lengths_shared) {
      ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.axis_lengths()[0], non_shared_axis_lengths[0]);
      ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.axis_lengths()[1], non_shared_axis_lengths[1]);
      ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.axis_lengths()[2], non_shared_axis_lengths[2]);
    }

    // Test that the data is modifiable
    // Add a constant value to the center and radius
    ngp_ellipsoid_view.center()[0] += add_value;
    ngp_ellipsoid_view.center()[1] -= add_value;
    ngp_ellipsoid_view.center()[2] *= add_value;
    ngp_ellipsoid_view.orientation()[0] += add_value;
    ngp_ellipsoid_view.orientation()[1] -= add_value;
    ngp_ellipsoid_view.orientation()[2] *= add_value;
    ngp_ellipsoid_view.orientation()[3] += 2 * add_value;
    if constexpr (!is_axis_lengths_shared) {
      ngp_ellipsoid_view.axis_lengths()[0] += 3 * add_value;
      ngp_ellipsoid_view.axis_lengths()[1] -= 4 * add_value;
      ngp_ellipsoid_view.axis_lengths()[2] *= 5 * add_value;
    }
    ASSERT_DOUBLE_EQ(ngp_center_field(ellipsoid_index, 0), old_non_shared_center[0] + add_value);
    ASSERT_DOUBLE_EQ(ngp_center_field(ellipsoid_index, 1), old_non_shared_center[1] - add_value);
    ASSERT_DOUBLE_EQ(ngp_center_field(ellipsoid_index, 2), old_non_shared_center[2] * add_value);
    ASSERT_DOUBLE_EQ(ngp_orientation_field(ellipsoid_index, 0), old_non_shared_orientation[0] + add_value);
    ASSERT_DOUBLE_EQ(ngp_orientation_field(ellipsoid_index, 1), old_non_shared_orientation[1] - add_value);
    ASSERT_DOUBLE_EQ(ngp_orientation_field(ellipsoid_index, 2), old_non_shared_orientation[2] * add_value);
    ASSERT_DOUBLE_EQ(ngp_orientation_field(ellipsoid_index, 3), old_non_shared_orientation[3] + 2 * add_value);
    if constexpr (is_axis_lengths_shared) {
      ASSERT_DOUBLE_EQ(axis_lengths[0], old_axis_lengths[0] + 3 * add_value);
      ASSERT_DOUBLE_EQ(axis_lengths[1], old_axis_lengths[1] - 4 * add_value);
      ASSERT_DOUBLE_EQ(axis_lengths[2], old_axis_lengths[2] * 5 * add_value);
    } else {
      ASSERT_DOUBLE_EQ(ngp_axis_lengths_field(ellipsoid_index, 0), old_non_shared_axis_lengths[0] + 3 * add_value);
      ASSERT_DOUBLE_EQ(ngp_axis_lengths_field(ellipsoid_index, 1), old_non_shared_axis_lengths[1] - 4 * add_value);
      ASSERT_DOUBLE_EQ(ngp_axis_lengths_field(ellipsoid_index, 2), old_non_shared_axis_lengths[2] * 5 * add_value);
    }
  } else {
    auto ngp_ellipsoid_view = create_ngp_ellipsoid_entity_view(ngp_ellipsoid_data, ellipsoid_index);
    ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.center()[0], non_shared_center[0]);
    ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.center()[1], non_shared_center[1]);
    ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.center()[2], non_shared_center[2]);
    ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.orientation()[0], non_shared_orientation[0]);
    ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.orientation()[1], non_shared_orientation[1]);
    ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.orientation()[2], non_shared_orientation[2]);
    ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.orientation()[3], non_shared_orientation[3]);
    if constexpr (!is_axis_lengths_shared) {
      ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.axis_lengths()[0], non_shared_axis_lengths[0]);
      ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.axis_lengths()[1], non_shared_axis_lengths[1]);
      ASSERT_DOUBLE_EQ(ngp_ellipsoid_view.axis_lengths()[2], non_shared_axis_lengths[2]);
    }

    // Test that the data is modifiable
    // Add a constant value to the center and radius
    ngp_ellipsoid_view.center()[0] += add_value;
    ngp_ellipsoid_view.center()[1] -= add_value;
    ngp_ellipsoid_view.center()[2] *= add_value;
    ngp_ellipsoid_view.orientation()[0] += add_value;
    ngp_ellipsoid_view.orientation()[1] -= add_value;
    ngp_ellipsoid_view.orientation()[2] *= add_value;
    ngp_ellipsoid_view.orientation()[3] += 2 * add_value;
    if constexpr (!is_axis_lengths_shared) {
      ngp_ellipsoid_view.axis_lengths()[0] += 3 * add_value;
      ngp_ellipsoid_view.axis_lengths()[1] -= 4 * add_value;
      ngp_ellipsoid_view.axis_lengths()[2] *= 5 * add_value;
    }
    stk::mesh::Entity node = bulk_data.begin_nodes(ellipsoid)[0];
    stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(node);

    ASSERT_EQ(ngp_center_field(node_index, 0), old_non_shared_center[0] + add_value);
    ASSERT_EQ(ngp_center_field(node_index, 1), old_non_shared_center[1] - add_value);
    ASSERT_EQ(ngp_center_field(node_index, 2), old_non_shared_center[2] * add_value);
    ASSERT_EQ(ngp_orientation_field(ellipsoid_index, 0), old_non_shared_orientation[0] + add_value);
    ASSERT_EQ(ngp_orientation_field(ellipsoid_index, 1), old_non_shared_orientation[1] - add_value);
    ASSERT_EQ(ngp_orientation_field(ellipsoid_index, 2), old_non_shared_orientation[2] * add_value);
    ASSERT_EQ(ngp_orientation_field(ellipsoid_index, 3), old_non_shared_orientation[3] + 2 * add_value);
    if constexpr (!is_axis_lengths_shared) {
      ASSERT_EQ(ngp_axis_lengths_field(ellipsoid_index, 0), old_non_shared_axis_lengths[0] + 3 * add_value);
      ASSERT_EQ(ngp_axis_lengths_field(ellipsoid_index, 1), old_non_shared_axis_lengths[1] - 4 * add_value);
      ASSERT_EQ(ngp_axis_lengths_field(ellipsoid_index, 2), old_non_shared_axis_lengths[2] * 5 * add_value);
    }
  }
}

template <stk::topology::topology_t OurTopology>
void test_point_data(stk::mesh::BulkData& bulk_data,  //
                     stk::mesh::Entity point_entity,  //
                     stk::mesh::Field<double>& center_field) {
  stk::topology our_topology = OurTopology;
  stk::topology::rank_t point_rank = our_topology.rank();
  ASSERT_TRUE(bulk_data.is_valid(point_entity));
  ASSERT_TRUE(point_rank == stk::topology::NODE_RANK || point_rank == stk::topology::ELEMENT_RANK);

  // The shared data for the point
  Point<double> center{1.1, 2.2, 3.3};

  // Test the regular point data to ensure that the stored shared data/fields are as expected
  auto point_data = create_point_data<double, OurTopology>(bulk_data, center_field);
  ASSERT_EQ(&point_data.node_coords_data(), &center_field);

  // Same test for the NGP point data
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
  stk::mesh::NgpField<double>& ngp_center_field = stk::mesh::get_updated_ngp_field<double>(center_field);
  auto ngp_point_data = create_ngp_point_data<double, OurTopology>(bulk_data, ngp_center_field);

  ASSERT_EQ(&ngp_point_data.node_coords_data(), &ngp_center_field);

  // Set the center and radius data for the point directly via their fields
  const Point<double> non_shared_center{0.1, 0.2, 0.3};

  Point<double> old_center = center;
  Point<double> old_non_shared_center = non_shared_center;
  if (point_rank == stk::topology::NODE_RANK) {
    stk::mesh::field_data(center_field, point_entity)[0] = non_shared_center[0];
    stk::mesh::field_data(center_field, point_entity)[1] = non_shared_center[1];
    stk::mesh::field_data(center_field, point_entity)[2] = non_shared_center[2];
  } else {
    ASSERT_EQ(bulk_data.num_nodes(point_entity), 1u);
    stk::mesh::Entity node = bulk_data.begin_nodes(point_entity)[0];
    stk::mesh::field_data(center_field, node)[0] = non_shared_center[0];
    stk::mesh::field_data(center_field, node)[1] = non_shared_center[1];
    stk::mesh::field_data(center_field, node)[2] = non_shared_center[2];
  }

  // Test that the point data properly views the updated fields

  const double add_value = 1.1;
  if (point_rank == stk::topology::NODE_RANK) {
    // Test that the data is modifiable
    auto point_view = create_point_entity_view(point_data, point_entity);
    ASSERT_NEAR(point_view[0], non_shared_center[0], 1e-12);
    ASSERT_NEAR(point_view[1], non_shared_center[1], 1e-12);
    ASSERT_NEAR(point_view[2], non_shared_center[2], 1e-12);

    point_view[0] += add_value;
    point_view[1] -= add_value;
    point_view[2] *= add_value;

    ASSERT_NEAR(stk::mesh::field_data(center_field, point_entity)[0], old_non_shared_center[0] + add_value, 1e-12);
    ASSERT_NEAR(stk::mesh::field_data(center_field, point_entity)[1], old_non_shared_center[1] - add_value, 1e-12);
    ASSERT_NEAR(stk::mesh::field_data(center_field, point_entity)[2], old_non_shared_center[2] * add_value, 1e-12);

    // Remove the added value
    point_view[0] -= add_value;
    point_view[1] += add_value;
    point_view[2] /= add_value;
  } else {
    // Test that the data is modifiable
    auto point_view = create_point_entity_view(point_data, point_entity);
    ASSERT_NEAR(point_view[0], non_shared_center[0], 1e-12);
    ASSERT_NEAR(point_view[1], non_shared_center[1], 1e-12);
    ASSERT_NEAR(point_view[2], non_shared_center[2], 1e-12);

    point_view[0] += add_value;
    point_view[1] -= add_value;
    point_view[2] *= add_value;

    ASSERT_EQ(bulk_data.num_nodes(point_entity), 1u);
    stk::mesh::Entity node = bulk_data.begin_nodes(point_entity)[0];
    ASSERT_NEAR(stk::mesh::field_data(center_field, node)[0], old_non_shared_center[0] + add_value, 1e-12);
    ASSERT_NEAR(stk::mesh::field_data(center_field, node)[1], old_non_shared_center[1] - add_value, 1e-12);
    ASSERT_NEAR(stk::mesh::field_data(center_field, node)[2], old_non_shared_center[2] * add_value, 1e-12);

    // Remove the added value
    point_view[0] -= add_value;
    point_view[1] += add_value;
    point_view[2] /= add_value;
  }

  if (point_rank == stk::topology::NODE_RANK) {
    // Test that the NGP point data properly views the updated fields
    stk::mesh::FastMeshIndex point_index = ngp_mesh.fast_mesh_index(point_entity);
    auto ngp_point_view = create_ngp_point_entity_view(ngp_point_data, point_index);
    ASSERT_NEAR(ngp_point_view[0], non_shared_center[0], 1e-12);
    ASSERT_NEAR(ngp_point_view[1], non_shared_center[1], 1e-12);
    ASSERT_NEAR(ngp_point_view[2], non_shared_center[2], 1e-12);

    // Test that the data is modifiable
    // Add a constant value to the center and radius
    ngp_point_view[0] += add_value;
    ngp_point_view[1] -= add_value;
    ngp_point_view[2] *= add_value;
    ASSERT_NEAR(ngp_center_field(point_index, 0), old_non_shared_center[0] + add_value, 1e-12);
    ASSERT_NEAR(ngp_center_field(point_index, 1), old_non_shared_center[1] - add_value, 1e-12);
    ASSERT_NEAR(ngp_center_field(point_index, 2), old_non_shared_center[2] * add_value, 1e-12);
  } else {
    // Test that the NGP point data properly views the updated fields
    stk::mesh::FastMeshIndex point_index = ngp_mesh.fast_mesh_index(point_entity);
    stk::mesh::Entity node = bulk_data.begin_nodes(point_entity)[0];
    stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(node);
    auto ngp_point_view = create_ngp_point_entity_view(ngp_point_data, point_index);
    ASSERT_NEAR(ngp_point_view[0], non_shared_center[0], 1e-12);
    ASSERT_NEAR(ngp_point_view[1], non_shared_center[1], 1e-12);
    ASSERT_NEAR(ngp_point_view[2], non_shared_center[2], 1e-12);

    // Test that the data is modifiable
    // Add a constant value to the center and radius
    ngp_point_view[0] += add_value;
    ngp_point_view[1] -= add_value;
    ngp_point_view[2] *= add_value;
    ASSERT_NEAR(ngp_center_field(node_index, 0), old_non_shared_center[0] + add_value, 1e-12);
    ASSERT_NEAR(ngp_center_field(node_index, 1), old_non_shared_center[1] - add_value, 1e-12);
    ASSERT_NEAR(ngp_center_field(node_index, 2), old_non_shared_center[2] * add_value, 1e-12);
  }
}

template <stk::topology::topology_t OurTopology>
void test_line_data(stk::mesh::BulkData& bulk_data,          //
                    stk::mesh::Entity line,                  //
                    stk::mesh::Field<double>& center_field,  //
                    stk::mesh::Field<double>& direction_field) {
  stk::topology our_topology = OurTopology;
  stk::topology::rank_t line_rank = our_topology.rank();
  ASSERT_TRUE(bulk_data.is_valid(line));
  ASSERT_TRUE(bulk_data.bucket(line).topology().rank() == line_rank);
  ASSERT_TRUE(line_rank == stk::topology::ELEMENT_RANK || line_rank == stk::topology::NODE_RANK);

  // The shared data for the line
  Point<double> direction{44.4, 55.5, 66.6};  // not a valid unit vector but that's okay for this test
  Point<double> center{1.1, 2.2, 3.3};

  // Test the regular line data to ensure that the stored shared data/fields are as expected
  auto line_data = create_line_data<double, OurTopology>(bulk_data, center_field, direction_field);
  ASSERT_EQ(&line_data.center_data(), &center_field);
  ASSERT_EQ(&line_data.direction_data(), &direction_field);

  // Same test for the NGP line data
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
  stk::mesh::NgpField<double>& ngp_center_field = stk::mesh::get_updated_ngp_field<double>(center_field);
  stk::mesh::NgpField<double>& ngp_direction_field = stk::mesh::get_updated_ngp_field<double>(direction_field);
  auto ngp_line_data = create_ngp_line_data<double, OurTopology>(ngp_mesh, ngp_center_field, ngp_direction_field);

  ASSERT_EQ(&ngp_line_data.center_data(), &ngp_center_field);
  ASSERT_EQ(&ngp_line_data.direction_data(), &ngp_direction_field);

  // Set the center and direction data for the line directly via their fields
  const Point<double> non_shared_direction{1.2, 3.4, 5.6};
  const Point<double> non_shared_center{7.7, 8.8, 9.9};

  Point<double> old_center = center;
  Point<double> old_non_shared_center = non_shared_center;
  Point<double> old_direction = direction;
  Point<double> old_non_shared_direction = non_shared_direction;
  if (line_rank == stk::topology::NODE_RANK) {
    stk::mesh::field_data(center_field, line)[0] = non_shared_center[0];
    stk::mesh::field_data(center_field, line)[1] = non_shared_center[1];
    stk::mesh::field_data(center_field, line)[2] = non_shared_center[2];
    stk::mesh::field_data(direction_field, line)[0] = non_shared_direction[0];
    stk::mesh::field_data(direction_field, line)[1] = non_shared_direction[1];
    stk::mesh::field_data(direction_field, line)[2] = non_shared_direction[2];
  } else {
    ASSERT_EQ(bulk_data.num_nodes(line), 1);
    stk::mesh::Entity node = bulk_data.begin_nodes(line)[0];
    ASSERT_TRUE(bulk_data.is_valid(node));
    stk::mesh::field_data(center_field, node)[0] = non_shared_center[0];
    stk::mesh::field_data(center_field, node)[1] = non_shared_center[1];
    stk::mesh::field_data(center_field, node)[2] = non_shared_center[2];
    stk::mesh::field_data(direction_field, line)[0] = non_shared_direction[0];
    stk::mesh::field_data(direction_field, line)[1] = non_shared_direction[1];
    stk::mesh::field_data(direction_field, line)[2] = non_shared_direction[2];
  }

  // Test that the line data properly views the updated fields

  // Test that the data is modifiable
  // Add and then remove a constant value to the center and direction
  const double add_value = 1.1;
  if (line_rank == stk::topology::NODE_RANK) {
    auto line_view = create_line_entity_view(line_data, line);
    ASSERT_NEAR(line_view.center()[0], non_shared_center[0], 1e-12);
    ASSERT_NEAR(line_view.center()[1], non_shared_center[1], 1e-12);
    ASSERT_NEAR(line_view.center()[2], non_shared_center[2], 1e-12);
    ASSERT_NEAR(line_view.direction()[0], non_shared_direction[0], 1e-12);
    ASSERT_NEAR(line_view.direction()[1], non_shared_direction[1], 1e-12);
    ASSERT_NEAR(line_view.direction()[2], non_shared_direction[2], 1e-12);
    line_view.center()[0] += add_value;
    line_view.center()[1] -= add_value;
    line_view.center()[2] *= add_value;
    line_view.direction()[0] += 2 * add_value;
    line_view.direction()[1] -= 2 * add_value;
    line_view.direction()[2] *= 2 * add_value;

    ASSERT_NEAR(stk::mesh::field_data(center_field, line)[0], old_non_shared_center[0] + add_value, 1e-12);
    ASSERT_NEAR(stk::mesh::field_data(center_field, line)[1], old_non_shared_center[1] - add_value, 1e-12);
    ASSERT_NEAR(stk::mesh::field_data(center_field, line)[2], old_non_shared_center[2] * add_value, 1e-12);
    ASSERT_NEAR(stk::mesh::field_data(direction_field, line)[0], old_non_shared_direction[0] + 2 * add_value, 1e-12);
    ASSERT_NEAR(stk::mesh::field_data(direction_field, line)[1], old_non_shared_direction[1] - 2 * add_value, 1e-12);
    ASSERT_NEAR(stk::mesh::field_data(direction_field, line)[2], old_non_shared_direction[2] * 2 * add_value, 1e-12);

    // Remove the added value
    line_view.center()[0] -= add_value;
    line_view.center()[1] += add_value;
    line_view.center()[2] /= add_value;
    line_view.direction()[0] -= 2 * add_value;
    line_view.direction()[1] += 2 * add_value;
    line_view.direction()[2] /= 2 * add_value;
  } else {
    auto line_view = create_line_entity_view(line_data, line);
    ASSERT_NEAR(line_view.center()[0], non_shared_center[0], 1e-12);
    ASSERT_NEAR(line_view.center()[1], non_shared_center[1], 1e-12);
    ASSERT_NEAR(line_view.center()[2], non_shared_center[2], 1e-12);
    ASSERT_NEAR(line_view.direction()[0], non_shared_direction[0], 1e-12);
    ASSERT_NEAR(line_view.direction()[1], non_shared_direction[1], 1e-12);
    ASSERT_NEAR(line_view.direction()[2], non_shared_direction[2], 1e-12);

    line_view.center()[0] += add_value;
    line_view.center()[1] -= add_value;
    line_view.center()[2] *= add_value;
    line_view.direction()[0] += 2 * add_value;
    line_view.direction()[1] -= 2 * add_value;
    line_view.direction()[2] *= 2 * add_value;

    ASSERT_EQ(bulk_data.num_nodes(line), 1);
    stk::mesh::Entity node = bulk_data.begin_nodes(line)[0];
    ASSERT_TRUE(bulk_data.is_valid(node));
    ASSERT_NEAR(stk::mesh::field_data(center_field, node)[0], old_non_shared_center[0] + add_value, 1e-12);
    ASSERT_NEAR(stk::mesh::field_data(center_field, node)[1], old_non_shared_center[1] - add_value, 1e-12);
    ASSERT_NEAR(stk::mesh::field_data(center_field, node)[2], old_non_shared_center[2] * add_value, 1e-12);
    ASSERT_NEAR(stk::mesh::field_data(direction_field, line)[0], old_non_shared_direction[0] + 2 * add_value, 1e-12);
    ASSERT_NEAR(stk::mesh::field_data(direction_field, line)[1], old_non_shared_direction[1] - 2 * add_value, 1e-12);
    ASSERT_NEAR(stk::mesh::field_data(direction_field, line)[2], old_non_shared_direction[2] * 2 * add_value, 1e-12);

    // Remove the added value
    line_view.center()[0] -= add_value;
    line_view.center()[1] += add_value;
    line_view.center()[2] /= add_value;
    line_view.direction()[0] -= 2 * add_value;
    line_view.direction()[1] += 2 * add_value;
    line_view.direction()[2] /= 2 * add_value;
  }

  // Test that the NGP line data properly views the updated fields
  stk::mesh::FastMeshIndex line_index = ngp_mesh.fast_mesh_index(line);
  if (line_rank == stk::topology::NODE_RANK) {
    auto ngp_line_view = create_ngp_line_entity_view(ngp_line_data, line_index);
    ASSERT_NEAR(ngp_line_view.center()[0], non_shared_center[0], 1e-12);
    ASSERT_NEAR(ngp_line_view.center()[1], non_shared_center[1], 1e-12);
    ASSERT_NEAR(ngp_line_view.center()[2], non_shared_center[2], 1e-12);
    ASSERT_NEAR(ngp_line_view.direction()[0], non_shared_direction[0], 1e-12);
    ASSERT_NEAR(ngp_line_view.direction()[1], non_shared_direction[1], 1e-12);
    ASSERT_NEAR(ngp_line_view.direction()[2], non_shared_direction[2], 1e-12);

    // Test that the data is modifiable
    ngp_line_view.center()[0] += add_value;
    ngp_line_view.center()[1] -= add_value;
    ngp_line_view.center()[2] *= add_value;
    ngp_line_view.direction()[0] += 2 * add_value;
    ngp_line_view.direction()[1] -= 2 * add_value;
    ngp_line_view.direction()[2] *= 2 * add_value;
    ASSERT_NEAR(ngp_center_field(line_index, 0), old_non_shared_center[0] + add_value, 1e-12);
    ASSERT_NEAR(ngp_center_field(line_index, 1), old_non_shared_center[1] - add_value, 1e-12);
    ASSERT_NEAR(ngp_center_field(line_index, 2), old_non_shared_center[2] * add_value, 1e-12);
    ASSERT_NEAR(ngp_direction_field(line_index, 0), old_non_shared_direction[0] + 2 * add_value, 1e-12);
    ASSERT_NEAR(ngp_direction_field(line_index, 1), old_non_shared_direction[1] - 2 * add_value, 1e-12);
    ASSERT_NEAR(ngp_direction_field(line_index, 2), old_non_shared_direction[2] * 2 * add_value, 1e-12);
  } else {
    auto ngp_line_view = create_ngp_line_entity_view(ngp_line_data, line_index);
    ASSERT_NEAR(ngp_line_view.center()[0], non_shared_center[0], 1e-12);
    ASSERT_NEAR(ngp_line_view.center()[1], non_shared_center[1], 1e-12);
    ASSERT_NEAR(ngp_line_view.center()[2], non_shared_center[2], 1e-12);
    ASSERT_NEAR(ngp_line_view.direction()[0], non_shared_direction[0], 1e-12);
    ASSERT_NEAR(ngp_line_view.direction()[1], non_shared_direction[1], 1e-12);
    ASSERT_NEAR(ngp_line_view.direction()[2], non_shared_direction[2], 1e-12);

    // Test that the data is modifiable
    ngp_line_view.center()[0] += add_value;
    ngp_line_view.center()[1] -= add_value;
    ngp_line_view.center()[2] *= add_value;
    ngp_line_view.direction()[0] += 2 * add_value;
    ngp_line_view.direction()[1] -= 2 * add_value;
    ngp_line_view.direction()[2] *= 2 * add_value;
    stk::mesh::Entity node = bulk_data.begin_nodes(line)[0];
    stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(node);
    ASSERT_NEAR(ngp_center_field(node_index, 0), old_non_shared_center[0] + add_value, 1e-12);
    ASSERT_NEAR(ngp_center_field(node_index, 1), old_non_shared_center[1] - add_value, 1e-12);
    ASSERT_NEAR(ngp_center_field(node_index, 2), old_non_shared_center[2] * add_value, 1e-12);
    ASSERT_NEAR(ngp_direction_field(line_index, 0), old_non_shared_direction[0] + 2 * add_value, 1e-12);
    ASSERT_NEAR(ngp_direction_field(line_index, 1), old_non_shared_direction[1] - 2 * add_value, 1e-12);
    ASSERT_NEAR(ngp_direction_field(line_index, 2), old_non_shared_direction[2] * 2 * add_value, 1e-12);
  }
}

template <stk::topology::topology_t OurTopology>
void test_line_segment_data(stk::mesh::BulkData& bulk_data,  //
                            stk::mesh::Entity line_segment,  //
                            stk::mesh::Field<double>& node_coords_field) {
  stk::topology our_topology = OurTopology;
  stk::topology::rank_t line_segment_rank = our_topology.rank();
  ASSERT_TRUE(bulk_data.is_valid(line_segment));
  ASSERT_TRUE(bulk_data.bucket(line_segment).topology().rank() == line_segment_rank);
  ASSERT_TRUE(line_segment_rank == stk::topology::ELEM_RANK) << "For now, we only support element rank line segments.";

  ASSERT_EQ(bulk_data.num_nodes(line_segment), 2);
  stk::mesh::Entity start_node = bulk_data.begin_nodes(line_segment)[0];
  stk::mesh::Entity end_node = bulk_data.begin_nodes(line_segment)[1];
  ASSERT_TRUE(bulk_data.is_valid(start_node));
  ASSERT_TRUE(bulk_data.is_valid(end_node));

  // Test the regular line_segment data to ensure that the stored shared data/fields are as expected
  auto line_segment_data = create_line_segment_data<double, OurTopology>(bulk_data, node_coords_field);
  ASSERT_EQ(&line_segment_data.node_coords_data(), &node_coords_field);

  // Same test for the NGP line_segment data
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
  stk::mesh::NgpField<double>& ngp_node_coords_field = stk::mesh::get_updated_ngp_field<double>(node_coords_field);
  auto ngp_line_segment_data = create_ngp_line_segment_data<double, OurTopology>(ngp_mesh, ngp_node_coords_field);
  ASSERT_EQ(&ngp_line_segment_data.node_coords_data(), &ngp_node_coords_field);

  // Set the node_coords for the line_segment directly via their fields
  const Point<double> non_shared_start_node_coords{7.7, 8.8, 9.9};
  const Point<double> non_shared_end_node_coords{1.1, 2.2, 3.3};

  Point<double> old_non_shared_start_node_coords = non_shared_start_node_coords;
  Point<double> old_non_shared_end_node_coords = non_shared_end_node_coords;
  stk::mesh::field_data(node_coords_field, start_node)[0] = non_shared_start_node_coords[0];
  stk::mesh::field_data(node_coords_field, start_node)[1] = non_shared_start_node_coords[1];
  stk::mesh::field_data(node_coords_field, start_node)[2] = non_shared_start_node_coords[2];
  stk::mesh::field_data(node_coords_field, end_node)[0] = non_shared_end_node_coords[0];
  stk::mesh::field_data(node_coords_field, end_node)[1] = non_shared_end_node_coords[1];
  stk::mesh::field_data(node_coords_field, end_node)[2] = non_shared_end_node_coords[2];

  // Test that the line_segment data properly views the updated fields

  // Test that the data is modifiable
  // Add and then remove a constant value to the node_coords
  const double add_value = 1.1;
  auto line_segment_view = create_line_segment_entity_view(line_segment_data, line_segment);
  ASSERT_NEAR(line_segment_view.start()[0], non_shared_start_node_coords[0], 1e-12);
  ASSERT_NEAR(line_segment_view.start()[1], non_shared_start_node_coords[1], 1e-12);
  ASSERT_NEAR(line_segment_view.start()[2], non_shared_start_node_coords[2], 1e-12);
  ASSERT_NEAR(line_segment_view.end()[0], non_shared_end_node_coords[0], 1e-12);
  ASSERT_NEAR(line_segment_view.end()[1], non_shared_end_node_coords[1], 1e-12);
  ASSERT_NEAR(line_segment_view.end()[2], non_shared_end_node_coords[2], 1e-12);
  line_segment_view.start()[0] += add_value;
  line_segment_view.start()[1] -= add_value;
  line_segment_view.start()[2] *= add_value;
  line_segment_view.end()[0] += 2 * add_value;
  line_segment_view.end()[1] -= 2 * add_value;
  line_segment_view.end()[2] *= 2 * add_value;

  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, start_node)[0], old_non_shared_start_node_coords[0] + add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, start_node)[1], old_non_shared_start_node_coords[1] - add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, start_node)[2], old_non_shared_start_node_coords[2] * add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, end_node)[0], old_non_shared_end_node_coords[0] + 2 * add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, end_node)[1], old_non_shared_end_node_coords[1] - 2 * add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, end_node)[2], old_non_shared_end_node_coords[2] * 2 * add_value,
              1e-12);

  // Remove the added value
  line_segment_view.start()[0] -= add_value;
  line_segment_view.start()[1] += add_value;
  line_segment_view.start()[2] /= add_value;
  line_segment_view.end()[0] -= 2 * add_value;
  line_segment_view.end()[1] += 2 * add_value;
  line_segment_view.end()[2] /= 2 * add_value;

  // Test that the NGP line_segment data properly views the updated fields
  stk::mesh::FastMeshIndex line_segment_index = ngp_mesh.fast_mesh_index(line_segment);
  stk::mesh::FastMeshIndex start_node_index = ngp_mesh.fast_mesh_index(start_node);
  stk::mesh::FastMeshIndex end_node_index = ngp_mesh.fast_mesh_index(end_node);
  auto ngp_line_segment_view = create_ngp_line_segment_entity_view(ngp_line_segment_data, line_segment_index);
  ASSERT_NEAR(ngp_line_segment_view.start()[0], non_shared_start_node_coords[0], 1e-12);
  ASSERT_NEAR(ngp_line_segment_view.start()[1], non_shared_start_node_coords[1], 1e-12);
  ASSERT_NEAR(ngp_line_segment_view.start()[2], non_shared_start_node_coords[2], 1e-12);
  ASSERT_NEAR(ngp_line_segment_view.end()[0], non_shared_end_node_coords[0], 1e-12);
  ASSERT_NEAR(ngp_line_segment_view.end()[1], non_shared_end_node_coords[1], 1e-12);
  ASSERT_NEAR(ngp_line_segment_view.end()[2], non_shared_end_node_coords[2], 1e-12);

  // Test that the data is modifiable
  ngp_line_segment_view.start()[0] += add_value;
  ngp_line_segment_view.start()[1] -= add_value;
  ngp_line_segment_view.start()[2] *= add_value;
  ngp_line_segment_view.end()[0] += 2 * add_value;
  ngp_line_segment_view.end()[1] -= 2 * add_value;
  ngp_line_segment_view.end()[2] *= 2 * add_value;
  ASSERT_NEAR(ngp_node_coords_field(start_node_index, 0), old_non_shared_start_node_coords[0] + add_value, 1e-12);
  ASSERT_NEAR(ngp_node_coords_field(start_node_index, 1), old_non_shared_start_node_coords[1] - add_value, 1e-12);
  ASSERT_NEAR(ngp_node_coords_field(start_node_index, 2), old_non_shared_start_node_coords[2] * add_value, 1e-12);
  ASSERT_NEAR(ngp_node_coords_field(end_node_index, 0), old_non_shared_end_node_coords[0] + 2 * add_value, 1e-12);
  ASSERT_NEAR(ngp_node_coords_field(end_node_index, 1), old_non_shared_end_node_coords[1] - 2 * add_value, 1e-12);
  ASSERT_NEAR(ngp_node_coords_field(end_node_index, 2), old_non_shared_end_node_coords[2] * 2 * add_value, 1e-12);
}

template <stk::topology::topology_t OurTopology>
void test_v_segment_data(stk::mesh::BulkData& bulk_data,  //
                         stk::mesh::Entity v_segment,     //
                         stk::mesh::Field<double>& node_coords_field) {
  stk::topology our_topology = OurTopology;
  stk::topology::rank_t v_segment_rank = our_topology.rank();
  ASSERT_TRUE(bulk_data.is_valid(v_segment));
  ASSERT_TRUE(bulk_data.bucket(v_segment).topology().rank() == v_segment_rank);
  ASSERT_TRUE(v_segment_rank == stk::topology::ELEM_RANK) << "For now, we only support element rank line segments.";

  ASSERT_EQ(bulk_data.num_nodes(v_segment), 3);
  stk::mesh::Entity start_node = bulk_data.begin_nodes(v_segment)[0];
  stk::mesh::Entity middle_node = bulk_data.begin_nodes(v_segment)[1];
  stk::mesh::Entity end_node = bulk_data.begin_nodes(v_segment)[2];
  ASSERT_TRUE(bulk_data.is_valid(start_node));
  ASSERT_TRUE(bulk_data.is_valid(middle_node));
  ASSERT_TRUE(bulk_data.is_valid(end_node));

  // Test the regular v_segment data to ensure that the stored shared data/fields are as expected
  auto v_segment_data = create_v_segment_data<double, OurTopology>(bulk_data, node_coords_field);
  ASSERT_EQ(&v_segment_data.node_coords_data(), &node_coords_field);

  // Same test for the NGP v_segment data
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
  stk::mesh::NgpField<double>& ngp_node_coords_field = stk::mesh::get_updated_ngp_field<double>(node_coords_field);
  auto ngp_v_segment_data = create_ngp_v_segment_data<double, OurTopology>(ngp_mesh, ngp_node_coords_field);
  ASSERT_EQ(&ngp_v_segment_data.node_coords_data(), &ngp_node_coords_field);

  // Set the node_coords for the v_segment directly via their fields
  const Point<double> non_shared_start_node_coords{7.7, 8.8, 9.9};
  const Point<double> non_shared_middle_node_coords{4.4, 5.5, 6.6};
  const Point<double> non_shared_end_node_coords{1.1, 2.2, 3.3};

  Point<double> old_non_shared_start_node_coords = non_shared_start_node_coords;
  Point<double> old_non_shared_middle_node_coords = non_shared_middle_node_coords;
  Point<double> old_non_shared_end_node_coords = non_shared_end_node_coords;
  stk::mesh::field_data(node_coords_field, start_node)[0] = non_shared_start_node_coords[0];
  stk::mesh::field_data(node_coords_field, start_node)[1] = non_shared_start_node_coords[1];
  stk::mesh::field_data(node_coords_field, start_node)[2] = non_shared_start_node_coords[2];
  stk::mesh::field_data(node_coords_field, middle_node)[0] = non_shared_middle_node_coords[0];
  stk::mesh::field_data(node_coords_field, middle_node)[1] = non_shared_middle_node_coords[1];
  stk::mesh::field_data(node_coords_field, middle_node)[2] = non_shared_middle_node_coords[2];
  stk::mesh::field_data(node_coords_field, end_node)[0] = non_shared_end_node_coords[0];
  stk::mesh::field_data(node_coords_field, end_node)[1] = non_shared_end_node_coords[1];
  stk::mesh::field_data(node_coords_field, end_node)[2] = non_shared_end_node_coords[2];

  // Test that the v_segment data properly views the updated fields

  // Test that the data is modifiable
  // Add and then remove a constant value to the node_coords
  const double add_value = 1.1;
  auto v_segment_view = create_v_segment_entity_view(v_segment_data, v_segment);
  ASSERT_NEAR(v_segment_view.start()[0], non_shared_start_node_coords[0], 1e-12);
  ASSERT_NEAR(v_segment_view.start()[1], non_shared_start_node_coords[1], 1e-12);
  ASSERT_NEAR(v_segment_view.start()[2], non_shared_start_node_coords[2], 1e-12);
  ASSERT_NEAR(v_segment_view.middle()[0], non_shared_middle_node_coords[0], 1e-12);
  ASSERT_NEAR(v_segment_view.middle()[1], non_shared_middle_node_coords[1], 1e-12);
  ASSERT_NEAR(v_segment_view.middle()[2], non_shared_middle_node_coords[2], 1e-12);
  ASSERT_NEAR(v_segment_view.end()[0], non_shared_end_node_coords[0], 1e-12);
  ASSERT_NEAR(v_segment_view.end()[1], non_shared_end_node_coords[1], 1e-12);
  ASSERT_NEAR(v_segment_view.end()[2], non_shared_end_node_coords[2], 1e-12);
  v_segment_view.start()[0] += add_value;
  v_segment_view.start()[1] -= add_value;
  v_segment_view.start()[2] *= add_value;
  v_segment_view.middle()[0] += 2 * add_value;
  v_segment_view.middle()[1] -= 2 * add_value;
  v_segment_view.middle()[2] *= 2 * add_value;
  v_segment_view.end()[0] += 2 * add_value;
  v_segment_view.end()[1] -= 2 * add_value;
  v_segment_view.end()[2] *= 2 * add_value;

  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, start_node)[0], old_non_shared_start_node_coords[0] + add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, start_node)[1], old_non_shared_start_node_coords[1] - add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, start_node)[2], old_non_shared_start_node_coords[2] * add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, middle_node)[0],
              old_non_shared_middle_node_coords[0] + 2 * add_value, 1e-12);
  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, middle_node)[1],
              old_non_shared_middle_node_coords[1] - 2 * add_value, 1e-12);
  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, middle_node)[2],
              old_non_shared_middle_node_coords[2] * 2 * add_value, 1e-12);
  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, end_node)[0], old_non_shared_end_node_coords[0] + 2 * add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, end_node)[1], old_non_shared_end_node_coords[1] - 2 * add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, end_node)[2], old_non_shared_end_node_coords[2] * 2 * add_value,
              1e-12);

  // Remove the added value
  v_segment_view.start()[0] -= add_value;
  v_segment_view.start()[1] += add_value;
  v_segment_view.start()[2] /= add_value;
  v_segment_view.middle()[0] -= 2 * add_value;
  v_segment_view.middle()[1] += 2 * add_value;
  v_segment_view.middle()[2] /= 2 * add_value;
  v_segment_view.end()[0] -= 2 * add_value;
  v_segment_view.end()[1] += 2 * add_value;
  v_segment_view.end()[2] /= 2 * add_value;

  // Test that the NGP v_segment data properly views the updated fields
  stk::mesh::FastMeshIndex v_segment_index = ngp_mesh.fast_mesh_index(v_segment);
  stk::mesh::FastMeshIndex start_node_index = ngp_mesh.fast_mesh_index(start_node);
  stk::mesh::FastMeshIndex middle_node_index = ngp_mesh.fast_mesh_index(middle_node);
  stk::mesh::FastMeshIndex end_node_index = ngp_mesh.fast_mesh_index(end_node);
  auto ngp_v_segment_view = create_ngp_v_segment_entity_view(ngp_v_segment_data, v_segment_index);
  ASSERT_NEAR(ngp_v_segment_view.start()[0], non_shared_start_node_coords[0], 1e-12);
  ASSERT_NEAR(ngp_v_segment_view.start()[1], non_shared_start_node_coords[1], 1e-12);
  ASSERT_NEAR(ngp_v_segment_view.start()[2], non_shared_start_node_coords[2], 1e-12);
  ASSERT_NEAR(ngp_v_segment_view.middle()[0], non_shared_middle_node_coords[0], 1e-12);
  ASSERT_NEAR(ngp_v_segment_view.middle()[1], non_shared_middle_node_coords[1], 1e-12);
  ASSERT_NEAR(ngp_v_segment_view.middle()[2], non_shared_middle_node_coords[2], 1e-12);
  ASSERT_NEAR(ngp_v_segment_view.end()[0], non_shared_end_node_coords[0], 1e-12);
  ASSERT_NEAR(ngp_v_segment_view.end()[1], non_shared_end_node_coords[1], 1e-12);
  ASSERT_NEAR(ngp_v_segment_view.end()[2], non_shared_end_node_coords[2], 1e-12);

  // Test that the data is modifiable
  ngp_v_segment_view.start()[0] += add_value;
  ngp_v_segment_view.start()[1] -= add_value;
  ngp_v_segment_view.start()[2] *= add_value;
  ngp_v_segment_view.middle()[0] += 2 * add_value;
  ngp_v_segment_view.middle()[1] -= 2 * add_value;
  ngp_v_segment_view.middle()[2] *= 2 * add_value;
  ngp_v_segment_view.end()[0] += 2 * add_value;
  ngp_v_segment_view.end()[1] -= 2 * add_value;
  ngp_v_segment_view.end()[2] *= 2 * add_value;
  ASSERT_NEAR(ngp_node_coords_field(start_node_index, 0), old_non_shared_start_node_coords[0] + add_value, 1e-12);
  ASSERT_NEAR(ngp_node_coords_field(start_node_index, 1), old_non_shared_start_node_coords[1] - add_value, 1e-12);
  ASSERT_NEAR(ngp_node_coords_field(start_node_index, 2), old_non_shared_start_node_coords[2] * add_value, 1e-12);
  ASSERT_NEAR(ngp_node_coords_field(middle_node_index, 0), old_non_shared_middle_node_coords[0] + 2 * add_value, 1e-12);
  ASSERT_NEAR(ngp_node_coords_field(middle_node_index, 1), old_non_shared_middle_node_coords[1] - 2 * add_value, 1e-12);
  ASSERT_NEAR(ngp_node_coords_field(middle_node_index, 2), old_non_shared_middle_node_coords[2] * 2 * add_value, 1e-12);
  ASSERT_NEAR(ngp_node_coords_field(end_node_index, 0), old_non_shared_end_node_coords[0] + 2 * add_value, 1e-12);
  ASSERT_NEAR(ngp_node_coords_field(end_node_index, 1), old_non_shared_end_node_coords[1] - 2 * add_value, 1e-12);
  ASSERT_NEAR(ngp_node_coords_field(end_node_index, 2), old_non_shared_end_node_coords[2] * 2 * add_value, 1e-12);
}

template <stk::topology::topology_t OurTopology, bool is_radius_shared,
          bool is_length_shared>
void test_spherocylinder_data(stk::mesh::BulkData& bulk_data,               //
                              stk::mesh::Entity spherocylinder,             //
                              stk::mesh::Field<double>& center_field,       //
                              stk::mesh::Field<double>& orientation_field,  //
                              stk::mesh::Field<double>& radius_field,       //
                              stk::mesh::Field<double>& length_field) {
  stk::topology our_topology = OurTopology;
  stk::topology::rank_t spherocylinder_rank = our_topology.rank();
  ASSERT_TRUE(bulk_data.is_valid(spherocylinder));
  ASSERT_TRUE(bulk_data.bucket(spherocylinder).topology().rank() == spherocylinder_rank);
  ASSERT_TRUE(spherocylinder_rank == stk::topology::ELEMENT_RANK || spherocylinder_rank == stk::topology::NODE_RANK);

  // The shared data for the spherocylinder
  mundy::math::Quaternion<double> orientation{0.1, 0.2, 0.3,
                                              0.4};  // Not a valid unit quaternion but that's fine for this test.
  double radius = 1.01;
  double length = 2.02;

  // Test the regular spherocylinder data to ensure that the stored shared data/fields are as expected
  auto spherocylinder_data =
      create_spherocylinder_data<double, OurTopology>(bulk_data,                                                    //
                                                      center_field,                                                 //
                                                      orientation_field,                                            //
                                                      get_first_or_second<is_radius_shared>(radius, radius_field),  //
                                                      get_first_or_second<is_length_shared>(length, length_field));
  ASSERT_EQ(&spherocylinder_data.center_data(), &center_field);
  ASSERT_EQ(&spherocylinder_data.orientation_data(), &orientation_field);
  if constexpr (is_radius_shared) {
    ASSERT_EQ(&spherocylinder_data.radius_data(), &radius);
  } else {
    ASSERT_EQ(&spherocylinder_data.radius_data(), &radius_field);
  }
  if constexpr (is_length_shared) {
    ASSERT_EQ(&spherocylinder_data.length_data(), &length);
  } else {
    ASSERT_EQ(&spherocylinder_data.length_data(), &length_field);
  }

  // Same test for the NGP spherocylinder data
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
  stk::mesh::NgpField<double>& ngp_center_field = stk::mesh::get_updated_ngp_field<double>(center_field);
  stk::mesh::NgpField<double>& ngp_orientation_field = stk::mesh::get_updated_ngp_field<double>(orientation_field);
  stk::mesh::NgpField<double>& ngp_radius_field = stk::mesh::get_updated_ngp_field<double>(radius_field);
  stk::mesh::NgpField<double>& ngp_length_field = stk::mesh::get_updated_ngp_field<double>(length_field);
  auto ngp_spherocylinder_data = create_ngp_spherocylinder_data<double, OurTopology>(
      ngp_mesh,                                                         //
      ngp_center_field,                                                 //
      ngp_orientation_field,                                            //
      get_first_or_second<is_radius_shared>(radius, ngp_radius_field),  //
      get_first_or_second<is_length_shared>(length, ngp_length_field));

  ASSERT_EQ(&ngp_spherocylinder_data.center_data(), &ngp_center_field);
  ASSERT_EQ(&ngp_spherocylinder_data.orientation_data(), &ngp_orientation_field);
  if constexpr (is_radius_shared) {
    ASSERT_EQ(&ngp_spherocylinder_data.radius_data(), &radius);
  } else {
    ASSERT_EQ(&ngp_spherocylinder_data.radius_data(), &ngp_radius_field);
  }
  if constexpr (is_length_shared) {
    ASSERT_EQ(&ngp_spherocylinder_data.length_data(), &length);
  } else {
    ASSERT_EQ(&ngp_spherocylinder_data.length_data(), &ngp_length_field);
  }

  // Set the center and radius data for the spherocylinder directly via their fields
  const Point<double> non_shared_center{7.7, 8.8, 9.9};
  const mundy::math::Quaternion<double> non_shared_orientation{0.5, 0.6, 0.7, 0.8};
  const double non_shared_radius = 4.04;
  const double non_shared_length = 5.05;

  Point<double> old_non_shared_center = non_shared_center;
  mundy::math::Quaternion<double> old_orientation = orientation;
  mundy::math::Quaternion<double> old_non_shared_orientation = non_shared_orientation;
  double old_radius = radius;
  double old_non_shared_radius = non_shared_radius;
  double old_length = length;
  double old_non_shared_length = non_shared_length;

  if (spherocylinder_rank == stk::topology::NODE_RANK) {
    stk::mesh::field_data(center_field, spherocylinder)[0] = non_shared_center[0];
    stk::mesh::field_data(center_field, spherocylinder)[1] = non_shared_center[1];
    stk::mesh::field_data(center_field, spherocylinder)[2] = non_shared_center[2];
    stk::mesh::field_data(orientation_field, spherocylinder)[0] = non_shared_orientation[0];
    stk::mesh::field_data(orientation_field, spherocylinder)[1] = non_shared_orientation[1];
    stk::mesh::field_data(orientation_field, spherocylinder)[2] = non_shared_orientation[2];
    stk::mesh::field_data(orientation_field, spherocylinder)[3] = non_shared_orientation[3];
    if constexpr (!is_radius_shared) {
      stk::mesh::field_data(radius_field, spherocylinder)[0] = non_shared_radius;
    }
    if constexpr (!is_length_shared) {
      stk::mesh::field_data(length_field, spherocylinder)[0] = non_shared_length;
    }
  } else {
    ASSERT_EQ(bulk_data.num_nodes(spherocylinder), 1);
    stk::mesh::Entity node = bulk_data.begin_nodes(spherocylinder)[0];
    ASSERT_TRUE(bulk_data.is_valid(node));
    stk::mesh::field_data(center_field, node)[0] = non_shared_center[0];
    stk::mesh::field_data(center_field, node)[1] = non_shared_center[1];
    stk::mesh::field_data(center_field, node)[2] = non_shared_center[2];
    stk::mesh::field_data(orientation_field, spherocylinder)[0] = non_shared_orientation[0];
    stk::mesh::field_data(orientation_field, spherocylinder)[1] = non_shared_orientation[1];
    stk::mesh::field_data(orientation_field, spherocylinder)[2] = non_shared_orientation[2];
    stk::mesh::field_data(orientation_field, spherocylinder)[3] = non_shared_orientation[3];
    if constexpr (!is_radius_shared) {
      stk::mesh::field_data(radius_field, spherocylinder)[0] = non_shared_radius;
    }
    if constexpr (!is_length_shared) {
      stk::mesh::field_data(length_field, spherocylinder)[0] = non_shared_length;
    }
  }

  // Test that the spherocylinder data properly views the updated fields

  // Test that the data is modifiable
  const double add_value = 1.1;
  if (spherocylinder_rank == stk::topology::NODE_RANK) {
    auto spherocylinder_view = create_spherocylinder_entity_view(spherocylinder_data, spherocylinder);
    ASSERT_DOUBLE_EQ(spherocylinder_view.center()[0], non_shared_center[0]);
    ASSERT_DOUBLE_EQ(spherocylinder_view.center()[1], non_shared_center[1]);
    ASSERT_DOUBLE_EQ(spherocylinder_view.center()[2], non_shared_center[2]);
    ASSERT_DOUBLE_EQ(spherocylinder_view.orientation()[0], non_shared_orientation[0]);
    ASSERT_DOUBLE_EQ(spherocylinder_view.orientation()[1], non_shared_orientation[1]);
    ASSERT_DOUBLE_EQ(spherocylinder_view.orientation()[2], non_shared_orientation[2]);
    ASSERT_DOUBLE_EQ(spherocylinder_view.orientation()[3], non_shared_orientation[3]);
    if constexpr (!is_radius_shared) {
      ASSERT_DOUBLE_EQ(spherocylinder_view.radius(), non_shared_radius);
    }
    if constexpr (!is_length_shared) {
      ASSERT_DOUBLE_EQ(spherocylinder_view.length(), non_shared_length);
    }

    spherocylinder_view.center()[0] += add_value;
    spherocylinder_view.center()[1] -= add_value;
    spherocylinder_view.center()[2] *= add_value;
    spherocylinder_view.orientation()[0] += add_value;
    spherocylinder_view.orientation()[1] -= add_value;
    spherocylinder_view.orientation()[2] *= add_value;
    spherocylinder_view.orientation()[3] += 2 * add_value;
    if constexpr (!is_radius_shared) {
      spherocylinder_view.radius() += 3 * add_value;
    }
    if constexpr (!is_length_shared) {
      spherocylinder_view.length() += 4 * add_value;
    }

    ASSERT_DOUBLE_EQ(stk::mesh::field_data(center_field, spherocylinder)[0], old_non_shared_center[0] + add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(center_field, spherocylinder)[1], old_non_shared_center[1] - add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(center_field, spherocylinder)[2], old_non_shared_center[2] * add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(orientation_field, spherocylinder)[0],
                     old_non_shared_orientation[0] + add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(orientation_field, spherocylinder)[1],
                     old_non_shared_orientation[1] - add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(orientation_field, spherocylinder)[2],
                     old_non_shared_orientation[2] * add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(orientation_field, spherocylinder)[3],
                     old_non_shared_orientation[3] + 2 * add_value);
    if constexpr (!is_radius_shared) {
      ASSERT_DOUBLE_EQ(stk::mesh::field_data(radius_field, spherocylinder)[0], old_non_shared_radius + 3 * add_value);
    }
    if constexpr (!is_length_shared) {
      ASSERT_DOUBLE_EQ(stk::mesh::field_data(length_field, spherocylinder)[0], old_non_shared_length + 4 * add_value);
    }

    // Remove the added value
    spherocylinder_view.center()[0] -= add_value;
    spherocylinder_view.center()[1] += add_value;
    spherocylinder_view.center()[2] /= add_value;
    spherocylinder_view.orientation()[0] -= add_value;
    spherocylinder_view.orientation()[1] += add_value;
    spherocylinder_view.orientation()[2] /= add_value;
    spherocylinder_view.orientation()[3] -= 2 * add_value;
    if constexpr (!is_radius_shared) {
      spherocylinder_view.radius() -= 3 * add_value;
    }
    if constexpr (!is_length_shared) {
      spherocylinder_view.length() -= 4 * add_value;
    }
  } else {
    auto spherocylinder_view = create_spherocylinder_entity_view(spherocylinder_data, spherocylinder);
    ASSERT_DOUBLE_EQ(spherocylinder_view.center()[0], non_shared_center[0]);
    ASSERT_DOUBLE_EQ(spherocylinder_view.center()[1], non_shared_center[1]);
    ASSERT_DOUBLE_EQ(spherocylinder_view.center()[2], non_shared_center[2]);
    ASSERT_DOUBLE_EQ(spherocylinder_view.orientation()[0], non_shared_orientation[0]);
    ASSERT_DOUBLE_EQ(spherocylinder_view.orientation()[1], non_shared_orientation[1]);
    ASSERT_DOUBLE_EQ(spherocylinder_view.orientation()[2], non_shared_orientation[2]);
    ASSERT_DOUBLE_EQ(spherocylinder_view.orientation()[3], non_shared_orientation[3]);
    if constexpr (!is_radius_shared) {
      ASSERT_DOUBLE_EQ(spherocylinder_view.radius(), non_shared_radius);
    }
    if constexpr (!is_length_shared) {
      ASSERT_DOUBLE_EQ(spherocylinder_view.length(), non_shared_length);
    }

    spherocylinder_view.center()[0] += add_value;
    spherocylinder_view.center()[1] -= add_value;
    spherocylinder_view.center()[2] *= add_value;
    spherocylinder_view.orientation()[0] += add_value;
    spherocylinder_view.orientation()[1] -= add_value;
    spherocylinder_view.orientation()[2] *= add_value;
    spherocylinder_view.orientation()[3] += 2 * add_value;
    if constexpr (!is_radius_shared) {
      spherocylinder_view.radius() += 3 * add_value;
    }
    if constexpr (!is_length_shared) {
      spherocylinder_view.length() += 4 * add_value;
    }

    ASSERT_EQ(bulk_data.num_nodes(spherocylinder), 1);
    stk::mesh::Entity node = bulk_data.begin_nodes(spherocylinder)[0];
    ASSERT_TRUE(bulk_data.is_valid(node));
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(center_field, node)[0], old_non_shared_center[0] + add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(center_field, node)[1], old_non_shared_center[1] - add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(center_field, node)[2], old_non_shared_center[2] * add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(orientation_field, spherocylinder)[0],
                     old_non_shared_orientation[0] + add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(orientation_field, spherocylinder)[1],
                     old_non_shared_orientation[1] - add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(orientation_field, spherocylinder)[2],
                     old_non_shared_orientation[2] * add_value);
    ASSERT_DOUBLE_EQ(stk::mesh::field_data(orientation_field, spherocylinder)[3],
                     old_non_shared_orientation[3] + 2 * add_value);
    if constexpr (!is_radius_shared) {
      ASSERT_DOUBLE_EQ(stk::mesh::field_data(radius_field, spherocylinder)[0], old_non_shared_radius + 3 * add_value);
    }
    if constexpr (!is_length_shared) {
      ASSERT_DOUBLE_EQ(stk::mesh::field_data(length_field, spherocylinder)[0], old_non_shared_length + 4 * add_value);
    }

    // Remove the added value
    spherocylinder_view.center()[0] -= add_value;
    spherocylinder_view.center()[1] += add_value;
    spherocylinder_view.center()[2] /= add_value;
    spherocylinder_view.orientation()[0] -= add_value;
    spherocylinder_view.orientation()[1] += add_value;
    spherocylinder_view.orientation()[2] /= add_value;
    spherocylinder_view.orientation()[3] -= 2 * add_value;
    if constexpr (!is_radius_shared) {
      spherocylinder_view.radius() -= 3 * add_value;
    }
    if constexpr (!is_length_shared) {
      spherocylinder_view.length() -= 4 * add_value;
    }
  }

  // Test that the NGP spherocylinder data properly views the updated fields
  stk::mesh::FastMeshIndex spherocylinder_index = ngp_mesh.fast_mesh_index(spherocylinder);
  if (spherocylinder_rank == stk::topology::NODE_RANK) {
    auto ngp_spherocylinder_view = create_ngp_spherocylinder_entity_view(ngp_spherocylinder_data, spherocylinder_index);
    ASSERT_DOUBLE_EQ(ngp_spherocylinder_view.center()[0], non_shared_center[0]);
    ASSERT_DOUBLE_EQ(ngp_spherocylinder_view.center()[1], non_shared_center[1]);
    ASSERT_DOUBLE_EQ(ngp_spherocylinder_view.center()[2], non_shared_center[2]);
    ASSERT_DOUBLE_EQ(ngp_spherocylinder_view.orientation()[0], non_shared_orientation[0]);
    ASSERT_DOUBLE_EQ(ngp_spherocylinder_view.orientation()[1], non_shared_orientation[1]);
    ASSERT_DOUBLE_EQ(ngp_spherocylinder_view.orientation()[2], non_shared_orientation[2]);
    ASSERT_DOUBLE_EQ(ngp_spherocylinder_view.orientation()[3], non_shared_orientation[3]);
    if constexpr (!is_radius_shared) {
      ASSERT_DOUBLE_EQ(ngp_spherocylinder_view.radius(), non_shared_radius);
    }
    if constexpr (!is_length_shared) {
      ASSERT_DOUBLE_EQ(ngp_spherocylinder_view.length(), non_shared_length);
    }

    // Test that the data is modifiable
    // Add a constant value to the center and radius
    ngp_spherocylinder_view.center()[0] += add_value;
    ngp_spherocylinder_view.center()[1] -= add_value;
    ngp_spherocylinder_view.center()[2] *= add_value;
    ngp_spherocylinder_view.orientation()[0] += add_value;
    ngp_spherocylinder_view.orientation()[1] -= add_value;
    ngp_spherocylinder_view.orientation()[2] *= add_value;
    ngp_spherocylinder_view.orientation()[3] += 2 * add_value;
    if constexpr (!is_radius_shared) {
      ngp_spherocylinder_view.radius() += 3 * add_value;
    }
    if constexpr (!is_length_shared) {
      ngp_spherocylinder_view.length() += 4 * add_value;
    }
    ASSERT_DOUBLE_EQ(ngp_center_field(spherocylinder_index, 0), old_non_shared_center[0] + add_value);
    ASSERT_DOUBLE_EQ(ngp_center_field(spherocylinder_index, 1), old_non_shared_center[1] - add_value);
    ASSERT_DOUBLE_EQ(ngp_center_field(spherocylinder_index, 2), old_non_shared_center[2] * add_value);
    ASSERT_DOUBLE_EQ(ngp_orientation_field(spherocylinder_index, 0), old_non_shared_orientation[0] + add_value);
    ASSERT_DOUBLE_EQ(ngp_orientation_field(spherocylinder_index, 1), old_non_shared_orientation[1] - add_value);
    ASSERT_DOUBLE_EQ(ngp_orientation_field(spherocylinder_index, 2), old_non_shared_orientation[2] * add_value);
    ASSERT_DOUBLE_EQ(ngp_orientation_field(spherocylinder_index, 3), old_non_shared_orientation[3] + 2 * add_value);
    if constexpr (!is_radius_shared) {
      ASSERT_DOUBLE_EQ(ngp_radius_field(spherocylinder_index, 0), old_non_shared_radius + 3 * add_value);
    }
    if constexpr (!is_length_shared) {
      ASSERT_DOUBLE_EQ(ngp_length_field(spherocylinder_index, 0), old_non_shared_length + 4 * add_value);
    }
  } else {
    auto ngp_spherocylinder_view = create_ngp_spherocylinder_entity_view(ngp_spherocylinder_data, spherocylinder_index);
    ASSERT_DOUBLE_EQ(ngp_spherocylinder_view.center()[0], non_shared_center[0]);
    ASSERT_DOUBLE_EQ(ngp_spherocylinder_view.center()[1], non_shared_center[1]);
    ASSERT_DOUBLE_EQ(ngp_spherocylinder_view.center()[2], non_shared_center[2]);
    ASSERT_DOUBLE_EQ(ngp_spherocylinder_view.orientation()[0], non_shared_orientation[0]);
    ASSERT_DOUBLE_EQ(ngp_spherocylinder_view.orientation()[1], non_shared_orientation[1]);
    ASSERT_DOUBLE_EQ(ngp_spherocylinder_view.orientation()[2], non_shared_orientation[2]);
    ASSERT_DOUBLE_EQ(ngp_spherocylinder_view.orientation()[3], non_shared_orientation[3]);
    if constexpr (!is_radius_shared) {
      ASSERT_DOUBLE_EQ(ngp_spherocylinder_view.radius(), non_shared_radius);
    }
    if constexpr (!is_length_shared) {
      ASSERT_DOUBLE_EQ(ngp_spherocylinder_view.length(), non_shared_length);
    }

    // Test that the data is modifiable
    // Add a constant value to the center and radius
    ngp_spherocylinder_view.center()[0] += add_value;
    ngp_spherocylinder_view.center()[1] -= add_value;
    ngp_spherocylinder_view.center()[2] *= add_value;
    ngp_spherocylinder_view.orientation()[0] += add_value;
    ngp_spherocylinder_view.orientation()[1] -= add_value;
    ngp_spherocylinder_view.orientation()[2] *= add_value;
    ngp_spherocylinder_view.orientation()[3] += 2 * add_value;
    if constexpr (!is_radius_shared) {
      ngp_spherocylinder_view.radius() += 3 * add_value;
    }
    if constexpr (!is_length_shared) {
      ngp_spherocylinder_view.length() += 4 * add_value;
    }

    stk::mesh::Entity node = bulk_data.begin_nodes(spherocylinder)[0];
    stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(node);
    ASSERT_EQ(ngp_center_field(node_index, 0), old_non_shared_center[0] + add_value);
    ASSERT_EQ(ngp_center_field(node_index, 1), old_non_shared_center[1] - add_value);
    ASSERT_EQ(ngp_center_field(node_index, 2), old_non_shared_center[2] * add_value);
    ASSERT_EQ(ngp_orientation_field(spherocylinder_index, 0), old_non_shared_orientation[0] + add_value);
    ASSERT_EQ(ngp_orientation_field(spherocylinder_index, 1), old_non_shared_orientation[1] - add_value);
    ASSERT_EQ(ngp_orientation_field(spherocylinder_index, 2), old_non_shared_orientation[2] * add_value);
    ASSERT_EQ(ngp_orientation_field(spherocylinder_index, 3), old_non_shared_orientation[3] + 2 * add_value);
    if constexpr (!is_radius_shared) {
      ASSERT_EQ(ngp_radius_field(spherocylinder_index, 0), old_non_shared_radius + 3 * add_value);
    }
    if constexpr (!is_length_shared) {
      ASSERT_EQ(ngp_length_field(spherocylinder_index, 0), old_non_shared_length + 4 * add_value);
    }
  }
}

template <stk::topology::topology_t OurTopology, bool is_radius_shared>
void test_spherocylinder_segment_data(stk::mesh::BulkData& bulk_data,               //
                                      stk::mesh::Entity spherocylinder_segment,     //
                                      stk::mesh::Field<double>& node_coords_field,  //
                                      stk::mesh::Field<double>& radius_field) {
  stk::topology our_topology = OurTopology;
  stk::topology::rank_t spherocylinder_segment_rank = our_topology.rank();
  ASSERT_TRUE(bulk_data.is_valid(spherocylinder_segment));
  ASSERT_TRUE(bulk_data.bucket(spherocylinder_segment).topology().rank() == spherocylinder_segment_rank);
  ASSERT_TRUE(spherocylinder_segment_rank == stk::topology::ELEM_RANK)
      << "For now, we only support element rank spherocylinder segments.";

  ASSERT_EQ(bulk_data.num_nodes(spherocylinder_segment), 2);
  stk::mesh::Entity start_node = bulk_data.begin_nodes(spherocylinder_segment)[0];
  stk::mesh::Entity end_node = bulk_data.begin_nodes(spherocylinder_segment)[1];
  ASSERT_TRUE(bulk_data.is_valid(start_node));
  ASSERT_TRUE(bulk_data.is_valid(end_node));

  // Shared data for the spherocylinder_segment
  double radius = 1.1;

  // Test the regular spherocylinder_segment data to ensure that the stored shared data/fields are as expected
  auto spherocylinder_segment_data = create_spherocylinder_segment_data<double, OurTopology>(
      bulk_data, node_coords_field, get_first_or_second<is_radius_shared>(radius, radius_field));
  ASSERT_EQ(&spherocylinder_segment_data.node_coords_data(), &node_coords_field);
  if constexpr (is_radius_shared) {
    ASSERT_EQ(&spherocylinder_segment_data.radius_data(), &radius);
  } else {
    ASSERT_EQ(&spherocylinder_segment_data.radius_data(), &radius_field);
  }

  // Same test for the NGP spherocylinder_segment data
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
  stk::mesh::NgpField<double>& ngp_node_coords_field = stk::mesh::get_updated_ngp_field<double>(node_coords_field);
  stk::mesh::NgpField<double>& ngp_radius_field = stk::mesh::get_updated_ngp_field<double>(radius_field);
  auto ngp_spherocylinder_segment_data = create_ngp_spherocylinder_segment_data<double, OurTopology>(
      ngp_mesh, ngp_node_coords_field, get_first_or_second<is_radius_shared>(radius, ngp_radius_field));
  ASSERT_EQ(&ngp_spherocylinder_segment_data.node_coords_data(), &ngp_node_coords_field);
  if constexpr (is_radius_shared) {
    ASSERT_EQ(&ngp_spherocylinder_segment_data.radius_data(), &radius);
  } else {
    ASSERT_EQ(&ngp_spherocylinder_segment_data.radius_data(), &ngp_radius_field);
  }

  // Set the node_coords for the spherocylinder_segment directly via their fields
  const Point<double> non_shared_start_node_coords{7.7, 8.8, 9.9};
  const Point<double> non_shared_end_node_coords{1.1, 2.2, 3.3};
  const double non_shared_radius = 2.2;

  Point<double> old_non_shared_start_node_coords = non_shared_start_node_coords;
  Point<double> old_non_shared_end_node_coords = non_shared_end_node_coords;
  double old_radius = radius;
  double old_non_shared_radius = non_shared_radius;
  stk::mesh::field_data(node_coords_field, start_node)[0] = non_shared_start_node_coords[0];
  stk::mesh::field_data(node_coords_field, start_node)[1] = non_shared_start_node_coords[1];
  stk::mesh::field_data(node_coords_field, start_node)[2] = non_shared_start_node_coords[2];
  stk::mesh::field_data(node_coords_field, end_node)[0] = non_shared_end_node_coords[0];
  stk::mesh::field_data(node_coords_field, end_node)[1] = non_shared_end_node_coords[1];
  stk::mesh::field_data(node_coords_field, end_node)[2] = non_shared_end_node_coords[2];
  if constexpr (!is_radius_shared) {
    stk::mesh::field_data(radius_field, spherocylinder_segment)[0] = non_shared_radius;
  }

  // Test that the spherocylinder_segment data properly views the updated fields

  // Test that the data is modifiable
  // Add and then remove a constant value to the node_coords
  const double add_value = 1.1;
  auto spherocylinder_segment_view =
      create_spherocylinder_segment_entity_view(spherocylinder_segment_data, spherocylinder_segment);
  ASSERT_NEAR(spherocylinder_segment_view.start()[0], non_shared_start_node_coords[0], 1e-12);
  ASSERT_NEAR(spherocylinder_segment_view.start()[1], non_shared_start_node_coords[1], 1e-12);
  ASSERT_NEAR(spherocylinder_segment_view.start()[2], non_shared_start_node_coords[2], 1e-12);
  ASSERT_NEAR(spherocylinder_segment_view.end()[0], non_shared_end_node_coords[0], 1e-12);
  ASSERT_NEAR(spherocylinder_segment_view.end()[1], non_shared_end_node_coords[1], 1e-12);
  ASSERT_NEAR(spherocylinder_segment_view.end()[2], non_shared_end_node_coords[2], 1e-12);
  if constexpr (is_radius_shared) {
    ASSERT_NEAR(spherocylinder_segment_view.radius(), radius, 1e-12);
  } else {
    ASSERT_NEAR(spherocylinder_segment_view.radius(), non_shared_radius, 1e-12);
  }

  spherocylinder_segment_view.start()[0] += add_value;
  spherocylinder_segment_view.start()[1] -= add_value;
  spherocylinder_segment_view.start()[2] *= add_value;
  spherocylinder_segment_view.end()[0] += 2 * add_value;
  spherocylinder_segment_view.end()[1] -= 2 * add_value;
  spherocylinder_segment_view.end()[2] *= 2 * add_value;
  spherocylinder_segment_view.radius() += 3 * add_value;

  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, start_node)[0], old_non_shared_start_node_coords[0] + add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, start_node)[1], old_non_shared_start_node_coords[1] - add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, start_node)[2], old_non_shared_start_node_coords[2] * add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, end_node)[0], old_non_shared_end_node_coords[0] + 2 * add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, end_node)[1], old_non_shared_end_node_coords[1] - 2 * add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(node_coords_field, end_node)[2], old_non_shared_end_node_coords[2] * 2 * add_value,
              1e-12);
  if constexpr (is_radius_shared) {
    ASSERT_NEAR(radius, old_radius + 3 * add_value, 1e-12);
  } else {
    ASSERT_NEAR(stk::mesh::field_data(radius_field, spherocylinder_segment)[0], old_non_shared_radius + 3 * add_value,
                1e-12);
  }

  // Remove the added value
  spherocylinder_segment_view.start()[0] -= add_value;
  spherocylinder_segment_view.start()[1] += add_value;
  spherocylinder_segment_view.start()[2] /= add_value;
  spherocylinder_segment_view.end()[0] -= 2 * add_value;
  spherocylinder_segment_view.end()[1] += 2 * add_value;
  spherocylinder_segment_view.end()[2] /= 2 * add_value;
  spherocylinder_segment_view.radius() -= 3 * add_value;

  // Test that the NGP spherocylinder_segment data properly views the updated fields
  stk::mesh::FastMeshIndex spherocylinder_segment_index = ngp_mesh.fast_mesh_index(spherocylinder_segment);
  stk::mesh::FastMeshIndex start_node_index = ngp_mesh.fast_mesh_index(start_node);
  stk::mesh::FastMeshIndex end_node_index = ngp_mesh.fast_mesh_index(end_node);
  auto ngp_spherocylinder_segment_view =
      create_ngp_spherocylinder_segment_entity_view(ngp_spherocylinder_segment_data, spherocylinder_segment_index);
  ASSERT_NEAR(ngp_spherocylinder_segment_view.start()[0], non_shared_start_node_coords[0], 1e-12);
  ASSERT_NEAR(ngp_spherocylinder_segment_view.start()[1], non_shared_start_node_coords[1], 1e-12);
  ASSERT_NEAR(ngp_spherocylinder_segment_view.start()[2], non_shared_start_node_coords[2], 1e-12);
  ASSERT_NEAR(ngp_spherocylinder_segment_view.end()[0], non_shared_end_node_coords[0], 1e-12);
  ASSERT_NEAR(ngp_spherocylinder_segment_view.end()[1], non_shared_end_node_coords[1], 1e-12);
  ASSERT_NEAR(ngp_spherocylinder_segment_view.end()[2], non_shared_end_node_coords[2], 1e-12);
  if constexpr (is_radius_shared) {
    ASSERT_NEAR(ngp_spherocylinder_segment_view.radius(), radius, 1e-12);
  } else {
    ASSERT_NEAR(ngp_spherocylinder_segment_view.radius(), non_shared_radius, 1e-12);
  }

  // Test that the data is modifiable
  ngp_spherocylinder_segment_view.start()[0] += add_value;
  ngp_spherocylinder_segment_view.start()[1] -= add_value;
  ngp_spherocylinder_segment_view.start()[2] *= add_value;
  ngp_spherocylinder_segment_view.end()[0] += 2 * add_value;
  ngp_spherocylinder_segment_view.end()[1] -= 2 * add_value;
  ngp_spherocylinder_segment_view.end()[2] *= 2 * add_value;
  ngp_spherocylinder_segment_view.radius() += 3 * add_value;
  ASSERT_NEAR(ngp_node_coords_field(start_node_index, 0), old_non_shared_start_node_coords[0] + add_value, 1e-12);
  ASSERT_NEAR(ngp_node_coords_field(start_node_index, 1), old_non_shared_start_node_coords[1] - add_value, 1e-12);
  ASSERT_NEAR(ngp_node_coords_field(start_node_index, 2), old_non_shared_start_node_coords[2] * add_value, 1e-12);
  ASSERT_NEAR(ngp_node_coords_field(end_node_index, 0), old_non_shared_end_node_coords[0] + 2 * add_value, 1e-12);
  ASSERT_NEAR(ngp_node_coords_field(end_node_index, 1), old_non_shared_end_node_coords[1] - 2 * add_value, 1e-12);
  ASSERT_NEAR(ngp_node_coords_field(end_node_index, 2), old_non_shared_end_node_coords[2] * 2 * add_value, 1e-12);
  if constexpr (is_radius_shared) {
    ASSERT_NEAR(radius, old_radius + 3 * add_value, 1e-12);
  } else {
    ASSERT_NEAR(ngp_radius_field(spherocylinder_segment_index, 0), old_non_shared_radius + 3 * add_value, 1e-12);
  }
}

TEST(Aggregates, SphereData) {
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  meta_data.set_coordinate_field_name("COORDS");
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  stk::mesh::Part& sphere_part = meta_data.declare_part_with_topology("spheres", stk::topology::PARTICLE);
  stk::mesh::Part& node_sphere_part = meta_data.declare_part_with_topology("node_spheres", stk::topology::NODE);
  stk::mesh::Field<double>& node_coords_field =
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "coordinates");
  stk::mesh::Field<double>& node_radius_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "radius");
  stk::mesh::Field<double>& elem_radius_field = meta_data.declare_field<double>(stk::topology::ELEM_RANK, "radius");

  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(node_radius_field, node_sphere_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_radius_field, sphere_part, 1, nullptr);
  meta_data.commit();

  // Create the node and element spheres
  bulk_data.modification_begin();
  stk::mesh::Entity sphere = bulk_data.declare_element(1, stk::mesh::PartVector{&sphere_part});
  stk::mesh::Entity node = bulk_data.declare_node(1);
  stk::mesh::Entity node_sphere = bulk_data.declare_node(2, stk::mesh::PartVector{&node_sphere_part});
  bulk_data.declare_relation(sphere, node, 0);
  bulk_data.modification_end();

  test_sphere_data<stk::topology::NODE, true>(bulk_data, node_sphere, node_coords_field, node_radius_field);
  test_sphere_data<stk::topology::NODE, false>(bulk_data, node_sphere, node_coords_field, node_radius_field);
  test_sphere_data<stk::topology::PARTICLE, true>(bulk_data, sphere, node_coords_field, elem_radius_field);
  test_sphere_data<stk::topology::PARTICLE, false>(bulk_data, sphere, node_coords_field, elem_radius_field);
}

TEST(Aggregates, EllipsoidData) {
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  meta_data.set_coordinate_field_name("COORDS");
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  stk::mesh::Part& ellipsoid_part = meta_data.declare_part_with_topology("ellipsoids", stk::topology::PARTICLE);
  stk::mesh::Part& node_ellipsoid_part = meta_data.declare_part_with_topology("node_ellipsoids", stk::topology::NODE);

  stk::mesh::Field<double>& node_coords_field =
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "coordinates");
  stk::mesh::Field<double>& node_orientation_field =
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "orientation");
  stk::mesh::Field<double>& node_axis_lengths_field =
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "axis_lengths");
  stk::mesh::Field<double>& elem_orientation_field =
      meta_data.declare_field<double>(stk::topology::ELEM_RANK, "orientation");
  stk::mesh::Field<double>& elem_axis_lengths_field =
      meta_data.declare_field<double>(stk::topology::ELEM_RANK, "axis_lengths");

  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(node_orientation_field, node_ellipsoid_part, 4, nullptr);
  stk::mesh::put_field_on_mesh(node_axis_lengths_field, node_ellipsoid_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_orientation_field, ellipsoid_part, 4, nullptr);
  stk::mesh::put_field_on_mesh(elem_axis_lengths_field, ellipsoid_part, 3, nullptr);
  meta_data.commit();

  // Create the node and element ellipsoids
  bulk_data.modification_begin();
  stk::mesh::Entity ellipsoid = bulk_data.declare_element(1, stk::mesh::PartVector{&ellipsoid_part});
  stk::mesh::Entity node = bulk_data.declare_node(1);
  stk::mesh::Entity node_ellipsoid = bulk_data.declare_node(2, stk::mesh::PartVector{&node_ellipsoid_part});
  bulk_data.declare_relation(ellipsoid, node, 0);
  bulk_data.modification_end();

  test_ellipsoid_data<stk::topology::NODE, true>(bulk_data, node_ellipsoid, node_coords_field, node_orientation_field,
                                                 node_axis_lengths_field);
  test_ellipsoid_data<stk::topology::NODE, false>(bulk_data, node_ellipsoid, node_coords_field, node_orientation_field,
                                                  node_axis_lengths_field);

  test_ellipsoid_data<stk::topology::PARTICLE, true>(bulk_data, ellipsoid, node_coords_field, elem_orientation_field,
                                                     elem_axis_lengths_field);
  test_ellipsoid_data<stk::topology::PARTICLE, false>(bulk_data, ellipsoid, node_coords_field, elem_orientation_field,
                                                      elem_axis_lengths_field);
}

TEST(Aggregates, PointData) {
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  stk::mesh::Part& node_part = meta_data.declare_part_with_topology("node_point_part", stk::topology::NODE);
  stk::mesh::Part& elem_part = meta_data.declare_part_with_topology("elem_point_part", stk::topology::PARTICLE);
  stk::mesh::Field<double>& node_center_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "center");

  stk::mesh::put_field_on_mesh(node_center_field, meta_data.universal_part(), 3, nullptr);
  meta_data.commit();

  bulk_data.modification_begin();
  stk::mesh::Entity point_node = bulk_data.declare_node(1, stk::mesh::PartVector{&node_part});

  stk::mesh::Entity node = bulk_data.declare_node(2);
  stk::mesh::Entity point_elem = bulk_data.declare_element(2, stk::mesh::PartVector{&elem_part});
  bulk_data.declare_relation(point_elem, node, 0);
  bulk_data.modification_end();

  test_point_data<stk::topology::NODE>(bulk_data, point_node, node_center_field);
  test_point_data<stk::topology::PARTICLE>(bulk_data, point_elem, node_center_field);
}

TEST(Aggregates, LineData) {
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  stk::mesh::Part& node_part = meta_data.declare_part_with_topology("node_point_part", stk::topology::NODE);
  stk::mesh::Part& elem_part = meta_data.declare_part_with_topology("elem_point_part", stk::topology::PARTICLE);
  stk::mesh::Field<double>& node_center_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "center");
  stk::mesh::Field<double>& node_direction_field =
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "direction");
  stk::mesh::Field<double>& elem_direction_field =
      meta_data.declare_field<double>(stk::topology::ELEM_RANK, "direction");

  stk::mesh::put_field_on_mesh(node_center_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(node_direction_field, node_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_direction_field, elem_part, 3, nullptr);
  meta_data.commit();

  bulk_data.modification_begin();
  stk::mesh::Entity line_node = bulk_data.declare_node(1, stk::mesh::PartVector{&node_part});

  stk::mesh::Entity node = bulk_data.declare_node(2);
  stk::mesh::Entity line_elem = bulk_data.declare_element(1, stk::mesh::PartVector{&elem_part});
  bulk_data.declare_relation(line_elem, node, 0);
  bulk_data.modification_end();

  test_line_data<stk::topology::NODE>(bulk_data, line_node, node_center_field, node_direction_field);
  test_line_data<stk::topology::PARTICLE>(bulk_data, line_elem, node_center_field, elem_direction_field);
}

TEST(Aggregates, LineSegmentData) {
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  stk::mesh::Part& line_segment_part = meta_data.declare_part_with_topology("line_segments", stk::topology::BEAM_2);
  stk::mesh::Field<double>& node_coords_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "coords");
  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);
  meta_data.commit();

  bulk_data.modification_begin();
  stk::mesh::Entity start_node = bulk_data.declare_node(1);
  stk::mesh::Entity end_node = bulk_data.declare_node(2);
  stk::mesh::Entity line_segment = bulk_data.declare_element(1, stk::mesh::PartVector{&line_segment_part});
  bulk_data.declare_relation(line_segment, start_node, 0);
  bulk_data.declare_relation(line_segment, end_node, 1);
  bulk_data.modification_end();

  test_line_segment_data<stk::topology::BEAM_2>(bulk_data, line_segment, node_coords_field);
  test_line_segment_data<stk::topology::BEAM_2>(bulk_data, line_segment, node_coords_field);
}

TEST(Aggregates, VSegmentData) {
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  stk::mesh::Part& v_segment_part = meta_data.declare_part_with_topology("v_segments", stk::topology::SHELL_TRI_3);
  stk::mesh::Field<double>& node_coords_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "coords");
  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);
  meta_data.commit();

  bulk_data.modification_begin();
  stk::mesh::Entity start_node = bulk_data.declare_node(1);
  stk::mesh::Entity middle_node = bulk_data.declare_node(2);
  stk::mesh::Entity end_node = bulk_data.declare_node(3);
  stk::mesh::Entity v_segment = bulk_data.declare_element(1, stk::mesh::PartVector{&v_segment_part});
  bulk_data.declare_relation(v_segment, start_node, 0);
  bulk_data.declare_relation(v_segment, middle_node, 1);
  bulk_data.declare_relation(v_segment, end_node, 2);
  bulk_data.modification_end();

  test_v_segment_data<stk::topology::SHELL_TRI_3>(bulk_data, v_segment, node_coords_field);
  test_v_segment_data<stk::topology::SHELL_TRI_3>(bulk_data, v_segment, node_coords_field);
}

TEST(Aggregates, SpherocylinderData) {
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  meta_data.set_coordinate_field_name("COORDS");
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  stk::mesh::Part& spherocylinder_part =
      meta_data.declare_part_with_topology("spherocylinders", stk::topology::PARTICLE);
  stk::mesh::Part& node_spherocylinder_part =
      meta_data.declare_part_with_topology("node_spherocylinders", stk::topology::NODE);

  stk::mesh::Field<double>& node_coords_field =
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "coordinates");
  stk::mesh::Field<double>& node_orientation_field =
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "orientation");
  stk::mesh::Field<double>& node_radius_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "radius");
  stk::mesh::Field<double>& node_length_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "length");
  stk::mesh::Field<double>& elem_orientation_field =
      meta_data.declare_field<double>(stk::topology::ELEM_RANK, "orientation");
  stk::mesh::Field<double>& elem_radius_field = meta_data.declare_field<double>(stk::topology::ELEM_RANK, "radius");
  stk::mesh::Field<double>& elem_length_field = meta_data.declare_field<double>(stk::topology::ELEM_RANK, "length");

  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(node_orientation_field, node_spherocylinder_part, 4, nullptr);
  stk::mesh::put_field_on_mesh(node_radius_field, node_spherocylinder_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_length_field, node_spherocylinder_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_orientation_field, spherocylinder_part, 4, nullptr);
  stk::mesh::put_field_on_mesh(elem_radius_field, spherocylinder_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_length_field, spherocylinder_part, 1, nullptr);
  meta_data.commit();

  // Create the node and element spherocylinders
  bulk_data.modification_begin();
  stk::mesh::Entity spherocylinder = bulk_data.declare_element(1, stk::mesh::PartVector{&spherocylinder_part});
  stk::mesh::Entity node = bulk_data.declare_node(1);
  stk::mesh::Entity node_spherocylinder = bulk_data.declare_node(2, stk::mesh::PartVector{&node_spherocylinder_part});
  bulk_data.declare_relation(spherocylinder, node, 0);
  bulk_data.modification_end();

  test_spherocylinder_data<stk::topology::NODE, true, true>(
      bulk_data, node_spherocylinder, node_coords_field, node_orientation_field, node_radius_field, node_length_field);
  test_spherocylinder_data<stk::topology::NODE, false, true>(
      bulk_data, node_spherocylinder, node_coords_field, node_orientation_field, node_radius_field, node_length_field);
  test_spherocylinder_data<stk::topology::NODE, true, false>(
      bulk_data, node_spherocylinder, node_coords_field, node_orientation_field, node_radius_field, node_length_field);
  test_spherocylinder_data<stk::topology::NODE, false, false>(
      bulk_data, node_spherocylinder, node_coords_field, node_orientation_field, node_radius_field, node_length_field);

  test_spherocylinder_data<stk::topology::PARTICLE, true, true>(
      bulk_data, spherocylinder, node_coords_field, elem_orientation_field, elem_radius_field, elem_length_field);
  test_spherocylinder_data<stk::topology::PARTICLE, false, true>(
      bulk_data, spherocylinder, node_coords_field, elem_orientation_field, elem_radius_field, elem_length_field);
  test_spherocylinder_data<stk::topology::PARTICLE, true, false>(
      bulk_data, spherocylinder, node_coords_field, elem_orientation_field, elem_radius_field, elem_length_field);
  test_spherocylinder_data<stk::topology::PARTICLE, false, false>(
      bulk_data, spherocylinder, node_coords_field, elem_orientation_field, elem_radius_field, elem_length_field);
}

TEST(Aggregates, SpherocylinderSegmentData) {
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  stk::mesh::Part& spherocylinder_segment_part =
      meta_data.declare_part_with_topology("segments", stk::topology::BEAM_2);
  stk::mesh::Field<double>& node_coords_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "coords");
  stk::mesh::Field<double>& radius_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "radius");
  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(radius_field, meta_data.universal_part(), 1, nullptr);
  meta_data.commit();

  bulk_data.modification_begin();
  stk::mesh::Entity start_node = bulk_data.declare_node(1);
  stk::mesh::Entity end_node = bulk_data.declare_node(2);
  stk::mesh::Entity segment = bulk_data.declare_element(1, stk::mesh::PartVector{&spherocylinder_segment_part});
  bulk_data.declare_relation(segment, start_node, 0);
  bulk_data.declare_relation(segment, end_node, 1);
  bulk_data.modification_end();

  test_spherocylinder_segment_data<stk::topology::BEAM_2, true>(bulk_data, segment, node_coords_field, radius_field);
  test_spherocylinder_segment_data<stk::topology::BEAM_2, false>(bulk_data, segment, node_coords_field, radius_field);
}

}  // namespace

}  // namespace geom

}  // namespace mundy
