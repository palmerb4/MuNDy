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

void test_aabb_data(stk::mesh::BulkData& bulk_data,  //
                    stk::mesh::Entity aabb_entity,   //
                    stk::mesh::Field<double>& aabb_field) {
  ASSERT_TRUE(bulk_data.is_valid(aabb_entity));

  // The shared data for the aabb
  AABB<double> aabb{Point<double>{1.1, 2.2, 3.3}, Point<double>{4.4, 5.5, 6.6}};

  // Test the regular aabb data to ensure that the stored shared data/fields are as expected
  auto aabb_data = create_aabb_data<double>(bulk_data, aabb_field);
  ASSERT_EQ(&aabb_data.aabb_data(), &aabb_field);

  // Same test for the NGP aabb data
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
  stk::mesh::NgpField<double>& ngp_aabb_field = stk::mesh::get_updated_ngp_field<double>(aabb_field);
  auto ngp_aabb_data = create_ngp_aabb_data<double>(bulk_data, ngp_aabb_field);
  ASSERT_EQ(&ngp_aabb_data.aabb_data(), &ngp_aabb_field);

  // Set the center and radius data for the aabb directly via their fields
  const AABB<double> non_shared_aabb{Point<double>{0.1, 0.2, 0.3}, Point<double>{0.4, 0.5, 0.6}};

  AABB<double> old_aabb = aabb;
  AABB<double> old_non_shared_aabb = non_shared_aabb;
  stk::mesh::field_data(aabb_field, aabb_entity)[0] = non_shared_aabb.min_corner()[0];
  stk::mesh::field_data(aabb_field, aabb_entity)[1] = non_shared_aabb.min_corner()[1];
  stk::mesh::field_data(aabb_field, aabb_entity)[2] = non_shared_aabb.min_corner()[2];
  stk::mesh::field_data(aabb_field, aabb_entity)[3] = non_shared_aabb.max_corner()[0];
  stk::mesh::field_data(aabb_field, aabb_entity)[4] = non_shared_aabb.max_corner()[1];
  stk::mesh::field_data(aabb_field, aabb_entity)[5] = non_shared_aabb.max_corner()[2];

  // Test that the aabb data properly views the updated fields

  // Test that the data is modifiable
  const double add_value = 1.1;
  auto aabb_view = create_aabb_entity_view(aabb_data, aabb_entity);
  ASSERT_NEAR(aabb_view.min_corner()[0], non_shared_aabb.min_corner()[0], 1e-12);
  ASSERT_NEAR(aabb_view.min_corner()[1], non_shared_aabb.min_corner()[1], 1e-12);
  ASSERT_NEAR(aabb_view.min_corner()[2], non_shared_aabb.min_corner()[2], 1e-12);
  ASSERT_NEAR(aabb_view.max_corner()[0], non_shared_aabb.max_corner()[0], 1e-12);
  ASSERT_NEAR(aabb_view.max_corner()[1], non_shared_aabb.max_corner()[1], 1e-12);
  ASSERT_NEAR(aabb_view.max_corner()[2], non_shared_aabb.max_corner()[2], 1e-12);

  aabb_view.min_corner()[0] += add_value;
  aabb_view.min_corner()[1] -= add_value;
  aabb_view.min_corner()[2] *= add_value;
  aabb_view.max_corner()[0] += 2.0 * add_value;
  aabb_view.max_corner()[1] -= 2.0 * add_value;
  aabb_view.max_corner()[2] *= 2.0 * add_value;

  ASSERT_NEAR(stk::mesh::field_data(aabb_field, aabb_entity)[0], old_non_shared_aabb.min_corner()[0] + add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(aabb_field, aabb_entity)[1], old_non_shared_aabb.min_corner()[1] - add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(aabb_field, aabb_entity)[2], old_non_shared_aabb.min_corner()[2] * add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(aabb_field, aabb_entity)[3], old_non_shared_aabb.max_corner()[0] + 2.0 * add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(aabb_field, aabb_entity)[4], old_non_shared_aabb.max_corner()[1] - 2.0 * add_value,
              1e-12);
  ASSERT_NEAR(stk::mesh::field_data(aabb_field, aabb_entity)[5], old_non_shared_aabb.max_corner()[2] * 2.0 * add_value,
              1e-12);

  // Remove the added value
  aabb_view.min_corner()[0] -= add_value;
  aabb_view.min_corner()[1] += add_value;
  aabb_view.min_corner()[2] /= add_value;
  aabb_view.max_corner()[0] -= 2.0 * add_value;
  aabb_view.max_corner()[1] += 2.0 * add_value;
  aabb_view.max_corner()[2] /= 2.0 * add_value;

  // Test that the NGP aabb data properly views the updated fields
  stk::mesh::FastMeshIndex aabb_index = ngp_mesh.fast_mesh_index(aabb_entity);
  auto ngp_aabb_view = create_ngp_aabb_entity_view(ngp_aabb_data, aabb_index);
  ASSERT_NEAR(ngp_aabb_view.min_corner()[0], non_shared_aabb.min_corner()[0], 1e-12);
  ASSERT_NEAR(ngp_aabb_view.min_corner()[1], non_shared_aabb.min_corner()[1], 1e-12);
  ASSERT_NEAR(ngp_aabb_view.min_corner()[2], non_shared_aabb.min_corner()[2], 1e-12);
  ASSERT_NEAR(ngp_aabb_view.max_corner()[0], non_shared_aabb.max_corner()[0], 1e-12);
  ASSERT_NEAR(ngp_aabb_view.max_corner()[1], non_shared_aabb.max_corner()[1], 1e-12);
  ASSERT_NEAR(ngp_aabb_view.max_corner()[2], non_shared_aabb.max_corner()[2], 1e-12);

  // Test that the data is modifiable
  // Add a constant value to the center and radius
  ngp_aabb_view.min_corner()[0] += add_value;
  ngp_aabb_view.min_corner()[1] -= add_value;
  ngp_aabb_view.min_corner()[2] *= add_value;
  ngp_aabb_view.max_corner()[0] += 2.0 * add_value;
  ngp_aabb_view.max_corner()[1] -= 2.0 * add_value;
  ngp_aabb_view.max_corner()[2] *= 2.0 * add_value;
  ASSERT_NEAR(ngp_aabb_field(aabb_index, 0), old_non_shared_aabb.min_corner()[0] + add_value, 1e-12);
  ASSERT_NEAR(ngp_aabb_field(aabb_index, 1), old_non_shared_aabb.min_corner()[1] - add_value, 1e-12);
  ASSERT_NEAR(ngp_aabb_field(aabb_index, 2), old_non_shared_aabb.min_corner()[2] * add_value, 1e-12);
  ASSERT_NEAR(ngp_aabb_field(aabb_index, 3), old_non_shared_aabb.max_corner()[0] + 2.0 * add_value, 1e-12);
  ASSERT_NEAR(ngp_aabb_field(aabb_index, 4), old_non_shared_aabb.max_corner()[1] - 2.0 * add_value, 1e-12);
  ASSERT_NEAR(ngp_aabb_field(aabb_index, 5), old_non_shared_aabb.max_corner()[2] * 2.0 * add_value, 1e-12);
}

TEST(Aggregates, AABBData) {
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  stk::mesh::Part& node_part = meta_data.declare_part("node_aabb_part", stk::topology::NODE_RANK);
  stk::mesh::Part& elem_part = meta_data.declare_part("elem_aabb_part", stk::topology::ELEMENT_RANK);
  stk::mesh::Field<double>& node_aabb_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "aabb");
  stk::mesh::Field<double>& elem_aabb_field = meta_data.declare_field<double>(stk::topology::ELEMENT_RANK, "aabb");

  stk::mesh::put_field_on_mesh(node_aabb_field, node_part, 6, nullptr);
  stk::mesh::put_field_on_mesh(elem_aabb_field, elem_part, 6, nullptr);
  meta_data.commit();

  bulk_data.modification_begin();
  stk::mesh::Entity node = bulk_data.declare_node(1, stk::mesh::PartVector{&node_part});
  stk::mesh::Entity elem = bulk_data.declare_element(1, stk::mesh::PartVector{&elem_part});
  bulk_data.modification_end();

  test_aabb_data(bulk_data, node, node_aabb_field);
  test_aabb_data(bulk_data, elem, elem_aabb_field);
}

}  // namespace

}  // namespace geom

}  // namespace mundy
