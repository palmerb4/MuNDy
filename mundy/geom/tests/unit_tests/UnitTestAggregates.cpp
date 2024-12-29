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
  stk::mesh::Field<double>& elem_radius_field = meta_data.declare_field<double>(stk::topology::ELEMENT_RANK, "radius");
  stk::mesh::Field<double>& node_coords_field =
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "coordinates");
 
  stk::mesh::put_field_on_mesh(elem_radius_field, sphere_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_coords_field, sphere_part, 3, nullptr);
  meta_data.commit(); 
 
  // Test the creation of the SphereData aggregate.
  const double shared_radius = 1.0;
  const Point<double> shared_center{1.1, 2.2, 3.3};
  auto sphere_data_both_fields = create_sphere_data<double>(elem_radius_field, node_coords_field);
  auto sphere_data_shared_radius = create_sphere_data<double>(shared_radius, node_coords_field);
  auto sphere_data_shared_center = create_sphere_data<double>(elem_radius_field, shared_center);
  auto sphere_data_both_shared = create_sphere_data<double>(shared_radius, shared_center);

  ASSERT_EQ(&sphere_data_both_fields.radius, &elem_radius_field);
  ASSERT_EQ(&sphere_data_both_fields.center, &node_coords_field);
  ASSERT_EQ(&sphere_data_shared_radius.radius, &shared_radius);
  ASSERT_EQ(&sphere_data_shared_radius.center, &node_coords_field);
  ASSERT_EQ(&sphere_data_shared_center.radius, &elem_radius_field);
  ASSERT_EQ(&sphere_data_shared_center.center, &shared_center);
  ASSERT_EQ(&sphere_data_both_shared.radius, &shared_radius);
  ASSERT_EQ(&sphere_data_both_shared.center, &shared_center);

  // Test the creation of the NgpSphereData aggregate.
  // directly
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
  stk::mesh::NgpField<double>& ngp_elem_radius_field = stk::mesh::get_updated_ngp_field<double>(elem_radius_field);
  stk::mesh::NgpField<double>& ngp_node_coords_field = stk::mesh::get_updated_ngp_field<double>(node_coords_field);
  auto ngp_sphere_data_both_fields1 = create_ngp_sphere_data<double>(ngp_elem_radius_field, ngp_node_coords_field);
  auto ngp_sphere_data_shared_radius1 = create_ngp_sphere_data<double>(shared_radius, ngp_node_coords_field);
  auto ngp_sphere_data_shared_center1 = create_ngp_sphere_data<double>(ngp_elem_radius_field, shared_center);
  auto ngp_sphere_data_both_shared1 = create_ngp_sphere_data<double>(shared_radius, shared_center);

  // via get_updated_ngp_data
  auto ngp_sphere_data_both_fields2 = get_updated_ngp_data(sphere_data_both_fields);
  auto ngp_sphere_data_shared_radius2 = get_updated_ngp_data(sphere_data_shared_radius);
  auto ngp_sphere_data_shared_center2 = get_updated_ngp_data(sphere_data_shared_center);
  auto ngp_sphere_data_both_shared2 = get_updated_ngp_data(sphere_data_both_shared);

  // Check that aggregate stores references to the fields/variables
  ASSERT_EQ(&ngp_sphere_data_both_fields1.radius, &ngp_elem_radius_field);
  ASSERT_EQ(&ngp_sphere_data_both_fields1.center, &ngp_node_coords_field);
  ASSERT_EQ(&ngp_sphere_data_shared_radius1.radius, &shared_radius);
  ASSERT_EQ(&ngp_sphere_data_shared_radius1.center, &ngp_node_coords_field);
  ASSERT_EQ(&ngp_sphere_data_shared_center1.radius, &ngp_elem_radius_field);
  ASSERT_EQ(&ngp_sphere_data_shared_center1.center, &shared_center);
  ASSERT_EQ(&ngp_sphere_data_both_shared1.radius, &shared_radius);
  ASSERT_EQ(&ngp_sphere_data_both_shared1.center, &shared_center);

  ASSERT_EQ(&ngp_sphere_data_both_fields2.radius, &ngp_elem_radius_field);
  ASSERT_EQ(&ngp_sphere_data_both_fields2.center, &ngp_node_coords_field);
  ASSERT_EQ(&ngp_sphere_data_shared_radius2.radius, &shared_radius);
  ASSERT_EQ(&ngp_sphere_data_shared_radius2.center, &ngp_node_coords_field);
  ASSERT_EQ(&ngp_sphere_data_shared_center2.radius, &ngp_elem_radius_field);
  ASSERT_EQ(&ngp_sphere_data_shared_center2.center, &shared_center);
  ASSERT_EQ(&ngp_sphere_data_both_shared2.radius, &shared_radius);
  ASSERT_EQ(&ngp_sphere_data_both_shared2.center, &shared_center);

  // Test the SphereEntityView.
  bulk_data.modification_begin();
  stk::mesh::Entity sphere = bulk_data.declare_element(1, stk::mesh::PartVector{&sphere_part});
  stk::mesh::Entity node = bulk_data.declare_node(1);
  bulk_data.declare_relation(sphere, node, 0);
  bulk_data.modification_end();

  const double non_shared_radius = 2.0;
  const Point<double> non_shared_center{7.7, 8.8, 9.9};
  stk::mesh::field_data(elem_radius_field, sphere)[0] = non_shared_radius;
  stk::mesh::field_data(node_coords_field, node)[0] = non_shared_center[0];
  stk::mesh::field_data(node_coords_field, node)[1] = non_shared_center[1];
  stk::mesh::field_data(node_coords_field, node)[2] = non_shared_center[2];

  auto sphere_view_both_fields = create_sphere_entity_view(bulk_data, sphere_data_both_fields, sphere);
  auto sphere_view_shared_radius = create_sphere_entity_view(bulk_data, sphere_data_shared_radius, sphere);
  auto sphere_view_shared_center = create_sphere_entity_view(bulk_data, sphere_data_shared_center, sphere);
  auto sphere_view_both_shared = create_sphere_entity_view(bulk_data, sphere_data_both_shared, sphere);

  ASSERT_EQ(sphere_view_both_fields.radius(), non_shared_radius);
  ASSERT_EQ(sphere_view_both_fields.center()[0], non_shared_center[0]);
  ASSERT_EQ(sphere_view_both_fields.center()[1], non_shared_center[1]);
  ASSERT_EQ(sphere_view_both_fields.center()[2], non_shared_center[2]);
  ASSERT_EQ(sphere_view_shared_radius.radius(), shared_radius);
  ASSERT_EQ(sphere_view_shared_radius.center()[0], non_shared_center[0]);
  ASSERT_EQ(sphere_view_shared_radius.center()[1], non_shared_center[1]);
  ASSERT_EQ(sphere_view_shared_radius.center()[2], non_shared_center[2]);
  ASSERT_EQ(sphere_view_shared_center.radius(), non_shared_radius);
  ASSERT_EQ(sphere_view_shared_center.center()[0], shared_center[0]);
  ASSERT_EQ(sphere_view_shared_center.center()[1], shared_center[1]);
  ASSERT_EQ(sphere_view_shared_center.center()[2], shared_center[2]);
  ASSERT_EQ(sphere_view_both_shared.radius(), shared_radius);
  ASSERT_EQ(sphere_view_both_shared.center()[0], shared_center[0]);
  ASSERT_EQ(sphere_view_both_shared.center()[1], shared_center[1]);
  ASSERT_EQ(sphere_view_both_shared.center()[2], shared_center[2]);

  stk::mesh::FastMeshIndex sphere_index = ngp_mesh.fast_mesh_index(sphere);
  auto ngp_sphere_view_both_fields1 = create_ngp_sphere_entity_view(ngp_mesh, ngp_sphere_data_both_fields1, sphere_index);
  auto ngp_sphere_view_shared_radius1 = create_ngp_sphere_entity_view(ngp_mesh, ngp_sphere_data_shared_radius1, sphere_index);
  auto ngp_sphere_view_shared_center1 = create_ngp_sphere_entity_view(ngp_mesh, ngp_sphere_data_shared_center1, sphere_index);
  auto ngp_sphere_view_both_shared1 = create_ngp_sphere_entity_view(ngp_mesh, ngp_sphere_data_both_shared1, sphere_index);

  ASSERT_EQ(ngp_sphere_view_both_fields1.radius(), non_shared_radius);
  ASSERT_EQ(ngp_sphere_view_both_fields1.center()[0], non_shared_center[0]);
  ASSERT_EQ(ngp_sphere_view_both_fields1.center()[1], non_shared_center[1]);
  ASSERT_EQ(ngp_sphere_view_both_fields1.center()[2], non_shared_center[2]);
  ASSERT_EQ(ngp_sphere_view_shared_radius1.radius(), shared_radius);
  ASSERT_EQ(ngp_sphere_view_shared_radius1.center()[0], non_shared_center[0]);
  ASSERT_EQ(ngp_sphere_view_shared_radius1.center()[1], non_shared_center[1]);
  ASSERT_EQ(ngp_sphere_view_shared_radius1.center()[2], non_shared_center[2]);
  ASSERT_EQ(ngp_sphere_view_shared_center1.radius(), non_shared_radius);
  ASSERT_EQ(ngp_sphere_view_shared_center1.center()[0], shared_center[0]);
  ASSERT_EQ(ngp_sphere_view_shared_center1.center()[1], shared_center[1]);
  ASSERT_EQ(ngp_sphere_view_shared_center1.center()[2], shared_center[2]);
  ASSERT_EQ(ngp_sphere_view_both_shared1.radius(), shared_radius);
  ASSERT_EQ(ngp_sphere_view_both_shared1.center()[0], shared_center[0]);
  ASSERT_EQ(ngp_sphere_view_both_shared1.center()[1], shared_center[1]);
  ASSERT_EQ(ngp_sphere_view_both_shared1.center()[2], shared_center[2]);
}

}  // namespace

}  // namespace geom

}  // namespace mundy
