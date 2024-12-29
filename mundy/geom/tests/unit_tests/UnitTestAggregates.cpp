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
  stk::mesh::NgpField<double>& ngp_elem_radius_field = stk::mesh::get_updated_ngp_field<double>(elem_radius_field);
  stk::mesh::NgpField<double>& ngp_node_coords_field = stk::mesh::get_updated_ngp_field<double>(node_coords_field);

  const double shared_radius = 1.0;
  const Point<double> shared_center{0.0, 0.0, 0.0};
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

  // NGP directly
  auto ngp_sphere_data_both_fields1 = create_ngp_sphere_data<double>(ngp_elem_radius_field, ngp_node_coords_field);
  auto ngp_sphere_data_shared_radius1 = create_ngp_sphere_data<double>(shared_radius, ngp_node_coords_field);
  auto ngp_sphere_data_shared_center1 = create_ngp_sphere_data<double>(ngp_elem_radius_field, shared_center);
  auto ngp_sphere_data_both_shared1 = create_ngp_sphere_data<double>(shared_radius, shared_center);

  // NGP via get_updated_ngp_data
  auto ngp_sphere_data_both_fields2 = get_updated_ngp_data(sphere_data_both_fields);
  auto ngp_sphere_data_shared_radius2 = get_updated_ngp_data(sphere_data_shared_radius);
  auto ngp_sphere_data_shared_center2 = get_updated_ngp_data(sphere_data_shared_center);
  auto ngp_sphere_data_both_shared2 = get_updated_ngp_data(sphere_data_both_shared);

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
}

}  // namespace

}  // namespace geom

}  // namespace mundy
