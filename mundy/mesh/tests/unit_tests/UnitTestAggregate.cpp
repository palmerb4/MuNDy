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

// C++ core libs
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move, std::pair, std::make_pair
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector

// Mundy libs
#include <mundy_math/Vector3.hpp>      // for mundy::math::Vector3
#include <mundy_mesh/Aggregate.hpp>    // for mundy::mesh::Aggregate
#include <mundy_mesh/BulkData.hpp>     // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>  // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>     // for mundy::mesh::MetaData

namespace mundy {

namespace mesh {

namespace {

// template <typename Radius, typename Center>
// struct move_spheres {
//   /// \brief Apply this functor to a single sphere object
//   void operator()(auto& sphere_view) const {
//     // We'll fetch the index
//     std::size_t i = sphere_view.entity();

//     // We'll fetch the center
//     double& c = sphere_view.template get<Center>(0);

//     // We'll fetch the radius (from radii[i])
//     float& r = sphere_view.template get<Radius>();

//     std::cout << "Entity " << i << ": center = " << c << ", radius = " << r << "\n";

//     // Maybe we update them
//     c += 0.1 * i;
//     r *= 1.01f;
//   }

//   /// \brief Apply this functor to all entities in a sphere aggregate
//   /// Syncs all components to the owning space and marks marks modified components.
//   void apply_to(const auto& sphere_data, const stk::mesh::Selector& subset_selector) {
//     sphere_data.sync_to_device<Radius, Center>();
//     sphere_data.template for_each((*this), subset_selector);
//     sphere_data.modified_on_device<Center>();
//   }
// };

TEST(UnitTestAggregate, BasicUsageIsAsExpected) {
  // Setup
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  stk::mesh::Part& sphere_part = meta_data.declare_part_with_topology("SPHERES", stk::topology::PARTICLE);
  stk::mesh::Field<double>& node_center_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "CENTER");
  stk::mesh::Field<double>& elem_radius_field = meta_data.declare_field<double>(stk::topology::ELEM_RANK, "RADIUS");

  stk::mesh::put_field_on_mesh(node_center_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_radius_field, sphere_part, 1, nullptr);
  meta_data.commit();

  bulk_data.modification_begin();
  stk::mesh::Entity node1 = bulk_data.declare_node(1);
  stk::mesh::Entity elem1 = bulk_data.declare_element(1, stk::mesh::PartVector{&sphere_part});
  bulk_data.declare_relation(elem1, node1, 0);
  bulk_data.modification_end();

  // Populate the center and radius
  math::Vector3<double> expected_center{1.0, 2.0, 3.0};
  double expected_radius = 0.5;
  stk::mesh::field_data(node_center_field, node1)[0] = expected_center[0];
  stk::mesh::field_data(node_center_field, node1)[1] = expected_center[1];
  stk::mesh::field_data(node_center_field, node1)[2] = expected_center[2];
  stk::mesh::field_data(elem_radius_field, elem1)[0] = expected_radius;

  // Create the accessors
  auto center_accessor = Vector3FieldComponent(node_center_field);
  auto radius_accessor = ScalarFieldComponent(elem_radius_field);

  // Fetch the data for the entity via the accessor's operator()
  auto center = center_accessor(elem1);
  double& radius = radius_accessor(node1);
  EXPECT_DOUBLE_EQ(center[0], expected_center[0]);
  EXPECT_DOUBLE_EQ(center[1], expected_center[1]);
  EXPECT_DOUBLE_EQ(center[2], expected_center[2]);
  EXPECT_DOUBLE_EQ(radius, expected_radius);

  // Create an aggregate for the spheres
  const auto collision_sphere_data = make_aggregate<stk::topology::PARTICLE>(bulk_data, sphere_part)
                                         .add_component<CENTER, stk::topology::NODE_RANK>(center_accessor)
                                         .add_component<COLLISION_RADIUS, stk::topology::ELEM_RANK>(radius_accessor);

  // Get an accessor-independent view of the entity's data
  auto sphere_view = collision_sphere_data.get_view(elem1);
  EXPECT_EQ(sphere_view.entity(), elem1);
  EXPECT_EQ(sphere_view.rank(), stk::topology::ELEM_RANK);
  EXPECT_EQ(sphere_view.topology(), stk::topology::PARTICLE);
  unsigned center_node_con_ordinal = 0;
  auto also_center = sphere_view.get<CENTER>(center_node_con_ordinal);
  double& also_radius = sphere_view.get<COLLISION_RADIUS>();
  EXPECT_DOUBLE_EQ(also_center[0], expected_center[0]);
  EXPECT_DOUBLE_EQ(also_center[1], expected_center[1]);
  EXPECT_DOUBLE_EQ(also_center[2], expected_center[2]);
  EXPECT_DOUBLE_EQ(also_radius, expected_radius);

  collision_sphere_data.for_each([&expected_center, &expected_radius, &elem1](auto& other_sphere_view) {
    auto c = other_sphere_view.template get<CENTER>(0);
    double& r = other_sphere_view.template get<COLLISION_RADIUS>();

    // There is only one sphere, so we can perform the same checks as above
    EXPECT_DOUBLE_EQ(c[0], expected_center[0]);
    EXPECT_DOUBLE_EQ(c[1], expected_center[1]);
    EXPECT_DOUBLE_EQ(c[2], expected_center[2]);
    EXPECT_DOUBLE_EQ(r, expected_radius);
    EXPECT_EQ(other_sphere_view.entity(), elem1);
    EXPECT_EQ(other_sphere_view.rank(), stk::topology::ELEM_RANK);
    EXPECT_EQ(other_sphere_view.topology(), stk::topology::PARTICLE);
  });
}

TEST(UnitTestAggregate, CanonicalExample) {
  // Setup
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  stk::mesh::Part& sphere_part = meta_data.declare_part_with_topology("SPHERES", stk::topology::PARTICLE);
  stk::mesh::Field<double>& node_center_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "CENTER");
  stk::mesh::Field<double>& node_force_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "FORCE");
  stk::mesh::Field<double>& node_velocity_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "VELOCITY");

  stk::mesh::Field<double>& elem_radius_field = meta_data.declare_field<double>(stk::topology::ELEM_RANK, "RADIUS");
  stk::mesh::Field<double>& elem_mass_field = meta_data.declare_field<double>(stk::topology::ELEM_RANK, "MASS");

  stk::mesh::put_field_on_mesh(node_center_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(node_force_field, sphere_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_velocity_field, sphere_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_radius_field, sphere_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_mass_field, sphere_part, 1, nullptr);
  meta_data.commit();

  bulk_data.modification_begin();
  unsigned num_spheres = 10;
  for (unsigned i = 0; i < num_spheres; ++i) {
    // STK is 1 indexed for EntityIds
    stk::mesh::Entity node = bulk_data.declare_node(i+1);
    stk::mesh::Entity elem = bulk_data.declare_element(i+1, stk::mesh::PartVector{&sphere_part});
    bulk_data.declare_relation(elem, node, 0);

    // Populate the fields
    vector3_field_data(node_center_field, node).set(1.1 * i, 2.2 * i, 3.3);
    vector3_field_data(node_force_field, node).set(5.0, 6.0, 7.0);
    vector3_field_data(node_velocity_field, node).set(1.0, 2.0, 3.0);
    scalar_field_data(elem_radius_field, elem) = 0.5;
    scalar_field_data(elem_mass_field, elem) = 1.0;
  }
  bulk_data.modification_end();

  // Create the accessors
  auto center_accessor = Vector3FieldComponent(node_center_field);  
  auto force_accessor = Vector3FieldComponent(node_force_field);
  auto velocity_accessor = Vector3FieldComponent(node_velocity_field);
  auto radius_accessor = ScalarFieldComponent(elem_radius_field);
  auto mass_accessor = ScalarFieldComponent(elem_mass_field);

  // Create an aggregate for the spheres
  const auto sphere_data = make_aggregate<stk::topology::PARTICLE>(bulk_data, sphere_part)
                                         .add_component<CENTER, stk::topology::NODE_RANK>(center_accessor)
                                          .add_component<FORCE, stk::topology::NODE_RANK>(force_accessor)
                                          .add_component<VELOCITY, stk::topology::NODE_RANK>(velocity_accessor)
                                          .add_component<COLLISION_RADIUS, stk::topology::ELEM_RANK>(radius_accessor)
                                          .add_component<MASS, stk::topology::ELEM_RANK>(mass_accessor);
 
  // Move the spheres on the CPU
  // Note, data that is not fetched via .get is neither accessed nor modified
  double dt = 1e-5;
  sphere_data.for_each([dt](auto& sphere_view) {
    // Note: The zero is the center node connectivity ordinal.
    //  If you had, say, a BEAM_2 you could use .get<VELOCITY>(0) and .get<VELOCITY>(1)
    //  For the velocity of the left and right nodes, respectively.
    auto c = sphere_view.template get<CENTER>(0);
    auto v = sphere_view.template get<VELOCITY>(0);
    c += dt * v;
  });

  // Same but on GPU
  auto ngp_sphere_data = get_updated_ngp_aggregate(sphere_data);
  
  ngp_sphere_data.sync_to_device<CENTER, VELOCITY>();
  // ngp_sphere_data.for_each(KOKKOS_LAMBDA(auto& sphere_view) {
  //   auto c = sphere_view.template get<CENTER>(0);
  //   auto v = sphere_view.template get<VELOCITY>(0);
  //   c += dt * v;
  // });
  // ngp_sphere_data.modify_on_device<CENTER>();
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
