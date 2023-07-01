// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2023 Flatiron Institute
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

// C++ core libs
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/Field.hpp>         // for stk::mesh::Field
#include <stk_topology/topology.hpp>       // for stk::topology
#include <stk_util/parallel/Parallel.hpp>  // for stk::ParallelMachine

// Mundy libs
#include <mundy_mesh/BulkData.hpp>     // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>  // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>     // for mundy::mesh::MetaData

namespace mundy {

namespace mesh {

namespace {

// Note, STK's MetaData is thoughly tested, so we don't need to test it here. We just need to test our extensions to it.

TEST(MetaDataAttributes, DeclareFetchAndRemoveFieldAttribute) {
  // Create a dummy mesh.
  mundy::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  std::unique_ptr<mundy::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Create a dummy field.
  const std::string field_name = "field";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const unsigned field_dimension = 1;
  stk::mesh::Field<double> &field = meta_data.declare_field<double>(field_rank, field_name, field_dimension);

  // Create an attribute.
  std::any attribute = 3.14;

  // Declare the attribute.
  meta_data.declare_attribute(field, attribute);

  // Fetch the attribute and check that it is correct.
  ASSERT_NE(meta_data.get_attribute<double>(field), nullptr);
  EXPECT_EQ(*meta_data.get_attribute<double>(field), attribute);

  // Remove the attribute.
  bool attribute_successfully_removed = meta_data.remove_attribute<double>(field);
  ASSERT_TRUE(attribute_successfully_removed);

  // Check that the attribute is gone.
  EXPECT_EQ(meta_data.get_attribute<double>(field), nullptr);
}

TEST(MetaDataAttributes, DeclareFetchAndRemovePartAttribute) {
  // Create a dummy mesh.
  mundy::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  std::unique_ptr<mundy::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Create a dummy part.
  const std::string part_name = "part";
  stk::mesh::Part &part = meta_data.declare_part(part_name, stk::topology::NODE_RANK);

  // Create an attribute.
  std::any attribute = 3.14;

  // Declare the attribute.
  meta_data.declare_attribute(part, attribute);

  // Fetch the attribute and check that it is correct.
  ASSERT_NE(meta_data.get_attribute<double>(part), nullptr);
  EXPECT_EQ(*meta_data.get_attribute<double>(part), attribute);

  // Remove the attribute.
  bool attribute_successfully_removed = meta_data.remove_attribute<double>(part);
  ASSERT_TRUE(attribute_successfully_removed);

  // Check that the attribute is gone.
  EXPECT_EQ(meta_data.get_attribute<double>(part), nullptr);
}

TEST(MetaDataAttributes, DeclareFetchAndRemoveMeshAttribute) {
  // Create a dummy mesh.
  mundy::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  std::unique_ptr<mundy::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Create an attribute.
  std::any attribute = 3.14;

  // Declare the attribute.
  meta_data.declare_attribute(attribute);

  // Fetch the attribute and check that it is correct.
  ASSERT_NE(meta_data.get_attribute<double>(), nullptr);
  EXPECT_EQ(*meta_data.get_attribute<double>(), attribute);

  // Remove the attribute.
  bool attribute_successfully_removed = meta_data.remove_attribute<double>();
  ASSERT_TRUE(attribute_successfully_removed);

  // Check that the attribute is gone.
  EXPECT_EQ(meta_data.get_attribute<double>(), nullptr);
}

struct UncopiableStruct {
  UncopiableStruct(const UncopiableStruct &) = delete;
  int value = 1;
};  // UncopiableStruct

TEST(MetaDataAttributes, DeclareFetchAndRemoveUncopiableFieldAttribute) {
  // Create a dummy mesh.
  mundy::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  std::unique_ptr<mundy::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Create a dummy field.
  const std::string field_name = "field";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const unsigned field_dimension = 1;
  stk::mesh::Field<double> &field = meta_data.declare_field<double>(field_rank, field_name, field_dimension);

  // Create an uncopiable attribute.
  std::any uncopiable_attribute = UncopiableStruct();

  // Declare the attribute with move symantics.
  meta_data.declare_attribute(field, std::move(uncopiable_attribute));

  // Fetch the attribute and check that it is correct.
  // Note, we can't use EXPECT_EQ because UncopiableStruct is not copyable.
  // Instead, we use EXPECT_TRUE to check that the attribute is not null and that it has the correct value.
  ASSERT_NE(meta_data.get_attribute<UncopiableStruct>(field), nullptr);
  EXPECT_EQ(meta_data.get_attribute<UncopiableStruct>(field)->value, 1);

  // Remove the attribute.
  bool attribute_successfully_removed = meta_data.remove_attribute<UncopiableStruct>(field);
  ASSERT_TRUE(attribute_successfully_removed);

  // Check that the attribute is gone.
  EXPECT_EQ(meta_data.get_attribute<UncopiableStruct>(field), nullptr);
}

TEST(MetaDataAttributes, DeclareFetchAndRemovePartAttribute) {
  // Create a dummy mesh.
  mundy::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  std::unique_ptr<mundy::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Create a dummy part.
  const std::string part_name = "part";
  stk::mesh::Part &part = meta_data.declare_part(part_name, stk::topology::NODE_RANK);

  // Create an attribute.
  std::any attribute = 3.14;

  // Create an uncopiable attribute.
  std::any uncopiable_attribute = UncopiableStruct();

  // Declare the attribute with move symantics.
  meta_data.declare_attribute(part, std::move(uncopiable_attribute));

  // Fetch the attribute and check that it is correct.
  // Note, we can't use EXPECT_EQ because UncopiableStruct is not copyable.
  // Instead, we use EXPECT_TRUE to check that the attribute is not null and that it has the correct value.
  ASSERT_NE(meta_data.get_attribute<UncopiableStruct>(part), nullptr);
  EXPECT_EQ(meta_data.get_attribute<UncopiableStruct>(part)->value, 1);

  // Remove the attribute.
  bool attribute_successfully_removed = meta_data.remove_attribute<UncopiableStruct>(part);
  ASSERT_TRUE(attribute_successfully_removed);

  // Check that the attribute is gone.
  EXPECT_EQ(meta_data.get_attribute<UncopiableStruct>(part), nullptr);
}

TEST(MetaDataAttributes, DeclareFetchAndRemoveMeshAttribute) {
  // Create a dummy mesh.
  mundy::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  std::unique_ptr<mundy::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Create an attribute.
  std::any attribute = 3.14;

  // Create an uncopiable attribute.
  std::any uncopiable_attribute = UncopiableStruct();

  // Declare the attribute with move symantics.
  meta_data.declare_attribute(std::move(uncopiable_attribute));

  // Fetch the attribute and check that it is correct.
  // Note, we can't use EXPECT_EQ because UncopiableStruct is not copyable.
  // Instead, we use EXPECT_TRUE to check that the attribute is not null and that it has the correct value.
  ASSERT_NE(meta_data.get_attribute<UncopiableStruct>(), nullptr);
  EXPECT_EQ(meta_data.get_attribute<UncopiableStruct>()->value, 1);

  // Remove the attribute.
  bool attribute_successfully_removed = meta_data.remove_attribute<UncopiableStruct>();
  ASSERT_TRUE(attribute_successfully_removed);

  // Check that the attribute is gone.
  EXPECT_EQ(meta_data.get_attribute<UncopiableStruct>(), nullptr);
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
