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

// External libs
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

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
  MeshBuilder builder(MPI_COMM_WORLD);
  std::unique_ptr<BulkData> bulk_data_ptr = builder.create_bulk_data();
  MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Create a dummy field.
  const std::string field_name = "field";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const unsigned field_dimension = 1;
  stk::mesh::Field<double> &field = meta_data.declare_field<double>(field_rank, field_name, field_dimension);

  // Create an attribute.
  std::any attribute1 = 3.14;
  std::any attribute2 = std::string("Hello world!");

  // Declare the attribute.
  meta_data.declare_attribute(field, attribute1);
  meta_data.declare_attribute(field, attribute2);

  // Fetch the attribute and check that it is correct.
  ASSERT_NE(meta_data.get_attribute<double>(field), nullptr);
  EXPECT_EQ(*meta_data.get_attribute<double>(field), std::any_cast<double>(attribute1));
  ASSERT_NE(meta_data.get_attribute<double>(field), nullptr);
  EXPECT_EQ(*meta_data.get_attribute<std::string>(field), std::any_cast<std::string>(attribute2));

  // Remove the attribute.
  bool attribute1_successfully_removed = meta_data.remove_attribute<double>(field);
  ASSERT_TRUE(attribute1_successfully_removed);
  bool attribute2_successfully_removed = meta_data.remove_attribute<std::string>(field);
  ASSERT_TRUE(attribute2_successfully_removed);

  // Check that the attribute is gone.
  EXPECT_EQ(meta_data.get_attribute<double>(field), nullptr);
  EXPECT_EQ(meta_data.get_attribute<std::string>(field), nullptr);
}

TEST(MetaDataAttributes, DeclareFetchAndRemovePartAttribute) {
  // Create a dummy mesh.
  MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  std::unique_ptr<BulkData> bulk_data_ptr = builder.create_bulk_data();
  MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Create a dummy part.
  // Note, you cannot declare a ranked part unless the spatial dimension has been set.
  const std::string part_name = "part";
  stk::mesh::Part &part = meta_data.declare_part(part_name, stk::topology::NODE_RANK);

  // Create an attribute.
  std::any attribute1 = 3.14;
  std::any attribute2 = std::string("Hello world!");

  // Declare the attribute.
  meta_data.declare_attribute(part, attribute1);
  meta_data.declare_attribute(part, attribute2);

  // Fetch the attribute and check that it is correct.
  ASSERT_NE(meta_data.get_attribute<double>(part), nullptr);
  EXPECT_EQ(*meta_data.get_attribute<double>(part), std::any_cast<double>(attribute1));
  ASSERT_NE(meta_data.get_attribute<double>(part), nullptr);
  EXPECT_EQ(*meta_data.get_attribute<std::string>(part), std::any_cast<std::string>(attribute2));

  // Remove the attribute.
  bool attribute1_successfully_removed = meta_data.remove_attribute<double>(part);
  ASSERT_TRUE(attribute1_successfully_removed);
  bool attribute2_successfully_removed = meta_data.remove_attribute<std::string>(part);
  ASSERT_TRUE(attribute2_successfully_removed);

  // Check that the attribute is gone.
  EXPECT_EQ(meta_data.get_attribute<double>(part), nullptr);
  EXPECT_EQ(meta_data.get_attribute<std::string>(part), nullptr);
}

TEST(MetaDataAttributes, DeclareFetchAndRemoveMeshAttribute) {
  // Create a dummy mesh.
  MeshBuilder builder(MPI_COMM_WORLD);
  std::unique_ptr<BulkData> bulk_data_ptr = builder.create_bulk_data();
  MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Create an attribute.
  std::any attribute1 = 3.14;
  std::any attribute2 = std::string("Hello world!");

  // Declare the attribute.
  meta_data.declare_attribute(attribute1);
  meta_data.declare_attribute(attribute2);

  // Fetch the attribute and check that it is correct.
  ASSERT_NE(meta_data.get_attribute<double>(), nullptr);
  EXPECT_EQ(*meta_data.get_attribute<double>(), std::any_cast<double>(attribute1));
  ASSERT_NE(meta_data.get_attribute<double>(), nullptr);
  EXPECT_EQ(*meta_data.get_attribute<std::string>(), std::any_cast<std::string>(attribute2));

  // Remove the attribute.
  bool attribute1_successfully_removed = meta_data.remove_attribute<double>();
  ASSERT_TRUE(attribute1_successfully_removed);
  bool attribute2_successfully_removed = meta_data.remove_attribute<std::string>();
  ASSERT_TRUE(attribute2_successfully_removed);

  // Check that the attribute is gone.
  EXPECT_EQ(meta_data.get_attribute<double>(), nullptr);
  EXPECT_EQ(meta_data.get_attribute<std::string>(), nullptr);
}

struct CountCopiesStruct {
  CountCopiesStruct() = default;  // Default constructable
  CountCopiesStruct(const CountCopiesStruct &) {
    ++num_copies;
  }  // Copy constructable
  CountCopiesStruct &operator=(const CountCopiesStruct &) {
    ++num_copies;
    return *this;
  }  // Copy assignable

  static int num_copies;
  int value = 1;
};  // CountCopiesStruct

int CountCopiesStruct::num_copies = 0;

TEST(MetaDataAttributes, DeclareFetchAndRemoveUncopiableFieldAttribute) {
  // Create a dummy mesh.
  MeshBuilder builder(MPI_COMM_WORLD);
  std::unique_ptr<BulkData> bulk_data_ptr = builder.create_bulk_data();
  MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Create a dummy field.
  const std::string field_name = "field";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const unsigned field_dimension = 1;
  stk::mesh::Field<double> &field = meta_data.declare_field<double>(field_rank, field_name, field_dimension);

  // Create an uncopiable attribute.
  // Note, std::any requires that the element stored within it is copyable.
  // So, we must wrap the uncopiable object in a std::shared_ptr.
  CountCopiesStruct::num_copies = 0;
  std::any uncopiable_attribute = std::make_shared<CountCopiesStruct>();

  // Declare the attribute with move symantics.
  meta_data.declare_attribute(field, std::move(uncopiable_attribute));

  // Fetch the attribute and check that it is correct.
  // Note, we can't use EXPECT_EQ because we are trying to avoid copies.
  // Instead, we use EXPECT_TRUE to check that the attribute is not null and that it has the correct value.
  // Note, meta_data.get_attribute returns a pointer to our attribute. So, we must dereference it to get the
  // underlying shared_ptr before fetching the value.
  ASSERT_NE(meta_data.get_attribute<std::shared_ptr<CountCopiesStruct>>(field), nullptr);
  std::shared_ptr<CountCopiesStruct> fetched_attribute =
      *meta_data.get_attribute<std::shared_ptr<CountCopiesStruct>>(field);
  EXPECT_EQ(fetched_attribute->value, 1);

  // Remove the attribute.
  bool attribute_successfully_removed = meta_data.remove_attribute<std::shared_ptr<CountCopiesStruct>>(field);
  ASSERT_TRUE(attribute_successfully_removed);

  // Check that the attribute is gone.
  EXPECT_EQ(meta_data.get_attribute<std::shared_ptr<CountCopiesStruct>>(field), nullptr);

  // Check that the attribute was never copied.
  EXPECT_EQ(CountCopiesStruct::num_copies, 0);
}

TEST(MetaDataAttributes, DeclareFetchAndRemoveUncopiablePartAttribute) {
  // Create a dummy mesh.
  MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  std::unique_ptr<BulkData> bulk_data_ptr = builder.create_bulk_data();
  MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Create a dummy part.
  // Note, you cannot declare a ranked part unless the spatial dimension has been set.
  const std::string part_name = "part";
  stk::mesh::Part &part = meta_data.declare_part(part_name, stk::topology::NODE_RANK);

  // Create an uncopiable attribute.
  // Note, std::any requires that the element stored within it is copyable.
  // So, we must wrap the uncopiable object in a std::shared_ptr.
  CountCopiesStruct::num_copies = 0;
  std::any uncopiable_attribute = std::make_shared<CountCopiesStruct>();

  // Declare the attribute with move symantics.
  meta_data.declare_attribute(part, std::move(uncopiable_attribute));

  // Fetch the attribute and check that it is correct.
  // Note, we can't use EXPECT_EQ because we are trying to avoid copies.
  // Instead, we use EXPECT_TRUE to check that the attribute is not null and that it has the correct value.
  // Note, meta_data.get_attribute returns a pointer to our attribute. So, we must dereference it to get the
  // underlying shared_ptr before fetching the value.
  ASSERT_NE(meta_data.get_attribute<std::shared_ptr<CountCopiesStruct>>(part), nullptr);
  std::shared_ptr<CountCopiesStruct> fetched_attribute =
      *meta_data.get_attribute<std::shared_ptr<CountCopiesStruct>>(part);
  EXPECT_EQ(fetched_attribute->value, 1);

  // Remove the attribute.
  bool attribute_successfully_removed = meta_data.remove_attribute<std::shared_ptr<CountCopiesStruct>>(part);
  ASSERT_TRUE(attribute_successfully_removed);

  // Check that the attribute is gone.
  EXPECT_EQ(meta_data.get_attribute<std::shared_ptr<CountCopiesStruct>>(part), nullptr);

  // Check that the attribute was never copied.
  EXPECT_EQ(CountCopiesStruct::num_copies, 0);
}

TEST(MetaDataAttributes, DeclareFetchAndRemoveUncopiableMeshAttribute) {
  // Create a dummy mesh.
  MeshBuilder builder(MPI_COMM_WORLD);
  std::unique_ptr<BulkData> bulk_data_ptr = builder.create_bulk_data();
  MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Create an uncopiable attribute.
  // Note, std::any requires that the element stored within it is copyable.
  // So, we must wrap the uncopiable object in a std::shared_ptr.
  CountCopiesStruct::num_copies = 0;
  std::any uncopiable_attribute = std::make_shared<CountCopiesStruct>();

  // Declare the attribute with move symantics.
  meta_data.declare_attribute(std::move(uncopiable_attribute));

  // Fetch the attribute and check that it is correct.
  // Note, we can't use EXPECT_EQ because we are trying to avoid copies.
  // Instead, we use EXPECT_TRUE to check that the attribute is not null and that it has the correct value.
  // Note, meta_data.get_attribute returns a pointer to our attribute. So, we must dereference it to get the
  // underlying shared_ptr before fetching the value.
  ASSERT_NE(meta_data.get_attribute<std::shared_ptr<CountCopiesStruct>>(), nullptr);
  std::shared_ptr<CountCopiesStruct> fetched_attribute = *meta_data.get_attribute<std::shared_ptr<CountCopiesStruct>>();
  EXPECT_EQ(fetched_attribute->value, 1);

  // Remove the attribute.
  bool attribute_successfully_removed = meta_data.remove_attribute<std::shared_ptr<CountCopiesStruct>>();
  ASSERT_TRUE(attribute_successfully_removed);

  // Check that the attribute is gone.
  EXPECT_EQ(meta_data.get_attribute<std::shared_ptr<CountCopiesStruct>>(), nullptr);

  // Check that the attribute was never copied.
  EXPECT_EQ(CountCopiesStruct::num_copies, 0);
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
