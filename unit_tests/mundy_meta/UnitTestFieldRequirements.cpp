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
#include <mundy_mesh/BulkData.hpp>               // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>            // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>               // for mundy::mesh::MetaData
#include <mundy_meta/FieldRequirements.hpp>      // for mundy::meta::FieldRequirements
#include <mundy_meta/FieldRequirementsBase.hpp>  // for mundy::meta::FieldRequirementsBase

namespace mundy {

namespace meta {

namespace {

/*
Fields are defined by their name, rank, dimension, number of states, type, and attributes. By default, FieldRequirements
only constrains the field's type, but partial requirements are allows; for example, a FieldRequirements object can
constrain field name and rank, but not number of components (aka. dimension). A FieldRequirements object can also be
used to generate a Field, given that the FieldRequirements object is fully specified. In the meantime, FieldRequirements
can be merged together to create a new FieldRequirements object that is the union of the two FieldRequirements objects.

The following tests check that the FieldRequirements object is working as expected. The tests are organized into
sections based on the functionality being tested. The sections are as follows:
  -# FieldRequirements object construction
  -# FieldRequirements object setting
  -# FieldRequirements object getting
  -# FieldRequirements object deleting
  -# FieldRequirements object merging
  -# FieldRequirements object declaring
*/

//! \name FieldRequirements object construction tests
//@{

TEST(FieldRequirementsConstruction, IsDefaultConstructable) {
  // Check that the default constructor works.
  using ExampleFieldType = double;
  ASSERT_NO_THROW(mundy::meta::FieldRequirements<ExampleFieldType>());
  auto field_reqs = mundy::meta::FieldRequirements<ExampleFieldType>();
  EXPECT_FALSE(field_reqs.is_fully_specified());
  EXPECT_FALSE(field_reqs.constrains_field_name());
  EXPECT_FALSE(field_reqs.constrains_field_rank());
  EXPECT_FALSE(field_reqs.constrains_field_dimension());
  EXPECT_FALSE(field_reqs.constrains_field_min_number_of_states());
}
//@}

//! \name FieldRequirements object setting tests
//@{

TEST(FieldRequirementsSetters, IsSettable) {
  // Check that the setters work.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs = mundy::meta::FieldRequirements<ExampleFieldType>();
  EXPECT_FALSE(field_reqs.constrains_field_name());
  EXPECT_FALSE(field_reqs.constrains_field_rank());
  EXPECT_FALSE(field_reqs.constrains_field_dimension());
  EXPECT_FALSE(field_reqs.constrains_field_min_number_of_states());
  EXPECT_FALSE(field_reqs.is_fully_specified());
  field_reqs.set_field_name(field_name);
  EXPECT_TRUE(field_reqs.constrains_field_name());
  EXPECT_FALSE(field_reqs.is_fully_specified());
  field_reqs.set_field_rank(field_rank);
  EXPECT_TRUE(field_reqs.constrains_field_rank());
  EXPECT_FALSE(field_reqs.is_fully_specified());
  field_reqs.set_field_dimension(field_dimension);
  EXPECT_TRUE(field_reqs.constrains_field_dimension());
  EXPECT_FALSE(field_reqs.is_fully_specified());
  field_reqs.set_field_min_number_of_states(field_min_number_of_states);
  EXPECT_TRUE(field_reqs.constrains_field_min_number_of_states());
  EXPECT_TRUE(field_reqs.is_fully_specified());
}

struct CountCopiesStruct {
  CountCopiesStruct() = default;                                    // Default constructable
  CountCopiesStruct(const CountCopiesStruct &) { ++num_copies; }    // Copy constructable
  CountCopiesStruct &operator=(const CountCopiesStruct &) { ++num_copies; return *this; }  // Copy assignable

  static int num_copies;
  int value = 1;
};  // CountCopiesStruct

int CountCopiesStruct::num_copies = 0;

TEST(FieldRequirementsAttributes, AddAttributesWithoutCopy) {
  // Check that the attribute adders work.

  // Create an uncopiable attribute.
  // Note, std::any requires that the element stored within it is copyable.
  // So, we must wrap the uncopiable object in a std::shared_ptr.
  CountCopiesStruct::num_copies = 0;
  std::any uncopiable_attribute = std::make_shared<CountCopiesStruct>();

  // Add the attribute to the FieldRequirements object.
  using ExampleFieldType = double;
  auto field_reqs = mundy::meta::FieldRequirements<ExampleFieldType>();
  ASSERT_NO_THROW(field_reqs.add_field_attribute(std::move(uncopiable_attribute)));
  // TODO(palmerb4): Add an attribute getter to FieldRequirements and check that the attribute was added correctly.
}
//@}

//! \name FieldRequirements object getting tests
//@{

TEST(FieldRequirementsGetters, IsGettable) {
  // Check that the getters work.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs = mundy::meta::FieldRequirements<ExampleFieldType>();
  EXPECT_THROW(field_reqs.get_field_name(), std::logic_error);
  EXPECT_THROW(field_reqs.get_field_rank(), std::logic_error);
  EXPECT_THROW(field_reqs.get_field_dimension(), std::logic_error);
  EXPECT_THROW(field_reqs.get_field_min_number_of_states(), std::logic_error);
  field_reqs.set_field_name(field_name);
  field_reqs.set_field_rank(field_rank);
  field_reqs.set_field_dimension(field_dimension);
  field_reqs.set_field_min_number_of_states(field_min_number_of_states);
  EXPECT_EQ(field_reqs.get_field_name(), field_name);
  EXPECT_EQ(field_reqs.get_field_rank(), field_rank);
  EXPECT_EQ(field_reqs.get_field_dimension(), field_dimension);
  EXPECT_EQ(field_reqs.get_field_min_number_of_states(), field_min_number_of_states);
  EXPECT_EQ(field_reqs.get_field_type_info(), typeid(ExampleFieldType));
}
//@}

//! \name FieldRequirements object deleting tests
//@{

TEST(FieldRequirementsDeleters, DeletersWorkProperly) {
  // Check that the getters work.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs = mundy::meta::FieldRequirements<ExampleFieldType>();
  field_reqs.set_field_name(field_name);
  field_reqs.set_field_rank(field_rank);
  field_reqs.set_field_dimension(field_dimension);
  field_reqs.set_field_min_number_of_states(field_min_number_of_states);
  field_reqs.delete_field_name();
  field_reqs.delete_field_rank();
  field_reqs.delete_field_dimension();
  field_reqs.delete_field_min_number_of_states();
  EXPECT_FALSE(field_reqs.constrains_field_name());
  EXPECT_FALSE(field_reqs.constrains_field_rank());
  EXPECT_FALSE(field_reqs.constrains_field_dimension());
  EXPECT_FALSE(field_reqs.constrains_field_min_number_of_states());
  EXPECT_FALSE(field_reqs.is_fully_specified());
}
//@}

//! \name FieldRequirements object merging tests
//@{

TEST(FieldRequirementsMerge, IsMergeable) {
  // Check that the merge function works.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto main_field_reqs = mundy::meta::FieldRequirements<ExampleFieldType>();
  auto other_field_reqs_ptr = std::make_shared<mundy::meta::FieldRequirements<ExampleFieldType>>();
  main_field_reqs.set_field_name(field_name);
  main_field_reqs.set_field_rank(field_rank);
  other_field_reqs_ptr->set_field_dimension(field_dimension);
  other_field_reqs_ptr->set_field_min_number_of_states(field_min_number_of_states);
  main_field_reqs.merge(other_field_reqs_ptr);
  EXPECT_TRUE(main_field_reqs.is_fully_specified());
  EXPECT_EQ(main_field_reqs.get_field_name(), field_name);
  EXPECT_EQ(main_field_reqs.get_field_rank(), field_rank);
  EXPECT_EQ(main_field_reqs.get_field_dimension(), field_dimension);
  EXPECT_EQ(main_field_reqs.get_field_min_number_of_states(), field_min_number_of_states);
  EXPECT_EQ(main_field_reqs.get_field_type_info(), typeid(ExampleFieldType));
}

TEST(FieldRequirementsMerge, IsMergeableWithVectorOfOtherRequirements) {
  // Check that the merge function works with a vector of other requirements.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto main_field_reqs = mundy::meta::FieldRequirements<ExampleFieldType>();
  auto other_field_reqs_ptr1 = std::make_shared<mundy::meta::FieldRequirements<ExampleFieldType>>();
  auto other_field_reqs_ptr2 = std::make_shared<mundy::meta::FieldRequirements<ExampleFieldType>>();
  auto other_field_reqs_ptr3 = std::make_shared<mundy::meta::FieldRequirements<ExampleFieldType>>();
  auto other_field_reqs_ptr4 = std::make_shared<mundy::meta::FieldRequirements<ExampleFieldType>>();
  other_field_reqs_ptr1->set_field_name(field_name);
  other_field_reqs_ptr2->set_field_rank(field_rank);
  other_field_reqs_ptr3->set_field_dimension(field_dimension);
  other_field_reqs_ptr4->set_field_min_number_of_states(field_min_number_of_states);
  main_field_reqs.merge({other_field_reqs_ptr1, other_field_reqs_ptr2, other_field_reqs_ptr3, other_field_reqs_ptr4});
  EXPECT_TRUE(main_field_reqs.is_fully_specified());
  EXPECT_EQ(main_field_reqs.get_field_name(), field_name);
  EXPECT_EQ(main_field_reqs.get_field_rank(), field_rank);
  EXPECT_EQ(main_field_reqs.get_field_dimension(), field_dimension);
  EXPECT_EQ(main_field_reqs.get_field_min_number_of_states(), field_min_number_of_states);
  EXPECT_EQ(main_field_reqs.get_field_type_info(), typeid(ExampleFieldType));
}

TEST(FieldRequirementsMerge, MergePropertlyHandlesNullptr) {
  // Check that the merge function works with a nullptr. It should do nothing.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto main_field_reqs = mundy::meta::FieldRequirements<ExampleFieldType>();
  std::shared_ptr<mundy::meta::FieldRequirementsBase> other_field_reqs_ptr = nullptr;
  EXPECT_NO_THROW(main_field_reqs.merge(other_field_reqs_ptr));
}

TEST(FieldRequirementsMerge, MergeProperlyHandlesConflicts) {
  // Check that the merge function works with a conflict. It should throw a logic error.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs = mundy::meta::FieldRequirements<ExampleFieldType>();
  field_reqs.set_field_name(field_name);
  field_reqs.set_field_rank(field_rank);
  field_reqs.set_field_dimension(field_dimension);
  field_reqs.set_field_min_number_of_states(field_min_number_of_states);

  // Check that the merge function throws a logic error if the field name, rank, or dimension are different.
  auto other_field_reqs_ptr = std::make_shared<mundy::meta::FieldRequirements<ExampleFieldType>>();
  other_field_reqs_ptr->set_field_name("other_field_name");
  EXPECT_THROW(field_reqs.merge(other_field_reqs_ptr), std::logic_error);
  other_field_reqs_ptr->delete_field_name();

  other_field_reqs_ptr->set_field_rank(stk::topology::EDGE_RANK);
  EXPECT_THROW(field_reqs.merge(other_field_reqs_ptr), std::logic_error);
  other_field_reqs_ptr->delete_field_rank();

  other_field_reqs_ptr->set_field_dimension(4);
  EXPECT_THROW(field_reqs.merge(other_field_reqs_ptr), std::logic_error);
  other_field_reqs_ptr->delete_field_dimension();

  // Notice that the field min number of states is always valid to merge.
  other_field_reqs_ptr->set_field_min_number_of_states(3);
  EXPECT_NO_THROW(field_reqs.merge(other_field_reqs_ptr));
  other_field_reqs_ptr->delete_field_min_number_of_states();
}

TEST(FieldRequirementsMerge, MergeProperlyHandlesDifferentTypes) {
  // Check that the merge function works with a conflict. It should throw a logic error.
  using ExampleFieldType1 = double;
  using ExampleFieldType2 = int;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs = mundy::meta::FieldRequirements<ExampleFieldType1>();
  field_reqs.set_field_name(field_name);
  field_reqs.set_field_rank(field_rank);
  field_reqs.set_field_dimension(field_dimension);
  field_reqs.set_field_min_number_of_states(field_min_number_of_states);

  // Notice that the field type is different. This should result in a logic error during the merge.
  auto other_field_reqs_ptr = std::make_shared<mundy::meta::FieldRequirements<ExampleFieldType2>>();
  other_field_reqs_ptr->set_field_name(field_name);
  other_field_reqs_ptr->set_field_rank(field_rank);
  other_field_reqs_ptr->set_field_dimension(field_dimension);
  other_field_reqs_ptr->set_field_min_number_of_states(field_min_number_of_states);
  EXPECT_THROW(field_reqs.merge(other_field_reqs_ptr), std::logic_error);
}

TEST(FieldRequirementsMerge, MergeProperlyHandlesMinNumStates) {
  // Check that the merge function works with a conflict. It should throw a logic error.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs = mundy::meta::FieldRequirements<ExampleFieldType>();
  field_reqs.set_field_name(field_name);
  field_reqs.set_field_rank(field_rank);
  field_reqs.set_field_dimension(field_dimension);
  field_reqs.set_field_min_number_of_states(field_min_number_of_states);

  // Create a new field requirements with a larger min number of states.
  auto other_field_reqs_ptr = std::make_shared<mundy::meta::FieldRequirements<ExampleFieldType>>();
  other_field_reqs_ptr->set_field_name(field_name);
  other_field_reqs_ptr->set_field_rank(field_rank);
  other_field_reqs_ptr->set_field_dimension(field_dimension);
  other_field_reqs_ptr->set_field_min_number_of_states(field_min_number_of_states + 1);

  // The merge should increase the min number of states to the maximum of the two values.
  field_reqs.merge(other_field_reqs_ptr);
  EXPECT_EQ(field_reqs.get_field_min_number_of_states(), field_min_number_of_states + 1);
  other_field_reqs_ptr->delete_field_min_number_of_states();
  other_field_reqs_ptr->set_field_min_number_of_states(field_min_number_of_states - 1);
  field_reqs.merge(other_field_reqs_ptr);
  EXPECT_EQ(field_reqs.get_field_min_number_of_states(), field_min_number_of_states + 1);
}
//@}

//! \name FieldRequirements object declaring tests
//@{

TEST(FieldRequirementsDeclare, DeclareOnPart) {
  // Check that the field requirements match the declared field.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs = mundy::meta::FieldRequirements<ExampleFieldType>();
  field_reqs.set_field_name(field_name);
  field_reqs.set_field_rank(field_rank);
  field_reqs.set_field_dimension(field_dimension);
  field_reqs.set_field_min_number_of_states(field_min_number_of_states);
  ASSERT_TRUE(field_reqs.is_fully_specified());

  // Create a dummy mesh with an example part.
  mundy::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  std::unique_ptr<mundy::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();
  stk::mesh::Part &example_part = meta_data.declare_part("example_part", field_rank);

  // Declare the field on the example part and validate that the resulting field matches the requirements.
  field_reqs.declare_field_on_part(&bulk_data_ptr->mesh_meta_data(), example_part);
  stk::mesh::Field<ExampleFieldType> *my_field_ptr = meta_data.get_field<ExampleFieldType>(field_rank, field_name);
  ASSERT_NE(my_field_ptr, nullptr);
  EXPECT_EQ(my_field_ptr->name(), field_name);
  EXPECT_EQ(my_field_ptr->entity_rank(), field_rank);
  EXPECT_EQ(my_field_ptr->max_size(), field_dimension);
  EXPECT_EQ(my_field_ptr->number_of_states(), field_min_number_of_states);
}

TEST(FieldRequirementsDeclare, DeclareOnEntireMesh) {
  // Check that the field requirements match the declared field.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs = mundy::meta::FieldRequirements<ExampleFieldType>();
  field_reqs.set_field_name(field_name);
  field_reqs.set_field_rank(field_rank);
  field_reqs.set_field_dimension(field_dimension);
  field_reqs.set_field_min_number_of_states(field_min_number_of_states);
  ASSERT_TRUE(field_reqs.is_fully_specified());

  // Create a dummy mesh with an example part.
  mundy::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  std::unique_ptr<mundy::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Declare the field on the example part and validate that the resulting field matches the requirements.
  field_reqs.declare_field_on_entire_mesh(&bulk_data_ptr->mesh_meta_data());
  stk::mesh::Field<ExampleFieldType> *my_field_ptr = meta_data.get_field<ExampleFieldType>(field_rank, field_name);
  ASSERT_NE(my_field_ptr, nullptr);
  EXPECT_EQ(my_field_ptr->name(), field_name);
  EXPECT_EQ(my_field_ptr->entity_rank(), field_rank);
  EXPECT_EQ(my_field_ptr->max_size(), field_dimension);
  EXPECT_EQ(my_field_ptr->number_of_states(), field_min_number_of_states);
}
//@}

}  // namespace

}  // namespace meta

}  // namespace mundy
