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
#include <mundy_meta/PartRequirements.hpp>       // for mundy::meta::PartRequirements

namespace mundy {

namespace meta {

namespace {

/*
Parts are defined by their name, rank (or topology), attributes, fields, and sub-parts. By default, PartRequirements
doesn't constrain any of these, but partial requirements are allowed; for example, a PartRequirements object can
constrain part name and rank, but not attributes. A PartRequirements object can also be used to generate a Part, given
that the PartRequirements object is fully specified; here, fully specified only requires that the part_name is set. In
the meantime, PartRequirements can be merged together to create a new PartRequirements object that is the union of the
two PartRequirements objects.

The following tests check that the PartRequirements object is working as expected. The tests are organized into
sections based on the functionality being tested. The sections are as follows:
  -# PartRequirements object construction
  -# PartRequirements object setting
  -# PartRequirements object getting
  -# PartRequirements object deleting
  -# PartRequirements object merging
  -# PartRequirements object declaring
*/

//! \name PartRequirements object construction
//@{

TEST(PartRequirementsConstruction, IsDefaultConstructible) {
  // Check that PartRequirements is default constructible
  ASSERT_NO_THROW(PartRequirements part_reqs);
}

TEST(PartRequirementsConstruction, IsConstructibleWithPartName) {
  // Check that PartRequirements is constructible with a part name
  ASSERT_NO_THROW(PartRequirements part_reqs("part_name"));
}

TEST(PartRequirementsConstruction, IsConstructibleWithPartNameAndRank) {
  // Check that PartRequirements is constructible with a part name and rank
  ASSERT_NO_THROW(PartRequirements part_reqs("part_name", stk::topology::NODE_RANK));
}

TEST(PartRequirementsConstruction, IsConstructibleWithPartNameAndTopology) {
  // Check that PartRequirements is constructible with a part name and topology
  ASSERT_NO_THROW(PartRequirements part_reqs("part_name", stk::topology::NODE));
}
//@}

//! \name PartRequirements object setting tests
//@{

TEST(PartRequirementsSetters, IsNameAndRankSettable) {
  // Check that the setters work.
  // Note: PartRequirements is fully specified if the part name is set, and will throw an exception you attempt to set
  // the part topology when the rank is already set.
  const std::string part_name = "part_name";
  const stk::topology::rank_t part_rank = stk::topology::NODE_RANK;
  const stk::topology::topology_t part_topology = stk::topology::NODE;
  PartRequirements part_reqs;
  EXPECT_FALSE(part_reqs.constrains_part_name());
  EXPECT_FALSE(part_reqs.constrains_part_topology());
  EXPECT_FALSE(part_reqs.constrains_part_rank());
  EXPECT_FALSE(part_reqs.is_fully_specified());
  part_reqs.set_part_name(part_name);
  EXPECT_TRUE(part_reqs.constrains_part_name());
  EXPECT_TRUE(part_reqs.is_fully_specified());
  part_reqs.set_part_rank(part_rank);
  ASSERT_TRUE(part_reqs.constrains_part_rank());
  EXPECT_THROW(part_reqs.set_part_topology(part_topology), std::logic_error);
}

TEST(PartRequirementsSetters, IsNameAndTopologySettable) {
  // Check that the setters work.
  // Note: PartRequirements is fully specified if the part name is set, and will throw an exception you attempt to set
  // the part rank when the topology is already set.
  const std::string part_name = "part_name";
  const stk::topology::rank_t part_rank = stk::topology::NODE_RANK;
  const stk::topology::topology_t part_topology = stk::topology::NODE;
  PartRequirements part_reqs;
  EXPECT_FALSE(part_reqs.constrains_part_name());
  EXPECT_FALSE(part_reqs.constrains_part_topology());
  EXPECT_FALSE(part_reqs.constrains_part_rank());
  EXPECT_FALSE(part_reqs.is_fully_specified());
  part_reqs.set_part_name(part_name);
  EXPECT_TRUE(part_reqs.constrains_part_name());
  EXPECT_TRUE(part_reqs.is_fully_specified());
  part_reqs.set_part_topology(part_topology);
  ASSERT_TRUE(part_reqs.constrains_part_topology());
  EXPECT_THROW(part_reqs.set_part_rank(part_rank), std::logic_error);
}

TEST(PartRequirementsSetters, AddFieldReqs) {
  // Check that field requirements can be added.

  // Create a dummy field requirements object.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs_ptr = std::make_shared<FieldRequirements<ExampleFieldType>>(field_name, field_rank, field_dimension,
                                                                              field_min_number_of_states);

  // Create a PartRequirements object and add the field requirements.
  PartRequirements part_reqs;
  part_reqs.set_part_name("part_name");
  part_reqs.add_field_reqs(field_reqs_ptr);
  // TODO(palmerb4): Add a getter for the field requirements and check that they are set correctly.
}

TEST(PartRequirementsSetters, AddSubpartRequirements) {
  // Check that subparts can be added.

  // Create a dummy subpart requirements.
  auto subpart_reqs_ptr = std::make_shared<PartRequirements>("subpart_name");

  // Create a PartRequirements object and add the subpart.
  auto part_reqs_ptr = std::make_shared<PartRequirements>("part_name");
  part_reqs_ptr->add_subpart_reqs(subpart_reqs_ptr);
  // TODO(palmerb4): Add a getter for the subpart requirements and check that they are set correctly.
}

TEST(PartRequirementsSetters, AddPartAttribute) {
  // Check that part attributes can be added.

  // Create a dummy part attribute.
  std::any part_attribute = 3.14159265358979323846;

  // Create a PartRequirements object and add the part attribute.
  PartRequirements part_reqs;
  part_reqs.set_part_name("part_name");
  part_reqs.add_part_attribute(part_attribute);

  // Check that the part attribute was added correctly.
  // TODO(palmerb4): Add a getter for the part attributes and check that they are set correctly.
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

TEST(PartRequirementsSetters, AddPartAttributeWithoutCopy) {
  // Check that part attributes can be added with perfect forwarding.

  // Create an uncopiable attribute.
  // Note, std::any requires that the element stored within it is copyable.
  // So, we must wrap the uncopiable object in a std::shared_ptr.
  CountCopiesStruct::num_copies = 0;
  std::any uncopiable_attribute = std::make_shared<CountCopiesStruct>();

  // Create a PartRequirements object and add the part attribute.
  PartRequirements part_reqs;
  part_reqs.set_part_name("part_name");
  part_reqs.add_part_attribute(std::move(uncopiable_attribute));

  // Check that the part attribute was added correctly.
  // TODO(palmerb4): Add a getter for the part attributes and check that they are set correctly.
}
//@}

//! \name PartRequirements object getting tests
//@{

TEST(PartRequirementsGetters, IsNameAndRankGettable) {
  // Check that the getters work.
  const std::string part_name = "part_name";
  const stk::topology::rank_t part_rank = stk::topology::NODE_RANK;
  PartRequirements part_reqs(part_name, part_rank);
  EXPECT_EQ(part_name, part_reqs.get_part_name());
  EXPECT_EQ(part_rank, part_reqs.get_part_rank());
}

TEST(PartRequirementsGetters, IsNameAndTopologyGettable) {
  // Check that the getters work.
  const std::string part_name = "part_name";
  const stk::topology part_topology = stk::topology::NODE;
  PartRequirements part_reqs(part_name, part_topology);
  EXPECT_EQ(part_name, part_reqs.get_part_name());
  EXPECT_EQ(part_topology, part_reqs.get_part_topology());
}
//@}

//! \name PartRequirements object deleting tests
//@{

TEST(PartRequirementsDeleters, DeletersWorkProperly) {
  // Check that the deleters work.
  const std::string part_name = "part_name";
  const stk::topology::rank_t part_rank = stk::topology::NODE_RANK;
  const stk::topology part_topology = stk::topology::NODE;
  PartRequirements part_reqs;
  part_reqs.set_part_name(part_name);
  part_reqs.set_part_rank(part_rank);
  ASSERT_TRUE(part_reqs.constrains_part_name());
  ASSERT_TRUE(part_reqs.constrains_part_rank());
  part_reqs.delete_part_name();
  part_reqs.delete_part_rank();
  EXPECT_FALSE(part_reqs.constrains_part_name());
  EXPECT_FALSE(part_reqs.constrains_part_rank());
  EXPECT_FALSE(part_reqs.is_fully_specified());
  part_reqs.set_part_topology(part_topology);
  ASSERT_TRUE(part_reqs.constrains_part_topology());
  part_reqs.delete_part_topology();
  EXPECT_FALSE(part_reqs.constrains_part_topology());
}
//@}

//! \name PartRequirements object merging tests
//@{

TEST(PartRequirementsMerge, IsNameMergable) {
  // Check that the merge function works when both parts have the same name.
  const std::string part_name = "part_name";
  auto part_reqs1_ptr = std::make_shared<PartRequirements>(part_name);
  auto part_reqs2_ptr = std::make_shared<PartRequirements>(part_name);
  part_reqs1_ptr->merge(part_reqs2_ptr);
  EXPECT_TRUE(part_reqs1_ptr->constrains_part_name());
}

TEST(PartRequirementsMerge, IsRankMergable) {
  // Check that the merge function works when both parts have the same rank.
  const stk::topology::rank_t part_rank = stk::topology::NODE_RANK;
  auto part_reqs1_ptr = std::make_shared<PartRequirements>();
  auto part_reqs2_ptr = std::make_shared<PartRequirements>();
  part_reqs1_ptr->set_part_rank(part_rank);
  part_reqs2_ptr->set_part_rank(part_rank);
  part_reqs1_ptr->merge(part_reqs2_ptr);
  EXPECT_TRUE(part_reqs1_ptr->constrains_part_rank());
}

TEST(PartRequirementsMerge, IsTopologyMergable) {
  // Check that the merge function works when both parts have the same topology.
  const stk::topology part_topology = stk::topology::NODE;
  auto part_reqs1_ptr = std::make_shared<PartRequirements>();
  auto part_reqs2_ptr = std::make_shared<PartRequirements>();
  part_reqs1_ptr->set_part_topology(part_topology);
  part_reqs2_ptr->set_part_topology(part_topology);
  part_reqs1_ptr->merge(part_reqs2_ptr);
  EXPECT_TRUE(part_reqs1_ptr->constrains_part_topology());
}

TEST(PartRequirementsMerge, IsNameAndRankMergable) {
  // Check that the merge function works when both parts have the same name and rank.
  const std::string part_name = "part_name";
  const stk::topology::rank_t part_rank = stk::topology::NODE_RANK;
  auto part_reqs1_ptr = std::make_shared<PartRequirements>(part_name, part_rank);
  auto part_reqs2_ptr = std::make_shared<PartRequirements>(part_name, part_rank);
  part_reqs1_ptr->merge(part_reqs1_ptr);
  EXPECT_TRUE(part_reqs1_ptr->constrains_part_name());
  EXPECT_TRUE(part_reqs1_ptr->constrains_part_rank());
}

TEST(PartRequirementsMerge, IsNameAndTopologyMergable) {
  // Check that the merge function works when both parts have the same name and topology.
  const std::string part_name = "part_name";
  const stk::topology part_topology = stk::topology::NODE;
  auto part_reqs1_ptr = std::make_shared<PartRequirements>(part_name, part_topology);
  auto part_reqs2_ptr = std::make_shared<PartRequirements>(part_name, part_topology);
  part_reqs1_ptr->merge(part_reqs2_ptr);
  EXPECT_TRUE(part_reqs1_ptr->constrains_part_name());
  EXPECT_TRUE(part_reqs1_ptr->constrains_part_topology());
}

TEST(PartRequirementsMerge, AreSubpartsMergable) {
  /* Check that the merge function properly merges subparts.
  The setup for this test is as follows:
  part1 (name=part_name)
    subpart1 (name=)
    subpart2 (name=B)
  part2 (name=part_name)
    subpart3 (name=B)
    subpart4 (name=C)
  merged12: part1 merged w/ part2 (name=part_name)
    subpart1 (name=A)
    subpart2 merged w/ subpart3 (name=B)
    subpart3 (name=C)
  */
  auto part_reqs1_ptr = std::make_shared<PartRequirements>("part_name");
  auto part_reqs2_ptr = std::make_shared<PartRequirements>("part_name");
  auto subpart_reqs1_ptr = std::make_shared<PartRequirements>("A");
  auto subpart_reqs2_ptr = std::make_shared<PartRequirements>("B");
  auto subpart_reqs3_ptr = std::make_shared<PartRequirements>("B");
  auto subpart_reqs4_ptr = std::make_shared<PartRequirements>("C");
  part_reqs1_ptr->add_subpart_reqs(subpart_reqs1_ptr);
  part_reqs1_ptr->add_subpart_reqs(subpart_reqs2_ptr);
  part_reqs2_ptr->add_subpart_reqs(subpart_reqs3_ptr);
  part_reqs2_ptr->add_subpart_reqs(subpart_reqs4_ptr);
  part_reqs1_ptr->merge(part_reqs2_ptr);
  // TODO(palmerb4): Use the subpart getters to check that the subparts were merged correctly.
}

TEST(PartRequirementsMerge, AreFieldsMergable) {
  /* Check that the merge function properly merges fields.
  The setup for this test is as follows:
  part1 (name=part_name)
    field1 (name=a, rank=NODE, dimension=3, min_number_of_states=1)
    field2 (name=b, rank=NODE, dimension=3, min_number_of_states=2)
    field3 (name=c, rank=ELEMENT, dimension=3, min_number_of_states=3)
  part2: (name=part_name)
    field4 (name=b, rank=NODE, dimension=3, min_number_of_states=4)
    field5 (name=c, rank=NODE, dimension=3, min_number_of_states=5)
  merged12: part1 merged w/ part2 (name=part_name)
    field1 (name=a, rank=NODE, dimension=3, min_number_of_states=1)
    field2 merged w/ field4 (name=b, rank=NODE, dimension=3, min_number_of_states=4)
    field3 (name=c, rank=ELEMENT, dimension=3, min_number_of_states=3)
    field5 (name=c, rank=NODE, dimension=3, min_number_of_states=5)

  Note, only fields with the same name AND rank will be merged. As a result, field2 will be merged with field 4 (this
  should increase it's minimum number of states to 4), but field3 will not be merged with field 5 because they don't
  have the same rank.
  */

  // Setup the dummy fields.
  using ExampleFieldType = double;

  auto field_reqs1_ptr =
      std::make_shared<FieldRequirements<ExampleFieldType>>("field1", stk::topology::NODE_RANK, 3, 1);
  auto field_reqs2_ptr =
      std::make_shared<FieldRequirements<ExampleFieldType>>("field2", stk::topology::NODE_RANK, 3, 2);
  auto field_reqs3_ptr =
      std::make_shared<FieldRequirements<ExampleFieldType>>("field3", stk::topology::ELEMENT_RANK, 3, 3);
  auto field_reqs4_ptr =
      std::make_shared<FieldRequirements<ExampleFieldType>>("field4", stk::topology::NODE_RANK, 3, 4);
  auto field_reqs5_ptr =
      std::make_shared<FieldRequirements<ExampleFieldType>>("field5", stk::topology::NODE_RANK, 3, 5);

  // Setup the part requirements according to the diagram above.
  auto part_reqs1_ptr = std::make_shared<PartRequirements>("part_name");
  auto part_reqs2_ptr = std::make_shared<PartRequirements>("part_name");
  part_reqs1_ptr->add_field_reqs(field_reqs1_ptr);
  part_reqs1_ptr->add_field_reqs(field_reqs2_ptr);
  part_reqs1_ptr->add_field_reqs(field_reqs3_ptr);
  part_reqs2_ptr->add_field_reqs(field_reqs4_ptr);
  part_reqs2_ptr->add_field_reqs(field_reqs5_ptr);

  // Merge the part requirements and check that the fields were merged correctly.
  part_reqs1_ptr->merge(part_reqs2_ptr);
  // TODO(palmerb4): Use the field getters to check that the fields were merged correctly.
}

TEST(PartRequirementsMerge, ArePartAttributesMergable) {
  /* Check that the merge function properly merges part attributes.
  The setup for this test is as follows:
  part1 (name=part_name)
    attribute1 (type=int)
    attribute2 (type=double)
  part2: (name=part_name)
    attribute3 (type=double)
    attribute4 (type=std::string)
  merged12: part1 merged w/ part2 (name=part_name)
    attribute1 (type=int)
    attribute2 merged w/ attribute3 (type=double)
    attribute4 (type=std::string)

  Note, only attributes with the same name will be merged. As a result, attribute2 will be merged with attribute 4
  (this should increase it's minimum number of states to 4), but attribute3 will not be merged with attribute 5 because
  they don't have the same name.
  */

  // Setup the dummy attributes.
  std::any attribute1 = 1;
  std::any attribute2 = 3.14159265358979323846;
  std::any attribute3 = 3.14159265358979323846;
  std::any attribute4 = "something";

  // Setup the part requirements according to the diagram above.
  auto part_reqs1_ptr = std::make_shared<PartRequirements>("part_name");
  auto part_reqs2_ptr = std::make_shared<PartRequirements>("part_name");

  part_reqs1_ptr->add_part_attribute(attribute1);
  part_reqs1_ptr->add_part_attribute(attribute2);
  part_reqs2_ptr->add_part_attribute(attribute3);
  part_reqs2_ptr->add_part_attribute(attribute4);

  // Merge the mesh requirements and check that the attributes were merged correctly.
  part_reqs1_ptr->merge(part_reqs2_ptr);
  // TODO(palmerb4): Use the attribute getters to check that the attributes were merged correctly.
}

TEST(PartRequirementsMerge, AreSubpartsAndTheirFieldsMergable) {
  /* Check that the merge function properly merges subparts and their fields.
  The setup for this test is as follows:
  part1: (name=part_name)
    subpart1 (name=A)
      field1 (name=a, rank=NODE, dimension=3, min_number_of_states=1)
    subpart2 (name=B)
      field2 (name=b, rank=NODE, dimension=3, min_number_of_states=2)
      field3 (name=c, rank=ELEMENT, dimension=3, min_number_of_states=3)
  part2: (name=part_name)
    subpart3 (name=B)
      field4 (name=b, rank=NODE, dimension=3, min_number_of_states=4)
      field5 (name=c, rank=NODE, dimension=3, min_number_of_states=5)
    subpart4 (name=C)
      field6 (name=d, rank=ELEMENT, dimension=3, min_number_of_states=6)
  merged12: part1 merged w/ part2 (name=part_name)
    subpart1 (name=A)
      field1 (name=a, rank=NODE, dimension=3, min_number_of_states=1)
    subpart2 merged w/ subpart3 (name=B)
      field2 merged w/ field4 (name=b, rank=NODE, dimension=3, min_number_of_states=4)
      field3 (name=c, rank=ELEMENT, dimension=3, min_number_of_states=3)
      field5 (name=c, rank=NODE, dimension=3, min_number_of_states=5)
    subpart4 (name=C)
      field6 (name=d, rank=ELEMENT, dimension=3, min_number_of_states=6)
  */

  // Setup the dummy fields.
  using ExampleFieldType = double;
  auto field_reqs1_ptr = std::make_shared<FieldRequirements<ExampleFieldType>>("a", stk::topology::NODE_RANK, 3, 1);
  auto field_reqs2_ptr = std::make_shared<FieldRequirements<ExampleFieldType>>("b", stk::topology::NODE_RANK, 3, 2);
  auto field_reqs3_ptr = std::make_shared<FieldRequirements<ExampleFieldType>>("c", stk::topology::ELEMENT_RANK, 3, 3);
  auto field_reqs4_ptr = std::make_shared<FieldRequirements<ExampleFieldType>>("b", stk::topology::NODE_RANK, 3, 4);
  auto field_reqs5_ptr = std::make_shared<FieldRequirements<ExampleFieldType>>("c", stk::topology::NODE_RANK, 3, 5);
  auto field_reqs6_ptr = std::make_shared<FieldRequirements<ExampleFieldType>>("d", stk::topology::ELEMENT_RANK, 3, 6);

  // Setup the subpart requirements according to the diagram above.
  auto subpart_reqs1_ptr = std::make_shared<PartRequirements>("A");
  auto subpart_reqs2_ptr = std::make_shared<PartRequirements>("B");
  auto subpart_reqs3_ptr = std::make_shared<PartRequirements>("B");
  auto subpart_reqs4_ptr = std::make_shared<PartRequirements>("C");
  subpart_reqs1_ptr->add_field_reqs(field_reqs1_ptr);
  subpart_reqs2_ptr->add_field_reqs(field_reqs2_ptr);
  subpart_reqs2_ptr->add_field_reqs(field_reqs3_ptr);
  subpart_reqs3_ptr->add_field_reqs(field_reqs4_ptr);
  subpart_reqs3_ptr->add_field_reqs(field_reqs5_ptr);
  subpart_reqs4_ptr->add_field_reqs(field_reqs6_ptr);

  // Setup the part requirements according to the diagram above.
  auto part_reqs1_ptr = std::make_shared<PartRequirements>("part_name");
  auto part_reqs2_ptr = std::make_shared<PartRequirements>("part_name");
  part_reqs1_ptr->add_subpart_reqs(subpart_reqs1_ptr);
  part_reqs1_ptr->add_subpart_reqs(subpart_reqs2_ptr);
  part_reqs2_ptr->add_subpart_reqs(subpart_reqs3_ptr);
  part_reqs2_ptr->add_subpart_reqs(subpart_reqs4_ptr);

  // Merge the mesh requirements and check that the attributes were merged correctly.
  part_reqs1_ptr->merge(part_reqs2_ptr);
  // TODO(palmerb4): Use the attribute getters to check that the attributes were merged correctly.
}

TEST(PartRequirementsMerge, AreSubpartsAndTheirAttributesMergable) {
  /* Check that the merge function properly merges subparts and their attributes.
  The setup for this test is as follows:
  part1 (name=part_name)
    subpart1 (name=A)
      attribute1 (type=bool)
    subpart2 (name=B)
      attribute2 (type=int)
      attribute3 (type=double)
  part2 (name=part_name)
    subpart3 (name=B)
      attribute4 (type=double)
      attribute5 (type=std::string)
    subpart4 (name=C)
      attribute6 (type=int)
  merged12
    subpart1 (name=A)
      attribute1 (type=bool)
    subpart2 merged w/ subpart3 (name=B)
      attribute2 (type=int)
      attribute3 merged w/ attribute4 (type=double)
      attribute5 (type=std::string)
    subpart4 (name=C)
      attribute6 (type=int)
  */

  // Setup the dummy attributes.
  std::any attribute1 = false;
  std::any attribute2 = 1;
  std::any attribute3 = 3.14159265358979323846;
  std::any attribute4 = 3.14159265358979323846;
  std::any attribute5 = "something";
  std::any attribute6 = 8675309;

  // Setup the subpart requirements according to the diagram above.
  auto subpart_reqs1_ptr = std::make_shared<PartRequirements>("A");
  auto subpart_reqs2_ptr = std::make_shared<PartRequirements>("B");
  auto subpart_reqs3_ptr = std::make_shared<PartRequirements>("B");
  auto subpart_reqs4_ptr = std::make_shared<PartRequirements>("C");
  subpart_reqs1_ptr->add_part_attribute(attribute1);
  subpart_reqs2_ptr->add_part_attribute(attribute2);
  subpart_reqs2_ptr->add_part_attribute(attribute3);
  subpart_reqs3_ptr->add_part_attribute(attribute4);
  subpart_reqs3_ptr->add_part_attribute(attribute5);
  subpart_reqs4_ptr->add_part_attribute(attribute6);

  // Setup the part requirements according to the diagram above.
  auto part_reqs1_ptr = std::make_shared<PartRequirements>("part_name");
  auto part_reqs2_ptr = std::make_shared<PartRequirements>("part_name");
  part_reqs1_ptr->add_subpart_reqs(subpart_reqs1_ptr);
  part_reqs1_ptr->add_subpart_reqs(subpart_reqs2_ptr);
  part_reqs2_ptr->add_subpart_reqs(subpart_reqs3_ptr);
  part_reqs2_ptr->add_subpart_reqs(subpart_reqs4_ptr);

  // Merge the mesh requirements and check that the attributes were merged correctly.
  part_reqs1_ptr->merge(part_reqs2_ptr);
  // TODO(palmerb4): Use the attribute getters to check that the attributes were merged correctly.
}

TEST(PartRequirementsMerge, MergePropertlyHandlesNullptr) {
  // Check that the merge function properly handles nullptrs. It should be a no-op.

  // Setup the part requirements.
  PartRequirements part_reqs("part_name");

  // Perform the merge.
  ASSERT_NO_THROW(part_reqs.merge(nullptr));
}

TEST(PartRequirementsMerge, MergeProperlyHandlesConflicts) {
  // Check that the merge function throws a logic error if the part name, rank, or topology are different.
  auto part_reqs1_ptr = std::make_shared<PartRequirements>();
  auto part_reqs2_ptr = std::make_shared<PartRequirements>();

  // Throw if name is different.
  part_reqs1_ptr->set_part_name("part_name");
  part_reqs2_ptr->set_part_name("other_part_name");
  EXPECT_THROW(part_reqs1_ptr->merge(part_reqs2_ptr), std::logic_error);
  part_reqs1_ptr->delete_part_name();
  part_reqs2_ptr->delete_part_name();

  // Throw if rank is different.
  part_reqs1_ptr->set_part_rank(stk::topology::ELEMENT_RANK);
  part_reqs2_ptr->set_part_rank(stk::topology::EDGE_RANK);
  EXPECT_THROW(part_reqs1_ptr->merge(part_reqs2_ptr), std::logic_error);
  part_reqs1_ptr->delete_part_rank();
  part_reqs2_ptr->delete_part_rank();

  // Throw if topology is different.
  part_reqs1_ptr->set_part_topology(stk::topology::LINE_2);
  part_reqs2_ptr->set_part_topology(stk::topology::LINE_3);
  EXPECT_THROW(part_reqs1_ptr->merge(part_reqs2_ptr), std::logic_error);
  part_reqs1_ptr->delete_part_topology();
  part_reqs2_ptr->delete_part_topology();
}
//@}

//! \name PartRequirements object declaring tests
//@{

TEST(PartRequirementsDeclare, DeclarePartWithName) {
  // Check that the declare_part_on_mesh function properly declares a part on a mesh.

  // Create a dummy mesh.
  mundy::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  std::unique_ptr<mundy::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Declare a part on the mesh using the PartRequirements object.
  PartRequirements part_reqs("part_name");
  ASSERT_TRUE(part_reqs.is_fully_specified());
  ASSERT_NO_THROW(part_reqs.declare_part_on_mesh(&meta_data));

  // Check that the part was declared on the mesh.
  stk::mesh::Part *part = meta_data.get_part("part_name");
  ASSERT_NE(part, nullptr);
}

TEST(PartRequirementsDeclare, DeclarePartWithRank) {
  // Check that the declare_part_on_mesh function properly declares a part with a rank on a mesh.

  // Create a dummy mesh.
  mundy::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  std::unique_ptr<mundy::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Declare a part on the mesh using the PartRequirements object.
  // Note, you cannot declare a ranked part unless the spatial dimension has been set.
  PartRequirements part_reqs("part_name", stk::topology::NODE_RANK);
  ASSERT_TRUE(part_reqs.is_fully_specified());
  ASSERT_NO_THROW(part_reqs.declare_part_on_mesh(&meta_data));

  // Check that the part was declared on the mesh.
  stk::mesh::Part *part = meta_data.get_part("part_name");
  ASSERT_NE(part, nullptr);
  ASSERT_EQ(part->primary_entity_rank(), stk::topology::NODE_RANK);
}

TEST(PartRequirementsDeclare, DeclarePartWithTopology) {
  // Check that the declare_part_on_mesh function properly declares a part with a topology on a mesh.

  // Create a dummy mesh.
  mundy::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  std::unique_ptr<mundy::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Declare a part on the mesh using the PartRequirements object.
  // Note, you cannot declare a ranked part unless the spatial dimension has been set.
  PartRequirements part_reqs("part_name", stk::topology::NODE);
  ASSERT_TRUE(part_reqs.is_fully_specified());
  ASSERT_NO_THROW(part_reqs.declare_part_on_mesh(&meta_data));

  // Check that the part was declared on the mesh.
  stk::mesh::Part *part = meta_data.get_part("part_name");
  ASSERT_NE(part, nullptr);
  ASSERT_EQ(part->topology(), stk::topology::NODE);
}

TEST(PartRequirementsDeclare, DeclarePartWithNameAndFields) {
  // Check that the declare_part_on_part function properly declares a part with fields on a mesh.

  // Create a dummy mesh.
  mundy::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  std::unique_ptr<mundy::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Create a dummy field requirements object.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs_ptr = std::make_shared<FieldRequirements<ExampleFieldType>>(field_name, field_rank, field_dimension,
                                                                              field_min_number_of_states);
  ASSERT_TRUE(field_reqs_ptr->is_fully_specified());

  // Declare a part on the mesh using the PartRequirements object.
  PartRequirements part_reqs("part_name");
  part_reqs.add_field_reqs(field_reqs_ptr);
  ASSERT_TRUE(part_reqs.is_fully_specified());
  ASSERT_NO_THROW(part_reqs.declare_part_on_mesh(&meta_data));

  // Check that the part and field were declared on the mesh.
  stk::mesh::Part *part = meta_data.get_part("part_name");
  ASSERT_NE(part, nullptr);
  ASSERT_NO_THROW(meta_data.get_field<ExampleFieldType>(field_rank, field_name));
}

TEST(PartRequirementsDeclare, DeclarePartWithNameAndSubparts) {
  // Check that the declare_part_on_part function properly declares a part with fields on a mesh.

  // Create a dummy mesh.
  mundy::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  std::unique_ptr<mundy::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Create a dummy subpart requirements object.
  auto subpart_reqs_ptr = std::make_shared<PartRequirements>("subpart_name");
  ASSERT_TRUE(subpart_reqs_ptr->is_fully_specified());

  // Declare a part on the mesh using the PartRequirements object.
  auto part_reqs_ptr = std::make_shared<PartRequirements>("part_name");
  part_reqs_ptr->add_subpart_reqs(subpart_reqs_ptr);
  ASSERT_TRUE(part_reqs_ptr->is_fully_specified());
  ASSERT_NO_THROW(part_reqs_ptr->declare_part_on_mesh(&meta_data));

  // Check that the part and subpart were declared on the mesh.
  stk::mesh::Part *part = meta_data.get_part("part_name");
  ASSERT_NE(part, nullptr);
  stk::mesh::Part *subpart = meta_data.get_part("subpart_name");
  ASSERT_NE(subpart, nullptr);
}

TEST(PartRequirementsDeclare, DeclarePartWithNameAndAttributes) {
  // Check that the declare_part_on_part function properly declares a part with fields on a mesh.

  // Create a dummy mesh.
  mundy::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  std::unique_ptr<mundy::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Create a dummy attribute requirements object.
  std::any attribute = std::string("something");
  ASSERT_TRUE(attribute.has_value());
  ASSERT_EQ(std::any_cast<std::string>(attribute), "something");

  // Declare a part on the mesh using the PartRequirements object.
  PartRequirements part_reqs("part_name");
  part_reqs.add_part_attribute(attribute);
  ASSERT_TRUE(part_reqs.is_fully_specified());
  ASSERT_NO_THROW(part_reqs.declare_part_on_mesh(&meta_data));

  // Check that the part and attribute were declared on the mesh.
  stk::mesh::Part *part_ptr = meta_data.get_part("part_name");
  ASSERT_NE(part_ptr, nullptr);
  ASSERT_NE(meta_data.get_attribute<std::string>(*part_ptr), nullptr);
  ASSERT_TRUE(*meta_data.get_attribute<std::string>(*part_ptr) == std::any_cast<std::string>(attribute));
}

TEST(PartRequirementsDeclare, DeclarePartWithNameAndFieldsAndSubpartsAndAttributes) {
  /* Check that the declare_part_on_part function properly declares a realistic part on a mesh.
  The setup for this test is as follows:
  part:
    subpart:
      field:
        field_attribute
      subpart_attribute
  */

  // Create a dummy mesh.
  mundy::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  std::unique_ptr<mundy::mesh::BulkData> bulk_data_ptr = builder.create_bulk_data();
  mundy::mesh::MetaData &meta_data = bulk_data_ptr->mesh_meta_data();

  // Create some dummy attributes.
  std::any field_attribute = std::string("something");
  std::any subpart_attribute = std::string("something else");

  // Create a dummy field requirements object.
  using ExampleFieldType = double;
  const std::string field_name = "field_name";
  const stk::topology::rank_t field_rank = stk::topology::NODE_RANK;
  const int field_dimension = 3;
  const int field_min_number_of_states = 2;
  auto field_reqs_ptr = std::make_shared<FieldRequirements<ExampleFieldType>>(field_name, field_rank, field_dimension,
                                                                              field_min_number_of_states);
  field_reqs_ptr->add_field_attribute(field_attribute);
  ASSERT_TRUE(field_reqs_ptr->is_fully_specified());

  // Create a dummy subpart requirements object.
  auto subpart_reqs_ptr = std::make_shared<PartRequirements>("subpart_name");
  subpart_reqs_ptr->add_field_reqs(field_reqs_ptr);
  subpart_reqs_ptr->add_part_attribute(subpart_attribute);
  ASSERT_TRUE(subpart_reqs_ptr->is_fully_specified());

  // Declare a part on the mesh using the PartRequirements object.
  auto part_reqs_ptr = std::make_shared<PartRequirements>("part_name");
  part_reqs_ptr->add_subpart_reqs(subpart_reqs_ptr);
  ASSERT_TRUE(part_reqs_ptr->is_fully_specified());
  ASSERT_NO_THROW(part_reqs_ptr->declare_part_on_mesh(&meta_data));

  // Check that the part, field, subpart, and attributes were declared on the mesh.
  // TODO(palmerb4): Check that the subpart is a subpart of the part.
  stk::mesh::Part *part_ptr = meta_data.get_part("part_name");
  ASSERT_NE(part_ptr, nullptr);
  stk::mesh::Part *subpart_ptr = meta_data.get_part("subpart_name");
  ASSERT_NE(subpart_ptr, nullptr);
  stk::mesh::FieldBase *field_ptr = meta_data.get_field(field_rank, field_name);
  ASSERT_NE(field_ptr, nullptr);
  ASSERT_NE(meta_data.get_attribute<std::string>(*field_ptr), nullptr);
  ASSERT_NE(meta_data.get_attribute<std::string>(*subpart_ptr), nullptr);
  ASSERT_TRUE(*meta_data.get_attribute<std::string>(*field_ptr) == std::any_cast<std::string>(field_attribute));
  ASSERT_TRUE(*meta_data.get_attribute<std::string>(*subpart_ptr) == std::any_cast<std::string>(subpart_attribute));
}

}  // namespace

}  // namespace meta

}  // namespace mundy
