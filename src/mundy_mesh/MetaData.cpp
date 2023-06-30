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

/// \file MetaData.cpp
/// \brief Definition of the MetaData class

// C++ core libs
#include <algorithm>    // for std::max
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <typeindex>    // for std::type_index
#include <utility>      // for std::make_pair

// Trilinos libs
#include <Teuchos_ParameterList.hpp>     // for Teuchos::ParameterList
#include <Teuchos_TestForException.hpp>  // for TEUCHOS_TEST_FOR_EXCEPTION
#include <stk_mesh/base/Field.hpp>       // for stk::mesh::Field
#include <stk_mesh/base/Part.hpp>        // for stk::mesh::Part
#include <stk_topology/topology.hpp>     // for stk::topology

// Mundy libs
#include <mundy_mesh/MetaData.hpp>  // for mundy::mesh::MetaData

namespace mundy {

namespace mesh {

// \name Constructor
//{

MetaData::MetaData() : stk::mesh::MetaData() {
}

MetaData::MetaData(size_t spatial_dimension, const std::vector<std::string> &rank_names)
    : stk::mesh::MetaData(spatial_dimension, rank_names) {
}
//}

// \name Actions
//{

void MetaData::declare_attribute(const stk::mesh::FieldBase &field, const std::any &attribute) {
  std::type_index attribute_type_index = std::type_index(attribute.type());
  const unsigned field_id = field.mesh_meta_data_ordinal();

  const bool field_has_attributes = (field_to_field_attributes_map_.count(field_id) != 0);
  if (field_has_attributes) {
    const bool attribute_is_unique = (field_to_field_attributes_map_[field_id].count(attribute_type_index) == 0);
    TEUCHOS_TEST_FOR_EXCEPTION(attribute_is_unique, std::invalid_argument,
                               "MetaData: An attribute with the same type as the provided attribute already "
                               "exists on the given field.");
  } else {
    field_to_field_attributes_map_.insert(std::make_pair(field_id, std::map<std::type_index, std::any>()));
  }

  field_to_field_attributes_map_[field_id].insert(std::make_pair(attribute_type_index, attribute));
}

void MetaData::declare_attribute(const stk::mesh::Part &part, const std::any &attribute) {
  std::type_index attribute_type_index = std::type_index(attribute.type());
  const unsigned part_id = part.id();

  const bool part_has_attributes = (part_to_part_attributes_map_.count(part_id) != 0);
  if (part_has_attributes) {
    const bool attribute_is_unique = (part_to_part_attributes_map_[part_id].count(attribute_type_index) == 0);
    TEUCHOS_TEST_FOR_EXCEPTION(
        attribute_is_unique, std::invalid_argument,
        "MetaData: An attribute with the same type as the provided attribute already exists on the given part.");
  } else {
    part_to_part_attributes_map_.insert(std::make_pair(part_id, std::map<std::type_index, std::any>()));
  }

  part_to_part_attributes_map_[part_id].insert(std::make_pair(attribute_type_index, attribute));
}

void MetaData::declare_attribute(const std::any &attribute) {
  std::type_index attribute_type_index = std::type_index(attribute.type());

  const bool attribute_is_unique = (mesh_attributes_map_.count(attribute_type_index) == 0);
  TEUCHOS_TEST_FOR_EXCEPTION(
      attribute_is_unique, std::invalid_argument,
      "MetaData: An attribute with the same type as the provided attribute already exists on this mesh.");

  mesh_attributes_map_.insert(std::make_pair(attribute_type_index, attribute));
}
//}

}  // namespace mesh

}  // namespace mundy
