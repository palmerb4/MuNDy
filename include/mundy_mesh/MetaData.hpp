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

#ifndef MUNDY_MESH_METADATA_HPP_
#define MUNDY_MESH_METADATA_HPP_

/// \file MetaData.hpp
/// \brief Declaration of the MetaData class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <typeindex>    // for std::type_index
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part

namespace mundy {

namespace mesh {

/// \class MetaData
/// \brief A extension of STK's MetaData, better suited for some of Mundy's requirements.
///
/// For now, this extension modifies how attributes are created and stored.
class MetaData : public stk::mesh::MetaData {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Construct a meta data manager to own parts and fields.
  MetaData();

  /// \brief Construct a meta data manager to own parts and fields.
  explicit MetaData(size_t spatial_dimension, const std::vector<std::string> &rank_names = std::vector<std::string>());

  /// \brief Destructor.
  virtual ~MetaData();
  //@}

  //! \name Actions
  //@{

  /// @brief Declare an attribute on the given field.
  /// @param field The field which should contain the given attribute.
  /// @param attribute The given attribute. Must have a unique type not shared by other attributes on the given
  /// field.
  void declare_attribute(const stk::mesh::FieldBase &field, const std::any &attribute);

  /// @brief Declare an attribute on the given part.
  /// @param part The part which should contain the given attribute.
  /// @param attribute The given attribute. Must have a unique type not shared by other attributes on the given part.
  void declare_attribute(const stk::mesh::Part &part, const std::any &attribute);

  /// @brief Declare an attribute on the mesh itself.
  /// @param attribute The given attribute. Must have a unique type not shared by other attributes on the mesh.
  void declare_attribute(const std::any &attribute);

  /// @brief Attempt to remove an attribute from the provided field.
  /// @tparam AttributeTypeToRemove Type of the attribute to attempt to remove.
  /// @param field The given field whose attribute we are trying to remove.
  /// @return A flag indicating if the attribute existed on the given field or not.
  template <typename AttributeTypeToRemove>
  bool remove_attribute(const stk::mesh::FieldBase &field);

  /// @brief Attempt to remove an attribute from the provided part.
  /// @tparam AttributeTypeToRemove Type of the attribute to attempt to remove.
  /// @param part The given part whose attribute we are trying to remove.
  /// @return A flag indicating if the attribute existed on the given part or not.
  template <typename AttributeTypeToRemove>
  bool remove_attribute(const stk::mesh::Part &part);

  /// @brief Attempt to remove an attribute from this mesh.
  /// @tparam AttributeTypeToRemove Type of the attribute to attempt to remove.
  /// @return A flag indicating if the attribute existed on this mesh or not.
  template <typename AttributeTypeToRemove>
  bool remove_attribute();

  /// @brief Attempt to fetch an attribute with the provided type. Will return nullptr if type doesn't exist on the
  /// given field.
  /// @tparam AttributeTypeToFetch The attribute type to fetch.
  /// @param field The given field whose attribute we are trying to fetch.
  /// @return A pointer to the internally maintained attribute.
  template <class AttributeTypeToFetch>
  AttributeTypeToFetch &get_attribute(const stk::mesh::FieldBase &field);

  /// @brief Attempt to fetch an attribute with the provided type. Will return nullptr if type doesn't exist on the
  /// given part.
  /// @tparam AttributeTypeToFetch The attribute type to fetch.
  /// @param part The given part whose attribute we are trying to fetch.
  /// @return A pointer to the internally maintained attribute.
  template <class AttributeTypeToFetch>
  AttributeTypeToFetch &get_attribute(const stk::mesh::Part &part);

  /// @brief Attempt to fetch an attribute with the provided type. Will return nullptr if type doesn't exist on the
  /// current mesh.
  /// @tparam AttributeTypeToFetch The attribute type to fetch.
  /// @return A pointer to the internally maintained attribute.
  template <class AttributeTypeToFetch>
  AttributeTypeToFetch &get_attribute();
  //@}

 private:
  //! \name Internal members
  //@{

  std::map<unsigned, std::map<std::type_index, std::any>> field_to_field_attributes_map_;
  std::map<unsigned, std::map<std::type_index, std::any>> part_to_part_attributes_map_;
  std::map<std::type_index, std::any> mesh_attributes_map_;
  //@}
};  // MetaData

//! \name Template implementations
//@{

// \name Actions
//{

template <typename AttributeTypeToRemove>
bool MetaData::remove_attribute(const stk::mesh::FieldBase &field) {
  std::type_index attribute_type_index = std::type_index(typeid(AttributeTypeToRemove));
  const unsigned field_id = field.mesh_meta_data_ordinal();

  const bool field_has_attributes = (field_to_field_attributes_map_.count(field_id) != 0);
  if (field_has_attributes) {
    const bool attribute_exists = (field_to_field_attributes_map_[field_id].count(attribute_type_index) != 0);
    if (attribute_exists) {
      field_to_field_attributes_map_[field_id].erase(attribute_type_index);
      return true;
    } else {
      // The attribute doesn't exist, nothing to remove.
      return false;
    }
  } else {
    // The field has no attributes, so I'm pretty certain the provided attribute doesn't exist.
    return false;
  }
}

template <typename AttributeTypeToRemove>
bool MetaData::remove_attribute(const stk::mesh::Part &part) {
  std::type_index attribute_type_index = std::type_index(typeid(AttributeTypeToRemove));
  const unsigned part_id = part.id();

  // TODO(palmerb4): Attributes should be inherited. Check if any of our parents are in the list.
  const bool part_has_attributes = (part_to_part_attributes_map_.count(part_id) != 0);
  if (part_has_attributes) {
    const bool attribute_exists = (part_to_part_attributes_map_[part_id].count(attribute_type_index) != 0);
    if (attribute_exists) {
      part_to_part_attributes_map_[part_id].erase(attribute_type_index);
      return true;
    } else {
      // The attribute doesn't exist, nothing to remove.
      return false;
    }
  } else {
    // The part has no attributes, so I'm pretty certain the provided attribute doesn't exist.
    return false;
  }
}

template <typename AttributeTypeToRemove>
bool MetaData::remove_attribute() {
  std::type_index attribute_type_index = std::type_index(typeid(AttributeTypeToRemove));

  const bool attribute_exists = (mesh_attributes_map_.count(attribute_type_index) != 0);
  if (attribute_exists) {
    mesh_attributes_map_.erase(attribute_type_index);
    return true;
  } else {
    // The attribute doesn't exist, nothing to remove.
    return false;
  }
}

template <class AttributeTypeToFetch>
AttributeTypeToFetch &MetaData::get_attribute(const stk::mesh::FieldBase &field) {
  std::type_index attribute_type_index = std::type_index(typeid(AttributeTypeToRemove));
  const unsigned field_id = field.mesh_meta_data_ordinal();

  const bool field_has_attributes = (field_to_field_attributes_map_.count(field_id) != 0);
  if (field_has_attributes) {
    const bool attribute_exists = (field_to_field_attributes_map_[field_id].count(attribute_type_index) != 0);
    if (attribute_exists) {
      // Attempt to cast the attribute to the requested type.
      return std::any_cast<AttributeTypeToFetch &>(field_to_field_attributes_map_[field_id][attribute_type_index]);
    }
  }

  // Attribute doesn't exist. Returning nullptr.
  return nullptr;
}

template <class AttributeTypeToFetch>
AttributeTypeToFetch &MetaData::get_attribute(const stk::mesh::Part &part) {
  std::type_index attribute_type_index = std::type_index(typeid(AttributeTypeToRemove));
  const unsigned part_id = part.id();

  const bool part_has_attributes = (part_to_part_attributes_map_.count(part_id) != 0);
  if (part_has_attributes) {
    const bool attribute_exists = (part_to_part_attributes_map_[part_id].count(attribute_type_index) != 0);
    if (attribute_exists) {
      // Attempt to cast the attribute to the requested type.
      return std::any_cast<AttributeTypeToFetch &>(part_to_part_attributes_map_[part_id][attribute_type_index]);
    }
  }

  // Attribute doesn't exist. Returning nullptr.
  return nullptr;
}

template <class AttributeTypeToFetch>
AttributeTypeToFetch &MetaData::get_attribute() {
  std::type_index attribute_type_index = std::type_index(typeid(AttributeTypeToRemove));

  const bool attribute_exists = (mesh_attributes_map_.count(attribute_type_index) != 0);
  if (attribute_exists) {
    // Attempt to cast the attribute to the requested type.
    return std::any_cast<AttributeTypeToFetch &>(mesh_attributes_map_[attribute_type_index]);
  }

  // Attribute doesn't exist. Returning nullptr.
  return nullptr;
}
//}

//@}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_METADATA_HPP_
