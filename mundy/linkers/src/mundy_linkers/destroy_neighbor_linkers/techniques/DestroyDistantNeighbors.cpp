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

/// \file DestroyDistantNeighbors.cpp
/// \brief Definition of GenerateNeighborLinkers's DestroyDistantNeighbors technique.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>      // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>        // for stk::mesh::Field, stl::mesh::field_data
#include <stk_mesh/base/GetEntities.hpp>  // for stk::mesh::count_entities
#include <stk_search/BoundingBox.hpp>     // for stk::search::Box
#include <stk_search/CoarseSearch.hpp>    // for stk::search::coarse_search
#include <stk_search/SearchMethod.hpp>    // for stk::search::KDTREE

// Mundy libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_linkers/Linkers.hpp>    // for mundy::linkers::declare_constraint_relations_to_family_tree_with_sharing
#include <mundy_linkers/destroy_neighbor_linkers/techniques/DestroyDistantNeighbors.hpp>  // for mundy::linkers::...::DestroyDistantNeighbors
#include <mundy_mesh/BulkData.hpp>                                                        // for mundy::mesh::BulkData
#include <mundy_mesh/utils/DestroyFlaggedEntities.hpp>  // for mundy::mesh::utils::destroy_flagged_entities

namespace mundy {

namespace linkers {

namespace destroy_neighbor_linkers {

namespace techniques {

namespace {

bool aabbs_overlap(const double *aabb_i, const double *aabb_j) {
  // Check overlap in the x dimension
  bool overlap_in_x = aabb_i[3] >= aabb_j[0] && aabb_j[3] >= aabb_i[0];

  // Check overlap in the y dimension
  bool overlap_in_y = aabb_i[4] >= aabb_j[1] && aabb_j[4] >= aabb_i[1];

  // Check overlap in the z dimension
  bool overlap_in_z = aabb_i[5] >= aabb_j[2] && aabb_j[5] >= aabb_i[2];

  // If there is overlap in all three dimensions, the AABBs overlap
  return overlap_in_x && overlap_in_y && overlap_in_z;
}

}  // namespace

// \name Constructors and destructor
//{

DestroyDistantNeighbors::DestroyDistantNeighbors(mundy::mesh::BulkData *const bulk_data_ptr,
                                                 const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument,
                     "DestroyDistantNeighbors: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(DestroyDistantNeighbors::get_valid_fixed_params());

  // Get the field pointers.
  const std::string element_aabb_field_name = valid_fixed_params.get<std::string>("element_aabb_field_name");
  const std::string linker_destroy_flag_field_name =
      valid_fixed_params.get<std::string>("linker_destroy_flag_field_name");
  element_aabb_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_aabb_field_name);
  linker_destroy_flag_field_ptr_ =
      meta_data_ptr_->get_field<int>(stk::topology::CONSTRAINT_RANK, linker_destroy_flag_field_name);

  auto field_exists = [](const stk::mesh::FieldBase *field_ptr, const std::string &field_name) {
    MUNDY_THROW_ASSERT(
        field_ptr != nullptr, std::invalid_argument,
        "DestroyDistantNeighbors: Expected a field with name '" << field_name << "' but field does not exist.");
  };  // field_exists

  field_exists(element_aabb_field_ptr_, element_aabb_field_name);
  field_exists(linker_destroy_flag_field_ptr_, linker_destroy_flag_field_name);

  // Get the part pointers.
  const Teuchos::Array<std::string> valid_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
  const Teuchos::Array<std::string> valid_connected_source_and_target_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_connected_source_and_target_part_names");
  auto parts_from_names = [](mundy::mesh::MetaData &meta_data,
                             const Teuchos::Array<std::string> &part_names) -> std::vector<stk::mesh::Part *> {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      std::cout << "part_name: " << part_name << std::endl;
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_ASSERT(
          part != nullptr, std::invalid_argument,
          "DestroyDistantNeighbors: Expected a part with name '" << part_name << "' but part does not exist.");
      parts.push_back(part);
    }
    return parts;
  };  // parts_from_names

  valid_linker_entity_part_ptrs_ = parts_from_names(*meta_data_ptr_, valid_entity_part_names);
  valid_source_target_entity_part_ptrs_ =
      parts_from_names(*meta_data_ptr_, valid_connected_source_and_target_part_names);
}
//}

// \name Setters
//{

void DestroyDistantNeighbors::set_mutable_params([[maybe_unused]] const Teuchos::ParameterList &mutable_params) {
}
//}

// \name Getters
//{

std::vector<stk::mesh::Part *> DestroyDistantNeighbors::get_valid_entity_parts() const {
  return valid_linker_entity_part_ptrs_;
}
//}

// \name Actions
//{

void DestroyDistantNeighbors::execute(const stk::mesh::Selector &input_selector) {
  // Step 0: Populate the AABB's of our ghosted elements.
  stk::mesh::communicate_field_data(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr_), {element_aabb_field_ptr_});

  // Step 1: Loop over each locally owned linker in the input selector and mark them for destruction if the AABBs of
  // their source and target connected elements don't overlap.
  const stk::mesh::Field<double> &element_aabb_field = *element_aabb_field_ptr_;
  const stk::mesh::Field<int> &linker_destroy_flag_field = *linker_destroy_flag_field_ptr_;
  const stk::mesh::Selector locally_owned_input_selector =
      input_selector & bulk_data_ptr_->mesh_meta_data().locally_owned_part();

  stk::mesh::for_each_entity_run(
      *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_), stk::topology::CONSTRAINT_RANK, locally_owned_input_selector,
      [&element_aabb_field, &linker_destroy_flag_field]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                                        const stk::mesh::Entity &linker) {
        // Get the source and target entities of the linker.
        const stk::mesh::Entity *source_target_elements = bulk_data.begin(linker, stk::topology::ELEMENT_RANK);
        const stk::mesh::Entity &source_entity = source_target_elements[0];
        const stk::mesh::Entity &target_entity = source_target_elements[1];

        // Get the AABBs of the source and target entities.
        const double *source_aabb = stk::mesh::field_data(element_aabb_field, source_entity);
        const double *target_aabb = stk::mesh::field_data(element_aabb_field, target_entity);

        // Mark the linker for destruction if the AABBs don't overlap.
        bool do_aabbs_overlap = aabbs_overlap(source_aabb, target_aabb);
        stk::mesh::field_data(linker_destroy_flag_field, linker)[0] = !do_aabbs_overlap;
      });

  // Step 2: Destroy the linkers marked for destruction.
  bulk_data_ptr_->modification_begin();
  const int value_that_indicates_destruction = 1;
  mundy::mesh::utils::destroy_flagged_entities(*bulk_data_ptr_, stk::topology::CONSTRAINT_RANK,
                                               locally_owned_input_selector, linker_destroy_flag_field,
                                               value_that_indicates_destruction);
  bulk_data_ptr_->modification_end();
}
//}

}  // namespace techniques

}  // namespace destroy_neighbor_linkers

}  // namespace linkers

}  // namespace mundy