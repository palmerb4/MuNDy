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

/// \file STKSearch.cpp
/// \brief Definition of GenerateNeighborLinkers's STKSearch technique.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data

// Mundy libs
#include <mundy_core/throw_assert.hpp>                  // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                      // for mundy::mesh::BulkData
#include <mundy_shape/compute_aabb/kernels/Sphere.hpp>  // for mundy::shape::compute_aabb::kernels::Sphere
#include <mundy_shape/shapes/Spheres.hpp>               // for mundy::shape::shapes::Spheres

namespace mundy {

namespace linker {

namespace generate_neighbor_linkers {

namespace techniques {

// \name Constructors and destructor
//{

STKSearch::STKSearch(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument, "STKSearch: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(STKSearch::get_valid_fixed_params());

  // Get the field pointers.
  const std::string element_aabb_field_name = valid_fixed_params.get<std::string>("element_aabb_field_name");
  element_aabb_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_aabb_field_name);
  MUNDY_THROW_ASSERT(element_aabb_field_ptr_ != nullptr, std::invalid_argument,
                     "STKSearch: element_aabb_field_ptr_ cannot be a nullptr. Check that the field exists.");

  // Get the valid source and target entity parts.
  const Teuchos::Array<std::string> valid_source_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_source_entity_part_names");
  const Teuchos::Array<std::string> valid_target_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_target_entity_part_names");

  auto parts_from_names = [&meta_data_ptr_](const Teuchos::Array<std::string> &part_names) {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      stk::mesh::Part *part = meta_data_ptr_->get_part(part_name);
      MUNDY_THROW_ASSERT(part != nullptr, std::invalid_argument,
                         "STKSearch: part " << part_name << " cannot be a nullptr. Check that the part exists.");
      parts.push_back(part);
    }
    return parts;
  };

  valid_source_entity_parts_ = parts_from_names(valid_source_entity_part_names);
  valid_target_entity_parts_ = parts_from_names(valid_target_entity_part_names);
}
//}

// \name Setters
//{

void STKSearch::set_mutable_params([[maybe_unused]] const Teuchos::ParameterList &mutable_params) {
}
//}

// \name Getters
//{

std::vector<stk::mesh::Part *> get_valid_source_entity_parts() const {
  return valid_source_entity_parts_;
}

std::vector<stk::mesh::Part *> get_valid_target_entity_parts() const {
  return valid_target_entity_parts_;
}
//}

// \name Actions
//{

void STKSearch::execute(const stk::mesh::Selector &domain_input_selector,
                        const stk::mesh::Selector &range_input_selector) {
  // Step 1: Copy the AABBs, owning procs, and entity keys into a separate contiguous vector for the sources and
  // targets.
  using SearchIdentProc = stk::search::IdentProc<stk::mesh::EntityKey>;
  using SphereIdVector = std::vector<std::pair<stk::search::Sphere<double>, SearchIdentProc>>;
  using BoxIdVector = std::vector<std::pair<stk::search::Box<double>, SearchIdentProc>>;
  using SearchIdPairVector = std::vector<std::pair<SearchIdentProc, SearchIdentProc>>;

  BoxIdVector source_boxes;
  BoxIdVector target_boxes;

  auto fill_box_id_vector = [&bulk_data_ptr_, &meta_data_ptr_, &element_aabb_field_ptr_](
                                const stk::mesh::Selector &selector, BoxIdVector &boxes) {
    const int rank = bulk_data_ptr_->parallel_rank();
    const size_t num_local_elements =
        stk::mesh::count_entities(*bulk_data_ptr_, stk::topology::ELEMENT_RANK, meta_data.locally_owned_part());
    boxes.reserve(num_local_elements);

    const stk::mesh::BucketVector &element_buckets = bulk_data_ptr_->get_buckets(
        stk::topology::ELEMENT_RANK, stk::mesh::Selector(meta_data.locally_owned_part()) & selector);
    for (size_t bucket_idx = 0; bucket_idx < element_buckets.size(); ++bucket_idx) {
      stk::mesh::Bucket &element_bucket = *element_buckets[bucket_idx];
      for (size_t elem_idx = 0; elem_idx < element_bucket.size(); ++elem_idx) {
        // Skip invalid entities.
        if (bulk_data_ptr_->is_valid(element_bucket[elem_idx])) {
          stk::mesh::Entity const &element = element_bucket[elem_idx];

          const double *aabb = stk::mesh::field_data(element_aabb_field, element);
          stk::search::Box<double> box(aabb[0], aabb[1], aabb[2], aabb[3], aabb[4], aabb[5]);

          SearchIdentProc search_id(bulk_data_ptr_->entity_key(element), rank);

          boxes.emplace_back(box, search_id);
        }
      }
    }
  };

  fill_box_id_vector(domain_input_selector, source_boxes);
  fill_box_id_vector(range_input_selector, target_boxes);

  // Step 2: Perform the search.
  SearchIdPairVector search_id_pairs;
  stk::search::coarse_search(source_boxes, target_boxes, stk::search::KDTREE, bulk_data_ptr_->parallel(),
                             search_id_pairs);

  // Step 3: Ghost our search results.
  bulk_data_ptr_->modification_begin();
  std::vector<stk::mesh::EntityProc> send_entities;
  for (const auto &search_id_pair : search_id_pairs) {
    stk::mesh::EntityKey source_entity_key = search_id_pair.first.id();
    stk::mesh::EntityKey target_entity_key = search_id_pair.second.id();

    stk::mesh::Entity source_entity = bulk_data_ptr_->get_entity(source_entity_key);
    stk::mesh::Entity target_entity = bulk_data_ptr_->get_entity(target_entity_key);

    bool is_source_owned = bulk_data_ptr_->bucket(source_entity).owned();
    bool is_target_owned = bulk_data_ptr_->bucket(target_entity).owned();
    int source_proc = searchResults[i].first.proc();
    int target_proc = searchResults[i].second.proc();
    if (is_source_owned && !is_target_owned) {
      // Sent the source entity to the target proc.
      send_entities.push_back(std::make_pair(source_entity, target_proc));
    } else if (!is_source_owned && is_target_owned) {
      // Ghost the target entity to the source proc.
      send_entities.push_back(std::make_pair(target_entity, source_proc));
    }
  }

  stk::mesh::Ghosting &ghosting = bulk_data_ptr_->create_ghosting("STKSearchGhosting");
  bulk_data_ptr_->change_ghosting(ghosting, send_entities);
  bulk_data_ptr_->modification_end();

  // Step 4: Create the neighbor linkers.
  bulk_data_ptr_->modification_begin();
  size_t num_linker_to_create = search_id_pairs.size();
  std::vector<size_t> requests(bulk_data_ptr_->mesh_meta_data().entity_rank_count(), 0);
  requests[stk::topology::CONSTRAINT_RANK] = num_linker_to_create;
  std::vector<stk::mesh::Entity> requested_entities;
  bulk_data_ptr_->generate_new_entities(requests, requested_entities);

  for (size_t i = 0; i < num_linker_to_create; ++i) {
    stk::mesh::EntityKey source_entity_key = search_id_pair[i].first.id();
    stk::mesh::EntityKey target_entity_key = search_id_pair[i].second.id();

    stk::mesh::Entity source_entity = bulk_data_ptr_->get_entity(source_entity_key);
    stk::mesh::Entity target_entity = bulk_data_ptr_->get_entity(target_entity_key);
    stk::mesh::Entity linker = requested_entities[i];
    declare_constraint_relations_to_family_tree_with_sharing(bulk_data_ptr_, linker, source_entity, target_entity);
  }
  bulk_data_ptr_->modification_end();
}
//}

}  // namespace techniques

}  // namespace generate_neighbor_linkers

}  // namespace linker

}  // namespace mundy