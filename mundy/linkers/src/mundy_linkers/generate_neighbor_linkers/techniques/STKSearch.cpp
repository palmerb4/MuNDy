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
#include <Teuchos_ParameterList.hpp>      // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>        // for stk::mesh::Field, stl::mesh::field_data
#include <stk_mesh/base/GetEntities.hpp>  // for stk::mesh::count_entities
#include <stk_search/BoundingBox.hpp>     // for stk::search::Box
#include <stk_search/CoarseSearch.hpp>    // for stk::search::coarse_search
#include <stk_search/SearchMethod.hpp>    // for stk::search::KDTREE

// Mundy libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_linkers/Linkers.hpp>     // for mundy::linkers::declare_constraint_relations_to_family_tree_with_sharing
#include <mundy_linkers/generate_neighbor_linkers/techniques/STKSearch.hpp>  // for mundy::linkers::...::STKSearch
#include <mundy_mesh/BulkData.hpp>                                          // for mundy::mesh::BulkData

namespace mundy {

namespace linkers {

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

  // Get the part pointers.
  neighbor_linkers_part_ptr_ = meta_data_ptr_->get_part("NEIGHBOR_LINKERS");
  MUNDY_THROW_ASSERT(neighbor_linkers_part_ptr_ != nullptr, std::invalid_argument,
                     "STKSearch: Expected a part with name 'NEIGHBOR_LINKERS' but part does not exist.");

  const Teuchos::Array<std::string> specialized_neighbor_linkers_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("specialized_neighbor_linkers_part_names");
  const Teuchos::Array<std::string> valid_source_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_source_entity_part_names");
  const Teuchos::Array<std::string> valid_target_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_target_entity_part_names");
  auto parts_from_names = [](mundy::mesh::MetaData &meta_data,
                             const Teuchos::Array<std::string> &part_names) -> std::vector<stk::mesh::Part *> {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      std::cout << "part_name: " << part_name << std::endl;
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_ASSERT(part != nullptr, std::invalid_argument,
                         "STKSearch: Expected a part with name '" << part_name << "' but part does not exist.");
      parts.push_back(part);
    }
    return parts;
  };

  specialized_neighbor_linkers_part_ptrs_ = parts_from_names(*meta_data_ptr_, specialized_neighbor_linkers_part_names);
  valid_source_entity_part_ptrs_ = parts_from_names(*meta_data_ptr_, valid_source_entity_part_names);
  valid_target_entity_part_ptrs_ = parts_from_names(*meta_data_ptr_, valid_target_entity_part_names);
}
//}

// \name Setters
//{

void STKSearch::set_mutable_params([[maybe_unused]] const Teuchos::ParameterList &mutable_params) {
}
//}

// \name Getters
//{

std::vector<stk::mesh::Part *> STKSearch::get_valid_source_entity_parts() const {
  return valid_source_entity_part_ptrs_;
}

std::vector<stk::mesh::Part *> STKSearch::get_valid_target_entity_parts() const {
  return valid_target_entity_part_ptrs_;
}
//}

// \name Actions
//{

void STKSearch::execute(const stk::mesh::Selector &domain_input_selector,
                        const stk::mesh::Selector &range_input_selector) {
  // Step 1: Copy the AABBs, owning procs, and entity keys into a separate contiguous vector for the sources and
  // targets.
  using SearchIdentProc = stk::search::IdentProc<stk::mesh::EntityKey>;
  using BoxIdVector = std::vector<std::pair<stk::search::Box<double>, SearchIdentProc>>;
  using SearchIdPairVector = std::vector<std::pair<SearchIdentProc, SearchIdentProc>>;

  BoxIdVector source_boxes;
  BoxIdVector target_boxes;

  auto fill_box_id_vector = [](mundy::mesh::BulkData &bulk_data, mundy::mesh::MetaData &meta_data,
                               const stk::mesh::Field<double> &element_aabb_field, const stk::mesh::Selector &selector,
                               BoxIdVector &boxes) {
    const int rank = bulk_data.parallel_rank();
    const size_t num_local_elements_in_selector = stk::mesh::count_selected_entities(
        meta_data.locally_owned_part() & selector, bulk_data.buckets(stk::topology::ELEMENT_RANK));
    boxes.reserve(num_local_elements_in_selector);

    const stk::mesh::BucketVector &element_buckets = bulk_data.get_buckets(
        stk::topology::ELEMENT_RANK, stk::mesh::Selector(meta_data.locally_owned_part()) & selector);
    for (size_t bucket_idx = 0; bucket_idx < element_buckets.size(); ++bucket_idx) {
      stk::mesh::Bucket &element_bucket = *element_buckets[bucket_idx];
      for (size_t elem_idx = 0; elem_idx < element_bucket.size(); ++elem_idx) {
        // Skip invalid entities.
        if (bulk_data.is_valid(element_bucket[elem_idx])) {
          stk::mesh::Entity const &element = element_bucket[elem_idx];

          const double *aabb = stk::mesh::field_data(element_aabb_field, element);
          stk::search::Box<double> box(aabb[0], aabb[1], aabb[2], aabb[3], aabb[4], aabb[5]);

          SearchIdentProc search_id(bulk_data.entity_key(element), rank);

          boxes.emplace_back(box, search_id);
        }
      }
    }
  };  // fill_box_id_vector

  fill_box_id_vector(*bulk_data_ptr_, *meta_data_ptr_, *element_aabb_field_ptr_, domain_input_selector, source_boxes);
  fill_box_id_vector(*bulk_data_ptr_, *meta_data_ptr_, *element_aabb_field_ptr_, range_input_selector, target_boxes);

  // Step 2: Perform the search.
  SearchIdPairVector search_id_pairs;
  stk::search::coarse_search(source_boxes, target_boxes, stk::search::KDTREE, bulk_data_ptr_->parallel(),
                             search_id_pairs);

  // Step 3: Ghost our search results.
  bulk_data_ptr_->modification_begin();
  std::vector<stk::mesh::EntityProc> send_entities;
  for (const auto &search_id_pair : search_id_pairs) {
    int source_proc = search_id_pair.first.proc();
    int target_proc = search_id_pair.second.proc();

    stk::mesh::EntityKey source_entity_key = search_id_pair.first.id();
    stk::mesh::EntityKey target_entity_key = search_id_pair.second.id();

    stk::mesh::Entity source_entity = bulk_data_ptr_->get_entity(source_entity_key);
    stk::mesh::Entity target_entity = bulk_data_ptr_->get_entity(target_entity_key);

    bool is_source_owned =
        bulk_data_ptr_->is_valid(source_entity) ? bulk_data_ptr_->bucket(source_entity).owned() : false;
    bool is_target_owned =
        bulk_data_ptr_->is_valid(target_entity) ? bulk_data_ptr_->bucket(target_entity).owned() : false;

    if (is_source_owned && !is_target_owned) {
      // Send the source entity to the target proc.
      send_entities.push_back(std::make_pair(source_entity, target_proc));
    } else if (!is_source_owned && is_target_owned) {
      // Send the target entity to the source proc.
      send_entities.push_back(std::make_pair(target_entity, source_proc));
    }
  }

  stk::mesh::Ghosting &ghosting = bulk_data_ptr_->create_ghosting("STKSearchGhosting");
  bulk_data_ptr_->change_ghosting(ghosting, send_entities);
  bulk_data_ptr_->modification_end();


  // Step 4: Check if a new linker should be created between the pair or not. We do not generate a linker if
  //  1. A linker already exists between the source and target entities. If it does, we add it to the specified neighbor
  //  linker parts instead of creating a new linker.
  //  2. The target entity has a GID greater than or equal to that of the source entity. This is to avoid creating
  //  duplicate linkers and to avoid self-intersections.
  //  3. The source entity is not owned. This also avoids creating duplicate linkers.
  bulk_data_ptr_->modification_begin();
  std::vector<int> new_linker_required(search_id_pairs.size(), true);  // int required for std::partial_sum
  for (size_t i = 0; i < search_id_pairs.size(); ++i) {
    stk::mesh::EntityKey source_entity_key = search_id_pairs[i].first.id();
    stk::mesh::EntityKey target_entity_key = search_id_pairs[i].second.id();

    stk::mesh::Entity source_entity = bulk_data_ptr_->get_entity(source_entity_key);
    stk::mesh::Entity target_entity = bulk_data_ptr_->get_entity(target_entity_key);

    bool is_source_owned =
        bulk_data_ptr_->is_valid(source_entity) ? bulk_data_ptr_->bucket(source_entity).owned() : false;

    // Avoid duplicate linkers and self-intersections.
    if (target_entity_key >= source_entity_key || !is_source_owned) {
      new_linker_required[i] = false;
      continue;
    }

    const auto num_connected_constraint_rank_entities_source =
        bulk_data_ptr_->num_connectivity(source_entity, stk::topology::CONSTRAINT_RANK);
    const auto num_connected_constraint_rank_entities_target =
        bulk_data_ptr_->num_connectivity(target_entity, stk::topology::CONSTRAINT_RANK);

    // If either the source or target entity has no constraint-rank entities, then no linkers can exist between them.
    if (num_connected_constraint_rank_entities_source == 0 || num_connected_constraint_rank_entities_target == 0) {
      continue;
    }

    const stk::mesh::Entity *source_constraint_entities =
        bulk_data_ptr_->begin(source_entity, stk::topology::CONSTRAINT_RANK);
    for (size_t j = 0; j < num_connected_constraint_rank_entities_source; ++j) {
      const stk::mesh::Entity &source_constraint_entity = source_constraint_entities[j];
      const bool is_source_constraint_entity_a_valid_linker =
          bulk_data_ptr_->is_valid(source_constraint_entity) &&
          bulk_data_ptr_->bucket(source_constraint_entity).member(*neighbor_linkers_part_ptr_);

      if (is_source_constraint_entity_a_valid_linker) {
        // The connected constraint-rank entity is a valid linker, is it connected to the target entity?
        const auto num_connected_entities_linker =
            bulk_data_ptr_->num_connectivity(source_constraint_entity, stk::topology::ELEMENT_RANK);
        const stk::mesh::Entity *connected_elements_linker =
            bulk_data_ptr_->begin(source_constraint_entity, stk::topology::ELEMENT_RANK);
        for (size_t k = 0; k < num_connected_entities_linker; ++k) {
          if (connected_elements_linker[k] == target_entity) {
            // Found the linker that connects the source and target entities.
            // Move it into the specialized neighbor linkers parts.
            bulk_data_ptr_->change_entity_parts(source_constraint_entity, specialized_neighbor_linkers_part_ptrs_);
            new_linker_required[i] = false;
            break;
          }
        }
      }
    }
  }

  // Step 5: Create the neighbor linkers.
  // The total number of linkers to create is the sum of the new_linker_required vector.
  // Because the total number of linkers does not match the number of search_id_pairs, we use an offset vector to allow
  // each pair to know its index in the requested_entities vector. This is just the cumulative sum of the
  // new_linker_required boolean vector minus 1.
  size_t num_linkers_to_create = std::reduce(new_linker_required.begin(), new_linker_required.end(), 0);
  std::vector<int> linker_offset(new_linker_required.size());
  std::partial_sum(new_linker_required.begin(), new_linker_required.end(), linker_offset.begin());
  std::transform(linker_offset.begin(), linker_offset.end(), linker_offset.begin(), [](int val) {
    return val - 1;
  });

  std::vector<size_t> requests(bulk_data_ptr_->mesh_meta_data().entity_rank_count(), 0);
  requests[stk::topology::CONSTRAINT_RANK] = num_linkers_to_create;
  std::vector<stk::mesh::Entity> requested_entities;
  bulk_data_ptr_->generate_new_entities(requests, requested_entities);
  for (size_t i = 0; i < requested_entities.size(); ++i) {
    bulk_data_ptr_->change_entity_parts(requested_entities[i], specialized_neighbor_linkers_part_ptrs_);
  }

  for (size_t i = 0; i < search_id_pairs.size(); ++i) {
    if (new_linker_required[i]) {
      stk::mesh::EntityKey source_entity_key = search_id_pairs[i].first.id();
      stk::mesh::EntityKey target_entity_key = search_id_pairs[i].second.id();
      stk::mesh::Entity source_entity = bulk_data_ptr_->get_entity(source_entity_key);
      stk::mesh::Entity target_entity = bulk_data_ptr_->get_entity(target_entity_key);
      stk::mesh::Entity linker_i = requested_entities[linker_offset[i]];

      // Connect the linker to the source and target entities.
      mundy::linkers::declare_constraint_relations_to_family_tree_with_sharing(bulk_data_ptr_, linker_i, source_entity,
                                                                              target_entity);
    }
  }
  bulk_data_ptr_->modification_end();
}
//}

}  // namespace techniques

}  // namespace generate_neighbor_linkers

}  // namespace linkers

}  // namespace mundy