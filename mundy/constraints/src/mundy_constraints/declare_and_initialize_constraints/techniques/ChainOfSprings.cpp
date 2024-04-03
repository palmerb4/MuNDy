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

/// \file ChainOfSprings.cpp
/// \brief Definition of DeclareAndInitConstraints' ChainOfSprings technique.

// C++ core libs
#include <iostream>  // for std::cout, std::endl
#include <memory>    // for std::shared_ptr, std::unique_ptr
#include <string>    // for std::string
#include <vector>    // for std::vector

// External libs
#include <openrand/philox.h>

// Trilinos libs
#include <Teuchos_ParameterList.hpp>      // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>        // for stk::mesh::Field, stl::mesh::field_data
#include <stk_mesh/base/GetEntities.hpp>  // for stk::mesh::count_entities
#include <stk_search/BoundingBox.hpp>     // for stk::search::Box
#include <stk_search/CoarseSearch.hpp>    // for stk::search::coarse_search
#include <stk_search/SearchMethod.hpp>    // for stk::search::KDTREE

// Mundy libs
#include <mundy_constraints/HookeanSprings.hpp>  // for mundy::constraints::HookeanSprings
#include <mundy_constraints/declare_and_initialize_constraints/techniques/ArchlengthCoordinateMapping.hpp>  // for mundy::constraints::...::ArchlengthCoordinateMapping
#include <mundy_constraints/declare_and_initialize_constraints/techniques/ChainOfSprings.hpp>  // for mundy::constraints::...::ChainOfSprings
#include <mundy_core/throw_assert.hpp>                                                         // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>  // for mundy::mesh::BulkData

namespace mundy {

namespace constraints {

namespace declare_and_initialize_constraints {

namespace techniques {

// \name Constructors and destructor
//{

ChainOfSprings::ChainOfSprings(mundy::mesh::BulkData *const bulk_data_ptr_, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr_), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument,
                     "ChainOfSprings: bulk_data_ptr_ cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(ChainOfSprings::get_valid_fixed_params());

  // Get the control parameters.
  generate_hookean_springs_ = valid_fixed_params.get<bool>("generate_hookean_springs");
  generate_angular_springs_ = valid_fixed_params.get<bool>("generate_angular_springs");

  // Get the field pointers.
  const std::string node_coord_field_name = mundy::constraints::HookeanSprings::get_node_coord_field_name();
  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);

  MUNDY_THROW_ASSERT(node_coord_field_ptr_ != nullptr, std::invalid_argument,
                     "ChainOfSprings: node_coord_field_ptr cannot be a nullptr. Check that the field exists.");

  // Get the part pointers.
  auto parts_from_names = [](mundy::mesh::MetaData &meta_data,
                             const Teuchos::Array<std::string> &part_names) -> std::vector<stk::mesh::Part *> {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      std::cout << "part_name: " << part_name << std::endl;
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_ASSERT(part != nullptr, std::invalid_argument,
                         "ChainOfSprings: Expected a part with name '" << part_name << "' but part does not exist.");
      parts.push_back(part);
    }
    return parts;
  };

  if (generate_hookean_springs_) {
    const Teuchos::Array<std::string> hookean_springs_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("hookean_springs_part_names");
    hookean_spring_part_ptrs_ = parts_from_names(*meta_data_ptr_, hookean_springs_part_names);
  }

  if (generate_angular_springs_) {
    const Teuchos::Array<std::string> angular_springs_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("angular_springs_part_names");
    angular_spring_part_ptrs_ = parts_from_names(*meta_data_ptr_, angular_springs_part_names);
  }
}
//}

// \name Setters
//{

void ChainOfSprings::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(ChainOfSprings::get_valid_mutable_params());

  num_nodes_ = valid_mutable_params.get<size_t>("num_nodes");
  num_hookean_springs_ = (generate_hookean_springs_ ? num_nodes_ - 1 : 0);
  num_angular_springs_ = (generate_angular_springs_ ? num_hookean_springs_ + num_nodes_ - 2 : 0);
  coordinate_map_ptr_ = valid_mutable_params.get<std::shared_ptr<ArchlengthCoordinateMapping>>("coordinate_mapping");
}
//}

// \name Getters
//{

stk::mesh::EntityId ChainOfSprings::get_node_id(const size_t &sequential_node_index) const {
  return node_id_start_ + sequential_node_index;
}

stk::mesh::EntityId ChainOfSprings::get_hookean_spring_id(const size_t &sequential_hookean_spring_index) const {
  return element_id_start_ + sequential_hookean_spring_index;
}

stk::mesh::EntityId ChainOfSprings::get_angular_spring_id(const size_t &sequential_angular_spring_index) const {
  return element_id_start_ + num_hookean_springs_ + sequential_angular_spring_index;
}

stk::mesh::Entity ChainOfSprings::get_node(const size_t &sequential_node_index) const {
  return bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, get_node_id(sequential_node_index));
}

stk::mesh::Entity ChainOfSprings::get_hookean_spring(const size_t &sequential_hookean_spring_index) const {
  return bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK,
                                    get_hookean_spring_id(sequential_hookean_spring_index));
}

stk::mesh::Entity ChainOfSprings::get_angular_spring(const size_t &sequential_angular_spring_index) const {
  return bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK,
                                    get_angular_spring_id(sequential_angular_spring_index));
}
//}

// \name Actions
//{

void ChainOfSprings::execute() {
  // Create the springs and their connected nodes, distributing the work across the ranks.
  const size_t rank = bulk_data_ptr_->parallel_rank();
  const size_t nodes_per_rank = num_nodes_ / bulk_data_ptr_->parallel_size();
  const size_t remainder = num_nodes_ % bulk_data_ptr_->parallel_size();
  const size_t start_node_index = rank * nodes_per_rank + std::min(rank, remainder);
  const size_t end_node_index = start_node_index + nodes_per_rank + (rank < remainder ? 1 : 0);

  // Concatenate the spring part pointers.
  std::vector<stk::mesh::Part *> spring_part_ptrs_;
  spring_part_ptrs_.insert(spring_part_ptrs_.end(), hookean_spring_part_ptrs_.begin(), hookean_spring_part_ptrs_.end());
  spring_part_ptrs_.insert(spring_part_ptrs_.end(), angular_spring_part_ptrs_.begin(), angular_spring_part_ptrs_.end());

  bulk_data_ptr_->modification_begin();
  openrand::Philox rng(1, 0);
  for (size_t i = start_node_index; i < end_node_index; ++i) {
    // Create the node.
    stk::mesh::EntityId our_node_id = get_node_id(i);
    stk::mesh::Entity node = bulk_data_ptr_->declare_node(our_node_id);
    bulk_data_ptr_->change_entity_parts(node, spring_part_ptrs_);

    // Set the node's coordinates using the given coordinate map.
    double *const node_coords = stk::mesh::field_data(*node_coord_field_ptr_, node);
    const auto [coord_x, coord_y, coord_z] = coordinate_map_ptr_->get_grid_coordinate({i});
    node_coords[0] = coord_x;
    node_coords[1] = coord_y;
    node_coords[2] = coord_z;
  }

  // Share the nodes with the neighboring ranks.
  // Note, node sharing is symmetric. If we don't own the node that we intend to share, we need to declare it before
  // marking it as shared. If we are rank 0, we share our final node with rank 1 and receive their first node. If we are
  // rank N, we share our first node with rank N - 1 and receive their final node. Otherwise, we share our first and
  // last nodes with the corresponding neighboring ranks and receive their corresponding nodes.
  if (bulk_data_ptr_->parallel_size() > 1) {
    if (rank == 0) {
      // Share the last node with rank 1.
      stk::mesh::Entity node = get_node(end_node_index - 1);
      bulk_data_ptr_->add_node_sharing(node, rank + 1);

      // Receive the first node from rank 1
      stk::mesh::EntityId received_node_id = get_node_id(end_node_index);
      stk::mesh::Entity received_node = bulk_data_ptr_->declare_node(received_node_id);
      bulk_data_ptr_->add_node_sharing(received_node, rank + 1);
    } else if (rank == bulk_data_ptr_->parallel_size() - 1) {
      // Share the first node with rank N - 1.
      stk::mesh::Entity node = get_node(start_node_index);
      bulk_data_ptr_->add_node_sharing(node, rank - 1);

      // Receive the last node from rank N - 1.
      stk::mesh::EntityId received_node_id = get_node_id(start_node_index - 1);
      stk::mesh::Entity received_node = bulk_data_ptr_->declare_node(received_node_id);
      bulk_data_ptr_->add_node_sharing(received_node, rank - 1);
    } else {
      // Share the first and last nodes with the corresponding neighboring ranks.
      stk::mesh::Entity first_node = get_node(start_node_index);
      stk::mesh::Entity last_node = get_node(end_node_index - 1);
      bulk_data_ptr_->add_node_sharing(first_node, rank - 1);
      bulk_data_ptr_->add_node_sharing(last_node, rank + 1);

      // Receive the corresponding nodes from the neighboring ranks.
      stk::mesh::EntityId received_first_node_id = get_node_id(start_node_index - 1);
      stk::mesh::EntityId received_last_node_id = get_node_id(end_node_index);
      stk::mesh::Entity received_first_node = bulk_data_ptr_->declare_node(received_first_node_id);
      stk::mesh::Entity received_last_node = bulk_data_ptr_->declare_node(received_last_node_id);
      bulk_data_ptr_->add_node_sharing(received_first_node, rank - 1);
      bulk_data_ptr_->add_node_sharing(received_last_node, rank + 1);
    }
  }

  // Now the hookean springs.
  if (generate_hookean_springs_) {
    const size_t start_spring_index = start_node_index;
    const size_t end_spring_index = end_node_index - 1;
    for (size_t i = start_spring_index; i < end_spring_index; ++i) {
      // Create the hookean spring.
      stk::mesh::EntityId spring_id = get_hookean_spring_id(i);
      stk::mesh::Entity spring = bulk_data_ptr_->declare_element(spring_id);
      bulk_data_ptr_->change_entity_parts(spring, hookean_spring_part_ptrs_);

      // Create the node and connect it to the spring.
      // To map our sequential index to the node sequential index, we connect to node i and i + 1.
      // node i --- spring i --- node i + 1
      stk::mesh::Entity node0 = get_node(i);
      stk::mesh::Entity node1 = get_node(i + 1);
      bulk_data_ptr_->declare_relation(spring, node0, 0);
      bulk_data_ptr_->declare_relation(spring, node1, 1);
    }
  }

  if (generate_angular_springs_) {
    const size_t start_spring_index = start_node_index + num_hookean_springs_;
    const size_t end_spring_index = end_node_index - 2;
    for (size_t i = start_spring_index; i < end_spring_index; ++i) {
      // Create the angular spring.
      stk::mesh::EntityId spring_id = get_angular_spring_id(i);
      stk::mesh::Entity spring = bulk_data_ptr_->declare_element(spring_id);
      bulk_data_ptr_->change_entity_parts(spring, angular_spring_part_ptrs_);

      // Create the nodes and connect them to the spring.
      // To map our sequential index to the node sequential index, we connect to node i, i - 1, and i + 1.
      // Our center node is node i. Note, the node ordinals for BEAM_3 are
      /* n1      n2
      //   \    /
      //    \  /
      //     n3
      */
      stk::mesh::Entity left_node = get_node(i - 1);
      stk::mesh::Entity center_node = get_node(i);
      stk::mesh::Entity right_node = get_node(i + 1);

      bulk_data_ptr_->declare_relation(spring, left_node, 0);
      bulk_data_ptr_->declare_relation(spring, right_node, 1);
      bulk_data_ptr_->declare_relation(spring, center_node, 2);
    }
  }

  bulk_data_ptr_->modification_end();
}
//}

}  // namespace techniques

}  // namespace declare_and_initialize_constraints

}  // namespace constraints

}  // namespace mundy