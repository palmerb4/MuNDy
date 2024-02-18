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

#ifndef MUNDY_SHAPE_LINKERS_HPP_
#define MUNDY_SHAPE_LINKERS_HPP_

/// \file Linkers.hpp
/// \brief Declaration of the Linkers class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_agent/AgentHierarchy.hpp>    // for mundy::agent::AgentHierarchy
#include <mundy_agent/RankedAssembly.hpp>    // for mundy::agent::RankedAssembly
#include <mundy_core/StringLiteral.hpp>      // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_meta/FieldRequirements.hpp>  // for mundy::meta::FieldRequirements
#include <mundy_meta/MeshRequirements.hpp>   // for mundy::meta::MeshRequirements
#include <mundy_meta/PartRequirements.hpp>   // for mundy::meta::PartRequirements

namespace mundy {

namespace linker {

/// In the current design, "linkers" are a constraint-rank Part with some set of requirements that endow the entities of
/// that part with the properties of a linker. By linker, we mean a constraint rank entity that connects a static number
/// of non-constraint rank entities and is used to communicate information between them or perform actions on them.
/// Think of linkers as mediators between objects, not objects themselves; a linker knows why two objects are connected
/// and how to use their information to perform actions. In this regard, linkers encode structural information that
/// cannot otherwise be expressed by static topological connections. A linker could be, for example, a means of
/// connecting a smooth surface to a node that lives on the surface, or a means of connecting two entities that are
/// physical neighbors. Neither of these connections can be easily expressed as a static topological connection.
///
/// Due to STK's design, linkers must connect to the nodes of the entities that they are connected to. If the entity
/// they connect to has lower rank connections, then the linker may optionally connect to them as a means of
/// accessing/modifying their data. By default, we connect a linker to all the lower rank entities of the entities that
/// it links. Think of this as the "family tree" of the entities. If you created a quad with a side-set face, then you
/// would go through the same process of connecting the quad to all the edges and nodes of the face.
///
/// In their current state, linkers are fully within the breadth of STK's design. For completeness, we will review the
/// nomenclature and core concepts as they relate to linkers.
///
/// As a reminder: A connection from a higher-rank entity to a lower-rank entity is referred to as a downward relation.
/// When a downward relation is declared (e.g., between an element and a node), STK Mesh, by default, creates the
/// corresponding upward relation (e.g., from the node to the element).
///
///   - Aura: Linkers with a downward connection to a shared (but not locally owned) entity are ghosted to the owning
///   process.
///   - Sharing: Nodes, edges, and faces that have a downward connection from an Entities are shared by the process that
///   owns that entity. Elements and Constraints cannot be shared.
///   - Upward connections: A downward connection from a linker to an entity automatically results in the creation of
///   the corresponding upward connection from the entity to the linker.
///
/// One of the principle results of the sharing concept is that nodes, edges, and faces are the only entities that
/// support reductions. Consequently, if each surface linker is supposed to, for example, take in the force at the
/// surface node and sum it to get the total force on the sphere, then the surface linker needs to write that
/// information directly to the sphere's node. Otherwise, there's no way to sum the force induced by multiple linkers
/// attached to the same sphere. This is why it's often better to store fields on nodes, edges, and faces rather than
/// elements or constraints. Instead, we tend to think of these higher-rank entities more as figure-heads that connected
/// and organize lower rank entities and their data than as entities that store data themselves.
///
/// The following is an example of a linker that connects a sphere to a node on its surface partitioned over two ranks.
///                     RANK0                            |                  RANK1
///                                     CONSTRAINT1(A)   |              CONSTRAINT1(LO)
///                                    |                 |             |      |        |
///           SPRING1(LO)             |                  |            |   SPHERE1(LO)   |
///          |           |           |                   |           |        |        |
/// NODE1(LO)             NODE2(LO,S)                    |   NODE2(S)         NODE3(LO)
///
/// or we could divide this schematic a bit differently:
///                     RANK0                            |                  RANK1
///                                     CONSTRAINT1(A)   |              CONSTRAINT1(LO)
///                                    |      |       |  |             |      |        |
///           SPRING1(LO)             |   SPHERE1(LO)  | |            |   SPHERE1(A)    |
///          |           |           |        |       |  |           |        |        |
/// NODE1(LO)             NODE2(LO,S)        NODE3(LO,S) |   NODE2(S)          NODE3(S)
///
/// LO: Locally owned
/// A: Aura'ed
/// S: Shared
///
/// \note Linkers require invariants to be maintained. For example, if a SurfaceLinker is only supposed to link an
/// sphere and a node that resides on its surface, then the first element should be the sphere and the first node the
/// surface node. All other connections should be the downward connections from the sphere and need not be in any
/// particular order. If you wish to fetch these connections in an order, use the sphere's static topology to access
/// them. Importantly, this means that certain errors will not be caught at compile time and may occur downstream such
/// as accidentally declaring a surface linker as an element of the NeighborLinker part and getting errors about
/// undefined fields down the road.
///
/// Each linker can be uniquely identified by either the linker's part or a fast unique identifier, namely linker_t.
/// \note linker_t is simply the agent_t associated with the linker. As a result, a linker_t will never equate to, for
/// example, a constraint_t since they are both agent_t's. You can think of this as a runtime extensible class enum.
using linker_t = mundy::agent::agent_t;

/// \class Linkers
/// \brief The static interface for all of Mundy's Linkers.
///
/// The design of this class is in accordance with the static interface requirements of mundy::agent::AgentFactory.
///
/// \note This class is a constraint rank assembly part containing all linkers. It is a subset of the AGENTS part.
class Linkers
    : public mundy::agent::RankedAssembly<mundy::core::make_string_literal("LINKERS"), stk::topology::CONSTRAINT_RANK,
                                          mundy::core::make_string_literal("AGENTS")> {};  // Linkers

/// \brief Declare the relations and their converse between a constraint-rank entity and the family tree of any number
/// of entities within the same mesh.
///
/// A parallel-local mesh modification operation.
///
/// Note, the relation-declarations must be symmetric across all sharers of the involved entities within a modification
/// cycle.
template <typename... Entities>
  requires(std::is_same_v<std::remove_cv_t<std::remove_reference_t<Entities>>, stk::mesh::Entity> && ...)
void declare_constraint_relations_to_family_tree_with_sharing(mundy::mesh::BulkData* const bulk_data_ptr,
                                                              const stk::mesh::Entity& from_constraint,
                                                              const Entities&... to_entities) {
  // For each entity to connect, we will declare a relation between the constraint rank entity and them. These will be
  // our "head" relations and will be the first relations in the constraint's relation list in the order they are given.
  // We'll use a counter to tell how many relations we've declared per rank. We will then declare relations between the
  // constraint and the family tree of the entities we're connecting. These will be the final relations in the
  // constraints's relation list and will not be in any particular order.
  //
  // To have proper sharing of low-rank entities, we want need to consider the following:
  //  - The constraint could be locally owned, shared, or aura'ed.
  //  - The entities we're connecting to could be locally owned, shared, or aura'ed.
  //
  // If the constraint is locally owned, then we need to declare sharing with non-locally owned NODE, EDGE, and
  // FACE-rank entities that we connect to. If the constraint is not locally owned, then we need to declare sharing with
  // the locally owned NODE, EDGE, and FACE-rank entities that we connect to.
  //
  // TODO(palmerb4): Test if having a non-locally owned constraint connects to non-locally owned entities is valid or
  // not.
  MUNDY_THROW_ASSERT(bulk_data_ptr->in_modifiable_state(), std::logic_error,
                     "declare_relation: The mesh must be in a modification cycle.");
  MUNDY_THROW_ASSERT(bulk_data_ptr->entity_rank(from_constraint) == stk::topology::CONSTRAINT_RANK, std::logic_error,
                     "declare_relation: The from entity must be constraint rank.");

  const bool is_from_constraint_locally_owned = bulk_data_ptr->bucket(from_constraint).owned();
  const int from_constraint_owner = bulk_data_ptr->parallel_owner_rank(from_constraint);

  std::vector<unsigned> ranked_relation_counter(bulk_data_ptr->mesh_meta_data().entity_rank_count(), 0);

  // Lambda function to declare a downward relation from the constraint to an entity.
  auto declare_relation_to_entity = [&](const stk::mesh::Entity& to_entity) {
    // Declare a relation between the constraint and the entity.
    bulk_data_ptr->declare_relation(from_constraint, to_entity,
                                    ranked_relation_counter[bulk_data_ptr->entity_rank(to_entity) + 1]++);
  };

  // Lambda function to declare downward relations from the constraint to the lower-ranked family tree of an entity.
  auto declare_relations_to_family_tree = [&](const stk::mesh::Entity& to_entity) {
    // Declare relations between the constraint and the family tree of the entity. Optionally, add sharing.
    for (stk::mesh::EntityRank lower_rank :
         {stk::topology::NODE_RANK, stk::topology::EDGE_RANK, stk::topology::FACE_RANK}) {
      const unsigned num_connection_of_lower_rank = bulk_data_ptr->num_connectivity(to_entity, lower_rank);
      const stk::mesh::Entity* connected_entities_of_lower_rank = bulk_data_ptr->begin(to_entity, lower_rank);
      for (unsigned i = 0; i < num_connection_of_lower_rank; ++i) {
        declare_relation_to_entity(connected_entities_of_lower_rank[i]);

        // If necessary, add sharing.
        const bool is_to_entity_locally_owned = bulk_data_ptr->bucket(connected_entities_of_lower_rank[i]).owned();
        if (is_from_constraint_locally_owned && !is_to_entity_locally_owned) {
          // We own the linker but not the entity we're connecting to.
          const int to_entity_owner = bulk_data_ptr->parallel_owner_rank(connected_entities_of_lower_rank[i]);
          bulk_data_ptr->add_node_sharing(connected_entities_of_lower_rank[i], to_entity_owner);
        } else if (!is_from_constraint_locally_owned && is_to_entity_locally_owned) {
          // We don't own the linker but we own the entity we're connecting to.
          bulk_data_ptr->add_node_sharing(connected_entities_of_lower_rank[i], from_constraint_owner);
        }
      }
    }
  };

  // Use a fold expression to link the constraint to all the entities we're connecting and then to the family tree of
  // all the entities we're connecting.
  (declare_relation_to_entity(to_entities), ...);
  (declare_relations_to_family_tree(to_entities), ...);
}

}  // namespace linker

}  // namespace mundy

#endif  // MUNDY_SHAPE_LINKERS_HPP_
