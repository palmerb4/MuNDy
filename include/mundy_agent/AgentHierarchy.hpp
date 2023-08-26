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

#ifndef MUNDY_AGENT_AGENTHEIRARCHY_HPP_
#define MUNDY_AGENT_AGENTHEIRARCHY_HPP_

/// \file AgentHeirarchy.hpp
/// \brief Declaration of the AgentHeirarchy class

// C++ core libs
#include <functional>  // for std::function
#include <iostream>
#include <map>        // for std::map
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <utility>    // for std::make_pair

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy/throw_assert.hpp>       // for MUNDY_THROW_ASSERT
#include <mundy_agent/Agent.hpp>        // for mundy::agent::Agent
#include <mundy_agent/IsAgentType.hpp>  // for mundy::agent::IsAgentType

namespace mundy {

namespace agent {

/// \brief Unique agent identifier
/// In the current design, an "agent" is a Part endowed with a set of default requirements. Each agent can be
/// uniquely identified by either the agent's part or a unique uint, namely agent_t.
///
/// \note First, agent_t is not a class enum, so comparing two agent_t's equates to comparing two unsigned ints (same as
/// regular enums). Second, agent_t is unique in that no two agents will share the same agent_t; however, the agent_t
/// assigned to an agent need not be the same between consecutive compilations of the code. This is caused by the static
/// initialization order fiasco.
using agent_t = unsigned;

/// \class AgentHeirarchy
/// \brief A factory containing generation routines for different Agent types that form a hierarchical structure.
///
/// Agents are used to streamlike the requirements generation process. Typically, an agent will hardcode only the bare
/// minimum requirements (e.g., all spheres have a radius and a center). All other requirements should be added at run-time.
/// Either one uses an agent directly, modifies the requirements of an agent, or inherits from/subsets an agent. Typically,
/// subsetting is more appropriate than direct modification since it's unlikely that you want ALL agents of a type to have a
/// certain field other than their default fields. If a subset occurs repeatedly in your problem, consider creating and
/// registering your own agent.
/// 
/// class ExampleAgent {
///  public:
///   //! \name Getters
///   //@{
/// 
///   /// \brief Get the ExampleAgent's name.
///   /// This name must be unique and not shared by any other shape.
///   static constexpr inline std::string_view get_name();
/// 
///   /// \brief Get the ExampleAgent's topology.
///   static constexpr inline stk::topology::topology_t get_topology();
/// 
///   /// \brief Add new part requirements to ALL members of this agent part.
///   /// These modifications are reflected in our mesh requirements.
///   static inline void add_part_requirements(std::shared_ptr<mundy::meta::PartRequirements> part_reqs_ptr);
/// 
///   /// \brief Add sub-part requirements.
///   /// These modifications are reflected in our mesh requirements.
///   static inline void add_subpart_requirements(std::shared_ptr<mundy::meta::PartRequirements> subpart_reqs_ptr);
/// 
///   /// \brief Get the mesh requirements for the ExampleAgent.
///   static inline std::shared_ptr<mundy::meta::MeshRequirements> get_mesh_requirements();
/// };  // ExampleAgent
///
/// This class is a tree of AgentFactories that can be used to generate the different types of agents. For example,
/// imagine a simple hierarchy of agents:
///                             Agents
///                            /      \
///                      Shapes        Constraints
///        /          /    |               |   \       \
/// Spheres Ellipsoids Spherocylinders  Springs Hingles Joints
///
/// In this example, every node in the tree knows its unique name and the name of its parent. Together, these names map
/// onto a agent_t that is unique accross all nodes in the tree. The pair of names or the corresponding agent_t can be
/// used to access the internal member functions of the registered agent in the hierarchy.
///
/// Because this factory is constructed at static initialization time, the order in which nodes in the tree
/// are registered is not guarenteed (due to the static initialization order fiasco). As a result, the id's of the nodes
/// in the tree are not guarenteed to be in the order you would expect and they need not be consistent between runs.
class AgentHeirarchy {
 public:
  //! \name Typedefs
  //@{

  /// \brief A function that returns an std::string_view.
  using StringViewGenerator = std::function<std::string_view()>;

  /// \brief A function that returns an stk::topology::topology_t.
  using TopologyGenerator = std::function<stk::topology::topology_t()>;

  /// \brief A function type that takes a parameter list and produces a vector of shared pointers to PartRequirements
  /// instances.
  using NewRequirementsGenerator = std::function<std::shared_ptr<MeshRequirements>(const Teuchos::ParameterList&)>;

  /// \brief A function that takes in a part reqirements and returns a void.
  using AddRequirementsGenerator = std::function<void(std::shared_ptr<PartRequirements>)>;
  //@}

  //! \name Getters
  //@{

  /// \brief Get the number of classes this factory recognizes.
  static size_t get_number_of_registered_types() {
    return number_of_registered_types_;
  }

  /// \brief Get if the provided name is valid or not
  /// \param name [in] A string name that may or may not correspond to a registered class.
  static bool is_valid(const std::string_view& name, const std::string_view& parent_name = "") {
    return is_valid(get_agent_type(name, parent_name));
  }

  /// \brief Get if the provided agent_type is valid or not
  /// \param agent_type [in] A agent_type that may or may not correspond to a registered class.
  static bool is_valid(const agent_t agent_type) {
    return get_name_map().count(agent_type) != 0;
  }

  /// \brief Throw if the provided name is invalid
  /// \param name [in] A string name that may or may not correspond to a registered class.
  static void assert_is_valid(const std::string_view& name, const std::string_view& parent_name = "") {
    MUNDY_THROW_ASSERT(is_valid(name, parent_name), std::invalid_argument,
                       "AgentFactory: The provided class's name '"
                           << name << "' is not valid.\n"
                           << "There are currently " << get_number_of_registered_types() << " registered classes.\n"
                           << "The hierarchy is:\n" << get_hierarchy_as_a_string() << "\n");
  }

  /// \brief Throw if the provided agent_type is invalid
  /// \param agent_type [in] A agent_type that may or may not correspond to a registered class.
  static void assert_is_valid(const agent_t agent_type) {
    MUNDY_THROW_ASSERT(is_valid(agent_type), std::invalid_argument,
                       "AgentFactory: The provided class's id '"
                           << agent_type << "' is not valid."
                           << "There are currently " << get_number_of_registered_types() << " registered classes.\n"
                           << "The hierarchy is:\n" << get_hierarchy_as_a_string() << "\n");
  }

  /// \brief Get the agent_type corresponding to a registered class with the given name.
  /// \param name [in] A string name that correspond to a registered class.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static agent_t get_agent_type(const std::string_view& name, const std::string_view& parent_name = "") {
    assert_is_valid(name, parent_name);
    return get_name_to_type_map()[name];
  }

  /// \brief Get the name corresponding to a registered class with the given agent_type.
  /// \param agent_type [in] A agent_type that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(agent_type) returns false
  static std::string_view get_name(const agent_t agent_type) {
    assert_is_valid(agent_type);
    return get_name_map()[agent_type];
  }

  /// \brief Get the topology corresponding to a registered class with the given agent_type.
  /// \param name [in] A string name that correspond to a registered class.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static stk::topology::topology_t get_topology(const std::string_view& name,
                                                const std::string_view& parent_name = "") {
    assert_is_valid(name, parent_name);
    const agent_t agent_type = get_agent_type(name);
    return get_topology(agent_type);
  }

  /// \brief Get the topology corresponding to a registered class with the given agent_type.
  /// \param agent_type [in] A agent_type that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(agent_type) returns false
  static stk::topology::topology_t get_topology(const agent_t agent_type) {
    assert_is_valid(agent_type);
    return get_topology_generator_map()[agent_type]();
  }

  /// \brief Add new part requirements to ALL members of the specified agent part.
  /// \param name [in] A string name that correspond to a registered class.
  /// \param part_reqs_ptr [in] A pointer to the part requirements to add.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static void add_part_requirements(const std::string_view& name, const std::string_view& parent_name = "",
                                    std::shared_ptr<mundy::meta::PartRequirements> part_reqs_ptr) {
    assert_is_valid(name, parent_name);
    const agent_t agent_type = get_agent_type(name);
    add_part_requirements(agent_type, part_reqs_ptr);
  }

  /// \brief Add new part requirements to ALL members of the specified agent part.
  /// \param agent_type [in] A agent_type that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(agent_type) returns false
  static void add_part_requirements(const agent_t agent_type,
                                    std::shared_ptr<mundy::meta::PartRequirements> part_reqs_ptr) {
    assert_is_valid(agent_type);
    get_part_requirements_generator_map()[agent_type](part_reqs_ptr);
  }

  /// \brief Add new sub-part requirements to ALL members of the specified agent part.
  /// \param name [in] A string name that correspond to a registered class.
  /// \param subpart_reqs_ptr [in] A pointer to the sub-part requirements to add.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static void add_subpart_requirements(const std::string_view& name, const std::string_view& parent_name = "",
                                       std::shared_ptr<mundy::meta::PartRequirements> subpart_reqs_ptr) {
    assert_is_valid(name, parent_name);
    const agent_t agent_type = get_agent_type(name);
    add_subpart_requirements(agent_type, subpart_reqs_ptr);
  }

  /// \brief Add new sub-part requirements to ALL members of the specified agent part.
  /// \param agent_type [in] A agent_type that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(agent_type) returns false
  static void add_subpart_requirements(const agent_t agent_type,
                                       std::shared_ptr<mundy::meta::PartRequirements> subpart_reqs_ptr) {
    assert_is_valid(agent_type);
    get_subpart_requirements_generator_map()[agent_type](subpart_reqs_ptr);
  }

  /// \brief Get the mesh requirements for the specified agent.
  /// \param name [in] A string name that correspond to a registered class.
  /// Throws an error if this name is not registered to an existing class, i.e., is_valid(name) returns false
  static std::shared_ptr<mundy::meta::MeshRequirements> get_mesh_requirements(
      const std::string_view& name, const std::string_view& parent_name = "") {
    assert_is_valid(name, parent_name);
    const agent_t agent_type = get_agent_type(name);
    return get_mesh_requirements(agent_type);
  }

  /// \brief Get the mesh requirements for the specified agent.
  /// \param agent_type [in] A agent_type that correspond to a registered class.
  /// Throws an error if this id is not registered to an existing class, i.e., is_valid(agent_type) returns false
  static std::shared_ptr<mundy::meta::MeshRequirements> get_mesh_requirements(const agent_t agent_type) {
    assert_is_valid(agent_type);
    return get_mesh_requirements_generator_map()[agent_type]();
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Register a new class. The key for the class is determined by its class identifier.
  template <typename ClassToRegister>
  static inline bool register_new_class() {
    // Check that the ClassToRegister has the desired interface.
    using Checker = IsAgentType<ClassToRegister>;
    static_assert(Checker::has_get_name,
                  "AgentFactory: The class to register doesn't have the correct get_name function.\n"
                  "See the documentation of AgentFactory for more information about the expected interface.");
    static_assert(Checker::has_get_parent_name,
                  "AgentFactory: The class to register doesn't have the correct get_parent_name function.\n"
                  "See the documentation of AgentFactory for more information about the expected interface.");
    static_assert(Checker::has_get_topology,
                  "AgentFactory: The class to register doesn't have the correct get_topology function.\n"
                  "See the documentation of AgentFactory for more information about the expected interface.");
    static_assert(Checker::has_add_part_requirements,
                  "AgentFactory: The class to register doesn't have the correct add_part_requirements function.\n"
                  "See the documentation of AgentFactory for more information about the expected interface.");
    static_assert(Checker::has_add_subpart_requirements,
                  "AgentFactory: The class to register doesn't have the correct add_subpart_requirements "
                  "function.\n"
                  "See the documentation of AgentFactory for more information about the expected interface.");
    static_assert(Checker::has_get_mesh_requirements,
                  "AgentFactory: The class to register doesn't have the correct get_mesh_requirements function.\n"
                  "See the documentation of AgentFactory for more information about the expected interface.");

    std::cout << "AgentFactory: Registering class " << ClassToRegister::get_name() << " with id "
              << number_of_registered_types_ << std::endl;

    // Ensure that the class name/parent name combo is unique.
    const std::string_view name = ClassToRegister::get_name();
    const std::string_view parent_name = ClassToRegister::get_parent_name();
    MUNDY_THROW_ASSERT(!is_valid(name, parent_name), std::invalid_argument,
                       "AgentFactory: The provided class's name '"
                           << name << "' already exists for parent node with name '" << parent_name << "'.");

    // Get the agent type for this class.
    number_of_registered_types_++;
    const agent_t agent_type = number_of_registered_types_ - 1;

    // Create the node in the hierarchy.
    StringTreeManager::create_node(agent_type, name, parent_name);

    // Register the class.
    get_name_pair_to_type_map().insert(std::make_pair(std::make_pair(name, parent_name), agent_type));
    get_name_map().insert(std::make_pair(agent_type, name));
    get_topology_generator_map().insert(std::make_pair(agent_type, ClassToRegister::get_topology));
    get_add_part_requirements_generator_map().insert(
        std::make_pair(agent_type, ClassToRegister::add_part_requirements));
    get_add_subpart_requirements_generator_map().insert(
        std::make_pair(agent_type, ClassToRegister::add_subpart_requirements));
    get_mesh_requirements_generator_map().insert(std::make_pair(agent_type, ClassToRegister::get_mesh_requirements));
    return true;
  }

  /// \brief Print the hierarchy.
  static void print_hierarchy(std::ostream& os = std::cout) {
    StringTreeManager::print_tree(os);
  }

  // \brief Get the hierarchy as a string.
  static std::string get_hierarchy_as_a_string() {
    std::stringstream ss;
    print_hierarchy(ss);
    return ss.str();
  }
  //@}

 private:
  //! \name Internal member
  //@{

  /// \brief The number of registered agent types.
  /// \note This is initialized to zero outside the class declaration.
  static agent_t number_of_registered_types_;
  //@}

  //! \name Internal hierarchy of agent names
  //@{

  /// \brief A helper class for generating a hierarchy of string names.
  class StringTreeNode {
   public:
    /// @brief Construct a new StringTreeNode object
    /// @param name [in] The name of this node.
    StringTreeNode(const unsigned id, const std::string_view& name, const std::string_view& parent_name = "")
        : id_(id), name_(name), parent_name_(parent_name) {
    }

    void add_child(const unsigned id, const std::string_view& name, const std::string_view& parent_name) {
      children_.push_back(std::make_shared<StringTreeNode>(id, name, parent_name));
    }

    void add_child(td::shared_ptr<StringTreeNode> child) {
      children_.push_back(child);
    }

    unsigned get_id() const {
      return id_;
    }

    std::string_view get_name() const {
      return name_;
    }

    std::string_view get_parent_name() const {
      return parent_name_;
    }

    std::shared_ptr<StringTreeNode> get_child(const std::string_view& name) const {
      for (const auto& child : children_) {
        if (child->get_name() == name) {
          return child;
        }
      }
      return nullptr;
    }

   private:
    const unsigned id_;
    const std::string_view name_;
    const std::string_view parent_name_;

    std::vector<std::shared_ptr<StringTreeNode>> children_;
  };  // StringTreeNode

  /// \brief A non-member builder class for \c StringTreeNode's that accounts for the static initialization order.
  class StringTreeManager {
   public:
    static std::shared_ptr<StringTreeNode> create_node(const unsigned id, const std::string_view& name,
                                                       const std::string_view& parent_name) {
      // Check if the node already exists.
      const auto node_iter = node_map_.find(id);
      if (node_iter != node_map_.end()) {
        return node_iter->second;
      }

      const auto root_node_iter = root_node_map_.find(id);
      if (root_node_iter != root_node_map_.end()) {
        return root_node_iter->second;
      }

      // Create the node.
      std::shared_ptr<StringTreeNode> node = std::make_shared<StringTreeNode>(name, parent_name);
      node_map_.insert(std::make_pair(id, node));

      // Store the node.
      if (!parent_name_.empty() && node_map_.count(parent_name) != 0) {
        // If the parent already exists, add the node to the parent's list of children.
        std::shared_ptr<StringTreeNode> parent_node = node_map_[parent_name];
        parent_node->add_child(node);
      } else if (!parent_name_.empty()) {
        // If the parent doesn't exist yet, add it to the orphaned node map.
        orphaned_node_map_.insert(std::make_pair(id, node));
      } else {
        // The parent is empty. This is a root node.
        root_node_map_.insert(name, node);
      }

      // Check if the orphaned nodes can be linked to the new node.
      for (auto orphaned_node_iter = orphaned_node_map_.begin(); orphaned_node_iter != orphaned_node_map_.end();) {
        if (orphaned_node_iter->first.second == name) {
          // This orphaned node is a child of the new node.
          std::shared_ptr<StringTreeNode> orphaned_node = orphaned_node_iter->second;
          node->add_child(orphaned_node);
          orphaned_node_map_.erase(orphaned_node_iter);
        } else {
          // This orphaned node is not a child of the new node.
          ++orphaned_node_iter;
        }
      }

      return node;
    }

    /// @brief Print the tree.
    static void print_tree(std::ostream& os = std::cout) {
      for (const auto& pair : root_node_map_) {
        print_tree(pair.second, os);
      }
    }

   private:
    /// @brief Print the tree.
    /// @param node The node in the tree to print.
    /// @param depth The depth of the node in the tree.
    static void print_tree(std::shared_ptr<StringTreeNode> node, std::ostream& os = std::cout, int depth = 0) {
      for (int i = 0; i < depth; ++i) {
        os << "  ";
      }
      os << "Name: " << node->get_name() << " | " << "ID: " << node->get_id() << std::endl;

      for (const auto& child : node->get_children()) {
        print_tree(child, os, depth + 1);
      }
    }

    static std::map<std::string_view, std::shared_ptr<StringTreeNode>> root_node_map_;
    static std::map<std::pair<std::string_view, std::string_view>, std::shared_ptr<StringTreeNode>> node_map_;
    static std::map<std::pair<std::string_view, std::string_view>, std::shared_ptr<StringTreeNode>> orphaned_node_map_;
  };  // StringTreeManager

  //@}

  //! \name Typedefs
  //@{

  /// \brief A map from a string_view to agent_type.
  using NamePairToTypeMap = std::map<std::pair<std::string_view, std::string_view>, agent_t>;

  /// \brief A map from a string_view to agent_type.
  using TypeToNameMap = std::map<agent_t, std::string_view>;

  /// \brief A map from agent_type to a function that returns a string view.
  using TypeToStringViewGeneratorMap = std::map<agent_t, StringViewGenerator>;

  /// \brief A map from agent_type to a function that returns an stk::topology::topology_t.
  using TypeToTopologyGeneratorMap = std::map<agent_t, TopologyGenerator>;

  /// \brief A map from agent_type to a function that takes in a part reqirements and returns a void.
  using TypeToAddRequirementsGeneratorMap = std::map<agent_t, AddRequirementsGenerator>;

  /// \brief A map from agent_type to a function that returns a class's mesh requirements.
  using TypeToNewRequirementsGeneratorMap = std::map<agent_t, NewRequirementsGenerator>;
  //@}

  //! \name Attributes
  //@{
  static NamePairToTypeMap& get_name_pair_to_type_map() {
    // Static: One and the same instance for all function calls.
    static NamePairToTypeMap name_pair_to_id_map;
    return name_pair_to_id_map;
  }

  static TypeToNameMap& get_name_map() {
    // Static: One and the same instance for all function calls.
    static TypeToNameMap name_map;
    return name_map;
  }

  static TypeToTopologyGeneratorMap& get_topology_generator_map() {
    // Static: One and the same instance for all function calls.
    static TypeToTopologyGeneratorMap topology_generator_map;
    return topology_generator_map;
  }

  static TypeToAddRequirementsGeneratorMap& get_add_part_requirements_generator_map() {
    // Static: One and the same instance for all function calls.
    static TypeToAddRequirementsGeneratorMap add_part_requirements_generator_map;
    return add_part_requirements_generator_map;
  }

  static TypeToAddRequirementsGeneratorMap& get_add_subpart_requirements_generator_map() {
    // Static: One and the same instance for all function calls.
    static TypeToAddRequirementsGeneratorMap add_subpart_requirements_generator_map;
    return add_subpart_requirements_generator_map;
  }

  static TypeToNewRequirementsGeneratorMap& get_mesh_requirements_generator_map() {
    // Static: One and the same instance for all function calls.
    static TypeToNewRequirementsGeneratorMap mesh_requirements_generator_map;
    return mesh_requirements_generator_map;
  }
  //@}
};  // AgentHeirarchy

}  // namespace agent

}  // namespace mundy

#endif  // MUNDY_AGENT_AGENTHEIRARCHY_HPP_
