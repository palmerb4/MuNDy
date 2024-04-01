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

#ifndef MUNDY_DRIVER_CONFIGURATOR_HPP_
#define MUNDY_DRIVER_CONFIGURATOR_HPP_

/// \file Configurator.hpp
/// \brief Declaration of the Configurator class

// C++ core libs
#include <string>

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy includes
#include <MundyDriver_config.hpp>                                     // for HAVE_MUNDYDRIVER_*
#include <mundy_meta/MeshRequirements.hpp>                            // for mundy::meta::MeshRequirements
#include <mundy_meta/MetaFactory.hpp>                                 // for mundy::meta::StringBasedMetaFactory
#include <mundy_meta/MetaMethodExecutionInterface.hpp>                // for mundy::meta::MetaMethodExecutionInterface
#include <mundy_meta/MetaMethodPairwiseSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodPairwiseSubsetExecutionInterface
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for MUNDY_REGISTER_METACLASS

#ifdef HAVE_MUNDYDRIVER_MUNDYAGENTS
#include <mundy_agents/Agents.hpp>             // for mundy::agents::Agents
#include <mundy_agents/HierarchyOfAgents.hpp>  // for mundy::agents::HierarchyOfAgents
#include <mundy_agents/RegisterAgents.hpp>     // MUNDY_REGISTER_AGENTS
#endif                                         // HAVE_MUNDYDRIVER_MUNDYAGENTS

#ifdef HAVE_MUNDYDRIVER_MUNDYCONSTRAINTS
#include <mundy_constraints/ComputeConstraintForcing.hpp>   // for mundy::constraints::ComputeConstraintForcing
#include <mundy_constraints/Constraints.hpp>                // for mundy::constraints::Constraints
#include <mundy_constraints/DeclareAndInitConstraints.hpp>  // for mundy::constraints::DeclareAndInitConstraints
#include <mundy_constraints/HookeanSprings.hpp>             // for mundy::constraints::HookeanSprings
#endif                                                      // HAVE_MUNDYDRIVER_MUNDYCONSTRAINTS

#ifdef HAVE_MUNDYDRIVER_MUNDYIO
#include <mundy_io/IOBroker.hpp>  // for mundy::io::IOBroker
#endif                            // HAVE_MUNDYDRIVER_MUNDYIO

#ifdef HAVE_MUNDYDRIVER_MUNDYLINKERS
#include <mundy_linkers/ComputeSignedSeparationDistanceAndContactNormal.hpp>  // for mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal
#include <mundy_linkers/DestroyNeighborLinkers.hpp>                  // for mundy::linkers::DestroyNeighborLinkers
#include <mundy_linkers/EvaluateLinkerPotentials.hpp>                // for mundy::linkers::EvaluateLinkerPotentials
#include <mundy_linkers/GenerateNeighborLinkers.hpp>                 // for mundy::linkers::GenerateNeighborLinkers
#include <mundy_linkers/LinkerPotentialForceMagnitudeReduction.hpp>  // for mundy::linkers::LinkerPotentialForceMagnitudeReduction
#include <mundy_linkers/Linkers.hpp>                                 // for mundy::linkers::Linkers
#include <mundy_linkers/NeighborLinkers.hpp>                         // for mundy::linkers::NeighborLinkers
#include <mundy_linkers/neighbor_linkers/SphereSphereLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SphereSphereLinkers
#endif                                                             // HAVE_MUNDYDRIVER_MUNDYLINKERS

#ifdef HAVE_MUNDYDRIVER_MUNDYSHAPES
#include <mundy_shapes/ComputeAABB.hpp>            // for mundy::shapes::ComputeAABB
#include <mundy_shapes/ComputeBoundingRadius.hpp>  // for mundy::shapes::ComputeBoundingRadius
#include <mundy_shapes/ComputeOBB.hpp>             // for mundy::shapes::ComputeOBB
#include <mundy_shapes/DeclareAndInitShapes.hpp>   // for mundy::shapes::DeclareAndInitShapes
#include <mundy_shapes/Spheres.hpp>                // for mundy::shapes::Spheres
#include <mundy_shapes/Spherocylinders.hpp>        // for mundy::shapes::Spherocylinders
#include <mundy_shapes/SpherocylinderSegments.hpp>  // for mundy::shapes::SpherocylinderSegments
#endif                                             // HAVE_MUNDYDRIVER_MUNDYSHAPES

// Class forward definitions (if we don't want the header)
// clang-format off
namespace mundy { namespace driver { class Driver; } }
// clang-format on

namespace mundy {

namespace driver {

/// \class Configurator
/// \brief Class for reading in a configuration file and configuring a driver (simulation context)
///
/// The Configurator is responsible for parsing the input configuration file (YAML, XML, etc) and either passing it onto
/// a Driver, or creating a temporary Driver that can be used to query enabled/registered metamethods.
class Configurator {
 public:
  //! \name Constructors and destructors
  //@{

  /// \brief Default constructor for configurator
  Configurator() {
  }

  /// \brief Teuchos paramter list constructor
  explicit Configurator(const Teuchos::ParameterList& param_list) : param_list_(param_list) {
  }

  /// \brief Configuration file constructor
  Configurator(const std::string& input_format, const std::string& input_filename);

  //@}

  //! \name Queries of Configurator
  //@{

  /// \brief Get the registered MetaMethodExecutionInterface
  static std::string get_registered_meta_method_execution_interface();

  /// \brief Get the registered MetaMethodSubsetExecutionInterface
  static std::string get_registered_meta_method_subset_execution_interface();

  /// \brief Get the registered MetaMethodPairwiseSubsetExecutionInterface
  static std::string get_registered_meta_method_pairwise_subset_execution_interface();

  /// \brief Get all registered classes
  static std::string get_registered_classes();

  //@}

  //! \name Parse input file (ParameterList)
  //@{

  /// \brief Parse the parameters and construct a driver (execution engine)
  void parse_parameters();

  /// \brief Parse the configuration portion of the parameters
  void parse_configuration(const Teuchos::ParameterList& config_params);

  /// \brief Parse and configure MetaMethodExecutionInterace methods
  void parse_metamethod(const std::string& method_type, const Teuchos::ParameterList& method_params);

  //@}

  //! \name Print/format
  //@{

  /// \brief Print the enabled MetaMethods and their fixed/mutable parameters
  void print_enabled_meta_methods();

  //@}

  //! \name Driver interactions
  //@{

  /// \brief Set the Driver instance
  void set_driver(std::shared_ptr<Driver> driver_ptr);

  /// \brief Generate the driver (entire)
  void generate_driver();

  /// \brief Generate mesh requirements on the driver
  void generate_mesh_requirements_driver();

  /// \brief Declare the mesh on the driver
  void declare_mesh_driver();

  /// \brief Commit the mesh on the driver
  void commit_mesh_driver();

  /// \brief Generate meta classes on driver
  void generate_meta_methods_driver();

  //@}

 private:
  //! \name Default parameters
  //@{

  /// \brief MetaMethod type names
  static constexpr std::array<std::string_view, 3> metamethod_types_ = {
      "meta_method_execution_interface", "meta_method_subset_execution_interface",
      "meta_method_pairwise_subset_execution_interface"};

  static constexpr std::string_view default_node_coordinates_field_name_ = "NODE_COORDINATES";

  //@}

  //! \name Internal members
  //@{

  /// \brief Teuchos ParameterList for global information
  Teuchos::ParameterList param_list_;

  /// \brief Number of dimensions
  int n_dim_ = 0;

  /// \brief Enabled MetaMethods [User-defined name](MetaMethod type, MetaMethod name, fixed parameters, mutable
  /// parameters)
  std::unordered_map<std::string, std::tuple<std::string, std::string, Teuchos::ParameterList, Teuchos::ParameterList>>
      enabled_meta_methods_;

  /// \brief Node coordinate field name
  std::string node_coordinates_field_name_;

  /// \brief Associated Driver instance
  //
  // TODO(cje): Think about if we want to hold a driver pointer ourselves, as the Configurator does not own the driver
  // in any way, or have it pass into all of the configuration methods. The reason to keep it on hand is so that we
  // aren't constantly writing down the Driver arguments, as well as incrementing the shared_ptr. The downside is that
  // it requires setting it on the inside of the class, and then not changing it, leading to undefined behavior.
  std::shared_ptr<Driver> driver_ptr_ = nullptr;

  //}
};  // Configurator

}  // namespace driver

}  // namespace mundy

#endif  // MUNDY_DRIVER_CONFIGURATOR_HPP_
