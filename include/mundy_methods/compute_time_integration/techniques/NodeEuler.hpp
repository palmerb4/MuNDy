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

#ifndef MUNDY_METHODS_COMPUTE_TIME_INTEGRATION_TECHNIQUES_NODEEULER_HPP_
#define MUNDY_METHODS_COMPUTE_TIME_INTEGRATION_TECHNIQUES_NODEEULER_HPP_

/// \file NodeEuler.hpp
/// \brief Declaration of ComputeTimeIntegration's NodeEuler technique.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>     // for Teuchos::ParameterList
#include <Teuchos_TestForException.hpp>  // for TEUCHOS_TEST_FOR_EXCEPTION
#include <stk_mesh/base/Entity.hpp>      // for stk::mesh::Entity
#include <stk_mesh/base/Part.hpp>        // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>    // for stk::mesh::Selector
#include <stk_topology/topology.hpp>     // for stk::topology

// Mundy libs
#include <mundy_meta/MetaFactory.hpp>                // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                 // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaMethod.hpp>                 // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>               // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/PartRequirements.hpp>           // for mundy::meta::PartRequirements
#include <mundy_mesh/BulkData.hpp>                   // for mundy::mesh::BulkData
#include <mundy_methods/ComputeTimeIntegration.hpp>  // for mundy::meta::ComputeTimeIntegration
#include <mundy_mesh/MetaData.hpp>          // for mundy::mesh::MetaData

namespace mundy {

namespace methods {

namespace compute_time_integration {

/// \class NodeEuler
/// \brief Method for computing the axis aligned boundary box of different parts.
class NodeEuler : public mundy::meta::MetaMethod<void, NodeEuler>,
                  public mundy::meta::MetaMethodRegistry<void, NodeEuler, ComputeTimeIntegration> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  NodeEuler() = delete;

  /// \brief Constructor
  NodeEuler(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list);
  //@}

  //! \name MetaMethod interface implementation
  //@{

  /// \brief Get the requirements that this method imposes upon each particle and/or constraint.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::vector<std::shared_ptr<mundy::meta::PartRequirements>> details_static_get_part_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_parameter_list) {
    // Validate the input params. Use default parameters for any parameter not given.
    // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
    Teuchos::ParameterList valid_fixed_parameter_list = fixed_parameter_list;
    valid_fixed_parameter_list.validateParametersAndSetDefaults(static_get_valid_fixed_params());

    // Create and store the required part params. One per input part.
    Teuchos::ParameterList &parts_parameter_list = valid_fixed_parameter_list.sublist("input_parts");
    const unsigned num_parts = parts_parameter_list.get<unsigned>("count");
    std::vector<std::shared_ptr<mundy::meta::PartRequirements>> part_requirements;
    for (size_t i = 0; i < num_parts; i++) {
      // Create a new parameter
      part_requirements.emplace_back(std::make_shared<mundy::meta::PartRequirements>());

      // Fetch the i'th part parameters
      Teuchos::ParameterList &part_parameter_list = parts_parameter_list.sublist("input_part_" + std::to_string(i));
      const std::string part_name = part_parameter_list.get<std::string>("name");

      // Add method-specific requirements.
      part_requirements[i]->set_part_name(part_name);
      part_requirements[i]->add_field_req(std::make_shared<mundy::meta::FieldRequirements<double>>(
          std::string(default_node_coord_field_name_), stk::topology::NODE_RANK, 3, 1));
      part_requirements[i]->add_field_req(std::make_shared<mundy::meta::FieldRequirements<double>>(
          std::string(default_node_velocity_field_name_), stk::topology::NODE_RANK, 3, 1));
      part_requirements[i]->add_field_req(std::make_shared<mundy::meta::FieldRequirements<double>>(
          std::string(default_node_omega_field_name_name_), stk::topology::NODE_RANK, 3, 1));
    }

    return part_requirements;
  }

  /// \brief Get the default fixed parameters for this class (those that impact the part requirements).
  static Teuchos::ParameterList details_static_get_valid_params() {
    static Teuchos::ParameterList default_fixed_parameter_list;
    default_fixed_parameter_list.set("node_coordinate_field_name", std::string(default_node_coord_field_name_),
                                     "Name of the node field containing the coordinate of the sphere's center.");
    default_fixed_parameter_list.set(
        "node_velocity_field_name", std::string(default_node_velocity_field_name_),
        "Name of the node field containing the translational velocity of the sphere's center.");
    default_fixed_parameter_list.set(
        "node_omega_field_name_name", std::string(default_node_omega_field_name_name_),
        "Name of the node field containing the rotational velocity of the sphere's center.");
    return default_fixed_parameter_list;
  }

  /// \brief Get the default transient parameters for this class (those that do not impact the part requirements).
  static Teuchos::ParameterList details_static_get_valid_params() {
    static Teuchos::ParameterList default_transient_parameter_list;
    default_transient_parameter_list.set("time_step_size", default_time_step_size_, "The numerical timestep size.");
    return default_transient_parameter_list;
  }

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c MetaMethod.
  static std::string details_static_get_class_identifier() {
    return std::string(class_identifier_);
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethodBase<void>> details_static_create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list) {
    return std::make_shared<NodeEuler>(bulk_data_ptr, fixed_parameter_list);
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  void execute(const stk::mesh::Selector &input_selector) override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr double default_time_step_size_ = -1;
  static constexpr std::string_view default_node_coord_field_name_ = "NODE_COORD";
  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
  static constexpr std::string_view default_node_omega_field_name_name_ = "NODE_OMEGA";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other methods in our MetaMethodRegistry.
  static constexpr std::string_view class_identifier_ = "NODE_EULER";

  /// \brief The BulkData objects this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData objects this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Number of parts that this method acts on.
  size_t num_parts_ = 0;

  /// \brief Vector of pointers to the parts that this class will act upon.
  std::vector<stk::mesh::Part *> part_ptr_vector_;

  /// \brief The numerical timestep size.
  double time_step_size_;

  /// \brief Name of the node field containing the coordinate of the sphere's center.
  std::string node_coord_field_name_;

  /// \brief Name of the node field containing the translational velocity of the sphere's center.
  std::string node_velocity_field_name_;

  /// \brief Name of the node field containing the rotational velocity of the sphere's center.
  std::string node_omega_field_name_;

  /// \brief Node field containing the coordinate of the sphere's center.
  stk::mesh::Field<double> *node_coord_field_ptr_ = nullptr;

  /// \brief Node field containing the translational velocity of the sphere's center.
  stk::mesh::Field<double> *node_velocity_field_ptr_ = nullptr;

  /// \brief Node field containing the rotational velocity of the sphere's center.
  stk::mesh::Field<double> *node_omega_field_ptr_ = nullptr;
  //@}
};  // NodeEuler

}  // namespace compute_time_integration

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTE_TIME_INTEGRATION_TECHNIQUES_NODEEULER_HPP_
