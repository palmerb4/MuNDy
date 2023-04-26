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

#ifndef MUNDY_METHODS_COMPUTECONSTRAINTPROJECTION_HPP_
#define MUNDY_METHODS_COMPUTECONSTRAINTPROJECTION_HPP_

/// \file ComputeConstraintProjection.hpp
/// \brief Declaration of the ComputeConstraintProjection class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>     // for Teuchos::ParameterList
#include <Teuchos_TestForException.hpp>  // for TEUCHOS_TEST_FOR_EXCEPTION
#include <stk_mesh/base/BulkData.hpp>    // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>      // for stk::mesh::Entity
#include <stk_mesh/base/Part.hpp>        // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>    // for stk::mesh::Selector
#include <stk_topology/topology.hpp>     // for stk::topology

// Mundy libs
#include <mundy_meta/MetaKernel.hpp>          // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaKernelFactory.hpp>   // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaMethod.hpp>          // for mundy::meta::MetaMethod
#include <mundy_meta/MetaMethodRegistry.hpp>  // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/PartRequirements.hpp>    // for mundy::meta::PartRequirements

namespace mundy {

namespace methods {

/// \class ComputeConstraintProjection
/// \brief Method for computing the axis aligned boundary box of different parts.
class ComputeConstraintProjection : public mundy::meta::MetaMethod<void, ComputeConstraintProjection>,
                                    public mundy::meta::MetaMethodRegistry<void, ComputeConstraintProjection> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeConstraintProjection() = delete;

  /// \brief Constructor
  ComputeConstraintProjection(stk::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &parameter_list);
  //@}

  //! \name MetaMethod interface implementation
  //@{

  /// \brief Get the requirements that this method imposes upon each particle and/or constraint.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::vector<std::shared_ptr<mundy::meta::PartRequirements>> details_get_part_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &parameter_list) {
    // Validate the input params. Use default parameters for any parameter not given.
    // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
    Teuchos::ParameterList valid_parameter_list = parameter_list;
    valid_parameter_list.validateParametersAndSetDefaults(this->get_valid_params());

    // Create and store the required part params.
    std::vector<mundy::meta::PartRequirements> part_requirements(num_parts_);
    for (int i = 0; i < num_parts_; i++) {
      // Add method-specific requirements.
      const std::string part_name = part_ptr_vector_[i]->name();
      part_requirements[i]->set_part_name(part_name);
      part_requirements[i]->set_part_rank(stk::topology::ELEMENT_RANK);

      // Fetch the parameters for this part's kernel.
      Teuchos::ParameterList &part_parameter_list = valid_parameter_list.sublist(part_name);
      Teuchos::ParameterList &part_kernel_parameter_list =
          part_parameter_list.sublist("kernels").sublist("compute_constraint_projection");

      // Validate the kernel params and fill in defaults.
      const std::string kernel_name = part_kernel_parameter_list.get<std::string>("name");
      part_kernel_parameter_list.validateParametersAndSetDefaults(
          mundy::meta::MetaKernelFactory<ComputeConstraintProjection>::get_valid_params(kernel_name));

      // Merge the kernel requirements.
      part_requirements[i]->merge(mundy::meta::MetaKernelFactory<ComputeConstraintProjection>::get_part_requirements(
          kernel_name, part_kernel_parameter_list));
    }

    return part_requirements;
  }

  /// \brief Get the default parameters for this class.
  static Teuchos::ParameterList details_get_valid_params() {
    static Teuchos::ParameterList default_parameter_list;
    Teuchos::ParameterList &kernel_params =
        default_parameter_list.sublist("kernels", false, "Sublist that defines the kernels and their parameters.");
    kernel_params.sublist("compute_constraint_projection", false,
                          "Sublist that defines the constraint projection kernel parameters.");
    return default_parameter_list;
  }

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c MetaMethod.
  static std::string details_get_class_identifier() {
    return "COMPUTE_CONSTRAINT_PROJECTION";
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethodBase<void>> details_create_new_instance(
      const stk::mesh::BulkData *bulk_data_ptr, const Teuchos::ParameterList &parameter_list) {
    return std::make_unique<ComputeConstraintProjection>(bulk_data_ptr, parameter_list);
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  void execute() override;
  //@}

 private:
  //! \name Internal members
  //@{

  /// \brief Number of parts that this method acts on.
  size_t num_parts_;

  /// \brief Vector of pointers to the parts that this class will act upon.
  std::vector<stk::mesh::Part *> &part_ptr_vector_;

  /// \brief Kernels corresponding to each of the specified parts.
  std::vector<std::shared_ptr<mundy::meta::MetaKernelBase<void>>> compute_constraint_projection_kernels_;
  //@}
};  // ComputeConstraintProjection

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTECONSTRAINTPROJECTION_HPP_
