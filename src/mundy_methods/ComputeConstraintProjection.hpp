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
class ComputeConstraintProjection : public MetaMethod<ComputeConstraintProjection, void>,
                                    public MetaMethodRegistry<ComputeConstraintProjection> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeConstraintProjection() = delete;

  /// \brief Constructor
  ComputeConstraintProjection(const stk::mesh::BulkData *bulk_data_ptr,
                              const std::vector<*stk::mesh::Part> &part_ptr_vector,
                              const Teuchos::ParameterList &parameter_list)
      : bulk_data_ptr_(bulk_data_ptr), part_ptr_vector_(part_ptr_vector), num_parts_(part_ptr_vector_.size()) {
    // The bulk data pointer must not be null.
    TEUCHOS_TEST_FOR_EXCEPTION(bulk_data_ptr_ == nullptr, std::invalid_argument,
                               "mundy::methods::ComputeConstraintProjection: bulk_data_ptr cannot be a nullptr.");

    // The parts cannot intersect.
    for (int i = 0; i < num_parts_; i++) {
      for (int j = 0; j < num_parts_; j++) {
        if (i = !j) {
          const bool parts_intersect = stk::mesh::intersect(*part_ptr_vector[i], *part_ptr_vector[j]);
          TEUCHOS_TEST_FOR_EXCEPTION(parts_intersect, std::invalid_argument,
                                     "mundy::methods::ComputeConstraintProjection: Part "
                                         << part_ptr_vector[i]->name() << " and "
                                         << "Part " << part_ptr_vector[j]->name() << "intersect.");
        }
      }
    }

    // Store the input parameters, use default parameters for any parameter not given.
    // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
    Teuchos::ParameterList valid_parameter_list = parameter_list;
    valid_parameter_list.validateParametersAndSetDefaults(this.get_valid_params());

    // Create and store the required kernels.
    for (int i = 0; i < num_parts_; i++) {
      // Fetch the parameters for this part's kernel
      const std::string part_name = part_ptr_vector_[i]->name();
      const Teuchos::ParameterList &part_parameter_list = valid_parameter_list.sublist(part_name);
      const Teuchos::ParameterList &part_kernel_parameter_list =
          part_parameter_list.sublist("kernels").sublist("compute_constraint_projection");

      // Create the kernel instance.
      const std::string kernel_name = part_kernel_parameter_list.get<std::string>("name");
      compute_constraint_projection_kernels_.push_back(
          MetaKernelFactory<ComputeConstraintProjection>::create_new_instance(kernel_name, bulk_data_ptr,
                                                                              part_kernel_parameter_list));
    }
  }
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
  static std::vector<std::unique_ptr<PartRequirements>> details_get_part_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &parameter_list) {
    // Validate the input params. Use default parameters for any parameter not given.
    // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
    Teuchos::ParameterList valid_parameter_list = parameter_list;
    valid_parameter_list.validateParametersAndSetDefaults(this.get_valid_params());

    // Create and store the required part params.
    std::vector<PartRequirements> part_requirements(num_parts_);
    for (int i = 0; i < num_parts_; i++) {
      // Add method-specific requirements.
      const std::string part_name = part_ptr_vector_[i]->name();
      part_requirements[i]->set_name(part_name);
      part_requirements[i]->set_rank(std::topology::ELEMENT_RANK);

      // Fetch the parameters for this part's kernel.
      Teuchos::ParameterList &part_parameter_list = valid_parameter_list.sublist(part_name);
      Teuchos::ParameterList &part_kernel_parameter_list =
          part_parameter_list.sublist("kernels").sublist("compute_constraint_projection");

      // Validate the kernel params and fill in defaults.
      const std::string kernel_name = part_kernel_parameter_list.get<std::string>("name");
      part_kernel_parameter_list.validateParametersAndSetDefaults(
          MetaKernelFactory<ComputeConstraintProjection>::get_valid_params(kernel_name));

      // Merge the kernel requirements.
      part_requirements[i]->merge(&MetaKernelFactory<ComputeConstraintProjection>::get_part_requirements(
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
  static std::string details_get_class_identifier() const {
    return "COMPUTE_CONSTRAINT_PROJECTION";
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  static std::unique_ptr<MetaMethodBase> details_create_new_instance(
      const stk::mesh::BulkData *bulk_data_ptr, const std::vector<*stk::mesh::Part> &part_ptr_vector,
      const Teuchos::ParameterList &parameter_list) const {
    return std::make_unique<ComputeConstraintProjection>(bulk_data_ptr, part_ptr_vector, parameter_list);
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  void execute() {
    for (int i = 0; i < num_parts_; i++) {
      const MetaKernel &compute_constraint_projection_kernel = compute_constraint_projection_kernels_[i];

      stk::mesh::Selector locally_owned_part =
          bulk_data_ptr->mesh_meta_data().locally_owned_part() && *part_ptr_vector_[i];
      stk::mesh::for_each_entity_run(
          *bulk_data_ptr, stk::topology::ELEM_RANK, locally_owned_part,
          [&compute_constraint_projection_kernel](const stk::mesh::BulkData &bulk_data, stk::mesh::Entity element) {
            compute_constraint_projection_kernel->execute(element);
          });
    }
  }
  //@}

 private:
  //! \name Internal members
  //@{

  /// \brief Number of parts that this method acts on.
  size_t num_parts_;

  /// \brief Vector of pointers to the parts that this class will act upon.
  std::vector<*stk::mesh::Part> &part_ptr_vector_;

  /// \brief Kernels corresponding to each of the specified parts.
  std::vector<std::shared_ptr<MetaKernelBase>> compute_constraint_projection_kernels_;
  //@}
};  // ComputeConstraintProjection

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTECONSTRAINTPROJECTION_HPP_
