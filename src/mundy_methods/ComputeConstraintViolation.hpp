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

#ifndef MUNDY_METHODS_COMPUTECONSTRAINTVIOLATION_HPP_
#define MUNDY_METHODS_COMPUTECONSTRAINTVIOLATION_HPP_

/// \file ComputeConstraintViolation.hpp
/// \brief Declaration of the ComputeConstraintViolation class

// clang-format off
#include <gtest/gtest.h>                             // for AssertHelper, etc
#include <mpi.h>                                     // for MPI_COMM_WORLD, etc
#include <stddef.h>                                  // for size_t
#include <vector>                                    // for vector, etc
#include <random>                                    // for rand
#include <memory>                                    // for shared_ptr
#include <string>                                    // for string
#include <type_traits>                               // for static_assert
#include <stk_math/StkVector.hpp>                    // for Vec
#include <stk_mesh/base/BulkData.hpp>                // for BulkData
#include <stk_mesh/base/MetaData.hpp>                // for MetaData
#include <stk_mesh/base/GetEntities.hpp>             // for count_selected_entities
#include <stk_mesh/base/Types.hpp>                   // for EntityVector, etc
#include <stk_mesh/base/Ghosting.hpp>                // for create_ghosting
#include <stk_io/WriteMesh.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_search/SearchMethod.hpp>               // for KDTREE
#include <stk_search/CoarseSearch.hpp>               // for coarse_search
#include <stk_search/BoundingBox.hpp>                // for Sphere, Box, tec.
#include <stk_balance/balance.hpp>                   // for balanceStkMesh
#include <stk_util/parallel/Parallel.hpp>            // for ParallelMachine
#include <stk_util/environment/WallTime.hpp>         // for wall_time
#include <stk_util/environment/perf_util.hpp>
#include <stk_unit_test_utils/BuildMesh.hpp>
#include "stk_util/util/ReportHandler.hpp"           // for ThrowAssert, etc
// clang-format on

namespace mundy {

namespace methods {

/// \class ComputeConstraintViolation
/// \brief Method for computing the axis aligned boundary box of different parts.
class ComputeConstraintViolation : public MetaMethod<ComputeConstraintViolation, void>,
                                   public MetaMethodRegistry<ComputeConstraintViolation> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeConstraintViolation() = delete;

  /// \brief Constructor
  ComputeConstraintViolation(const stk::mesh::BulkData *bulk_data_ptr,
                             const std::vector<*stk::mesh::Part> &part_ptr_vector,
                             const Teuchos::ParameterList &parameter_list)
      : bulk_data_ptr_(bulk_data_ptr), part_ptr_vector_(part_ptr_vector), num_parts_(part_ptr_vector_.size()) {
    // The bulk data pointer must not be null.
    TEUCHOS_TEST_FOR_EXCEPTION(bulk_data_ptr_ == nullptr, std::invalid_argument,
                               "mundy::methods::ComputeConstraintViolation: bulk_data_ptr cannot be a nullptr.");

    // The parts cannot intersect.
    for (int i = 0; i < num_parts_; i++) {
      for (int j = 0; j < num_parts_; j++) {
        if (i = !j) {
          const bool parts_intersect = stk::mesh::intersect(*part_ptr_vector[i], *part_ptr_vector[j]);
          TEUCHOS_TEST_FOR_EXCEPTION(parts_intersect, std::invalid_argument,
                                     "mundy::methods::ComputeConstraintViolation: Part "
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
          part_parameter_list.sublist("kernels").sublist("compute_constraint_violation");

      // Create the kernel instance.
      const std::string kernel_name = part_kernel_parameter_list.get<std::string>("name");
      compute_constraint_violation_kernels_.push_back(
          MetaKernelFactory<ComputeConstraintViolation>::create_new_instance(kernel_name, bulk_data_ptr,
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
          part_parameter_list.sublist("kernels").sublist("compute_constraint_violation");

      // Validate the kernel params and fill in defaults.
      const std::string kernel_name = part_kernel_parameter_list.get<std::string>("name");
      part_kernel_parameter_list.validateParametersAndSetDefaults(
          MetaKernelFactory<ComputeConstraintViolation>::get_valid_params(kernel_name));

      // Merge the kernel requirements.
      part_requirements[i]->merge(&MetaKernelFactory<ComputeConstraintViolation>::get_part_requirements(
          kernel_name, part_kernel_parameter_list));
    }

    return part_requirements;
  }

  /// \brief Get the default parameters for this class.
  static Teuchos::ParameterList details_get_valid_params() {
    static Teuchos::ParameterList default_parameter_list;
    Teuchos::ParameterList &kernel_params =
        default_parameter_list.sublist("kernels", false, "Sublist that defines the kernels and their parameters.");
    kernel_params.sublist("compute_constraint_violation", false,
                          "Sublist that defines the constraint violation kernel parameters.");
    return default_parameter_list;
  }

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c MetaMethod.
  static std::string details_get_class_identifier() const {
    return "COMPUTE_CONSTRAINT_VIOLATION";
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  static std::unique_ptr<MetaMethodBase> details_create_new_instance(const stk::mesh::BulkData *bulk_data_ptr,
                                                                 const std::vector<*stk::mesh::Part> &part_ptr_vector,
                                                                 const Teuchos::ParameterList &parameter_list) const {
    return std::make_unique<ComputeConstraintViolation>(bulk_data_ptr, part_ptr_vector, parameter_list);
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  void execute() {
    for (int i = 0; i < num_parts_; i++) {
      const MetaKernel &compute_constraint_violation_kernel = compute_constraint_violation_kernels_[i];

      stk::mesh::Selector locally_owned_part = meta_mesh.locally_owned_part() && *part_ptr_vector_[i];
      stk::mesh::for_each_entity_run(
          *bulk_data_ptr, stk::topology::ELEM_RANK, locally_owned_part,
          [&compute_constraint_violation_kernel](const stk::mesh::BulkData &bulk_data, stk::mesh::Entity element) {
            compute_constraint_violation_kernel->execute(element);
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
  std::vector<std::shared_ptr<MetaKernelBase>>  compute_constraint_violation_kernels_;
  //@}
};  // ComputeConstraintViolation

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTECONSTRAINTVIOLATION_HPP_
