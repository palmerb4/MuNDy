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

#ifndef MUNDY_METHODS_COMPUTEOBB_HPP_
#define MUNDY_METHODS_COMPUTEOBB_HPP_

/// \file ComputeOBB.hpp
/// \brief Declaration of the ComputeOBB class

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
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>          // for mundy::mesh::MetaData
#include <mundy_meta/MetaFactory.hpp>       // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaMethod.hpp>        // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>      // for mundy::meta::GlobalMetaMethodRegistry
#include <mundy_meta/MeshRequirements.hpp>  // for mundy::meta::MeshRequirements

namespace mundy {

namespace methods {

/// \class ComputeOBB
/// \brief Method for computing the axis aligned boundary box of different parts.
class ComputeOBB : public mundy::meta::MetaMethod<void, ComputeOBB>,
                   public mundy::meta::GlobalMetaMethodRegistry<void, ComputeOBB> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeOBB() = delete;

  /// \brief Constructor
  ComputeOBB(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name Typedefs
  //@{

  using OurKernelFactory = mundy::meta::MetaKernelFactory<void, ComputeOBB>;

  template<typename ClassToRegister>
  using OurKernelRegistry = mundy::meta::MetaKernelRegistry<void, ClassToRegister, ComputeOBB>;
  //@}

  //! \name MetaMethod interface implementation
  //@{

  /// \brief Get the requirements that this method imposes upon each particle and/or constraint.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c MeshRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::MeshRequirements> details_static_get_mesh_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_params) {
    // Validate the input params. Use default parameters for any parameter not given.
    // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(static_get_valid_fixed_params());

    // Create and store the required part params. One per input part.
    Teuchos::ParameterList &parts_params = valid_fixed_params.sublist("input_parts");
    const unsigned num_parts = parts_params.get<unsigned>("count");
    std::vector<std::shared_ptr<mundy::meta::PartRequirements>> part_requirements;
    for (size_t i = 0; i < num_parts; i++) {
      // Create a new parameter
      part_requirements.emplace_back(std::make_shared<mundy::meta::PartRequirements>());

      // Fetch the i'th part parameters
      Teuchos::ParameterList &part_params = parts_params.sublist("input_part_" + std::to_string(i));
      const std::string part_name = part_params.get<std::string>("name");

      // Add method-specific requirements.
      part_requirements[i]->set_part_name(part_name);
      part_requirements[i]->set_part_rank(stk::topology::ELEMENT_RANK);

      // Fetch the parameters for this part's kernel.
      Teuchos::ParameterList &part_kernel_params =
          part_params.sublist("kernels").sublist("compute_obb");

      // Validate the kernel params and fill in defaults.
      const std::string kernel_name = part_kernel_params.get<std::string>("name");
      part_kernel_params.validateParametersAndSetDefaults(
          mundy::meta::MetaKernelFactory<void, ComputeOBB>::get_valid_params(kernel_name));

      // Merge the kernel requirements.
      part_requirements[i]->merge(mundy::meta::MetaKernelFactory<void, ComputeOBB>::get_part_requirements(
          kernel_name, part_kernel_params));
    }

    return part_requirements;
  }

  /// \brief Validate the default fixed parameters for this class (those that impact the mesh requirements) and set
  /// their defaults.
  ///
  /// The only required parameter is "enabled_multibody_type_names" which must specify the name of at least one
  /// multibody type to enable. The compute_obb kernel associated with this type must be registered with our kernel
  /// factory.
  static void details_static_validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList const *fixed_params_ptr) {
    Teuchos::ParameterList params = *fixed_params_ptr;
    TEUCHOS_TEST_FOR_EXCEPTION(
        params.isParameter("enabled_multibody_type_names"), std::invalid_argument,
        "ComputeOBB: The provided parameter list must include the set of enabled multibody type names.");
    Teuchos::Array &enabled_multibody_type_names =
        params.get<Teuchos::Array<std::string>>("enabled_multibody_type_names");
    TEUCHOS_TEST_FOR_EXCEPTION(enabled_multibody_type_names.size() != 0, std::invalid_argument,
                               "ComputeOBB: The enabled multibody type names must not be empty.");

    Teuchos::ParameterList &kernel_params =
        fixed_params_ptr->sublist("kernels", false).sublist("compute_obb", false);
    for (const auto enabled_multibody_type_name : enabled_multibody_type_names) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          mundy::multibody::Factory::is_valid(enabled_multibody_type_name), std::invalid_argument,
          "ComputeOBB: Failed to find a multibody type with name (" << enabled_multibody_type_name << ").");
      TEUCHOS_TEST_FOR_EXCEPTION(
          OurKernelFactory::is_valid_key(enabled_multibody_type_name), std::invalid_argument,
          "ComputeOBB: Failed to find a compute_obb kernel associated with the provided multibody type name ("
              << enabled_multibody_type_name << ").");
      Teuchos::ParameterList &multibody_params = kernel_params.sublist(enabled_multibody_type_name, false);
      OurKernelFactory::validate_fixed_parameters_and_set_defaults(enabled_multibody_type_name, multibody_params);
    }
  }

  /// \brief Get the default mutable parameters for this class (those that do not impact the mesh requirements) and
  /// set their defaults.
  static void details_static_validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList const *mutable_params_ptr) {
  }

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c MetaMethod.
  static std::string details_static_get_class_identifier() {
    return std::string(class_identifier_);
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethodBase<void>> details_static_create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &parameter_list) {
    return std::make_shared<ComputeOBB>(bulk_data_ptr, parameter_list);
  }

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;
  //@}
  
  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  void execute(const stk::mesh::Selector &input_selector) override;
  //@}

 private:
  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other methods in our MetaMethodRegistry.
  static constexpr std::string_view class_identifier_ = "COMPUTE_OBB";

  /// \brief The BulkData objects this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData objects this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Number of parts that this method acts on.
  size_t num_parts_ = 0;

  /// \brief Vector of pointers to the parts that this class will act upon.
  std::vector<stk::mesh::Part *> part_ptr_vector_;

  /// \brief Kernels corresponding to each of the specified parts.
  std::vector<std::shared_ptr<mundy::meta::MetaKernelBase<void>>> compute_obb_kernel_ptrs_;
  //@}
};  // ComputeOBB

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTEOBB_HPP_
