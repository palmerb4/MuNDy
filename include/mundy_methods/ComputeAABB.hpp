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

#ifndef MUNDY_METHODS_COMPUTEAABB_HPP_
#define MUNDY_METHODS_COMPUTEAABB_HPP_

/// \file ComputeAABB.hpp
/// \brief Declaration of the ComputeAABB class

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
#include <mundy_meta/MeshRequirements.hpp>  // for mundy::meta::MeshRequirements
#include <mundy_meta/MetaFactory.hpp>       // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaMethod.hpp>        // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>      // for mundy::meta::GlobalMetaMethodRegistry

namespace mundy {

namespace methods {

/// \class ComputeAABB
/// \brief Method for computing the axis aligned boundary box of different parts.
class ComputeAABB : public mundy::meta::MetaMethod<void, ComputeAABB>,
                    public mundy::meta::GlobalMetaMethodRegistry<void, ComputeAABB> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeAABB() = delete;

  /// \brief Constructor
  ComputeAABB(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name Typedefs
  //@{

  using OurKernelFactory = mundy::meta::MetaKernelFactory<void, ComputeAABB>;

  template <typename ClassToRegister>
  using OurKernelRegistry = mundy::meta::MetaKernelRegistry<void, ClassToRegister, ComputeAABB>;
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
    static_validate_fixed_parameters_and_set_defaults(&valid_fixed_params);
    Teuchos::ParameterList &kernels_sublist = valid_fixed_params.sublist("kernels");
    const unsigned num_specified_kernels = kernels_sublist.get<unsigned>("count");

    std::shared_ptr<mundy::meta::MeshRequirements> mesh_requirements_ptr;
    for (size_t i = 0; i < num_specified_kernels; i++) {
      Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));
      const std::string kernel_name = kernel_params.get<std::string>("name");
      mesh_requirements_ptr->merge(OurKernelFactory::get_mesh_requirements(kernel_name, kernel_params));
    }

    return mesh_requirements_ptr;
  }

  /// \brief Validate the default fixed parameters for this class (those that impact the mesh requirements) and set
  /// their defaults.
  static void details_static_validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList const *fixed_params_ptr) {
    Teuchos::ParameterList params = *fixed_params_ptr;

    if (params.isSublist("kernels")) {
      // Only validate and fill parameters for the given kernels.
      Teuchos::ParameterList &kernels_sublist = params.sublist("kernels", true);
      const unsigned num_specified_kernels = kernels_sublist.get<unsigned>("count");
      for (size_t i = 0; i < num_specified_kernels; i++) {
        Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));
        const std::string kernel_name = kernel_params.get<std::string>("name");
        OurKernelFactory::validate_fixed_parameters_and_set_defaults(kernel_name, &kernel_params);
      }
    } else {
      // Validate and fill parameters for any kernel in our registry.
      Teuchos::ParameterList &kernels_sublist = params.sublist("kernel_params", false);
      const unsigned num_specified_kernels = OurKernelFactory::num_registered_classes();
      kernels_sublist.set("count", num_specified_kernels);
      int i = 0;
      for (auto &key : OurKernelFactory::get_keys()) {
        Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i), false);
        kernel_params.set("name", key);
        OurKernelFactory::validate_fixed_parameters_and_set_defaults(key, &kernel_params);
        i++;
      }
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
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<ComputeAABB>(bulk_data_ptr, fixed_params);
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
  static constexpr std::string_view class_identifier_ = "COMPUTE_AABB";

  /// \brief The BulkData objects this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData objects this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Number of parts that this method acts on.
  size_t num_parts_ = 0;

  /// \brief Vector of pointers to the parts that this class will act upon.
  std::vector<stk::mesh::Part *> part_ptr_vector_;

  /// \brief Kernels corresponding to each of the specified parts.
  std::vector<std::shared_ptr<mundy::meta::MetaKernelBase<void>>> compute_aabb_kernel_ptrs_;
  //@}
};  // ComputeAABB

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTEAABB_HPP_
