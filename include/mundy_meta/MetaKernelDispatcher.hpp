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

#ifndef MUNDY_META_KERNELDISPATCHER_HPP_
#define MUNDY_META_KERNELDISPATCHER_HPP_

/// \file FieldRequirements.hpp
/// \brief Declaration of the FieldRequirements class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>  // for stk::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector
#include <stk_topology/topology.hpp>        // for stk::topology

// Mundy libs
#include <mundy/throw_assert.hpp>           // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>          // for mundy::mesh::MetaData
#include <mundy_meta/MeshRequirements.hpp>  // for mundy::meta::MeshRequirements
#include <mundy_meta/MetaFactory.hpp>       // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel, mundy::meta::MetaKernel
#include <mundy_meta/MetaMethod.hpp>        // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>      // for MUNDY_REGISTER_METACLASS

namespace mundy {

namespace meta {

template <typename RegistryIdentifier>
class MetaKernelDispatcher : public mundy::meta::MetaMethod<void> {
 public:
  //! \name Typedefs
  //@{

  using RegistrationType = std::string_view;
  using PolymorphicBaseType = mundy::meta::MetaMethod<void>;
  using OurKernelFactory = mundy::meta::MetaKernelFactory<void, RegistryIdentifier>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  MetaKernelDispatcher() = delete;

  /// \brief Constructor
  MetaKernelDispatcher(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name MetaFactory static interface implementation
  //@{

  /// \brief Get the requirements that this method imposes upon each particle and/or constraint.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c MeshRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::MeshRequirements> get_mesh_requirements(
      const Teuchos::ParameterList &fixed_params) {
    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    validate_fixed_parameters_and_set_defaults(&valid_fixed_params);
    Teuchos::ParameterList &kernels_sublist = valid_fixed_params.sublist("kernels");
    const unsigned num_specified_kernels = kernels_sublist.get<unsigned>("count");

    auto mesh_requirements_ptr = std::make_shared<mundy::meta::MeshRequirements>();
    for (size_t i = 0; i < num_specified_kernels; i++) {
      Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));
      const std::string kernel_name = kernel_params.get<std::string>("name");
      mesh_requirements_ptr->merge(OurKernelFactory::get_mesh_requirements(kernel_name, kernel_params));
    }

    return mesh_requirements_ptr;
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void validate_fixed_parameters_and_set_defaults(Teuchos::ParameterList *const fixed_params_ptr) {
    if (fixed_params_ptr->isSublist("kernels")) {
      // Only validate and fill parameters for the given kernels.
      Teuchos::ParameterList &kernels_sublist = fixed_params_ptr->sublist("kernels", true);
      const unsigned num_specified_kernels = kernels_sublist.get<unsigned>("count");
      for (size_t i = 0; i < num_specified_kernels; i++) {
        Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));

        // Ensure that each kernel has a name and associated part name.
        const std::string kernel_name = kernel_params.get<std::string>("name");
        MUNDY_THROW_ASSERT(
            kernel_params.isParameter("part_name"), std::invalid_argument,
            "MetaKernelDispatcher: Missing required parameter 'part_name' for kernel " + std::to_string(i));
        const bool valid_type = kernel_params.INVALID_TEMPLATE_QUALIFIER isType<std::string>("part_name");
        MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                           "MetaKernelDispatcher: Type error. Given a parameter with name 'part_name'  for kernel " +
                               std::to_string(i) + " but with a type other than std::string");

        // Validate and fill parameters for this kernel.
        OurKernelFactory::validate_fixed_parameters_and_set_defaults(kernel_name, &kernel_params);
      }
    } else {
      // Validate and fill parameters for any kernel in our registry.
      Teuchos::ParameterList &kernels_sublist = fixed_params_ptr->sublist("kernels", false);
      const unsigned num_specified_kernels = OurKernelFactory::num_registered_classes();
      kernels_sublist.set("count", num_specified_kernels);
      int i = 0;
      for (auto &key : OurKernelFactory::get_keys()) {
        Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i), false);
        kernel_params.set("name", std::string(key));
        OurKernelFactory::validate_fixed_parameters_and_set_defaults(key, &kernel_params);
        i++;
      }
    }
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void validate_mutable_parameters_and_set_defaults(Teuchos::ParameterList *const mutable_params_ptr) {
    if (mutable_params_ptr->isSublist("kernels")) {
      // Only validate and fill parameters for the given kernels.
      Teuchos::ParameterList &kernels_sublist = mutable_params_ptr->sublist("kernels", true);
      const unsigned num_specified_kernels = kernels_sublist.get<unsigned>("count");
      for (size_t i = 0; i < num_specified_kernels; i++) {
        Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));

        // Ensure that each kernel has a name and associated part name.
        const std::string kernel_name = kernel_params.get<std::string>("name");
        MUNDY_THROW_ASSERT(
            kernel_params.isParameter("part_name"), std::invalid_argument,
            "MetaKernelDispatcher: Missing required parameter 'part_name' for kernel " + std::to_string(i));
        const bool valid_type = kernel_params.INVALID_TEMPLATE_QUALIFIER isType<std::string>("part_name");
        MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                           "MetaKernelDispatcher: Type error. Given a parameter with name 'part_name'  for kernel " +
                               std::to_string(i) + " but with a type other than std::string");

        // Validate and fill parameters for this kernel.
        OurKernelFactory::validate_mutable_parameters_and_set_defaults(kernel_name, &kernel_params);
      }
    } else {
      // Validate and fill parameters for any kernel in our registry.
      Teuchos::ParameterList &kernels_sublist = mutable_params_ptr->sublist("kernels", false);
      const unsigned num_specified_kernels = OurKernelFactory::num_registered_classes();
      kernels_sublist.set("count", num_specified_kernels);
      int i = 0;
      for (auto &key : OurKernelFactory::get_keys()) {
        Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i), false);
        kernel_params.set("name", std::string(key));
        OurKernelFactory::validate_mutable_parameters_and_set_defaults(key, &kernel_params);
        i++;
      }
    }
  }

  /// \brief Get the unique registration identifier. Ideally, this should be unique and not shared by any other \c
  /// MetaMethod.
  static RegistrationType get_registration_id() {
    return registration_id_;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethod<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<MetaKernelDispatcher<RegistryIdentifier>>(bulk_data_ptr, fixed_params);
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
  static constexpr std::string_view registration_id_ = "META_KERNEL_DISPATCHER";

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Number of active kernels.
  size_t num_active_kernels_ = 0;

  /// \brief Vector of part pointers, one for each active kernel.
  std::vector<stk::mesh::Part *> part_ptr_vector_;

  /// \brief Vector of kernels, one for each active kernel.
  std::vector<std::shared_ptr<mundy::meta::MetaKernel<void>>> kernel_ptrs_;
  //@}
};  // MetaKernelDispatcher

//! \name Template specializations
//@{

// \name Constructors and destructor
//{

template <typename RegistryIdentifier>
MetaKernelDispatcher<RegistryIdentifier>::MetaKernelDispatcher(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                          const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument,
                     "MetaKernelDispatcher: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

  // Parse the parameters
  Teuchos::ParameterList &kernels_sublist = valid_fixed_params.sublist("kernels", true);
  num_active_kernels_ = kernels_sublist.get<unsigned>("count");
  part_ptr_vector_.reserve(num_active_kernels_);
  kernel_ptrs_.reserve(num_active_kernels_);
  for (size_t i = 0; i < num_active_kernels_; i++) {
    Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));
    const std::string kernel_name = kernel_params.get<std::string>("name");
    const std::string associated_part_name = kernel_params.get<std::string>("part_name");
    part_ptr_vector_.push_back(meta_data_ptr_->get_part(associated_part_name));
    kernel_ptrs_.push_back(OurKernelFactory::create_new_instance(kernel_name, bulk_data_ptr_, kernel_params));
  }
}
//}

// \name MetaFactory static interface implementation
//{

template <typename RegistryIdentifier>
void MetaKernelDispatcher<RegistryIdentifier>::set_mutable_params(
    const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  validate_mutable_parameters_and_set_defaults(&valid_mutable_params);

  // Parse the parameters
  Teuchos::ParameterList &kernels_sublist = valid_mutable_params.sublist("kernels", true);
  MUNDY_THROW_ASSERT(num_active_kernels_ == kernels_sublist.get<unsigned>("count"), std::invalid_argument,
                     "MetaKernelDispatcher: Internal error. Mismatch between the stored kernel count\n"
                         << "and the parameter list kernel count. This should not happen.\n"
                         << "Please contact the development team.");
  for (size_t i = 0; i < num_active_kernels_; i++) {
    Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));
    kernel_ptrs_[i]->set_mutable_params(kernel_params);
  }
}
//}

// \name Actions
//{

template <typename RegistryIdentifier>
void MetaKernelDispatcher<RegistryIdentifier>::execute(const stk::mesh::Selector &input_selector) {
  for (size_t i = 0; i < num_active_kernels_; i++) {
    kernel_ptrs_[i]->setup();
  }

  for (size_t i = 0; i < num_active_kernels_; i++) {
    auto part_ptr_i = part_ptr_vector_[i];
    auto kernel_ptr_i = kernel_ptrs_[i];

    stk::mesh::Selector locally_owned_intersection_with_part_i =
        stk::mesh::Selector(meta_data_ptr_->locally_owned_part()) & stk::mesh::Selector(*part_ptr_i) & input_selector;

    stk::mesh::for_each_entity_run(
        *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_), stk::topology::ELEMENT_RANK,
        locally_owned_intersection_with_part_i,
        [&kernel_ptr_i]([[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &element) {
          kernel_ptr_i->execute(element);
        });
  }

  for (size_t i = 0; i < num_active_kernels_; i++) {
    kernel_ptrs_[i]->finalize();
  }
}
//}

//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_KERNELDISPATCHER_HPP_
