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
#include <mundy_core/throw_assert.hpp>      // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>          // for mundy::mesh::MetaData
#include <mundy_meta/MeshRequirements.hpp>  // for mundy::meta::MeshRequirements
#include <mundy_meta/MetaFactory.hpp>       // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel, mundy::meta::MetaKernel
#include <mundy_meta/MetaMethod.hpp>        // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>      // for MUNDY_REGISTER_METACLASS
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter

namespace mundy {

namespace meta {

/// \brief A class that dispatches the execution of multiple kernels.
///
/// This class is a concrete implementation of the \c MetaMethod interface. It provides a means of dispatching a
/// collection of kernels (each associated with potentially different parts) on a subset of the entities in the mesh. In
/// a sense, it's role is to streamline the registration of collections of kernels and to provide a means of executing
/// them in a single call. It uses subsetting and part unions to properly dispatch the kernels on the correct entities
/// AND (importantly) to guarantee that a kernel only acts on an entity once even if that entity is in multiple valid
/// parts.
///
/// \note All kernels within a MetaKernelDispatcher must act on entities of the same rank. Otherwise, it's not clear how to
/// properly dispatch the kernels in a way that guarantees that each entity is only acted upon once. For example, a user
/// might erroneously create a kernel that acts on a ELEMENT_RANKED entities in a PARTICLE topology part and another
/// that acts on NODE_RANK entities in the same part. If the NODE_RANK entity kernel changes the state of the elements, then
/// the order of the kernels will matter.
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
    const int num_specified_kernels = kernels_sublist.get<int>("count");

    auto mesh_requirements_ptr = std::make_shared<mundy::meta::MeshRequirements>();
    for (int i = 0; i < num_specified_kernels; i++) {
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
      const int num_specified_kernels = kernels_sublist.get<int>("count");
      for (int i = 0; i < num_specified_kernels; i++) {
        Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));

        // Ensure that each kernel has a name.
        mundy::meta::check_required_parameter<std::string>(&kernel_params, "name");

        // Validate and fill parameters for this kernel.
        const std::string kernel_name = kernel_params.get<std::string>("name");
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
      const int num_specified_kernels = kernels_sublist.get<int>("count");
      for (int i = 0; i < num_specified_kernels; i++) {
        Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));

        // Ensure that each kernel has a name.
        mundy::meta::check_required_parameter<std::string>(&kernel_params, "name");

        // Validate and fill parameters for this kernel.
        const std::string kernel_name = kernel_params.get<std::string>("name");
        OurKernelFactory::validate_mutable_parameters_and_set_defaults(kernel_name, &kernel_params);
      }
    } else {
      // Validate and fill parameters for any kernel in our registry.
      Teuchos::ParameterList &kernels_sublist = mutable_params_ptr->sublist("kernels", false);
      const int num_specified_kernels = OurKernelFactory::num_registered_classes();
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
  //@}

  //! \name Setters
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;
  //@}

  //! \name Getters
  //@{

  /// \brief Get valid entity parts for the kernel.
  /// By "valid entity parts," we mean the parts whose entities the kernel can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override;
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
  int num_active_kernels_ = 0;

  /// \brief The entity rank that the kernels acts on.
  stk::topology::rank_t kernel_entity_rank_ = stk::topology::INVALID_RANK;

  /// \brief The valid entity parts for the kernel.
  std::vector<stk::mesh::Part *> valid_entity_parts_;

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

  // Populate our internal members.
  Teuchos::ParameterList &kernels_sublist = valid_fixed_params.sublist("kernels", true);
  num_active_kernels_ = kernels_sublist.get<int>("count");
  if (num_active_kernels_ > 0) {
    kernel_ptrs_.reserve(num_active_kernels_);
    valid_entity_parts_.reserve(num_active_kernels_);
    for (int i = 0; i < num_active_kernels_; i++) {
      // Create the kernel and store it in the kernel_ptrs_ vector.
      Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));
      const std::string kernel_name = kernel_params.get<std::string>("name");
      kernel_ptrs_.push_back(OurKernelFactory::create_new_instance(kernel_name, bulk_data_ptr_, kernel_params));

      // Store the entity rank and ensure that it is the same for all kernels.
      if (i == 0) {
        kernel_entity_rank_ = kernel_ptrs_[0]->get_entity_rank();
      } else {
        MUNDY_THROW_ASSERT(kernel_ptrs_[i]->get_entity_rank() == kernel_entity_rank_, std::invalid_argument,
                          "MetaKernelDispatcher: All kernels in a dispatcher must act on entities of the same rank.");
      }

      // Get the valid entity parts for the kernel.
      auto valid_entity_parts_i = kernel_ptrs_[i]->get_valid_entity_parts();
      valid_entity_parts_.insert(valid_entity_parts_.end(), valid_entity_parts_i.begin(), valid_entity_parts_i.end());
    }
  }
}
//}

// \name MetaFactory static interface implementation
//{

template <typename RegistryIdentifier>
void MetaKernelDispatcher<RegistryIdentifier>::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  validate_mutable_parameters_and_set_defaults(&valid_mutable_params);

  // Parse the parameters
  Teuchos::ParameterList &kernels_sublist = valid_mutable_params.sublist("kernels", true);
  MUNDY_THROW_ASSERT(num_active_kernels_ == kernels_sublist.get<int>("count"), std::invalid_argument,
                     "MetaKernelDispatcher: Internal error. Mismatch between the stored kernel count\n"
                         << "and the parameter list kernel count. This should not happen.\n"
                         << "Please contact the development team.");
  for (int i = 0; i < num_active_kernels_; i++) {
    Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));
    kernel_ptrs_[i]->set_mutable_params(kernel_params);
  }
}
//}

// \name Getters
//{

template <typename RegistryIdentifier>
std::vector<stk::mesh::Part *> MetaKernelDispatcher<RegistryIdentifier>::get_valid_entity_parts() const {
  return valid_entity_parts_;
}
//}

// \name Actions
//{

template <typename RegistryIdentifier>
void MetaKernelDispatcher<RegistryIdentifier>::execute(const stk::mesh::Selector &input_selector) {
  for (int i = 0; i < num_active_kernels_; i++) {
    kernel_ptrs_[i]->setup();
  }

  for (int i = 0; i < num_active_kernels_; i++) {
    // For each kernel, we only want to evaluate the kernel ONCE for each entity in our valid entity parts.
    // We do so via taking the union of our valid entity parts and the input selector.
    auto kernel_ptr_i = kernel_ptrs_[i];
    auto valid_entity_parts_i = kernel_ptr_i->get_valid_entity_parts();

    stk::mesh::Selector locally_owned_intersection_with_valid_entity_parts =
        stk::mesh::Selector(meta_data_ptr_->locally_owned_part()) & input_selector;
    for (auto *part_ptr_i : valid_entity_parts_i) {
      locally_owned_intersection_with_valid_entity_parts &= *part_ptr_i;
    }

    stk::mesh::for_each_entity_run(
        *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_), kernel_entity_rank_,
        locally_owned_intersection_with_valid_entity_parts,
        [&kernel_ptr_i]([[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &entity) {
          kernel_ptr_i->execute(entity);
        });
  }

  for (int i = 0; i < num_active_kernels_; i++) {
    kernel_ptrs_[i]->finalize();
  }
}
//}

//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_KERNELDISPATCHER_HPP_
