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

#ifndef MUNDY_META_METATECHNIQUEDISPATCHER_HPP_
#define MUNDY_META_METATECHNIQUEDISPATCHER_HPP_

/// \file MetaTechniqueDispatcher.hpp
/// \brief Declaration of the MetaTechniqueDispatcher class

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
#include <mundy_core/StringLiteral.hpp>                       // for mundy::core::StringLiteral
#include <mundy_core/throw_assert.hpp>                        // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                            // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                            // for mundy::mesh::MetaData
#include <mundy_meta/MeshRequirements.hpp>                    // for mundy::meta::MeshRequirements
#include <mundy_meta/MetaFactory.hpp>                         // for mundy::meta::MetaMethodFactory
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel, mundy::meta::MetaKernel
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for MUNDY_REGISTER_METACLASS
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter

namespace mundy {

namespace meta {

// TODO(palmerb4): If string lookups turn egregiously slow, we need to modify this class to allow for arbitrary
// registration types, just like MetaFactory. For now, we'll default to using strings as the registration type.
template <typename TechniquePolymorphicBaseType,
          mundy::meta::RegistrationStringValueWrapper technique_factory_registration_string_value_wrapper,
          mundy::meta::RegistrationStringValueWrapper default_technique_name_wrapper>
class MetaTechniqueDispatcher {
 public:
  //! \name Typedefs
  //@{

  using OurMethodFactory = mundy::meta::StringBasedMetaFactory<TechniquePolymorphicBaseType,
                                                               technique_factory_registration_string_value_wrapper>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  MetaTechniqueDispatcher() = delete;

  /// \brief Constructor
  MetaTechniqueDispatcher(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(
        MetaTechniqueDispatcher<RegistryIdentifier, DefaultMethodStringLiteral>::get_valid_fixed_params());

    // Forward the inputs to the technique.
    const std::string technique_name = valid_fixed_params.get<std::string>("technique_name");
    Teuchos::ParameterList technique_params = valid_fixed_params.sublist(technique_name);
    technique_ptr_ = OurMethodFactory::create_new_instance(technique_name, bulk_data_ptr, technique_params);
  }
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
      [[maybe_unused]] const Teuchos::ParameterList &fixed_params) {
    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(
        MetaTechniqueDispatcher<RegistryIdentifier, DefaultMethodStringLiteral>::get_valid_fixed_params());

    // Fetch the technique sublist and return its parameters.
    const std::string technique_name = valid_fixed_params.get<std::string>("technique_name");
    const Teuchos::ParameterList &technique_params = valid_fixed_params.sublist(technique_name);
    return OurMethodFactory::get_mesh_requirements(technique_name, technique_params);
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList default_fixed_parameter_list;
    return get_valid_enabled_kernels_and_kernel_params(default_fixed_parameter_list,
                                                       [](const std::string &name) -> Teuchos::ParameterList {
                                                         return OurMethodFactory::get_valid_fixed_params(name);
                                                       });
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static Teuchos::ParameterList default_mutable_parameter_list;
    return get_valid_enabled_kernels_and_kernel_params(default_mutable_parameter_list,
                                                       [](const std::string &name) -> Teuchos::ParameterList {
                                                         return OurMethodFactory::get_valid_mutable_params(name);
                                                       });
  }

  /// \brief Get the unique registration identifier associated with our technique factory.
  static std::string get_technique_factory_registration_id() {
    return technique_factory_registration_string_value_wrapper.to_string();
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<DerivedType>(bulk_data_ptr, fixed_params);
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override {
    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_mutable_params = mutable_params;
    valid_mutable_params.validateParametersAndSetDefaults(
        MetaTechniqueDispatcher<RegistryIdentifier, DefaultMethodStringLiteral>::get_valid_mutable_params());

    // Forward the inputs to the technique.
    const std::string technique_name = valid_fixed_params.get<std::string>("technique_name");
    Teuchos::ParameterList technique_params = valid_fixed_params.sublist(technique_name);
    kernel_ptrs_[i]->set_mutable_params(kernel_params);
  }
  //@}

 private:
  //! \name Internal members
  //@{

  /// \brief Method corresponding to the specified technique.
  std::shared_ptr<TechniquePolymorphicBaseType> technique_ptr_;
  //@}

  //! \name Internal methods
  //@{

  /// @brief Get the valid enabled kernels and their parameters.
  /// @param get_kernel_params_func [in] A function that returns the valid parameters for a kernel given its name
  static Teuchos::ParameterList get_valid_enabled_kernels_and_kernel_params(
      const std::function<Teuchos::ParameterList(const std::string &)> &get_kernel_params_func) {
    Teuchos::ParameterList default_parameter_list;

    std::string valid_names = OurMethodFactory::get_keys_as_string();
    default_parameter_list.set("technique_name", default_technique_name_wrapper.value,
                               "The name of the technique to use. Valid names are: " + valid_names);

    // Because this is the valid params we list ALL possible parameters. We expect the parameters for
    // each technique to be a sublist of this list with the same name as the technique.
    for (auto &key : OurMethodFactory::get_keys()) {
      std::string valid_technique_name = key;
      Teuchos::ParameterList &technique_params = default_parameter_list.sublist(valid_technique_name);
      technique_params.setParameters(get_kernel_params_func(valid_technique_name));
    }

    return default_parameter_list;
  }
  //@}

};  // MetaTechniqueDispatcher

//! \brief Type specializations for different polymorphic base types.
//@{

template <mundy::meta::RegistrationStringValueWrapper registration_string_value_wrapper,
          mundy::core::StringLiteral default_technique_name_wrapper>
struct MetaMethodSubsetExecutionDispatcher
    : public mundy::meta::MetaMethodSubsetExecutionInterface<void>,
      public MetaTechniqueDispatcher<mundy::meta::MetaMethodSubsetExecutionInterface<void>,
                                     registration_string_value_wrapper, default_technique_name_wrapper> {
  using PolymorphicBaseType = mundy::meta::MetaMethodSubsetExecutionInterface<void>;

  //! \name MetaMethodSubsetExecutionInterface's interface implementation
  //@{

  /// \brief Get valid entity parts for the method.
  /// By "valid entity parts," we mean the parts whose entities this method can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override {
    return technique_ptr_->get_valid_entity_parts();
  }

  /// \brief Run the method's core calculation.
  void execute(const stk::mesh::Selector &input_selector) override {
    // Forward the inputs to the technique.
    technique_ptr_->execute(input_selector);
  }
  //@}
};  // MetaMethodSubsetExecutionDispatcher

template <mundy::meta::RegistrationStringValueWrapper registration_string_value_wrapper,
          mundy::core::StringLiteral default_technique_name_wrapper>
struct MetaMethodPairwiseSubsetExecutionDispatcher
    : public mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>,
      public MetaTechniqueDispatcher<mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>,
                                     registration_string_value_wrapper, default_technique_name_wrapper> {
  using PolymorphicBaseType = mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>;

  //! \name MetaMethodPairwiseSubsetExecutionInterface's interface implementation
  //@{

  /// \brief Get valid source entity parts for the method.
  /// By "valid source entity parts," we mean the parts whose entities this method can act on as source entities.
  std::vector<stk::mesh::Part *> get_valid_source_entity_parts() const override {
    return technique_ptr_->get_valid_source_entity_parts();
  }

  /// \brief Get valid target entity parts for the method.
  /// By "valid target entity parts," we mean the parts whose entities this method can act on as target entities.
  std::vector<stk::mesh::Part *> get_valid_target_entity_parts() const override {
    return technique_ptr_->get_valid_target_entity_parts();
  }

  /// \brief Run the method's core calculation.
  void execute(const stk::mesh::Selector &source_selector, const stk::mesh::Selector &target_selector) override {
    // Forward the inputs to the technique.
    technique_ptr_->execute(source_selector, target_selector);
  }
  //@}
};  // MetaMethodSubsetExecutionDispatcher
//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METATECHNIQUEDISPATCHER_HPP_
