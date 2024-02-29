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
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for MUNDY_REGISTER_METACLASS
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter

namespace mundy {

namespace meta {

// TODO(palmerb4): If string lookups turn egregiously slow, we need to modify this class to allow for arbitrary
// registration types, just like MetaFactory. For now, we'll default to using strings as the registration type.

/// \brief A helper class for defining MetaMethods whose only job is to dispatch different techniques for carrying out a
/// task with optional constrains on their fixed/mutable params.
///
/// Use this class to create a collection of techniques (each with their own specialized fixed/mutable params and mesh
/// requirements), constrain the fixed/mutable params of those techniques, and then execute the desired technique.
///
/// The DerivedType only has one requirement: it must contain a get_valid_forwarded_technique_fixed_params() and a
/// get_valid_forwarded_technique_mutable_params() method. These methods will specify the fixed and mutable parameters
/// that the method will accept and forward to the enabled technique. As such, we throw if any of the registered
/// techniques don't have these parameters in their valid fixed/mutable params.
template <typename DerivedType, typename TechniquePolymorphicBaseType,
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
        MetaTechniqueDispatcher<DerivedType, TechniquePolymorphicBaseType,
                                technique_factory_registration_string_value_wrapper,
                                default_technique_name_wrapper>::get_valid_fixed_params());

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
        MetaTechniqueDispatcher<DerivedType, TechniquePolymorphicBaseType,
                                technique_factory_registration_string_value_wrapper,
                                default_technique_name_wrapper>::get_valid_fixed_params());

    // At this point, the only parameters are the enabled technique name, the forwarded parameters for the enabled
    // technique, and the non-forwarded technique params within the technique sublists. We'll loop over all parameters
    // that aren't in the technique sublists and forward them to the enabled technique.
    for (Teuchos::ParameterList::ConstIterator i = valid_fixed_params.begin(); i != valid_fixed_params.end(); i++) {
      const std::string &param_name = valid_fixed_params.name(i);
      const Teuchos::ParameterEntry &param_entry = valid_fixed_params.getEntry(param_name);
      if (!valid_fixed_params.isSublist(param_name) && param_name != "technique_name") {
        for (int j = 0; j < OurMethodFactory::num_registered_classes(); j++) {
          const std::string technique_name = std::string(OurMethodFactory::get_keys()[j]);
          Teuchos::ParameterList &technique_params = valid_fixed_params.sublist(technique_name);
          technique_params.setEntry(param_name, param_entry);
        }
      }
    }

    // Fetch the technique sublist and return its parameters.
    const std::string enabled_technique_name = valid_fixed_params.get<std::string>("technique_name");
    const Teuchos::ParameterList &enabled_technique_params = valid_fixed_params.sublist(enabled_technique_name);
    return OurMethodFactory::get_mesh_requirements(enabled_technique_name, enabled_technique_params);
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    const Teuchos::ParameterList valid_forwarded_technique_fixed_params =
        DerivedType::get_valid_forwarded_technique_fixed_params();
    return get_valid_enabled_techniques_and_technique_params(
        [&valid_forwarded_technique_fixed_params](const std::string &technique_name) {
          Teuchos::ParameterList technique_params = OurMethodFactory::get_valid_fixed_params(technique_name);
          for (Teuchos::ParameterList::ConstIterator i = valid_forwarded_technique_fixed_params.begin();
               i != valid_forwarded_technique_fixed_params.end(); i++) {
            const std::string &parameter_name = valid_forwarded_technique_fixed_params.name(i);

            MUNDY_THROW_ASSERT(technique_params.isParameter(parameter_name), std::logic_error,
                               "MetaTechniqueDispatcher: The technique "
                                   << technique_name << " does not have the required (forwarded) parameter "
                                   << parameter_name << " in its valid fixed params.");
            technique_params.remove(parameter_name);
          }
          return technique_params;
        });
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    const Teuchos::ParameterList valid_forwarded_technique_mutable_params =
        DerivedType::get_valid_forwarded_technique_mutable_params();
    return get_valid_enabled_techniques_and_technique_params(
        [&valid_forwarded_technique_mutable_params](const std::string &technique_name) {
          Teuchos::ParameterList technique_params = OurMethodFactory::get_valid_mutable_params(technique_name);
          for (Teuchos::ParameterList::ConstIterator i = valid_forwarded_technique_mutable_params.begin();
               i != valid_forwarded_technique_mutable_params.end(); i++) {
            const std::string &parameter_name = valid_forwarded_technique_mutable_params.name(i);

            MUNDY_THROW_ASSERT(technique_params.isParameter(parameter_name), std::logic_error,
                               "MetaTechniqueDispatcher: The technique "
                                   << technique_name << " does not have the required (forwarded) parameter "
                                   << parameter_name << " in its valid mutable params.");
            technique_params.remove(parameter_name);
          }
          return technique_params;
        });
  }

  /// \brief Get the unique registration identifier associated with our technique factory.
  static std::string get_technique_factory_registration_id() {
    return technique_factory_registration_string_value_wrapper.to_string();
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override {
    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_mutable_params = mutable_params;
    valid_mutable_params.validateParametersAndSetDefaults(
        MetaTechniqueDispatcher<DerivedType, TechniquePolymorphicBaseType,
                                technique_factory_registration_string_value_wrapper,
                                default_technique_name_wrapper>::get_valid_mutable_params());

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

  /// @brief Get the valid enabled techniques and their parameters.
  /// @param get_technique_params_func [in] A function that returns the valid parameters for a technique given its name
  static Teuchos::ParameterList get_valid_enabled_techniques_and_technique_params(
      const std::function<Teuchos::ParameterList(const std::string &)> &get_technique_params_func) {
    Teuchos::ParameterList default_parameter_list;

    std::string valid_names = OurMethodFactory::get_keys_as_string();
    default_parameter_list.set("technique_name", default_technique_name_wrapper.value,
                               "The name of the technique to use. Valid names are: " + valid_names);

    // Because this is the valid params we list ALL possible parameters. We expect the parameters for
    // each technique to be a sublist of this list with the same name as the technique.
    for (auto &key : OurMethodFactory::get_keys()) {
      std::string valid_technique_name = key;
      Teuchos::ParameterList &technique_params = default_parameter_list.sublist(valid_technique_name);
      technique_params.setParameters(get_technique_params_func(valid_technique_name));
    }

    return default_parameter_list;
  }
  //@}

};  // MetaTechniqueDispatcher

//! \brief Type specializations for different polymorphic base types.
//@{

template <typename DerivedType, typename ReturnType,
          mundy::meta::RegistrationStringValueWrapper registration_string_value_wrapper,
          mundy::core::StringLiteral default_technique_name_wrapper>
struct MetaMethodSubsetExecutionDispatcher
    : public mundy::meta::MetaMethodSubsetExecutionInterface<ReturnType>,
      public MetaTechniqueDispatcher<DerivedType, mundy::meta::MetaMethodSubsetExecutionInterface<ReturnType>,
                                     registration_string_value_wrapper, default_technique_name_wrapper> {
  using PolymorphicBaseType = mundy::meta::MetaMethodSubsetExecutionInterface<ReturnType>;

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                  const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<DerivedType>(bulk_data_ptr, fixed_params);
  }

  //! \name MetaMethodSubsetExecutionInterface's interface implementation
  //@{

  /// \brief Get valid entity parts for the method.
  /// By "valid entity parts," we mean the parts whose entities this method can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override {
    return technique_ptr_->get_valid_entity_parts();
  }

  /// \brief Run the method's core calculation.
  ReturnType execute(const stk::mesh::Selector &input_selector) override {
    // Forward the inputs to the technique.
    technique_ptr_->execute(input_selector);
  }
  //@}
};  // MetaMethodSubsetExecutionDispatcher

template <typename DerivedType, typename ReturnType,
          mundy::meta::RegistrationStringValueWrapper registration_string_value_wrapper,
          mundy::core::StringLiteral default_technique_name_wrapper>
struct MetaMethodPairwiseSubsetExecutionDispatcher
    : public mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<ReturnType>,
      public MetaTechniqueDispatcher<DerivedType, mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<ReturnType>,
                                     registration_string_value_wrapper, default_technique_name_wrapper> {
  using PolymorphicBaseType = mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<ReturnType>;

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                  const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<DerivedType>(bulk_data_ptr, fixed_params);
  }

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
  ReturnType execute(const stk::mesh::Selector &source_selector, const stk::mesh::Selector &target_selector) override {
    // Forward the inputs to the technique.
    technique_ptr_->execute(source_selector, target_selector);
  }
  //@}
};  // MetaMethodSubsetExecutionDispatcher
//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METATECHNIQUEDISPATCHER_HPP_
