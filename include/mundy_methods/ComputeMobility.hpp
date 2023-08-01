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

#ifndef MUNDY_METHODS_COMPUTEMOBILITY_HPP_
#define MUNDY_METHODS_COMPUTEMOBILITY_HPP_

/// \file ComputeMobility.hpp
/// \brief Declaration of the ComputeMobility class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector
#include <stk_topology/topology.hpp>   // for stk::topology

// Mundy libs
#include <mundy/throw_assert.hpp>           // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>          // for mundy::mesh::MetaData
#include <mundy_meta/MeshRequirements.hpp>  // for mundy::meta::MeshRequirements
#include <mundy_meta/MetaFactory.hpp>       // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel, mundy::meta::MetaKernel
#include <mundy_meta/MetaMethod.hpp>        // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>      // for mundy::meta::GlobalMetaMethodRegistry

namespace mundy {

namespace methods {

/// \class ComputeMobility
/// \brief Method for mapping the body force on a rigid body to the rigid body velocity.
class ComputeMobility : public mundy::meta::MetaMethod<void> {
 public:
  //! \name Typedefs
  //@{
  
  using RegistrationType = std::string_view;
  using PolymorphicBaseType = mundy::meta::MetaMethod<void>;
  using OurMethodFactory = mundy::meta::MetaMethodFactory<void, ComputeMobility>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeMobility() = delete;

  /// \brief Constructor
  ComputeMobility(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
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
    validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

    // Fetch the technique sublist and return its parameters.
    Teuchos::ParameterList &technique_params = valid_fixed_params.sublist("technique");
    const std::string technique_name = technique_params.get<std::string>("name");

    return OurMethodFactory::get_mesh_requirements(technique_name, technique_params);
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const fixed_params_ptr) {
    // Fetch the technique sublist and return its parameters.
    Teuchos::ParameterList &technique_params = fixed_params_ptr->sublist("technique", false);
    if (technique_params.isParameter("name")) {
      const bool valid_type = technique_params.INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      MUNDY_THROW_ASSERT(
          valid_type, std::invalid_argument,
          "ComputeMobility: Type error. Given a parameter with name 'name' but with a type other than std::string");
    } else {
      technique_params.set("name", default_technique_name_, "The name of the technique to use.");
    }

    const std::string technique_name = technique_params.get<std::string>("name");
    OurMethodFactory::validate_fixed_parameters_and_set_defaults(technique_name, &technique_params);
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const mutable_params_ptr) {
    // Fetch the technique sublist and return its parameters.
    Teuchos::ParameterList &technique_params = mutable_params_ptr->sublist("technique", false);
    if (technique_params.isParameter("name")) {
      const bool valid_type = technique_params.INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      MUNDY_THROW_ASSERT(
          valid_type, std::invalid_argument,
          "ComputeMobility: Type error. Given a parameter with name 'name' but with a type other than std::string");
    } else {
      technique_params.set("name", default_technique_name_, "The name of the technique to use.");
    }

    const std::string technique_name = technique_params.get<std::string>("name");
    OurMethodFactory::validate_mutable_parameters_and_set_defaults(technique_name, &technique_params);
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
    return std::make_shared<ComputeMobility>(bulk_data_ptr, fixed_params);
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
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_technique_name_ = "RIGID_BODY_MOTION";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other methods in our MetaMethodRegistry.
  static constexpr std::string_view registration_id_ = "COMPUTE_MOBILITY";

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Method corresponding to the specified technique.
  std::shared_ptr<mundy::meta::MetaMethod<void>> technique_ptr_;
  //@}
};  // ComputeMobility

}  // namespace methods

}  // namespace mundy

//! \name Registration
//@{

/// @brief Register ComputeMobility with the global MetaMethodFactory.
MUNDY_REGISTER_METACLASS(mundy::methods::ComputeMobility, mundy::meta::GlobalMetaMethodFactory<void>)
//@}

#endif  // MUNDY_METHODS_COMPUTEMOBILITY_HPP_
