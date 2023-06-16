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
#include <Teuchos_ParameterList.hpp>     // for Teuchos::ParameterList
#include <Teuchos_TestForException.hpp>  // for TEUCHOS_TEST_FOR_EXCEPTION
#include <stk_mesh/base/Entity.hpp>      // for stk::mesh::Entity
#include <stk_mesh/base/Part.hpp>        // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>    // for stk::mesh::Selector
#include <stk_topology/topology.hpp>     // for stk::topology

// Mundy libs
#include <mundy_meta/MetaFactory.hpp>       // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>        // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaMethod.hpp>        // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>      // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/PartRequirements.hpp>  // for mundy::meta::PartRequirements
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>          // for mundy::mesh::MetaData

namespace mundy {

namespace methods {

/// \class ComputeMobility
/// \brief Method for mapping the body force on a rigid body to the rigid body velocity.
class ComputeMobility : public mundy::meta::MetaMethod<void, ComputeMobility>,
                        public mundy::meta::MetaMethodRegistry<void, ComputeMobility> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeMobility() = delete;

  /// \brief Constructor
  ComputeMobility(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list);
  //@}

  //! \name MetaMethod interface implementation
  //@{

  /// \brief Get the requirements that this method imposes upon each particle and/or constraint.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::vector<std::shared_ptr<mundy::meta::PartRequirements>> details_static_get_part_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_parameter_list) {
    // Validate the input params. Use default parameters for any parameter not given.
    // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
    Teuchos::ParameterList valid_fixed_parameter_list = fixed_parameter_list;
    valid_fixed_parameter_list.validateParametersAndSetDefaults(static_get_valid_fixed_params());

    // Fetch the technique sublist and return its parameters.
    Teuchos::ParameterList &technique_parameter_list = valid_fixed_parameter_list.sublist("technique");
    const std::string technique_name = technique_parameter_list.get<std::string>("name");

    return mundy::meta::MetaMethodFactory<void, ComputeMobility>::get_part_requirements(technique_name,
                                                                                        technique_parameter_list);
  }

  /// \brief Get the default fixed parameters for this class (those that impact the part requirements).
  static Teuchos::ParameterList details_static_get_valid_fixed_params() {
    static Teuchos::ParameterList default_fixed_parameter_list;
    default_fixed_parameter_list.sublist("technique", false,
                                         "Sublist that defines the technique to use and its parameters.");
    return default_fixed_parameter_list;
  }

  /// \brief Get the default transient parameters for this class (those that do not impact the part requirements).
  static Teuchos::ParameterList details_static_get_valid_transient_params() {
    static Teuchos::ParameterList default_transient_parameter_list;
    return default_transient_parameter_list;
  }

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c
  /// MetaMethod.
  static std::string details_static_get_class_identifier() {
    return std::string(class_identifier_);
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethodBase<void>> details_static_create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list) {
    return std::make_shared<ComputeMobility>(bulk_data_ptr, fixed_parameter_list);
  }
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
  static constexpr std::string_view class_identifier_ = "COMPUTE_MOBILITY";

  /// \brief The BulkData objects this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData objects this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Method corresponding to the specified technique.
  std::shared_ptr<mundy::meta::MetaMethodBase<void>> technique_ptr_;
  //@}
};  // ComputeMobility

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTEMOBILITY_HPP_
