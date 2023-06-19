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

#ifndef MUNDY_METHODS_RESOLVE_CONSTRAINTS_TECHNIQUES_NONSMOOTHLCP_HPP_
#define MUNDY_METHODS_RESOLVE_CONSTRAINTS_TECHNIQUES_NONSMOOTHLCP_HPP_

/// \file NonSmoothLCP.hpp
/// \brief Declaration of the NonSmoothLCP class

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
#include <mundy_mesh/BulkData.hpp>               // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>               // for mundy::mesh::MetaData
#include <mundy_meta/MetaFactory.hpp>            // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>             // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaMethod.hpp>             // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>           // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/PartRequirements.hpp>       // for mundy::meta::PartRequirements
#include <mundy_methods/ResolveConstraints.hpp>  // for mundy::methods::ResolveConstraints

namespace mundy {

namespace methods {

namespace techniques {

/// \class NonSmoothLCP
/// \brief Method for mapping the body force on a rigid body to the rigid body velocity.
class NonSmoothLCP : public mundy::meta::MetaMethod<void, NonSmoothLCP>,
                     public ResolveConstraints::OutMethodRegistry<NonSmoothLCP> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  NonSmoothLCP() = delete;

  /// \brief Constructor
  NonSmoothLCP(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name Typedefs
  //@{

  using OurMethodFactory = mundy::meta::MetaMethodFactory<void, NonSmoothLCP>;

  template <typename ClassToRegister>
  using OurMethodRegistry = mundy::meta::MetaMethodRegistry<void, ClassToRegister, NonSmoothLCP>;
  //@}

  //! \name MetaMethod interface implementation
  //@{

  /// \brief Get the requirements that this method imposes upon each particle and/or constraint.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::MeshRequirements>(
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

      // Fetch the parameters for this part's sub-methods.
      Teuchos::ParameterList &part_map_rbf_to_rbv_params =
          part_params.sublist("methods").sublist("map_rigid_body_force_to_rigid_body_velocity");
      Teuchos::ParameterList &part_map_rbv_to_sv_params =
          part_params.sublist("methods").sublist("map_rigid_body_velocity_to_surface_velocity");
      Teuchos::ParameterList &part_map_sf_to_rbf_params =
          part_params.sublist("methods").sublist("map_surface_force_to_rigid_body_force");

      // Validate the method params and fill in defaults.
      const std::string rbf_to_rbv_class_id = part_map_rbf_to_rbv_params.get<std::string>("class_id");
      part_map_rbf_to_rbv_params.validateParametersAndSetDefaults(
          mundy::meta::MetaMethodFactory<void, NonSmoothLCP>::get_valid_params(class_id));
      const std::string rbv_to_sv_class_id = part_map_rbv_to_sv_params.get<std::string>("class_id");
      part_map_rbv_to_sv_params.validateParametersAndSetDefaults(
          mundy::meta::MetaMethodFactory<void, NonSmoothLCP>::get_valid_params(class_id));
      const std::string sf_to_rbf_class_id = part_map_sf_to_rbf_params.get<std::string>("class_id");
      part_map_sf_to_rbf_params.validateParametersAndSetDefaults(
          mundy::meta::MetaMethodFactory<void, NonSmoothLCP>::get_valid_params(class_id));

      // Merge the method requirements.
      part_requirements[i]->merge(mundy::meta::MetaMethodFactory<void, NonSmoothLCP>::get_part_requirements(
          rbf_to_rbv_class_id, part_map_rbf_to_rbv_params));
      part_requirements[i]->merge(mundy::meta::MetaMethodFactory<void, NonSmoothLCP>::get_part_requirements(
          rbv_to_sv_class_id, part_map_rbv_to_sv_params));
      part_requirements[i]->merge(mundy::meta::MetaMethodFactory<void, NonSmoothLCP>::get_part_requirements(
          sf_to_rbf_class_id, part_map_sf_to_rbf_params));
    }

    return part_requirements;
  }

  /// \brief Get the default fixed parameters for this class (those that impact the part requirements).
  static Teuchos::ParameterList details_static_get_valid_fixed_params() {
    static Teuchos::ParameterList default_fixed_params;
    Teuchos::ParameterList &method_params = default_fixed_params.sublist(
        "methods", false, "Sublist that defines the sub-methods and their parameters.");
    kernel_params.sublist(
        "map_rigid_body_force_to_rigid_body_velocity", false,
        "Sublist that defines the parameters for mapping from rigid body force to rigid body velocity.");
    kernel_params.sublist(
        "map_rigid_body_velocity_to_surface_velocity", false,
        "Sublist that defines the parameters for mapping from rigid body velocity to surface velocity.");
    kernel_params.sublist("map_surface_force_to_rigid_body_force", false,
                          "Sublist that defines the parameters for mapping from surface force to rigid body force.");
    return default_fixed_params;
  }

  /// \brief Get the default mutable parameters for this class (those that do not impact the part requirements).
  static Teuchos::ParameterList details_static_get_valid_mutable_params() {
    static Teuchos::ParameterList default_mutable_params;
    return default_mutable_params;
  }

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c
  /// MetaMethod.
  static std::string details_static_get_class_identifier() {
    return std::string(class_identifier_);
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethodBase<void>> details_static_create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<NonSmoothLCP>(bulk_data_ptr, fixed_params);
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
  static constexpr std::string_view class_identifier_ = "NON_SMOOTH_LCP";

  /// \brief The BulkData objects this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData objects this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Method for mapping from rigid body force to rigid body velocity.
  std::shared_ptr<mundy::meta::MetaMethodBase<void>> map_rigid_body_force_to_rigid_body_velocity_method_ptr_;

  /// \brief Method for mapping from rigid body velocity to surface velocity.
  std::shared_ptr<mundy::meta::MetaMethodBase<void>> map_rigid_body_velocity_to_surface_velocity_method_ptr_;

  /// \brief Method for mapping from surface force to rigid body force.
  std::shared_ptr<mundy::meta::MetaMethodBase<void>> map_surface_force_to_rigid_body_force_method_ptr_;
  //@}
};  // NonSmoothLCP

}  // namespace techniques

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_RESOLVE_CONSTRAINTS_TECHNIQUES_NONSMOOTHLCP_HPP_
