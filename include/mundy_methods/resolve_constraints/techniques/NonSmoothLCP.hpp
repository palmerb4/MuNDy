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
  /// \note This method does not cache its return value, so every time you call this method, a new \c MeshRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::MeshRequirements> details_static_get_mesh_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_params) {
    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    static_validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

    // Fetch the parameters for this part's sub-methods.
    Teuchos::ParameterList &compute_constraint_forcing_params =
        valid_fixed_params.sublist("submethods").sublist("compute_constraint_forcing");
    Teuchos::ParameterList &compute_constraint_projection_params =
        valid_fixed_params.sublist("submethods").sublist("compute_constraint_projection");
    Teuchos::ParameterList &compute_constraint_residual_params =
        valid_fixed_params.sublist("submethods").sublist("compute_constraint_residual");
    Teuchos::ParameterList &compute_constraint_violation_params =
        valid_fixed_params.sublist("submethods").sublist("compute_constraint_violation");

    // Collect and merge the submethod requirements.
    auto mesh_reqs = std::make_shared<mundy::meta::MeshRequirements>();
    const std::string compute_constraint_forcing_name = compute_constraint_forcing_params.get<std::string>("name");
    const std::string compute_constraint_projection_name =
        compute_constraint_projection_params.get<std::string>("name");
    const std::string compute_constraint_residual_name = compute_constraint_residual_params.get<std::string>("name");
    const std::string compute_constraint_violation_name = compute_constraint_violation_params.get<std::string>("name");
    mesh_reqs->merge(
        OurMethodFactory::get_part_requirements(compute_constraint_forcing_name, compute_constraint_forcing_params));
    mesh_reqs->merge(OurMethodFactory::get_part_requirements(compute_constraint_projection_name,
                                                             compute_constraint_projection_params));
    mesh_reqs->merge(
        OurMethodFactory::get_part_requirements(compute_constraint_residual_name, compute_constraint_residual_params));
    mesh_reqs->merge(OurMethodFactory::get_part_requirements(compute_constraint_violation_name,
                                                             compute_constraint_violation_params));
    return mesh_reqs;
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void details_static_validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList const *fixed_params_ptr) {
    Teuchos::ParameterList &compute_constraint_forcing_params =
        part_params.sublist("submethods", false).sublist("compute_constraint_forcing", false);
    Teuchos::ParameterList &compute_constraint_projection_params =
        part_params.sublist("submethods", false).sublist("compute_constraint_projection", false);
    Teuchos::ParameterList &compute_constraint_residual_params =
        part_params.sublist("submethods", false).sublist("compute_constraint_residual", false);
    Teuchos::ParameterList &compute_constraint_violation_params =
        part_params.sublist("submethods", false).sublist("compute_constraint_violation", false);

    if (compute_constraint_forcing_params->isParameter("name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      TEUCHOS_TEST_FOR_EXCEPTION(
          valid_type, std::invalid_argument,
          "NonSmoothLCP: Type error. Given a compute_constraint_forcing parameter with name 'name' but "
              << "with a type other than std::string");
    } else {
      compute_constraint_forcing_params.set(
          "name", std::string(default_map_rbf_to_rbv_name_),
          "Name of the method for computing the force induced by the constraints on their nodes.");
    }

    if (map_rbv_to_sv_params.isParameter("name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      TEUCHOS_TEST_FOR_EXCEPTION(
          valid_type, std::invalid_argument,
          "NonSmoothLCP: Type error. Given a compute_constraint_projection parameter with name 'name' but "
              << "with a type other than std::string");
    } else {
      map_rbv_to_sv_params.set("name", std::string(default_map_rbf_to_rbv_name_),
                               "Name of the method for projecting the constraints onto the feasable solution space.");
    }

    if (map_sf_to_rbf_params.isParameter("name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      TEUCHOS_TEST_FOR_EXCEPTION(
          valid_type, std::invalid_argument,
          "NonSmoothLCP: Type error. Given a compute_constraint_residual parameter with name 'name' but "
              << "with a type other than std::string");
    } else {
      map_sf_to_rbf_params.set(
          "name", std::string(default_map_rbf_to_rbv_name_),
          "Name of the method for computing the global residual quantifying the constraint violation.");
    }

    if (map_sf_to_rbf_params.isParameter("name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      TEUCHOS_TEST_FOR_EXCEPTION(
          valid_type, std::invalid_argument,
          "NonSmoothLCP: Type error. Given a compute_constraint_violation parameter with name 'name' but "
              << "with a type other than std::string");
    } else {
      map_sf_to_rbf_params.set("name", std::string(default_map_rbf_to_rbv_name_),
                               "Name of the method for computing the amount of violation of all constraints.");
    }

    const std::string rbf_to_rbv_name = map_rbf_to_rbv_params.get<std::string>("name");
    const std::string rbv_to_sv_name = map_rbv_to_sv_params.get<std::string>("name");
    const std::string sf_to_rbf_name = map_sf_to_rbf_params.get<std::string>("name");
    OurMethodFactory::validate_fixed_parameters_and_set_defaults(rbf_to_rbv_name, &map_rbf_to_rbv_params);
    OurMethodFactory::validate_fixed_parameters_and_set_defaults(rbv_to_sv_name, &map_rbv_to_sv_params);
    OurMethodFactory::validate_fixed_parameters_and_set_defaults(sf_to_rbf_name, &map_sf_to_rbf_params);
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void details_static_validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList const *mutable_params_ptr) {
    Teuchos::ParameterList &compute_constraint_forcing_params =
        part_params.sublist("submethods", false).sublist("compute_constraint_forcing", false);
    Teuchos::ParameterList &compute_constraint_projection_params =
        part_params.sublist("submethods", false).sublist("compute_constraint_projection", false);
    Teuchos::ParameterList &compute_constraint_residual_params =
        part_params.sublist("submethods", false).sublist("compute_constraint_residual", false);
    Teuchos::ParameterList &compute_constraint_violation_params =
        part_params.sublist("submethods", false).sublist("compute_constraint_violation", false);

    if (compute_constraint_forcing_params->isParameter("name")) {
      const bool valid_type = mutable_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      TEUCHOS_TEST_FOR_EXCEPTION(
          valid_type, std::invalid_argument,
          "NonSmoothLCP: Type error. Given a compute_constraint_forcing parameter with name 'name' but "
              << "with a type other than std::string");
    } else {
      compute_constraint_forcing_params.set(
          "name", std::string(default_map_rbf_to_rbv_name_),
          "Name of the method for computing the force induced by the constraints on their nodes.");
    }

    if (map_rbv_to_sv_params.isParameter("name")) {
      const bool valid_type = mutable_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      TEUCHOS_TEST_FOR_EXCEPTION(
          valid_type, std::invalid_argument,
          "NonSmoothLCP: Type error. Given a compute_constraint_projection parameter with name 'name' but "
              << "with a type other than std::string");
    } else {
      map_rbv_to_sv_params.set("name", std::string(default_map_rbf_to_rbv_name_),
                               "Name of the method for projecting the constraints onto the feasable solution space.");
    }

    if (map_sf_to_rbf_params.isParameter("name")) {
      const bool valid_type = mutable_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      TEUCHOS_TEST_FOR_EXCEPTION(
          valid_type, std::invalid_argument,
          "NonSmoothLCP: Type error. Given a compute_constraint_residual parameter with name 'name' but "
              << "with a type other than std::string");
    } else {
      map_sf_to_rbf_params.set(
          "name", std::string(default_map_rbf_to_rbv_name_),
          "Name of the method for computing the global residual quantifying the constraint violation.");
    }

    if (map_sf_to_rbf_params.isParameter("name")) {
      const bool valid_type = mutable_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("name");
      TEUCHOS_TEST_FOR_EXCEPTION(
          valid_type, std::invalid_argument,
          "NonSmoothLCP: Type error. Given a compute_constraint_violation parameter with name 'name' but "
              << "with a type other than std::string");
    } else {
      map_sf_to_rbf_params.set("name", std::string(default_map_rbf_to_rbv_name_),
                               "Name of the method for computing the amount of violation of all constraints.");
    }

    const std::string rbf_to_rbv_name = map_rbf_to_rbv_params.get<std::string>("name");
    const std::string rbv_to_sv_name = map_rbv_to_sv_params.get<std::string>("name");
    const std::string sf_to_rbf_name = map_sf_to_rbf_params.get<std::string>("name");
    OurMethodFactory::validate_mutable_parameters_and_set_defaults(rbf_to_rbv_name, &map_rbf_to_rbv_params);
    OurMethodFactory::validate_mutable_parameters_and_set_defaults(rbv_to_sv_name, &map_rbv_to_sv_params);
    OurMethodFactory::validate_mutable_parameters_and_set_defaults(sf_to_rbf_name, &map_sf_to_rbf_params);
  }

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other
  /// \c MetaMethod.
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

  /// \brief Method for computing the force induced by the constraints on their nodes.
  std::shared_ptr<mundy::meta::MetaMethodBase<void>> compute_constraint_forcing_method_ptr_;

  /// \brief Method for projecting the constraints onto the feasable solution space.
  std::shared_ptr<mundy::meta::MetaMethodBase<void>> compute_constraint_projection_method_ptr_;

  /// \brief Method for computing the global residual quantifying the constraint violation.
  std::shared_ptr<mundy::meta::MetaMethodBase<void>> compute_constraint_residual_method_ptr_;

  /// \brief Method for computing the amount of violation of all constraints.
  std::shared_ptr<mundy::meta::MetaMethodBase<void>> compute_constraint_violation_method_ptr_;
  //@}
};  // NonSmoothLCP

}  // namespace techniques

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_RESOLVE_CONSTRAINTS_TECHNIQUES_NONSMOOTHLCP_HPP_
