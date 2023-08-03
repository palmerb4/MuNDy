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

#ifndef MUNDY_METHODS_RESOLVE_CONSTRAINTS_TECHNIQUES_NON_SMOOTH_LCP_COMPUTECONSTRAINTRESIDUAL_HPP_
#define MUNDY_METHODS_RESOLVE_CONSTRAINTS_TECHNIQUES_NON_SMOOTH_LCP_COMPUTECONSTRAINTRESIDUAL_HPP_

/// \file ComputeConstraintResidual.hpp
/// \brief Declaration of the ComputeConstraintResidual class

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
#include <mundy_meta/MetaRegistry.hpp>      // for mundy::meta::MetaMethodRegistry
#include <mundy_methods/resolve_constraints/techniques/non_smooth_lcp/ComputeConstraintResidual.hpp>  // for mundy::methods::...::non_smooth_lcp::ComputeConstraintResidual

namespace mundy {

namespace methods {

namespace resolve_constraints {

namespace techniques {

namespace non_smooth_lcp {

/// \class ComputeConstraintResidual
/// \brief Method for computing the axis aligned boundary box of different parts.
class ComputeConstraintResidual : public mundy::meta::MetaMethod<void> {
 public:
  //! \name Typedefs
  //@{

  using RegistrationType = std::string_view;
  using PolymorphicBaseType = mundy::meta::MetaMethod<void>;
  using OurKernelFactory = mundy::meta::MetaKernelFactory<void, ComputeConstraintResidual>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeConstraintResidual() = delete;

  /// \brief Constructor
  ComputeConstraintResidual(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
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
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

    // Fill the requirements using the given parameter list.
    // For now, we allow this method to assign these fields to all constraints.
    // TODO(palmerb4): Should we allow these fields to differ from multibody type to multibody type?
    std::string element_constraint_violation_field_name =
        valid_fixed_params.get<std::string>("element_constraint_violation_field_name");

    auto part_reqs = std::make_shared<mundy::meta::PartRequirements>();
    part_reqs->set_part_name("CONSTRAINT");
    part_reqs->set_part_rank(stk::topology::CONSTRAINT_RANK);
    part_reqs->put_multibody_part_attribute(mundy::multibody::MultibodyFactory::get_multibody_type("CONSTRAINT"));
    mesh_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
        element_constraint_violation_field_name, stk::topology::ELEMENT_RANK, 1, 1));

    auto mesh_reqs = std::make_shared<mundy::meta::MeshRequirements>();
    mesh_reqs->add_part_reqs(part_reqs);

    return mesh_reqs;
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const fixed_params_ptr) {
    if (fixed_params_ptr->isParameter("element_constraint_violation_field_name")) {
      const bool valid_type =
          fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("element_constraint_violation_field_name");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                         "ComputeConstraintResidual: Type error. Given a parameter with name "
                         "'element_constraint_violation_field_name' but "
                             << "with a type other than std::string");
    } else {
      fixed_params_ptr->set("element_constraint_violation_field_name",
                            std::string(default_element_constraint_violation_field_name_),
                            "Name of the element field containing the constraint's violation measure.");
    }
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const mutable_params_ptr) {
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
  static std::shared_ptr<mundy::meta::MetaMethod<double>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<ComputeConstraintResidual>(bulk_data_ptr, fixed_params);
  }

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  double execute() override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_element_constraint_violation_field_name_ = "ELEMENT_CONSTRAINT_VIOLATION";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other methods in our MetaMethodRegistry.
  static constexpr std::string_view registration_id_ = "COMPUTE_CONSTRAINT_RESIDUAL";

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Pointer to the part containing all multibody constraints.
  stk::mesh::Part *constraint_part_ptr_;

  /// \brief Name of the element field containing the constraint's violation measure.
  std::string element_constraint_violation_field_name_;

  /// \brief Element field containing the sphere radius.
  stk::mesh::Field<double> *element_constraint_violation_field_ptr_;
  //@}
};  // ComputeConstraintResidual

}  // namespace non_smooth_lcp

}  // namespace techniques

}  // namespace resolve_constraints

}  // namespace methods

}  // namespace mundy

//! \name Registration
//@{

/// @brief Register ComputeConstraintResidual with NonSmoothLCP's method factory.
MUNDY_REGISTER_METACLASS(mundy::methods::resolve_constraints::techniques::non_smooth_lcp::ComputeConstraintResidual,
                         mundy::methods::resolve_constraints::techniques::NonSmoothLCP::OurMethodFactory)
//}

#endif  // MUNDY_METHODS_RESOLVE_CONSTRAINTS_TECHNIQUES_NON_SMOOTH_LCP_COMPUTECONSTRAINTRESIDUAL_HPP_
