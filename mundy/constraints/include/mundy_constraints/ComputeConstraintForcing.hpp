// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
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

#ifndef MUNDY_CONSTRAINTS_COMPUTECONSTRAINTFORCING_HPP_
#define MUNDY_CONSTRAINTS_COMPUTECONSTRAINTFORCING_HPP_

/// \file ComputeConstraintForcing.hpp
/// \brief Declaration of the ComputeConstraintForcing class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_constraints/compute_constraint_forcing/kernels/HookeanSpringsKernel.hpp>
#include <mundy_core/StringLiteral.hpp>         // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_mesh/BulkData.hpp>              // for mundy::mesh::BulkData
#include <mundy_meta/MetaKernelDispatcher.hpp>  // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaRegistry.hpp>          // for mundy::meta::MetaMethodRegistry

namespace mundy {

namespace constraints {

/// \class ComputeConstraintForcing
/// \brief Method for computing the force exerted by a constraint onto its nodes.
class ComputeConstraintForcing
    : public mundy::meta::MetaKernelDispatcher<ComputeConstraintForcing,
                                               mundy::meta::make_registration_string("COMPUTE_CONSTRAINT_FORCING")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeConstraintForcing() = delete;

  /// \brief Constructor
  ComputeConstraintForcing(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
      : mundy::meta::MetaKernelDispatcher<ComputeConstraintForcing,
                                          mundy::meta::make_registration_string("COMPUTE_CONSTRAINT_FORCING")>(
            bulk_data_ptr, fixed_params) {
  }
  //@}

  //! \name MetaKernelDispatcher static interface implementation
  //@{

  /// \brief Get the valid fixed parameters that we require our techniques have.
  static Teuchos::ParameterList get_valid_required_kernel_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we require our techniques have.
  static Teuchos::ParameterList get_valid_required_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid fixed parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("node_force_field_name", std::string(default_node_force_field_name_),
                               "Name of the node force field to be used for storing the computed spring force.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_node_force_field_name_ = "NODE_FORCE";
  //@}
};  // ComputeConstraintForcing

}  // namespace constraints

}  // namespace mundy

//! \name Registration
//@{

/// @brief Register our default kernels
MUNDY_REGISTER_METACLASS("HOOKEAN_SPRINGS",
                         mundy::constraints::compute_constraint_forcing::kernels::HookeanSpringsKernel,
                         mundy::constraints::ComputeConstraintForcing::OurKernelFactory)
//@}

#endif  // MUNDY_CONSTRAINTS_COMPUTECONSTRAINTFORCING_HPP_
