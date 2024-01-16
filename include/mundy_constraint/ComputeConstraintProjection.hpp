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

#ifndef MUNDY_CONSTRAINT_COMPUTECONSTRAINTPROJECTION_HPP_
#define MUNDY_CONSTRAINT_COMPUTECONSTRAINTPROJECTION_HPP_

/// \file ComputeConstraintProjection.hpp
/// \brief Declaration of the ComputeConstraintProjection class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_mesh/BulkData.hpp>                                       // for mundy::mesh::BulkData
#include <mundy_meta/MetaKernelDispatcher.hpp>                           // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaRegistry.hpp>                                   // for mundy::meta::MetaMethodRegistry
#include <mundy_motion/resolve_constraints/techniques/NonSmoothLCP.hpp>  // for mundy::motion::...::NonSmoothLCP

namespace mundy {

namespace constraint {

/// \class ComputeConstraintProjection
/// \brief Method for computing the projection of the constraint Lagrange multiplier onto its feasible set.
class ComputeConstraintProjection : public mundy::meta::MetaKernelDispatcher<ComputeConstraintProjection> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeConstraintProjection() = delete;

  /// \brief Constructor
  ComputeConstraintProjection(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name MetaFactory static interface implementation
  //@{

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
    return std::make_shared<ComputeConstraintProjection>(bulk_data_ptr, fixed_params);
  }
  //@}

 private:
  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other methods in our MetaMethodRegistry.
  static constexpr std::string_view registration_id_ = "COMPUTE_CONSTRAINT_PROJECTION";
  //@}
};  // ComputeConstraintProjection

}  // namespace constraint

}  // namespace mundy

//! \name Registration
//@{

/// @brief Register ComputeConstraintProjection with NonSmoothLCP's method factory.
MUNDY_REGISTER_METACLASS(
    mundy::constraint::ComputeConstraintProjection,
    mundy::motion::resolve_constraints::techniques::NonSmoothLCP::OurConstraintProjectionMethodFactory)
//}

#endif  // MUNDY_CONSTRAINT_COMPUTECONSTRAINTPROJECTION_HPP_
