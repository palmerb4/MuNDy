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

#ifndef MUNDY_METHODS_RESOLVECONSTRAINTS_HPP_
#define MUNDY_METHODS_RESOLVECONSTRAINTS_HPP_

/// \file ResolveConstraints.hpp
/// \brief Declaration of the ResolveConstraints class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_core/StringLiteral.hpp>                 // for mundy::core::make_string_literal
#include <mundy_mesh/BulkData.hpp>                 // for mundy::mesh::BulkData
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>               // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>             // for mundy::meta::GlobalMetaMethodRegistry
#include <mundy_meta/MetaTechniqueDispatcher.hpp>  // for mundy::meta::MetaTechniqueDispatcher
#include <mundy_motion/resolve_constraints/techniques/NonSmoothLCP.hpp>  // for mundy::motion::...::NonSmoothLCP
#include <mundy_motion/resolve_constraints/techniques/PairwisePotential.hpp>  // for mundy::motion::...::PairwisePotential

namespace mundy {

namespace motion {

/// \class ResolveConstraints
/// \brief Method for resolving constraints between N bodies
class ResolveConstraints
    : public mundy::meta::MetaTechniqueDispatcher<ResolveConstraints, mundy::core::make_string_literal("NON_SMOOTH_LCP")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ResolveConstraints() = delete;

  /// \brief Constructor
  ResolveConstraints(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name MetaFactory static interface implementation
  //@{

  /// \brief Get the unique registration identifier. Ideally, this should be unique and not shared by any other \c
  /// MetaMethodSubsetExecutionInterface.
  static RegistrationType get_registration_id() {
    return registration_id_;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<ResolveConstraints>(bulk_data_ptr, fixed_params);
  }
  //@}

 private:
  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other methods in our MetaMethodRegistry.
  static constexpr std::string_view registration_id_ = "RESOLVE_CONSTRAINTS";
  //@}
};  // ResolveConstraints

}  // namespace motion

}  // namespace mundy

//! \name Registration
//@{

/// @brief Register ResolveConstraints with the global MetaMethodFactory.
MUNDY_REGISTER_METACLASS(mundy::motion::ResolveConstraints, mundy::meta::GlobalMetaMethodFactory<void>)

/// @brief Register our default techniques
MUNDY_REGISTER_METACLASS(mundy::motion::resolve_constraints::techniques::NonSmoothLCP,
                         mundy::motion::ResolveConstraints::OurMethodFactory)

MUNDY_REGISTER_METACLASS(mundy::motion::resolve_constraints::techniques::PairwisePotential,
                          mundy::motion::ResolveConstraints::OurMethodFactory)
//@}

#endif  // MUNDY_METHODS_RESOLVECONSTRAINTS_HPP_
