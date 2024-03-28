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

#ifndef MUNDY_CONSTRAINTS_DECLAREANDINITCONSTRAINTS_HPP_
#define MUNDY_CONSTRAINTS_DECLAREANDINITCONSTRAINTS_HPP_

/// \file DeclareAndInitConstraints.hpp
/// \brief Declaration of the DeclareAndInitConstraints class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_constraints/declare_and_initialize_constraints/techniques/ChainOfSprings.hpp>  // for mundy::constraints::declare_and_initialize_constraints::techniques::ChainOfSprings
#include <mundy_mesh/BulkData.hpp>                 // for mundy::mesh::BulkData
#include <mundy_meta/MetaRegistry.hpp>             // for mundy::meta::GlobalMetaMethodRegistry
#include <mundy_meta/MetaTechniqueDispatcher.hpp>  // for mundy::meta::MetaMethodExecutionDispatcher

namespace mundy {

namespace constraints {

/// \class DeclareAndInitConstraints
/// \brief Method for declaring and initializing constraints.
class DeclareAndInitConstraints
    : public mundy::meta::MetaMethodExecutionDispatcher<
          DeclareAndInitConstraints, void, mundy::meta::make_registration_string("DECLARE_AND_INIT_CONSTRAINTS"),
          mundy::meta::make_registration_string("NO_DEFAULT_TECHNIQUE")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  DeclareAndInitConstraints() = delete;

  /// \brief Constructor
  DeclareAndInitConstraints(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
      : mundy::meta::MetaMethodExecutionDispatcher<
            DeclareAndInitConstraints, void, mundy::meta::make_registration_string("DECLARE_AND_INIT_CONSTRAINTS"),
            mundy::meta::make_registration_string("NO_DEFAULT_TECHNIQUE")>(bulk_data_ptr, fixed_params) {
  }
  //@}

  //! \name MetaMethodExecutionDispatcher static interface implementation
  //@{

  /// \brief Get the valid fixed parameters that we will forward to the techniques.
  static Teuchos::ParameterList get_valid_forwarded_technique_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we will forward to the techniques.
  static Teuchos::ParameterList get_valid_forwarded_technique_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid fixed parameters that we require all techniques registered with our technique factory to
  /// have.
  static Teuchos::ParameterList get_valid_required_technique_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we require all techniques registered with our technique factory to
  /// have.
  static Teuchos::ParameterList get_valid_required_technique_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }
  //@}
};  // DeclareAndInitConstraints

}  // namespace constraints

}  // namespace mundy

//! \name Registration
//@{

/// @brief Register our default techniques
MUNDY_REGISTER_METACLASS("CHAIN_OF_SPRINGS",
                         mundy::constraints::declare_and_initialize_constraints::techniques::ChainOfSprings,
                         mundy::constraints::DeclareAndInitConstraints::OurTechniqueFactory)

//@}

#endif  // MUNDY_CONSTRAINTS_DECLAREANDINITCONSTRAINTS_HPP_
