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

#ifndef MUNDY_META_METAMETHOD_HPP_
#define MUNDY_META_METAMETHOD_HPP_

/// \file MetaMethod.hpp
/// \brief Declaration of the MetaMethod class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part

// Mundy libs
#include <mundy_mesh/BulkData.hpp>                              // for mundy::mesh::BulkData
#include <mundy_meta/HasMeshRequirementsAndIsRegisterable.hpp>  // for mundy::meta::HasMeshRequirementsAndIsRegisterable
#include <mundy_meta/PartRequirements.hpp>                      // for mundy::meta::PartRequirements

namespace mundy {

namespace meta {

/// \class MetaMethodBase
/// \brief The polymorphic interface which all \c MetaMethods will share.
///
/// This design pattern allows for \c MetaMethod to use CRTP to force derived classes to implement certain static
/// functions while also having a consistant polymorphic interface that allows different \c MetaMethods to be stored in
/// a vector of pointers.
///
/// \tparam ReturnType The return type of the execute function.
/// \tparam RegistrationType The type of this class's identifier.
template <typename ReturnType, typename RegistrationType = std::string>
class MetaMethodBase
    : virtual public HasMeshRequirementsAndIsRegisterableInterface<MetaMethodBase<ReturnType, RegistrationType>,
                                                                   RegistrationType> {
 public:
  //! \name Setters
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  virtual void set_mutable_params(const Teuchos::ParameterList &mutable_parameter_list) = 0;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  virtual ReturnType execute(const stk::mesh::Selector &input_selector) = 0;
  //@}
};  // MetaMethodBase

/// \class MetaMethod
/// \brief The static interface that encodes a class's assumptions about the structure and contents of an STK mesh.
///
/// This class follows the Curiously Recurring Template Pattern such that each class derived from \c MetaMethod must
/// implement the following static member functions
/// - \c details_static_get_mesh_requirements implementation of the \c static_get_mesh_requirements interface.
/// - \c details_static_get_valid_fixed_params implementation of the \c static_get_valid_fixed_params interface.
/// - \c details_static_get_valid_mutable_params implementation of the \c static_get_valid_mutable_params interface.
/// - \c details_static_get_class_identifier implementation of the \c static_get_class_identifier interface.
/// - \c details_static_create_new_instance implementation of the \c static_create_new_instance interface.
///
/// To keep these out of the public interface, we suggest that each details function be made private and
/// \c MetaMethod<DerivedMetaMethod> be made a friend of \c DerivedMetaMethod.
///
/// \note The _t in our template paramaters breaks our naming convention for types but is used to prevent template
/// shaddowing by internal typedefs.
///
/// \tparam ReturnType_t The return type of the execute function.
/// \tparam DerivedMetaMethod_t A class derived from \c MetaMethod that implements the desired interface.
/// \tparam RegistrationType_t The type of this class's identifier.
template <typename ReturnType_t, class DerivedMetaMethod_t, typename RegistrationType_t = std::string>
class MetaMethod
    : virtual public MetaMethodBase<ReturnType_t, RegistrationType_t>,
      public HasMeshRequirementsAndIsRegisterable<MetaMethod<ReturnType_t, DerivedMetaMethod_t>,
                                                  MetaMethodBase<ReturnType_t, RegistrationType_t>, RegistrationType_t> {
 public:
  //! \name Typedefs
  //@{

  using ReturnType = ReturnType_t;
  using RegistrationType = RegistrationType_t;
  using PolymorphicBase = MetaMethodBase<ReturnType, RegistrationType>;
  //@}

  //! \name Getters
  //@{

  /// \brief Get the requirements that this \c MetaMethod imposes upon each input part.
  ///
  /// The set part requirements returned by this function are meant to encode the assumptions made by this class
  /// with respect to the structure, topology, and fields of the STK mesh. These assumptions may vary
  /// based on parameters in the \c fixed_parameter_list but not the \c mutable_parameter_list.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  static std::vector<std::shared_ptr<MeshRequirements>> details_static_get_mesh_requirements(
      const Teuchos::ParameterList &fixed_parameter_list) {
    return DerivedMetaMethod_t::details_static_get_mesh_requirements(fixed_parameter_list);
  }

  /// \brief Get the valid fixed parameters and their default parameter list for this \c MetaMethod.
  static Teuchos::ParameterList details_static_get_valid_fixed_params() {
    return DerivedMetaMethod_t::details_static_get_valid_fixed_params();
  }

  /// \brief Get the valid mutable parameters and their default parameter list for this \c MetaMethod.
  static Teuchos::ParameterList details_static_get_valid_mutable_params() {
    return DerivedMetaMethod_t::details_static_get_valid_mutable_params();
  }

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c MetaMethod.
  static RegistrationType details_static_get_class_identifier() {
    return DerivedMetaMethod_t::details_static_get_class_identifier();
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  static std::shared_ptr<PolymorphicBase> details_static_create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list) {
    return DerivedMetaMethod_t::details_static_create_new_instance(bulk_data_ptr, fixed_parameter_list);
  }
  //@}
};  // MetaMethod

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METAMETHOD_HPP_
