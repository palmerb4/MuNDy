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

#ifndef MUNDY_META_HASMESHREQUIREMENTSANDISREGISTERABLE_HPP_
#define MUNDY_META_HASMESHREQUIREMENTSANDISREGISTERABLE_HPP_

/// \file HasMeshRequirementsAndIsRegisterable.hpp
/// \brief Declaration of the HasMeshRequirementsAndIsRegisterable class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part

// Mundy libs
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_meta/MeshRequirements.hpp>  // for mundy::meta::MeshRequirements

namespace mundy {

namespace meta {

/// \class HasMeshRequirementsAndIsRegisterableBase
/// \brief The polymorphic interface that encodes a class's assumptions about the structure and contents of an STK mesh.
///
/// This design pattern allows for \c HasMeshRequirementsAndIsRegisterable to use CRTP to force derived classes to
/// implement certain static functions while also having a consistant polymorphic interface that allows different
/// \c HasMeshRequirementsAndIsRegisterables to be stored in a vector of pointers.
///
/// \tparam BaseType The polymorphic base type returned by create_new_instance.
/// \tparam RegistrationType The type of this class's identifier.
template <typename BaseType, typename RegistrationType = std::string>
class HasMeshRequirementsAndIsRegisterableBase {
 public:
  //! \name Getters
  //@{

  /// \brief Get the requirements that this class imposes upon each input part.
  ///
  /// The set of part requirements returned by this function are meant to encode the assumptions made by this class
  /// with respect to the structure, topology, and fields of the STK mesh. These assumptions may vary
  /// based on parameters in the \c fixed_parameter_list but not the \c mutable_parameter_list.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  /// \note Fixed parameters are those that change the part requirements.
  virtual std::vector<std::shared_ptr<MeshRequirements>> get_part_requirements(
      const Teuchos::ParameterList &fixed_parameter_list) const = 0;

  /// \brief Get the valid fixed parameters and their default parameter list for this class.
  virtual Teuchos::ParameterList get_valid_fixed_params() const = 0;

  /// \brief Get the valid mutable parameters and their default parameter list for this class.
  virtual Teuchos::ParameterList get_valid_mutable_params() const = 0;

  /// \brief Get the class identifier.
  virtual RegistrationType get_class_identifier() const = 0;
  //@}

  //! \name Setters
  //@{

  /// \brief Set the mutable parameters. If a valid parameter is not provided, we use the default value.
  virtual Teuchos::ParameterList set_mutable_params(const Teuchos::ParameterList &mutable_parameter_list) const = 0;
  //@}

  //! \name Actions
  //@{

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  virtual std::shared_ptr<BaseType> create_new_instance(mundy::mesh::BulkData *const bulk_data_ptr,
                                                        const Teuchos::ParameterList &fixed_parameter_list) const = 0;
  //@}
};  // HasMeshRequirementsAndIsRegisterableBase

/// \class HasMeshRequirementsAndIsRegisterable
/// \brief The static interface that encodes a class's assumptions about the structure and contents of an STK mesh.
///
/// This class follows the Curiously Recurring Template Pattern such that each class derived from \c
/// HasMeshRequirementsAndIsRegisterable must implement the following static member functions
/// - \c details_static_get_part_requirements implementation of the \c static_get_part_requirements interface.
/// - \c details_static_get_valid_fixed_params implementation of the \c static_get_valid_fixed_params interface.
/// - \c details_static_get_valid_mutable_params implementation of the \c static_get_valid_mutable_params interface.
/// - \c details_static_get_class_identifier implementation of the \c static_get_class_identifier interface.
/// - \c details_static_create_new_instance implementation of the \c static_create_new_instance interface.
///
/// The derived class must also have a static PolymorphicBaseType type.
///
/// To keep these out of the public interface, we suggest that each details function be made private and
/// \c HasMeshRequirementsAndIsRegisterable<DerivedClass> be made a friend of \c DerivedClass.
///
/// \tparam DerivedClass A class derived from \c HasMeshRequirementsAndIsRegisterable that implements the desired
/// interface.
/// \tparam RegistrationType The type of this class's identifier.
template <class DerivedClass, typename RegistrationType = std::string>
class HasMeshRequirementsAndIsRegisterable
    : virtual public HasMeshRequirementsAndIsRegisterableBase<typename DerivedClass::PolymorphicBaseType,
                                                              RegistrationType> {
 public:
  //! \name Getters
  //@{

  /// \brief Get the requirements that this \c HasMeshRequirementsAndIsRegisterable imposes upon each input part.
  ///
  /// The set part requirements returned by this function are meant to encode the assumptions made by this class
  /// with respect to the structure, topology, and fields of the STK mesh. These assumptions may vary
  /// based on parameters in the \c fixed_parameter_list but not the \c mutable_parameter_list.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  std::vector<std::shared_ptr<MeshRequirements>> get_part_requirements(
      const Teuchos::ParameterList &fixed_parameter_list) const final;

  /// \brief Get the requirements that this \c HasMeshRequirementsAndIsRegisterable imposes upon each input part.
  ///
  /// The set part requirements returned by this function are meant to encode the assumptions made by this class
  /// with respect to the structure, topology, and fields of the STK mesh. These assumptions may vary
  /// based on parameters in the \c fixed_parameter_list but not the \c mutable_parameter_list.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  static std::vector<std::shared_ptr<MeshRequirements>> static_get_part_requirements(
      const Teuchos::ParameterList &fixed_parameter_list) {
    return DerivedClass::details_static_get_part_requirements(fixed_parameter_list);
  }

  /// \brief Get the valid fixed parameters and their default parameter list for this class.
  Teuchos::ParameterList get_valid_fixed_params() const final;

  /// \brief Get the valid mutable parameters and their default parameter list for this class.
  Teuchos::ParameterList get_valid_mutable_params() const final;

  /// \brief Get the valid fixed parameters and their default parameter list for this class.
  static Teuchos::ParameterList static_get_valid_fixed_params() {
    return DerivedClass::details_static_get_valid_fixed_params();
  }

  /// \brief Get the valid mutable parameters and their default parameter list for this class.
  static Teuchos::ParameterList static_get_valid_mutable_params() {
    return DerivedClass::details_static_get_valid_mutable_params();
  }

  /// \brief Get the unique class identifier.
  RegistrationType get_class_identifier() const final;

  /// \brief Get the unique class identifier.
  static RegistrationType static_get_class_identifier() {
    return DerivedClass::details_static_get_class_identifier();
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  std::shared_ptr<typename DerivedClass::PolymorphicBaseType> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list) const final;

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  static std::shared_ptr<typename DerivedClass::PolymorphicBaseType> static_create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list) {
    return DerivedClass::details_static_create_new_instance(bulk_data_ptr, fixed_parameter_list);
  }
  //@}
};  // HasMeshRequirementsAndIsRegisterable

//! \name Template implementations
//@{

// \name Getters
//{
template <class DerivedClass, typename RegistrationType>
std::vector<std::shared_ptr<MeshRequirements>>
HasMeshRequirementsAndIsRegisterable<DerivedClass, RegistrationType>::get_part_requirements(
    const Teuchos::ParameterList &fixed_parameter_list) const {
  return static_get_part_requirements(fixed_parameter_list);
}

template <class DerivedClass, typename RegistrationType>
Teuchos::ParameterList HasMeshRequirementsAndIsRegisterable<DerivedClass, RegistrationType>::get_valid_fixed_params()
    const {
  return static_get_valid_fixed_params();
}

template <class DerivedClass, typename RegistrationType>
Teuchos::ParameterList HasMeshRequirementsAndIsRegisterable<DerivedClass, RegistrationType>::get_valid_mutable_params()
    const {
  return static_get_valid_mutable_params();
}

template <class DerivedClass, typename RegistrationType>
RegistrationType HasMeshRequirementsAndIsRegisterable<DerivedClass, RegistrationType>::get_class_identifier() const {
  return static_get_class_identifier();
}
//}

// \name Actions
//{

template <class DerivedClass, typename RegistrationType>
std::shared_ptr<typename DerivedClass::PolymorphicBaseType>
HasMeshRequirementsAndIsRegisterable<DerivedClass, RegistrationType>::create_new_instance(
    mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list) const {
  return static_create_new_instance(bulk_data_ptr, fixed_parameter_list);
}
//}
//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_HASMESHREQUIREMENTSANDISREGISTERABLE_HPP_
