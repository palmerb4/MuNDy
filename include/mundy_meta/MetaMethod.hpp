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

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part

// Mundy libs
#include <mundy_meta/PartRequirements.hpp>  // for mundy::meta::PartRequirements

namespace mundy {

namespace meta {

/// \class MetaMethodBase
/// \brief The polymorphic interface which all \c MetaMethods will share.
///
/// This design pattern allows for \c MetaMethod to use CRTP to force derived classes to implement certain static
/// functions while also having a consistant polymoirphic interface that allows different \c MetaMethods to be stored in
/// a vector of pointers.
///
/// \tparam ReturnType The return type of the execute function.
template <typename ReturnType>
class MetaMethodBase {
 public:
  //! \name Getters
  //@{

  /// \brief Get the requirements that this \c MetaMethod imposes upon each input part.
  ///
  /// The set part requirements returned by this function are meant to encode the assumptions made by this \c MetaMethod
  /// with respect to the parts, topology, and fields input into the \c run function. These assumptions may vary
  /// based parameters in the \c parameter_list.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  virtual std::vector<std::shared_ptr<PartRequirements>> get_part_requirements(
      const Teuchos::ParameterList &parameter_list) const = 0;

  /// \brief Get the valid parameters and their default parameter list for this \c MetaMethod.
  virtual Teuchos::ParameterList get_valid_params() const = 0;

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c MetaMethod.
  virtual std::string get_class_identifier() const = 0;
  //@}

  //! \name Actions
  //@{

  /// \brief Generate a new instance of this class.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  virtual std::shared_ptr<MetaMethodBase<ReturnType>> create_new_instance(
      stk::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &parameter_list) const = 0;

  /// \brief Run the method's core calculation.
  virtual ReturnType execute() = 0;
  //@}
};  // MetaMethodBase

/// \class MetaMethod
/// \brief An abstract interface for all of Mundy's methods.
///
/// The goal of \c MetaMethod is to wrap a function that acts on Mundy's multibody hierarchy with a class that can
/// output the assumptions of the wrapped function with respect to the fields and structure of the hierarchy.
///
/// This class follows the Curiously Recurring Template Pattern such that each class derived from \c MetaMethod must
/// implement the following static member functions
///   - \c details_static_get_part_requirements implementation of the \c get_part_requirements interface.
///   - \c details_static_get_valid_params implementation of the \c get_valid_params interface.
///   - \c details_static_get_class_identifier implementation of the \c get_class_identifier interface.
///   - \c details_static_create_new_instance implementation of the \c create_new_instance interface.
///
/// To keep these out of the public interface, we suggest that each details function be made private and
/// \c MetaMethod<DerivedMetaMethod> be made a friend of \c DerivedMetaMethod.
///
/// \tparam DerivedMetaMethod A class derived from \c MetaMethod that implements the desired interface.
/// \tparam ReturnType The return type of the execute function.
template <typename ReturnType, class DerivedMetaMethod>
class MetaMethod : public MetaMethodBase<ReturnType> {
 public:
  //! \name Getters
  //@{

  /// \brief Get the requirements that this \c MetaMethod imposes upon each input part.
  ///
  /// The set part requirements returned by this function are meant to encode the assumptions made by this \c MetaMethod
  /// with respect to the parts, topology, and fields input into the \c run function. These assumptions may vary
  /// based parameters in the \c parameter_list.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  std::vector<std::shared_ptr<PartRequirements>> get_part_requirements(
      const Teuchos::ParameterList &parameter_list) const override final;

  /// \brief Get the requirements that this \c MetaMethod imposes upon each input part.
  ///
  /// The set part requirements returned by this function are meant to encode the assumptions made by this \c MetaMethod
  /// with respect to the parts, topology, and fields input into the \c run function. These assumptions may vary
  /// based parameters in the \c parameter_list.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  static std::vector<std::shared_ptr<PartRequirements>> static_get_part_requirements(
      const Teuchos::ParameterList &parameter_list) {
    return DerivedMetaMethod::details_static_get_part_requirements(parameter_list);
  }

  /// \brief Get the valid parameters and their default parameter list for this \c MetaMethod.
  Teuchos::ParameterList get_valid_params() const override final;

  /// \brief Get the valid parameters and their default parameter list for this \c MetaMethod.
  static Teuchos::ParameterList static_get_valid_params() {
    return DerivedMetaMethod::details_static_get_valid_params();
  }

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c MetaMethod.
  std::string get_class_identifier() const override final;

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c MetaMethod.
  static std::string static_get_class_identifier() {
    return DerivedMetaMethod::details_static_get_class_identifier();
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Generate a new instance of this class.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  std::shared_ptr<MetaMethodBase<ReturnType>> create_new_instance(
      stk::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &parameter_list) const override final;

  /// \brief Generate a new instance of this class.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  static std::shared_ptr<MetaMethodBase<ReturnType>> static_create_new_instance(
      stk::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &parameter_list) {
    return DerivedMetaMethod::details_static_create_new_instance(bulk_data_ptr, parameter_list);
  }

  /// \brief Run the method's core calculation.
  virtual ReturnType execute() = 0;
  //@}
};  // MetaMethod

//! \name Template implementations
//@{

// \name Getters
//{
template <typename ReturnType, class DerivedMetaMethod>
std::vector<std::shared_ptr<PartRequirements>> MetaMethod<ReturnType, DerivedMetaMethod>::get_part_requirements(
    const Teuchos::ParameterList &parameter_list) const {
  return static_get_part_requirements(parameter_list);
}

template <typename ReturnType, class DerivedMetaMethod>
Teuchos::ParameterList MetaMethod<ReturnType, DerivedMetaMethod>::get_valid_params() const {
  return static_get_valid_params();
}

template <typename ReturnType, class DerivedMetaMethod>
std::string MetaMethod<ReturnType, DerivedMetaMethod>::get_class_identifier() const {
  return static_get_class_identifier();
}
//}

// \name Actions
//{

template <typename ReturnType, class DerivedMetaMethod>
std::shared_ptr<MetaMethodBase<ReturnType>> MetaMethod<ReturnType, DerivedMetaMethod>::create_new_instance(
    stk::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &parameter_list) const {
  return static_create_new_instance(bulk_data_ptr, parameter_list);
}
//}
//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METAMETHOD_HPP_
