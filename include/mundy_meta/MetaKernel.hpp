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

#ifndef MUNDY_META_METAKERNEL_HPP_
#define MUNDY_META_METAKERNEL_HPP_

/// \file MetaKernel.hpp
/// \brief Declaration of the MetaKernel class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity

// Mundy libs
#include <mundy_meta/PartRequirements.hpp>  // for mundy::meta::PartRequirements

namespace mundy {

namespace meta {

/// \class MetaKernelBase
/// \brief The polymorphic interface which all \c MetaKernels will share.
///
/// This design pattern allows for \c MetaKernel to use CRTP to force derived classes to implement certain static
/// functions while also having a consistant polymoirphic interface that allows different \c MetaKernels to be stored in
/// a vector of pointers.
///
/// \tparam ReturnType The return type of the execute function.
template <typename ReturnType>
class MetaKernelBase {
 public:
  //! \name Attributes
  //@{

  /// \brief Get the requirements that this \c MetaKernel imposes upon each input part.
  ///
  /// The set part requirements returned by this function are meant to encode the assumptions made by this \c MetaKernel
  /// with respect to the parts, topology, and fields input into the \c run function. These assumptions may vary
  /// based parameters in the \c parameter_list.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  virtual std::vector<std::shared_ptr<PartRequirements>> get_part_requirements(
      const Teuchos::ParameterList& parameter_list) const = 0;

  /// \brief Get the valid parameters and their default parameter list for this \c MetaKernel.
  virtual Teuchos::ParameterList get_valid_params() const = 0;

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c MetaKernel.
  virtual std::string get_class_identifier() const = 0;
  //@}

  //! \name Actions
  //@{

  /// \brief Generate a new instance of this class.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  virtual std::shared_ptr<MetaKernelBase<ReturnType>> create_new_instance(
      const Teuchos::ParameterList& parameter_list) const = 0;

  /// \brief Run the kernel's core calculation.
  virtual ReturnType execute(const stk::mesh::Entity& entity) = 0;
  //@}
};  // MetaKernelBase

/// \class MetaKernel
/// \brief An abstract interface for all of Mundy's methods.
///
/// A \c MetaKernel represents an atomic unit of computation applied to a \b single multibody object in isolation
/// (sphere, spring, hinge, etc.). Through STK and Kokkos looping constructs, a \c MetaKernel can be efficiently applied
/// to large groups of multibody objects.
///
/// While \c MetaKernel only accepts a single multibody object, but Mundy offers two pairwise specializations:
///  - \c MetaKernelPairwise for pairwise anti-symmetric interaction between two multibody objects.
///  - \c MetaKernelPairwiseSymmetric for pairwise symmetric interaction between two multibody objects.
///
/// The goal of \c MetaKernel is to wrap a kernel that acts on an STK Element with a known multibody type.
/// The wrapper can output the assumptions of the wrapped kernel with respect to the fields and topology associated with
/// the provided element. Note, this element is part of some STK Part, so the output requirements are
/// \c PartRequirements for that part. Requirements cannot be applied at the element-level.
///
/// This class follows the Curiously Recurring Template Pattern such that each class derived from \c MetaKernel must
/// implement the following static member functions
///   - \c details_get_part_requirements implementation of the \c get_part_requirements interface.
///   class.
///   - \c details_get_valid_params implementation of the \c get_valid_params interface.
///   - \c details_get_class_identifier implementation of the \c get_class_identifier interface.
///   - \c details_create_new_instance implementation of the \c create_new_instance interface.
///
/// To keep these out of the public interface, we suggest that each details function be made private and
/// \c MetaMethod<DerivedMetaMethod> be made a friend of \c DerivedMetaMethod.
///
/// \tparam DerivedMetaKernel A class derived from \c MetaKernel that implements the desired interface.
/// \tparam ReturnType The return type of the execute function.
template <typename ReturnType, class DerivedMetaKernel>
class MetaKernel : public MetaKernelBase<ReturnType> {
 public:
  //! \name Getters
  //@{

  /// \brief Get the requirements that this \c MetaKernel imposes upon each input part.
  ///
  /// The set part requirements returned by this function are meant to encode the assumptions made by this \c MetaKernel
  /// with respect to the parts, topology, and fields input into the \c run function. These assumptions may vary
  /// based parameters in the \c parameter_list.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  std::vector<std::shared_ptr<PartRequirements>> get_part_requirements(
      const Teuchos::ParameterList& parameter_list) const override final;

  /// \brief Get the requirements that this \c MetaKernel imposes upon each input part.
  ///
  /// The set part requirements returned by this function are meant to encode the assumptions made by this \c MetaKernel
  /// with respect to the parts, topology, and fields input into the \c run function. These assumptions may vary
  /// based parameters in the \c parameter_list.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  static std::vector<std::shared_ptr<PartRequirements>> static_get_part_requirements(
      const Teuchos::ParameterList& parameter_list) {
    return DerivedMetaKernel::details_get_part_requirements(parameter_list);
  }

  /// \brief Get the valid parameters and their default parameter list for this \c MetaKernel.
  Teuchos::ParameterList get_valid_params() const override final;

  /// \brief Get the valid parameters and their default parameter list for this \c MetaKernel.
  static Teuchos::ParameterList static_get_valid_params() {
    return DerivedMetaKernel::details_get_valid_params();
  }

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c MetaKernel.
  std::string get_class_identifier() const override final;

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c MetaKernel.
  static std::string static_get_class_identifier() {
    return DerivedMetaKernel::details_get_class_identifier();
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Generate a new instance of this class.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  std::shared_ptr<MetaKernelBase<ReturnType>> create_new_instance(
      stk::mesh::BulkData* const bulk_data_ptr, const Teuchos::ParameterList& parameter_list) const override final;

  /// \brief Generate a new instance of this class.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  static std::shared_ptr<MetaKernelBase<ReturnType>> static_create_new_instance(
      stk::mesh::BulkData* const bulk_data_ptr, const Teuchos::ParameterList& parameter_list) {
    return DerivedMetaKernel::details_create_new_instance(bulk_data_ptr, parameter_list);
  }

  /// \brief Run the kernel's core calculation.
  virtual ReturnType execute(const stk::mesh::Entity& entity) override = 0;
  //@}
};  // MetaKernel

//! \name Template implementations
//@{

// \name Getters
//{

template <typename ReturnType, class DerivedMetaKernel>
std::vector<std::shared_ptr<PartRequirements>> MetaKernel<ReturnType, DerivedMetaKernel>::get_part_requirements(
    const Teuchos::ParameterList& parameter_list) const {
  return static_get_part_requirements(parameter_list);
}

template <typename ReturnType, class DerivedMetaKernel>
Teuchos::ParameterList MetaKernel<ReturnType, DerivedMetaKernel>::get_valid_params() const {
  return static_get_valid_params();
}

template <typename ReturnType, class DerivedMetaKernel>
std::string MetaKernel<ReturnType, DerivedMetaKernel>::get_class_identifier() const {
  return static_get_class_identifier();
}
//}

// \name Actions
//{

template <typename ReturnType, class DerivedMetaKernel>
std::shared_ptr<MetaKernelBase<ReturnType>> MetaKernel<ReturnType, DerivedMetaKernel>::create_new_instance(
    stk::mesh::BulkData* const bulk_data_ptr, const Teuchos::ParameterList& parameter_list) const {
  return static_create_new_instance(bulk_data_ptr, parameter_list);
}
//}
//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METAKERNEL_HPP_
