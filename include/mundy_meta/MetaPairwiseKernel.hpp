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

/// \file MetaPairwiseKernel.hpp
/// \brief Declaration of the MetaPairwiseKernel class

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

/// \class MetaPairwiseKernelBase
/// \brief The polymorphic interface which all \c MetaPairwiseKernels will share.
///
/// This design pattern allows for \c MetaPairwiseKernel to use CRTP to force derived classes to implement certain
/// static functions while also having a consistant polymorphic interface that allows different \c MetaPairwiseKernels
/// to be stored in a vector of pointers.
///
/// \tparam ReturnType The return type of the execute function.
template <typename ReturnType>
class MetaPairwiseKernelBase {
 public:
  //! \name Getters
  //@{

  /// \brief Get the requirements that this \c MetaPairwiseKernel imposes upon each input part.
  ///
  /// The set part requirements returned by this function are meant to encode the assumptions made by this \c
  /// MetaPairwiseKernel with respect to the parts, topology, and fields input into the \c execute function. These
  /// assumptions may vary based on parameters in the \c fixed_parameter_list.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  /// \note Fixed parameters are those that change the part requirements.
  virtual std::pair<std::shared_ptr<PartRequirements>, std::shared_ptr<PartRequirements>> get_part_requirements(
      const Teuchos::ParameterList& fixed_parameter_list) const = 0;

  /// \brief Get the valid fixed parameters and their default parameter list for this \c MetaPairwiseKernel.
  virtual Teuchos::ParameterList get_valid_fixed_params() const = 0;

  /// \brief Get the valid transient parameters and their default parameter list for this \c MetaPairwiseKernel.
  virtual Teuchos::ParameterList get_valid_transient_params() const = 0;

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c
  /// MetaPairwiseKernel.
  virtual std::string get_class_identifier() const = 0;
  //@}

  //! \name Setters
  //@{

  /// \brief Set the transient parameters. If a parameter is not provided, we use the default value.
  virtual Teuchos::ParameterList set_transient_params(const Teuchos::ParameterList& transient_parameter_list) const = 0;
  //@}

  //! \name Actions
  //@{

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  virtual std::shared_ptr<MetaPairwiseKernelBase<ReturnType>> create_new_instance(
      stk::mesh::BulkData* const bulk_data_ptr, const Teuchos::ParameterList& fixed_parameter_list) const = 0;

  /// \brief Run the kernel's core calculation.
  virtual ReturnType execute(const stk::mesh::Entity& entity) = 0;
  //@}
};  // MetaPairwiseKernelBase

/// \class MetaPairwiseKernel
/// \brief An abstract interface for all of Mundy's methods.
///
/// A \c MetaPairwiseKernel represents an atomic unit of computation applied to a \b single multibody object in
/// isolation (sphere, spring, hinge, etc.). Through STK and Kokkos looping constructs, a \c MetaPairwiseKernel can be
/// efficiently applied to large groups of multibody objects.
///
/// While \c MetaPairwiseKernel only accepts a single multibody object, but Mundy offers two pairwise specializations:
///  - \c MetaPairwiseKernelPairwise for pairwise anti-symmetric interaction between two multibody objects.
///  - \c MetaPairwiseKernelPairwiseSymmetric for pairwise symmetric interaction between two multibody objects.
///
/// The goal of \c MetaPairwiseKernel is to wrap a kernel that acts on an STK Element with a known multibody type.
/// The wrapper can output the assumptions of the wrapped kernel with respect to the fields and topology associated with
/// the provided element. Note, this element is part of some STK Part, so the output requirements are
/// \c PartRequirements for that part. Requirements cannot be applied at the element-level.
///
/// This class follows the Curiously Recurring Template Pattern such that each class derived from \c MetaPairwiseKernel
/// must implement the following static member functions
///   - \c details_static_get_part_requirements implementation of the \c get_part_requirements interface.
///   - \c details_static_get_valid_fixed_params implementation of the \c get_valid_fixed_params interface.
///   - \c details_static_get_valid_transient_params implementation of the \c get_valid_transient_params interface.
///   - \c details_static_get_class_identifier implementation of the \c get_class_identifier interface.
///   - \c details_static_create_new_instance implementation of the \c create_new_instance interface.
///
/// To keep these out of the public interface, we suggest that each details function be made private and
/// \c MetaMethod<DerivedMetaMethod> be made a friend of \c DerivedMetaMethod.
///
/// \tparam DerivedMetaPairwiseKernel A class derived from \c MetaPairwiseKernel that implements the desired interface.
/// \tparam ReturnType The return type of the execute function.
template <typename ReturnType, class DerivedMetaPairwiseKernel>
class MetaPairwiseKernel : public MetaPairwiseKernelBase<ReturnType> {
 public:
  //! \name Getters
  //@{

  /// \brief Get the requirements that this \c MetaPairwiseKernel imposes upon each input part.
  ///
  /// The set part requirements returned by this function are meant to encode the assumptions made by this \c
  /// MetaPairwiseKernel with respect to the parts, topology, and fields input into the \c run function. These
  /// assumptions may vary based on parameters in the \c fixed_parameter_list.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  /// \note Fixed parameters are those that change the part requirements.
  std::pair<std::shared_ptr<PartRequirements>, std::shared_ptr<PartRequirements>> get_part_requirements(
      const Teuchos::ParameterList& fixed_parameter_list) const final;

  /// \brief Get the requirements that this \c MetaPairwiseKernel imposes upon each input part.
  ///
  /// The set part requirements returned by this function are meant to encode the assumptions made by this \c
  /// MetaPairwiseKernel with respect to the parts, topology, and fields input into the \c run function. These
  /// assumptions may vary based on parameters in the \c fixed_parameter_list.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  /// \note Fixed parameters are those that change the part requirements.
  static std::pair<std::shared_ptr<PartRequirements>, std::shared_ptr<PartRequirements>> static_get_part_requirements(
      const Teuchos::ParameterList& fixed_parameter_list) {
    return DerivedMetaPairwiseKernel::details_static_get_part_requirements(fixed_parameter_list);
  }

  /// \brief Get the valid fixed parameters and their default parameter list for this \c MetaPairwiseKernel.
  /// \note Fixed parameters are those that change the part requirements and are fixed upon instantiation.
  Teuchos::ParameterList get_valid_fixed_params() const final;

  /// \brief Get the valid fixed parameters and their default parameter list for this \c MetaPairwiseKernel.
  static Teuchos::ParameterList static_get_valid_fixed_params() {
    return DerivedMetaPairwiseKernel::details_static_get_valid_fixed_params();
  }

  /// \brief Get the valid transient parameters and their default parameter list for this \c MetaPairwiseKernel.
  /// \note Transient parameters are those that have no impact on the part requirements and can be set after
  /// instantiation using \c set_transient_params.
  Teuchos::ParameterList get_valid_transient_params() const final;

  /// \brief Get the valid transient parameters and their default parameter list for this \c MetaPairwiseKernel.
  static Teuchos::ParameterList static_get_valid_transient_params() {
    return DerivedMetaPairwiseKernel::details_static_get_valid_transient_params();
  }

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c
  /// MetaPairwiseKernel.
  std::string get_class_identifier() const final;

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c
  /// MetaPairwiseKernel.
  static std::string static_get_class_identifier() {
    return DerivedMetaPairwiseKernel::details_static_get_class_identifier();
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the transient parameters. If a parameter is not provided, we use the default value.
  virtual Teuchos::ParameterList set_transient_params(
      const Teuchos::ParameterList& transient_parameter_list) const override = 0;
  //@}

  //! \name Actions
  //@{

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  std::shared_ptr<MetaPairwiseKernelBase<ReturnType>> create_new_instance(
      stk::mesh::BulkData* const bulk_data_ptr, const Teuchos::ParameterList& fixed_parameter_list) const final;

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  static std::shared_ptr<MetaPairwiseKernelBase<ReturnType>> static_create_new_instance(
      stk::mesh::BulkData* const bulk_data_ptr, const Teuchos::ParameterList& fixed_parameter_list) {
    return DerivedMetaPairwiseKernel::details_static_create_new_instance(bulk_data_ptr, fixed_parameter_list);
  }

  /// \brief Run the kernel's core calculation.
  virtual ReturnType execute(const stk::mesh::Entity& entity) override = 0;
  //@}
};  // MetaPairwiseKernel

//! \name Template implementations
//@{

// \name Getters
//{

template <typename ReturnType, class DerivedMetaPairwiseKernel>
std::pair<std::shared_ptr<PartRequirements>, std::shared_ptr<PartRequirements>>
MetaPairwiseKernel<ReturnType, DerivedMetaPairwiseKernel>::get_part_requirements(
    const Teuchos::ParameterList& fixed_parameter_list) const {
  return static_get_part_requirements(fixed_parameter_list);
}

template <typename ReturnType, class DerivedMetaPairwiseKernel>
Teuchos::ParameterList MetaPairwiseKernel<ReturnType, DerivedMetaPairwiseKernel>::get_valid_fixed_params() const {
  return static_get_valid_fixed_params();
}

template <typename ReturnType, class DerivedMetaPairwiseKernel>
Teuchos::ParameterList MetaPairwiseKernel<ReturnType, DerivedMetaPairwiseKernel>::get_valid_transient_params() const {
  return static_get_valid_transient_params();
}

template <typename ReturnType, class DerivedMetaPairwiseKernel>
std::string MetaPairwiseKernel<ReturnType, DerivedMetaPairwiseKernel>::get_class_identifier() const {
  return static_get_class_identifier();
}
//}

// \name Actions
//{

template <typename ReturnType, class DerivedMetaPairwiseKernel>
std::shared_ptr<MetaPairwiseKernelBase<ReturnType>>
MetaPairwiseKernel<ReturnType, DerivedMetaPairwiseKernel>::create_new_instance(
    stk::mesh::BulkData* const bulk_data_ptr, const Teuchos::ParameterList& fixed_parameter_list) const {
  return static_create_new_instance(bulk_data_ptr, fixed_parameter_list);
}
//}
//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METAKERNEL_HPP_
