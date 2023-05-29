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

#ifndef MUNDY_META_METAPAIRWISEKERNEL_HPP_
#define MUNDY_META_METAPAIRWISEKERNEL_HPP_

/// \file MetaPairwiseKernel.hpp
/// \brief Declaration of the MetaPairwiseKernel class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part

// Mundy libs
#include <mundy_meta/HasMeshRequirementsAndIsRegisterable.hpp>  // for mundy::meta::HasMeshRequirementsAndIsRegisterable
#include <mundy_meta/PartRequirements.hpp>                      // for mundy::meta::PartRequirements

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
/// \tparam RegistrationType The type of this class's identifier.
template <typename ReturnType, typename RegistrationType = std::string>
class MetaPairwiseKernelBase : virtual public HasMeshRequirementsAndIsRegisterableBase<RegistrationType> {
 public:
  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  virtual ReturnType execute(stk::mesh::Entity entity1, stk::mesh::Entity entity2) = 0;
  //@}
};  // MetaPairwiseKernelBase

/// \class MetaPairwiseKernel
/// \brief The static interface that encodes a class's assumptions about the structure and contents of an STK mesh.
///
/// This class follows the Curiously Recurring Template Pattern such that each class derived from \c MetaPairwiseKernel
/// must implement the following static member functions
/// - \c details_static_get_part_requirements implementation of the \c static_get_part_requirements interface.
/// - \c details_static_get_valid_fixed_params implementation of the \c static_get_valid_fixed_params interface.
/// - \c details_static_get_valid_transient_params implementation of the \c static_get_valid_transient_params interface.
/// - \c details_static_get_class_identifier implementation of the \c static_get_class_identifier interface.
/// - \c details_static_create_new_instance implementation of the \c static_create_new_instance interface.
///
/// To keep these out of the public interface, we suggest that each details function be made private and
/// \c MetaPairwiseKernel<DerivedMetaPairwiseKernel> be made a friend of \c DerivedMetaPairwiseKernel.
///
/// \tparam ReturnType The return type of the execute function.
/// \tparam DerivedMetaPairwiseKernel A class derived from \c MetaPairwiseKernel that implements the desired interface.
/// \tparam RegistrationType The type of this class's identifier.
template <typename ReturnType, class DerivedMetaPairwiseKernel, typename RegistrationType = std::string>
class MetaPairwiseKernel
    : virtual public MetaPairwiseKernelBase<ReturnType, RegistrationType>,
      virtual public HasMeshRequirementsAndIsRegisterable<MetaPairwiseKernel<ReturnType, DerivedMetaPairwiseKernel>,
                                                          RegistrationType> {
 public:
  //! \name Getters
  //@{

  /// \brief Get the requirements that this \c MetaPairwiseKernel imposes upon each input part.
  ///
  /// The set part requirements returned by this function are meant to encode the assumptions made by this class
  /// with respect to the structure, topology, and fields of the STK mesh. These assumptions may vary
  /// based on parameters in the \c fixed_parameter_list but not the \c transient_parameter_list.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  static std::vector<std::shared_ptr<PartRequirements>> details_static_get_part_requirements(
      const Teuchos::ParameterList &fixed_parameter_list) {
    return DerivedMetaPairwiseKernel::details_static_get_part_requirements(fixed_parameter_list);
  }

  /// \brief Get the valid fixed parameters and their default parameter list for this \c MetaPairwiseKernel.
  static Teuchos::ParameterList details_static_get_valid_fixed_params() {
    return DerivedMetaPairwiseKernel::details_static_get_valid_fixed_params();
  }

  /// \brief Get the valid transient parameters and their default parameter list for this \c MetaPairwiseKernel.
  static Teuchos::ParameterList details_static_get_valid_transient_params() {
    return DerivedMetaPairwiseKernel::details_static_get_valid_transient_params();
  }

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c
  /// MetaPairwiseKernel.
  static RegistrationType details_static_get_class_identifier() {
    return DerivedMetaPairwiseKernel::details_static_get_class_identifier();
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  static std::shared_ptr<MetaPairwiseKernelBase<ReturnType, RegistrationType>> details_static_create_new_instance(
      stk::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list) {
    return DerivedMetaPairwiseKernel::details_static_create_new_instance(bulk_data_ptr, fixed_parameter_list);
  }
  //@}
};  // MetaPairwiseKernel

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METAPAIRWISEKERNEL_HPP_
