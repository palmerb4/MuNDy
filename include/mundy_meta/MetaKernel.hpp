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

/// \class MetaKernelBase
/// \brief The polymorphic interface which all \c MetaKernels will share.
///
/// This design pattern allows for \c MetaKernel to use CRTP to force derived classes to implement certain static
/// functions while also having a consistant polymorphic interface that allows different \c MetaKernels to be stored in
/// a vector of pointers.
///
/// \tparam ReturnType The return type of the execute function.
/// \tparam RegistrationType The type of this class's identifier.
template <typename ReturnType, typename RegistrationType = std::string>
class MetaKernelBase
    : virtual public HasMeshRequirementsAndIsRegisterableBase<MetaKernelBase<ReturnType, RegistrationType>,
                                                                   RegistrationType> {
 public:
  //! \name Setters
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  virtual void set_mutable_params(const Teuchos::ParameterList &mutable_params) = 0;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the kernel's core calculation.
  virtual ReturnType execute(stk::mesh::Entity entity) = 0;
  //@}
};  // MetaKernelBase

/// \class MetaKernel
/// \brief The static interface that encodes a class's assumptions about the structure and contents of an STK mesh.
///
/// This class follows the Curiously Recurring Template Pattern such that each class derived from \c MetaKernel must
/// implement the following static member functions
/// - \c details_static_get_mesh_requirements implementation of the \c static_get_mesh_requirements interface.
/// - \c details_static_validate_fixed_parameters_and_set_defaults implementation of the
///     \c static_validate_fixed_parameters_and_set_defaults interface.
/// - \c details_static_validate_mutable_parameters_and_set_defaults implementation of the
///     \c static_validate_mutable_parameters_and_set_defaults interface.
/// - \c details_static_get_class_identifier implementation of the \c static_get_class_identifier interface.
/// - \c details_static_create_new_instance implementation of the \c static_create_new_instance interface.
///
/// To keep these out of the public interface, we suggest that each details function be made private and
/// \c MetaKernel<DerivedMetaKernel> be made a friend of \c DerivedMetaKernel.
///
/// \note The _t in our template paramaters breaks our naming convention for types but is used to prevent template
/// shaddowing by internal typedefs.
///
/// \tparam ReturnType_t The return type of the execute function.
/// \tparam DerivedMetaKernel_t A class derived from \c MetaKernel that implements the desired interface.
/// \tparam RegistrationType_t The type of this class's identifier.
template <typename ReturnType_t, class DerivedMetaKernel_t, typename RegistrationType_t = std::string>
class MetaKernel
    : virtual public MetaKernelBase<ReturnType_t, RegistrationType_t>,
      public HasMeshRequirementsAndIsRegisterable<MetaKernel<ReturnType_t, DerivedMetaKernel_t, RegistrationType_t>, 
                                                  MetaKernelBase<ReturnType_t, RegistrationType_t>,
                                                  RegistrationType_t> {
 public:
  //! \name Typedefs
  //@{

  using ReturnType = ReturnType_t;
  using RegistrationType = RegistrationType_t;
  using PolymorphicBase = MetaKernelBase<ReturnType, RegistrationType>;
  //@}

  //! \name Getters
  //@{

  /// \brief Get the requirements that this \c MetaKernel imposes upon each input part.
  ///
  /// The set part requirements returned by this function are meant to encode the assumptions made by this class
  /// with respect to the structure, topology, and fields of the STK mesh. These assumptions may vary
  /// based on parameters in the \c fixed_params but not the \c mutable_params.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  static std::shared_ptr<MeshRequirements> details_static_get_mesh_requirements(
      const Teuchos::ParameterList &fixed_params) {
    return DerivedMetaKernel_t::details_static_get_mesh_requirements(fixed_params);
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void static_validate_fixed_parameters_and_set_defaults(
      Teuchos::ParameterList const *fixed_params_ptr) {
    DerivedMetaKernel_t::static_validate_fixed_parameters_and_set_defaults(fixed_params_ptr);
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void details_static_validate_mutable_parameters_and_set_defaults(
      Teuchos::ParameterList const *mutable_params_ptr) {
    DerivedMetaKernel_t::details_static_validate_mutable_parameters_and_set_defaults(mutable_params_ptr);
  }

  /// \brief Get the unique class identifier. Here, 'unique' means with with respect to other class in our registere(s).
  static RegistrationType details_static_get_class_identifier() {
    return DerivedMetaKernel_t::details_static_get_class_identifier();
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  static std::shared_ptr<MetaKernelBase<ReturnType, RegistrationType>> details_static_create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return DerivedMetaKernel_t::details_static_create_new_instance(bulk_data_ptr, fixed_params);
  }
  //@}
};  // MetaKernel

//! \name Partial Specializations
//@{

/// \brief Partial specialization for MetaKernelBases, identified by a mundy multibody type.
template <typename ReturnType>
using MetaMultibodyKernelBase = MetaKernelBase<ReturnType, mundy::multibody::multibody_t>;

/// \brief Partial specialization for MetaKernels, identified by a mundy multibody type.
template <typename ReturnType, class DerivedMetaKernel>
using MetaMultibodyKernel = MetaKernel<ReturnType, DerivedMetaKernel, mundy::multibody::multibody_t>;

/// \brief Partial specialization for MetaKernels, identified by an stk topology type.
template <typename ReturnType>
using MetaTopologyKernelBase = MetaKernelBase<ReturnType, stk::topology::topology_t>;

/// \brief Partial specialization for MetaKernels, identified by an stk topology type.
template <typename ReturnType, class DerivedMetaKernel>
using MetaTopologyKernel = MetaKernel<ReturnType, DerivedMetaKernel, stk::topology::topology_t>;
//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METAKERNEL_HPP_
