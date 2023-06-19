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

#ifndef MUNDY_META_METAKWAYKERNEL_HPP_
#define MUNDY_META_METAKWAYKERNEL_HPP_

/// \file MetaKWayKernel.hpp
/// \brief Declaration of the MetaKWayKernel class

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

/// \class MetaKWayKernelBase
/// \brief The polymorphic interface which all \c MetaKWayKernels will share.
///
/// This design pattern allows for \c MetaKWayKernel to use CRTP to force derived classes to implement certain
/// static functions while also having a consistant polymorphic interface that allows different \c MetaKWayKernels
/// to be stored in a vector of pointers.
///
/// \tparam K The number of entities passed to execute.
/// \tparam ReturnType The return type of the execute function.
/// \tparam RegistrationType The type of this class's identifier.
template <std::size_t K, typename ReturnType, typename RegistrationType = std::string>
class MetaKWayKernelBase
    : virtual public HasMeshRequirementsAndIsRegisterableBase<MetaKWayKernelBase<K, ReturnType, RegistrationType>,
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
  virtual ReturnType execute(std::array<stk::mesh::Entity, K> entity_array) = 0;
  //@}
};  // MetaKWayKernelBase

/// \class MetaKWayKernel
/// \brief The static interface that encodes a class's assumptions about the structure and contents of an STK mesh.
///
/// This class follows the Curiously Recurring Template Pattern such that each class derived from \c MetaKWayKernel
/// must implement the following static member functions
/// - \c details_static_get_mesh_requirements implementation of the \c static_get_mesh_requirements interface.
/// - \c details_static_validate_fixed_parameters_and_set_defaults implementation of the
///     \c static_validate_fixed_parameters_and_set_defaults interface.
/// - \c details_static_validate_mutable_parameters_and_set_defaults implementation of the
///     \c static_validate_mutable_parameters_and_set_defaults interface.
/// - \c details_static_get_class_identifier implementation of the \c static_get_class_identifier interface.
/// - \c details_static_create_new_instance implementation of the \c static_create_new_instance interface.
///
/// To keep these out of the public interface, we suggest that each details function be made private and
/// \c MetaKWayKernel<DerivedMetaKWayKernel> be made a friend of \c DerivedMetaKWayKernel.
///
/// \note The _t in our template paramaters breaks our naming convention for types but is used to prevent template
/// shaddowing by internal typedefs.
///
/// \tparam N The number of entities passed to execute.
/// \tparam ReturnType_t The return type of the execute function.
/// \tparam DerivedMetaKWayKernel_t A class derived from \c MetaKWayKernel that implements the desired interface.
/// \tparam RegistrationType_t The type of this class's identifier.
template <std::size_t K, typename ReturnType_t, class DerivedMetaKWayKernel_t,
          typename RegistrationType_t = std::string>
class MetaKWayKernel : virtual public MetaKWayKernelBase<K, ReturnType_t, RegistrationType_t>,
                       public HasMeshRequirementsAndIsRegisterable<
                           MetaKWayKernel<K, ReturnType_t, DerivedMetaKWayKernel_t, RegistrationType_t>,
                           MetaKWayKernelBase<K, ReturnType_t, RegistrationType_t>, RegistrationType_t> {
 public:
  //! \name Typedefs
  //@{

  using ReturnType = ReturnType_t;
  using RegistrationType = RegistrationType_t;
  using PolymorphicBaseType = MetaKWayKernelBase<K, ReturnType, RegistrationType>;
  //@}

  //! \name Getters
  //@{

  /// \brief Get the requirements that this \c MetaKWayKernel imposes upon each input part.
  ///
  /// The set part requirements returned by this function are meant to encode the assumptions made by this class
  /// with respect to the structure, topology, and fields of the STK mesh. These assumptions may vary
  /// based on parameters in the \c fixed_params but not the \c mutable_params.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  static std::shared_ptr<MeshRequirements> details_static_get_mesh_requirements(
      const Teuchos::ParameterList &fixed_params) {
    return DerivedMetaKWayKernel_t::details_static_get_mesh_requirements(fixed_params);
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void details_static_validate_fixed_parameters_and_set_defaults(
      Teuchos::ParameterList const *fixed_params_ptr) {
    DerivedMetaKWayKernel_t::details_static_validate_fixed_parameters_and_set_defaults(fixed_params_ptr);
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void details_static_validate_mutable_parameters_and_set_defaults(
      Teuchos::ParameterList const *mutable_params_ptr) {
    DerivedMetaKWayKernel_t::details_static_validate_mutable_parameters_and_set_defaults(mutable_params_ptr);
  }

  /// \brief Get the unique class identifier. Here, 'unique' means with with respect to other class in our registere(s).
  static RegistrationType details_static_get_class_identifier() {
    return DerivedMetaKWayKernel_t::details_static_get_class_identifier();
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_valid_fixed_params.
  static std::shared_ptr<MetaKWayKernelBase<K, ReturnType, RegistrationType>> details_static_create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return DerivedMetaKWayKernel_t::details_static_create_new_instance(bulk_data_ptr, fixed_params);
  }
  //@}
};  // MetaKWayKernel

/// \name Type specializations for k-way inputs.
//@{

/// \brief Partial specialization for MetaKWayKernelBase, identified by a mundy multibody type.
template <std::size_t K, typename ReturnType>
using MetaMultibodyKWayKernelBase = MetaKWayKernelBase<K, ReturnType, std::array<mundy::multibody::multibody_t, K>>;

/// \brief Partial specialization for MetaKWayKernel, identified by a mundy multibody type.
template <std::size_t K, typename ReturnType, class DerivedMetaKWayKernel>
using MetaMultibodyKWayKernel =
    MetaKWayKernel<K, ReturnType, DerivedMetaKWayKernel, std::array<mundy::multibody::multibody_t, K>>;

/// \brief Partial specialization for MetaKWayKernelBase, identified by an stk topology type.
template <std::size_t K, typename ReturnType>
using MetaTopologyKWayKernelBase = MetaKWayKernelBase<K, ReturnType, std::array<stk::topology::topology_t, K>>;

/// \brief Partial specialization for MetaKWayKernel, identified by an stk topology type.
template <std::size_t K, typename ReturnType, class DerivedMetaKWayKernel>
using MetaTopologyKWayKernel =
    MetaKWayKernel<K, ReturnType, DerivedMetaKWayKernel, std::array<stk::topology::topology_t, K>>;
//@}

/// \name Type specializations for two way inputs.
//@{

/// \brief Partial specialization for MetaKWayKernelBases, with two input entities.
template <typename ReturnType, typename RegistrationType = std::string>
using MetaTwoWayKernelBase = MetaKWayKernelBase<2, ReturnType, RegistrationType>;

/// \brief Partial specialization for MetaKWayKernels, with two entities.
template <typename ReturnType, class DerivedMetaKWayKernel, typename RegistrationType = std::string>
using MetaTwoWayKernel = MetaKWayKernel<2, ReturnType, DerivedMetaKWayKernel, RegistrationType>;

/// \brief Partial specialization for MetaTwoWayKernelBase, identified by a mundy multibody type.
template <typename ReturnType>
using MetaMultibodyTwoWayKernelBase = MetaMultibodyKWayKernelBase<2, ReturnType>;

/// \brief Partial specialization for MetaTwoWayKernel, identified by a mundy multibody type.
template <typename ReturnType, class DerivedMetaKWayKernel>
using MetaMultibodyTwoWayKernel = MetaMultibodyKWayKernel<2, ReturnType, DerivedMetaKWayKernel>;

/// \brief Partial specialization for MetaTwoWayKernelBase, identified by an stk topology type.
template <typename ReturnType>
using MetaTopologyTwoWayKernelBase = MetaTopologyKWayKernelBase<2, ReturnType>;

/// \brief Partial specialization for MetaTwoWayKernel, identified by an stk topology type.
template <typename ReturnType, class DerivedMetaKWayKernel>
using MetaTopologyTwoWayKernel = MetaTopologyKWayKernel<2, ReturnType, DerivedMetaKWayKernel>;
//@}

/// \name Type specializations for three way inputs.
//@{

/// \brief Partial specialization for MetaKWayKernelBases, with three input entities.
template <typename ReturnType, typename RegistrationType = std::string>
using MetaThreeWayKernelBase = MetaKWayKernelBase<3, ReturnType, RegistrationType>;

/// \brief Partial specialization for MetaKWayKernels, with three entities.
template <typename ReturnType, class DerivedMetaKWayKernel, typename RegistrationType = std::string>
using MetaThreeWayKernel = MetaKWayKernel<3, ReturnType, DerivedMetaKWayKernel, RegistrationType>;

/// \brief Partial specialization for MetaThreeWayKernelBase, identified by a mundy multibody type.
template <typename ReturnType>
using MetaMultibodyThreeWayKernelBase = MetaMultibodyKWayKernelBase<3, ReturnType>;

/// \brief Partial specialization for MetaThreeWayKernel, identified by a mundy multibody type.
template <typename ReturnType, class DerivedMetaKWayKernel>
using MetaMultibodyThreeWayKernel = MetaMultibodyKWayKernel<3, ReturnType, DerivedMetaKWayKernel>;

/// \brief Partial specialization for MetaThreeWayKernelBase, identified by an stk topology type.
template <typename ReturnType>
using MetaTopologyThreeWayKernelBase = MetaTopologyKWayKernelBase<3, ReturnType>;

/// \brief Partial specialization for MetaThreeWayKernel, identified by an stk topology type.
template <typename ReturnType, class DerivedMetaKWayKernel>
using MetaTopologyThreeWayKernel = MetaTopologyKWayKernel<3, ReturnType, DerivedMetaKWayKernel>;
//@}
}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METAKWAYKERNEL_HPP_
