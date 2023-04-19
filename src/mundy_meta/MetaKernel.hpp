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
/// \brief A consistant base for all \c MetaKernels.
class MetaKernelBase {};  // MetaKernelBase

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
/// \tparam DerivedMetaKernel A class derived from \c MetaKernel that implements the desired interface.
/// \tparam ReturnType The return type of the execute function.
template <class DerivedMetaKernel, typename ReturnType,
          typename std::enable_if<std::is_base_of<MetaKernelBase, DerivedMetaKernel>::value, void>::type>
class MetaKernel : public Teuchos::Describable {
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
  static std::unique_ptr<PartRequirements> get_part_requirements(const Teuchos::ParameterList& parameter_list) const {
    return DerivedMetaKernel::details_get_part_requirements(parameter_list);
  }

  /// \brief Get the valid parameters and their default parameter list for this \c MetaKernel.
  static Teuchos::ParameterList get_valid_params() const {
    return DerivedMetaKernel::details_get_valid_params();
  }

  /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c MetaKernel.
  static std::string get_class_identifier() const {
    return DerivedMetaKernel::details_get_class_identifier();
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Generate a new instance of this class.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  static std::unique_ptr<MetaKernelBase> create_new_instance(const Teuchos::ParameterList& parameter_list) const {
    return DerivedMetaKernel::details_create_new_instance(parameter_list);
  }

  /// \brief Run the kernel's core calculation.
  virtual ReturnType execute(const stk::mesh::Entity& entity) = 0;
  //@}

};  // MetaKernel

}  // namespace meta

}  // namespace mundy

//}
#endif  // MUNDY_META_METAKERNEL_HPP_
