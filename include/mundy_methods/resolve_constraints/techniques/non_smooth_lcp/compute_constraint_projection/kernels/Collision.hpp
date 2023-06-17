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

#ifndef MUNDY_METHODS_RESOLVE_CONSTRAINTS_TECHNIQUES_NON_SMOOTH_LCP_COMPUTE_CONSTRAINT_PROJECTION_KERNELS_COLLISION_HPP_
#define MUNDY_METHODS_RESOLVE_CONSTRAINTS_TECHNIQUES_NON_SMOOTH_LCP_COMPUTE_CONSTRAINT_PROJECTION_KERNELS_COLLISION_HPP_

/// \file Collision.hpp
/// \brief Declaration of the ComputeConstraintProjection's Collision kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_mesh/BulkData.hpp>           // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>           // for mundy::mesh::MetaData
#include <mundy_meta/FieldRequirements.hpp>  // for mundy::meta::FieldRequirements
#include <mundy_meta/MetaFactory.hpp>        // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>         // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaRegistry.hpp>       // for mundy::meta::MetaKernelRegistry
#include <mundy_meta/PartRequirements.hpp>   // for mundy::meta::PartRequirements
#include <mundy_methods/resolve_constraints/techniques/non_smooth_lcp/ComputeConstraintProjection.hpp>  // for mundy::methods::...::ComputeConstraintProjection

namespace mundy {

namespace methods {

namespace resolve_constraints {

namespace techniques {

namespace non_smooth_lcp {

namespace compute_constraint_projection {

namespace kernels {

/// \class Collision
/// \brief Concrete implementation of \c MetaKernel for computing the axis aligned boundary box of spheres.
class Collision : public mundy::meta::MetaKernel<void, Collision>,
                  public ComputeConstraintProjection::OurKernelRegistry<Collision> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit Collision(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list);
  //@}

  //! \name MetaKernel interface implementation
  //@{

  /// \brief Get the requirements that this kernel imposes upon each particle and/or constraint.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::MeshRequirements>(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_parameter_list) {
    std::shared_ptr<mundy::meta::PartRequirements> required_part_params =
        std::make_shared<mundy::meta::PartRequirements>();
    required_part_params->set_part_topology(stk::topology::QUAD_4);
    required_part_params->add_field_req(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_element_lagrange_multiplier_field_name_), stk::topology::ELEMENT_RANK, 1, 1));
    return required_part_params;
  }

  /// \brief Get the default fixed parameters for this class (those that impact the part requirements).
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c ParameterList
  /// will be created. You can save the result yourself if you wish to reuse it.
  static Teuchos::ParameterList details_static_get_valid_fixed_params() {
    static Teuchos::ParameterList default_fixed_parameter_list;
    default_fixed_parameter_list.set("element_lagrange_multiplier_field_name",
                                     std::string(default_element_lagrange_multiplier_field_name_),
                                     "Name of the element field containing the constraint's Lagrange multiplier.");
    return default_fixed_parameter_list;
  }

  /// \brief Get the default mutable parameters for this class (those that do not impact the part requirements).
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c ParameterList
  /// will be created. You can save the result yourself if you wish to reuse it.
  static Teuchos::ParameterList details_static_get_valid_mutable_params() {
    static Teuchos::ParameterList default_mutable_parameter_list;
    return default_mutable_parameter_list;
  }

  /// \brief Get the unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our \c MetaKernelRegistry.
  static std::string details_static_get_class_identifier() {
    return std::string(class_identifier_);
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_parameter_list [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaKernelBase<void>> details_static_create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_parameter_list) {
    return std::make_shared<Collision>(bulk_data_ptr, fixed_parameter_list);
  }

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_parameter_list) override;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the kernel's core calculation.
  /// \param element [in] The element acted on by the kernel.
  void execute(const stk::mesh::Entity &element) override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_element_lagrange_multiplier_field_name_ = "ELEMENT_LAGRANGE_MULTIPLIER";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our MetaKernelRegistry.
  static constexpr std::string_view class_identifier_ = "COLLISION";

  /// \brief The BulkData objects this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData objects this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Name of the element field containing the constraint's Lagrange multiplier.
  std::string element_lagrange_multiplier_field_name_;

  /// \brief Element field containing the sphere radius.
  stk::mesh::Field<double> *element_lagrange_multiplier_field_ptr_;
  //@}
};  // Collision

}  // namespace kernels

}  // namespace compute_constraint_projection

}  // namespace non_smooth_lcp

}  // namespace techniques

}  // namespace resolve_constraints

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_RESOLVE_CONSTRAINTS_TECHNIQUES_NON_SMOOTH_LCP_COMPUTE_CONSTRAINT_PROJECTION_KERNELS_COLLISION_HPP_
