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

#ifndef MUNDY_METHODS_RESOLVE_CONSTRAINTS_TECHNIQUES_NON_SMOOTH_LCP_COMPUTE_CONSTRAINT_FORCING_KERNELS_COLLISION_HPP_
#define MUNDY_METHODS_RESOLVE_CONSTRAINTS_TECHNIQUES_NON_SMOOTH_LCP_COMPUTE_CONSTRAINT_FORCING_KERNELS_COLLISION_HPP_

/// \file Collision.hpp
/// \brief Declaration of the ComputeConstraintForcing's Collision kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field, stl::mesh::field_data
#include <stk_topology/topology.hpp>   // for stk::topology

// Mundy libs
#include <mundy_meta/FieldRequirements.hpp>   // for mundy::meta::FieldRequirements
#include <mundy_meta/MetaKernelPairwise.hpp>  // for mundy::meta::MetaKernelPairwise, mundy::meta::MetaKernelPairwiseBase
#include <mundy_meta/MetaKernelPairwiseFactory.hpp>   // for mundy::meta::MetaKernelPairwiseFactory
#include <mundy_meta/MetaKernelPairwiseRegistry.hpp>  // for mundy::meta::MetaKernelPairwiseRegistry
#include <mundy_meta/PartRequirements.hpp>            // for mundy::meta::PartRequirements
#include <mundy_methods/resolve_constraints/techniques/non_smooth_lcp/ComputeConstraintForcing.hpp>  // for mundy::methods::...::non_smooth_lcp::ComputeConstraintForcing

namespace mundy {

namespace methods {

namespace resolve_constraints {

namespace techniques {

namespace non_smooth_lcp {

namespace compute_constraint_forcing {

namespace kernels {

/// \class Collision
/// \brief Concrete implementation of \c MetaKernel for computing the axis aligned boundary box of spheres.
class Collision : public mundy::meta::MetaPairwiseKernel<void, Collision>,
                  public mundy::meta::MetaPairwiseKernelRegistry<void, Collision, ComputeConstraintForcing> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit Collision(stk::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &parameter_list);
  //@}

  //! \name MetaKernel interface implementation
  //@{

  /// \brief Get the requirements that this kernel imposes upon each particle and/or constraint.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::PartRequirements> details_static_get_part_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &parameter_list) {
    std::shared_ptr<mundy::meta::PartRequirements> required_part_params =
        std::make_shared<mundy::meta::PartRequirements>();
    required_part_params->set_part_topology(stk::topology::BEAM_2);
    required_collision_part_params->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_node_coord_field_name_), stk::topology::NODE_RANK, 3, 1));
    required_collision_part_params->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_node_force_field_name_), stk::topology::NODE_RANK, 3, 1));
    required_collision_part_params->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_element_lagrange_multiplier_field_name_), stk::topology::ELEMENT_RANK, 1, 1));
    return required_part_params;
  }

  /// \brief Get the default parameters for this class.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c ParameterList
  /// will be created. You can save the result yourself if you wish to reuse it.
  static Teuchos::ParameterList details_static_get_valid_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("node_coordinate_field_name", std::string(default_node_coord_field_name_),
                               "Name of the node field containing the coordinate of the sphere's center.");
    default_parameter_list.set("node_force_field_name", std::string(default_node_force_field_name_),
                               "Name of the node field containing the sphere's com force.");
    default_parameter_list.set("element_lagrange_multiplier_field_name",
                               std::string(default_element_lagrange_multiplier_field_name_),
                               "Name of the element field containing the constraint's Lagrange multiplier.");

    return default_parameter_list;
  }

  /// \brief Get the unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our \c MetaKernelPairwiseRegistry.
  static std::string details_static_get_class_identifier() {
    return std::string(class_identifier_);
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  static std::shared_ptr<mundy::meta::MetaKernelBase<void>> details_static_create_new_instance(
      stk::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &parameter_list) {
    return std::make_shared<Collision>(bulk_data_ptr, parameter_list);
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Run the kernel's core calculation.
  /// \param collision_node [in] A node connected to a collision element.
  void execute(const stk::mesh::Entity &collision_node) override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_node_coord_field_name_ = "NODE_COORD";
  static constexpr std::string_view default_node_force_field_name_ = "NODE_FORCE";
  static constexpr std::string_view default_element_lagrange_multiplier_field_name_ = "ELEMENT_LAGRANGE_MULTIPLIER";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our MetaKernelRegistry.
  static constexpr std::string_view class_identifier_ = "COLLISION";

  /// \brief The BulkData objects this class acts upon.
  stk::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData objects this class acts upon.
  stk::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Minimum allowable signed separation distance between colliding bodies.
  double minimum_allowable_separation_;

  /// \brief Name of the node field containing the coordinate of each node.
  std::string node_coord_field_name_;

  /// \brief Name of the node field containing the force on each node.
  std::string node_force_field_name_;

  /// \brief Name of the element field containing the constraint's Lagrange multiplier.
  std::string element_lagrange_multiplier_field_name_;

  /// \brief Node field containing the coordinate of the sphere's center.
  stk::mesh::Field<double> *node_coord_field_ptr_ = nullptr;

  /// \brief Node field containing the coordinate of the force on each node.
  stk::mesh::Field<double> *node_force_field_ptr_ = nullptr;

  /// \brief Element field containing the sphere radius.
  stk::mesh::Field<double> *element_lagrange_multiplier_field_ptr_ = nullptr;
  //@}
};  // Collision

}  // namespace kernels

}  // namespace compute_constraint_forcing

}  // namespace non_smooth_lcp

}  // namespace techniques

}  // namespace resolve_constraints

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_RESOLVE_CONSTRAINTS_TECHNIQUES_NON_SMOOTH_LCP_COMPUTE_CONSTRAINT_FORCING_KERNELS_COLLISION_HPP_
