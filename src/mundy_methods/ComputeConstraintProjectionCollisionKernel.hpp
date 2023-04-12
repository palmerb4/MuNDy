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

#ifndef MUNDY_METHODS_COMPUTECONSTRAINTPROJECTIONCOLLISIONKERNEL_HPP_
#define MUNDY_METHODS_COMPUTECONSTRAINTPROJECTIONCOLLISIONKERNEL_HPP_

/// \file ComputeConstraintProjectionCollisionKernel.hpp
/// \brief Declaration of the ComputeConstraintProjectionCollisionKernel class

// clang-format off
#include <gtest/gtest.h>                             // for AssertHelper, etc
#include <mpi.h>                                     // for MPI_COMM_WORLD, etc
#include <stddef.h>                                  // for size_t
#include <vector>                                    // for vector, etc
#include <random>                                    // for rand
#include <memory>                                    // for shared_ptr
#include <string>                                    // for string
#include <type_traits>                               // for static_assert
#include <stk_math/StkVector.hpp>                    // for Vec
#include <stk_mesh/base/BulkData.hpp>                // for BulkData
#include <stk_mesh/base/MetaData.hpp>                // for MetaData
#include <stk_mesh/base/GetEntities.hpp>             // for count_selected_entities
#include <stk_mesh/base/Types.hpp>                   // for EntityVector, etc
#include <stk_mesh/base/Ghosting.hpp>                // for create_ghosting
#include <stk_io/WriteMesh.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_search/SearchMethod.hpp>               // for KDTREE
#include <stk_search/CoarseSearch.hpp>               // for coarse_search
#include <stk_search/BoundingBox.hpp>                // for Sphere, Box, tec.
#include <stk_balance/balance.hpp>                   // for balanceStkMesh
#include <stk_util/parallel/Parallel.hpp>            // for ParallelMachine
#include <stk_util/environment/WallTime.hpp>         // for wall_time
#include <stk_util/environment/perf_util.hpp>
#include <stk_unit_test_utils/BuildMesh.hpp>
#include "stk_util/util/ReportHandler.hpp"           // for ThrowAssert, etc
// clang-format on

namespace mundy {

namespace methods {

/// \class ComputeConstraintProjectionCollisionKernel
/// \brief Concrete implementation of \c MetaKernel for computing the axis aligned boundary box of spheres.
class ComputeConstraintProjectionCollisionKernel
    : public MetaKernel<ComputeConstraintProjectionCollisionKernel>,
      public MetaKernelRegistry<ComputeConstraintProjectionCollisionKernel, ComputeAABB> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit ComputeConstraintProjectionCollisionKernel(const stk::mesh::BulkData *bulk_data_ptr,
                                                      const Teuchos::ParameterList &parameter_list) {
    // Store the input parameters, use default parameters for any parameter not given.
    // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
    parameter_list_ = parameter_list;
    parameter_list_.validateParametersAndSetDefaults(get_valid_params());

    // Fill the internal members using the internal parameter list
    radius_field_name_ = parameter_list_.get<std::string>("radius_field_name");
    bounding_radius_field_name_ = parameter_list_.get<std::string>("bounding_radius_field_name");
    buffer_distance_ = parameter_list_.get<std::string>("buffer_distance");

    // Store the input params.
    const stk::mesh::Field &node_coord_field =
        bulk_data_ptr->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name_);
    const stk::mesh::Field &radius_field =
        bulk_data_ptr->get_field<double>(stk::topology::ELEM_RANK, radius_field_name_);
    const stk::mesh::Field &aabb_field = bulk_data_ptr->get_field<double>(stk::topology::ELEM_RANK, aabb_field_name_);
  }
  //@}

  //! \name MetaKernel interface implementation
  //@{

  /// \brief Get the requirements that this manager imposes upon each particle and/or constraint.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartParams
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::unique_ptr<PartParams> details_get_part_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &parameter_list) {
    std::unique_ptr<PartParams> required_part_params = std::make_unique<PartParams>(std::topology::PARTICLE);
    required_part_params->add_field_params(
        std::make_unique<FieldParams<double>>("node_coord", std::topology::NODE_RANK, 3, 1));
    required_part_params->add_field_params(
        std::make_unique<FieldParams<double>>("node_normal_vec", std::topology::NODE_RANK, 3, 1));
    required_part_params->add_field_params(
        std::make_unique<FieldParams<double>>("lagrange_multiplier", std::topology::ELEMENT_RANK, 1, 1));
    required_part_params->add_field_params(
        std::make_unique<FieldParams<double>>("constraint_violation", std::topology::ELEMENT_RANK, 1, 1));
    return required_part_params;
  }

  /// \brief Get the default parameters for this class.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c ParameterList
  /// will be created. You can save the result yourself if you wish to reuse it.
  static Teuchos::ParameterList details_get_valid_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("minimum_allowable_separation", default_minimum_allowable_separation_,
                               "Minimum allowable separation distance between colliding bodies.");
    default_parameter_list.set(
        "node_coordinate_field_name", default_node_coord_field_name_,
        "Name of the node field containing the coordinate of the constraint's attachment points.");
    default_parameter_list.set(
        "node_normal_vector_field_name", default_node_normal_vec_field_name_,
        "Name of the node field containing the normal vector to the constraint's attachment point.");
    default_parameter_list.set("lagrange_multiplier_field_name", default_lagrange_multiplier_field_name_,
                               "Name of the element field containing the constraint's Lagrange multiplier.");
    default_parameter_list.set("constraint_violation_field_name", default_constraint_violation_field_name_,
                               "Name of the element field containing the constraint's violation measure.");
    return default_parameter_list;
  }
  //@}

  //! \name Actions
  //@{

  void execute(const stk::mesh::Entity &element) {
    stk::mesh::Entity const *nodes = bulk_data.begin_nodes(element);
    const double *contact_pointI = stk::mesh::field_data(*node_coord_field_ptr_, nodes[1]);
    const double *contact_pointJ = stk::mesh::field_data(*node_coord_field_ptr_, nodes[2]);
    const double *contact_normal_vecI = stk::mesh::field_data(*node_normal_vec_field_ptr_, nodes[1]);
    double *constraint_violation = stk::mesh::field_data(*constraint_violation_field_ptr_, element);

    constraint_violation[0] = contact_normal_vecI[0] * (contact_pointJ[0] - contact_pointI[0]) +
                              contact_normal_vecI[1] * (contact_pointJ[1] - contact_pointI[1]) +
                              contact_normal_vecI[2] * (contact_pointJ[2] - contact_pointI[2]) - min_allowable_sep_;
  }
  //@}

 private:
  //! \name Default parameters
  //@{
  static constexpr double default_minimum_allowable_separation_ = 0.0;
  static constexpr std::string default_node_coord_field_name_ = "NODE_COORD";
  static constexpr std::string default_node_normal_vec_field_name_ = "NODE_NORMAL_VEC";
  static constexpr std::string default_lagrange_multiplier_field_name_ = "LAGRANGE_MULTIPLIER";
  static constexpr std::string default_constraint_violation_field_name_ = "CONSTRAINT_VIOLATION";
  //@}

  //! \name Internal members
  //@{

  /// \brief Current parameter list with valid entries.
  Teuchos::ParameterList parameter_list_;

  /// \brief Minimum allowable separation distance between colliding bodies.
  double minimum_allowable_separation_;

  /// \brief Name of the node field containing the coordinate of the constraint's attachment points.
  std::string node_coord_field_name_;

  /// \brief Name of the node field containing the normal vector to the constraint's attachment point.
  std::string node_normal_vec_field_name_;

  /// \brief Name of the element field containing the constraint's Lagrange multiplier.
  std::string lagrange_multiplier_field_name_;

  /// \brief Name of the element field containing the constraint's violation measure.
  std::string constraint_violation_field_name_;

  /// \brief Node field containing the coordinate of the constraint's attachment points.
  stk::mesh::Field *node_coord_field_ptr_;

  /// \brief Element field within which the output axis-aligned boundary boxes will be written.
  ///
  /// Per convention, the normal vector points outward from the attached surface.
  stk::mesh::Field *node_normal_vec_field_ptr_;

  /// \brief Element field containing the sphere radius.
  stk::mesh::Field *lagrange_multiplier_field_ptr_;

  /// \brief Element field containing the sphere radius.
  stk::mesh::Field *constraint_violation_field_ptr_;

  //@}
};  // ComputeConstraintProjectionCollisionKernel

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTECONSTRAINTPROJECTIONCOLLISIONKERNEL_HPP_
