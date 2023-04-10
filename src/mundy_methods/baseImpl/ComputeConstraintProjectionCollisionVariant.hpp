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

#ifndef MUNDY_METHODS_COMPUTECONSTRAINTPROJECTIONCOLLISIONVARIANT_HPP_
#define MUNDY_METHODS_COMPUTECONSTRAINTPROJECTIONCOLLISIONVARIANT_HPP_

/// \file ComputeConstraintProjectionCollisionVariant.hpp
/// \brief Declaration of the ComputeConstraintProjectionCollisionVariant class

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

/// \class ComputeConstraintProjectionCollisionVariant
/// \brief Concrete implementation of \c MultibodyVariant for computing projection of the constraint's Lagrange
/// multiplier onto the feasible set of collision constraints.
class ComputeConstraintProjectionCollisionVariant
    : public MetaMethod<ComputeConstraintProjectionCollisionVariant>,
      public MetaMethodRegistry<ComputeConstraintProjectionCollisionVariant, ComputeConstraintProjection> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit ComputeConstraintProjectionCollisionVariant(const Teuchos::RCP<Teuchos::ParameterList> &parameter_list)
      : parameter_list_(parameter_list),
        node_coord_field_name_(params.get_value<std::string>("node_coord")),
        node_normal_vec_field_name_(params.get_value<std::string>("node_normal_vec")),
        lagrange_multiplier_field_name_(params.get_value<std::string>("lagrange_multiplier")),
        min_allowable_sep_(params.get_value<double>("minimum allowable separation")) {
  }

  //@}
  //! \name Attributes
  //@{

  /// \brief Get the requirements that this manager imposes upon each particle and/or constraint.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartParams
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::unique_ptr<PartParams> get_part_requirements(
      [[maybe_unused]] const Teuchos::RCP<Teuchos::ParameterList> &parameter_list) {
    std::unique_ptr<PartParams> required_part_params = std::make_unique<PartParams>("collision", std::topology::QUAD4);
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
  static Teuchos::RCP<Teuchos::ParameterList> get_valid_params() {
    Teuchos::RCP<Teuchos::ParameterList> default_parameter_list;
    default_parameter_list.set_param("minimum allowable separation", 0.0);
    default_parameter_list.set_param("node coordinate field name", "node_coord");
    default_parameter_list.set_param("node normal vector field name", "node_normal_vec");
    default_parameter_list.set_param("lagrange multiplier field name", "lagrange_multiplier");
    default_parameter_list.set_param("constraint violation field name", "constraint_violation");
    return default_parameter_list;
  }

  //@}

  //! \name Actions
  //@{
  execute(const stk::mesh::BulkData *bulk_data_ptr, const stk::mesh::Part &part) {
    const stk::mesh::Field &node_coord_field =
        bulk_data_ptr->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name_);
    const stk::mesh::Field &node_normal_vec_field =
        bulk_data_ptr->get_field<double>(stk::topology::NODE_RANK, node_normal_vec_field_name_);
    const stk::mesh::Field &constraint_violation_field =
        bulk_data_ptr->get_field<double>(stk::topology::ELEM_RANK, constraint_violation_field_name_);

    stk::mesh::Selector locally_owned_part = metaB.locally_owned_part() && part;
    stk::mesh::for_each_entity_run(
        *bulk_data_ptr, stk::topology::NODE_RANK, locally_owned_part,
        [&node_coord_field, &node_normal_vec_field, &lagrange_multiplier_field, &constraint_violation_field](
            const stk::mesh::BulkData &bulk_data, stk::mesh::Entity element) {
          stk::mesh::Entity const *nodes = bulk_data.begin_nodes(element);
          const double *contact_pointI = stk::mesh::field_data(node_coord_field, nodes[1]);
          const double *contact_pointJ = stk::mesh::field_data(node_coord_field, nodes[2]);
          const double *contact_normal_vecI = stk::mesh::field_data(node_normal_vec_field, nodes[1]);
          double *constraint_violation = stk::mesh::field_data(constraint_violation_field, element);

          constraint_violation[0] = contact_normal_vecI[0] * (contact_pointJ[0] - contact_pointI[0]) +
                                    contact_normal_vecI[1] * (contact_pointJ[1] - contact_pointI[1]) +
                                    contact_normal_vecI[2] * (contact_pointJ[2] - contact_pointI[2]) -
                                    min_allowable_sep_;
        });
  }

 private:
  const Teuchos::RCP<Teuchos::ParameterList> parameter_list_;
  const double min_allowable_sep_;
  const std::string node_coord_field_name_;
  const std::string node_normal_vec_field_name_;
  const std::string lagrange_multiplier_field_name_;
  const std::string constraint_violation_field_name_;
};  // ComputeConstraintProjectionCollisionVariant

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTECONSTRAINTPROJECTIONCOLLISIONVARIANT_HPP_
