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

#ifndef MUNDY_METHODS_COMPUTEOBBSPHEREVARIANT_HPP_
#define MUNDY_METHODS_COMPUTEOBBSPHEREVARIANT_HPP_

/// \file ComputeOBBSphereVariant.hpp
/// \brief Declaration of the ComputeOBBSphereVariant class

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

/// \class ComputeOBBSphereVariant
/// \brief Concrete implementation of \c MultibodyVariant for computing the object aligned bounding box of spheres.
class ComputeOBBSphereVariant : public MetaMethod<ComputeOBBSphereVariant>,
                                public MetaMethodRegistry<ComputeOBBSphereVariant, ComputeOBB> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit ComputeOBBSphereVariant(const stk::util::ParameterList &parameter_list)
      : parameter_list_(parameter_list),
        obb_field_name_(params.get_value<std::string>("obb field name")),
        node_coord_field_name_(params.get_value<std::string>("node_coord")),
        radius_field_name_(params.get_value<std::string>("radius field name")),
        buffer_distance_(params.get_value<double>("buffer distance")) {
  }

  //@}
  //! \name Attributes
  //@{

  /// \brief Get the requirements that this manager imposes upon each particle and/or constraint.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_default_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartParams
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::unique_ptr<PartParams> get_part_requirements(
      [[maybe_unused]] const stk::util::ParameterList &parameter_list) {
    std::unique_ptr<PartParams> required_part_params = std::make_unique<PartParams>("spheres", std::topology::PARTICLE);
    required_part_params->add_field_params("node_coord",
                                           std::make_unique<FieldParams<double>>(std::topology::NODE_RANK, 3, 1));
    required_part_params->add_field_params("radius",
                                           std::make_unique<FieldParams<double>>(std::topology::ELEMENT_RANK, 1, 1));
    required_part_params->add_field_params("obb",
                                           std::make_unique<FieldParams<double>>(std::topology::ELEMENT_RANK, 4, 1));
    return required_part_params;
  }

  /// \brief Get the default parameters for this class.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c ParameterList
  /// will be created. You can save the result yourself if you wish to reuse it.
  static stk::util::ParameterList get_default_params() {
    stk::util::ParameterList default_parameter_list;
    default_parameter_list.set_param("node coordinate field name", "node_coord");
    default_parameter_list.set_param("obb field name", "obb");
    default_parameter_list.set_param("radius field name", "radius");
    default_parameter_list.set_param("buffer distance", 0.0);
    return default_parameter_list;
  }

  //@}

  //! \name Actions
  //@{
  run(const stk::mesh::BulkData *bulk_data_ptr, const stk::mesh::Part &part) {
    const stk::mesh::Field &node_coord_field =
        bulk_data_ptr->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name_);
    const stk::mesh::Field &radius_field =
        bulk_data_ptr->get_field<double>(stk::topology::ELEM_RANK, radius_field_name_);
    const stk::mesh::Field &obb_field = bulk_data_ptr->get_field<double>(stk::topology::ELEM_RANK, obb_field_name_);

    stk::mesh::Selector locally_owned_part = metaB.locally_owned_part() && part;
    stk::mesh::for_each_entity_run(*bulk_data_ptr, stk::topology::NODE_RANK, locally_owned_part,
                                   [&aabb_field, &radius_field, &buffer_distance_](const stk::mesh::BulkData &bulk_data,
                                                                                   stk::mesh::Entity element) {
                                     stk::mesh::Entity const *nodes = bulk_data.begin_nodes(element);
                                     double *coords = stk::mesh::field_data(node_coord_field, nodes[0]);
                                     double *radius = stk::mesh::field_data(radius_field, element);
                                     double *obb = stk::mesh::field_data(obb_field, element);

                                     obb[0] = coords[0] - radius[0] - buffer_distance_;
                                     obb[1] = coords[1] - radius[0] - buffer_distance_;
                                     obb[2] = coords[2] - radius[0] - buffer_distance_;
                                     obb[3] = coords[0] + radius[0] + buffer_distance_;
                                     obb[4] = coords[1] + radius[0] + buffer_distance_;
                                     obb[5] = coords[2] + radius[0] + buffer_distance_;
                                   });
  }

 private:
  const stk::util::ParameterList parameter_list_;
  const double buffer_distance_;
  const std::string obb_field_name_;
  const std::string radius_field_name_;
  const std::string node_coord_field_name_;
};  // ComputeOBBSphereVariant

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTEOBBSPHEREVARIANT_HPP_
