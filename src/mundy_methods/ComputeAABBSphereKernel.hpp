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

#ifndef MUNDY_METHODS_COMPUTEAABBSPHEREKERNEL_HPP_
#define MUNDY_METHODS_COMPUTEAABBSPHEREKERNEL_HPP_

/// \file ComputeAABBSphereKernel.hpp
/// \brief Declaration of the ComputeAABBSphereKernel class

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

/// \class ComputeAABBSphereKernel
/// \brief Concrete implementation of \c MetaKernel for computing the axis aligned boundary box of spheres.
class ComputeAABBSphereKernel : public MetaKernel<ComputeAABBSphereKernel>,
                                public MetaKernelRegistry<ComputeAABBSphereKernel, ComputeAABB> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit ComputeAABBSphereKernel(const stk::mesh::BulkData *bulk_data_ptr,
                                   const Teuchos::ParameterList &parameter_list) {
    // Store the input parameters, use default parameters for any parameter not given.
    // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
    parameter_list_ = parameter_list;
    parameter_list_.validateParametersAndSetDefaults(get_valid_params());

    // Fill the internal members using the internal parameter list
    node_coord_field_name_ = parameter_list_.get<std::string>("node_coord_field_name");
    radius_field_name_ = parameter_list_.get<std::string>("radius_field_name");
    aabb_field_name_ = parameter_list_.get<std::string>("aabb_field_name");
    buffer_distance_ = parameter_list_.get<std::string>("buffer_distance");

    // Store the input params.
    const stk::mesh::Field &node_coord_field_ptr_ =
        *bulk_data_ptr->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name_);
    const stk::mesh::Field &radius_field_ptr_ =
        *bulk_data_ptr->get_field<double>(stk::topology::ELEM_RANK, radius_field_name_);
    const stk::mesh::Field &aabb_field_ptr_ =
        *bulk_data_ptr->get_field<double>(stk::topology::ELEM_RANK, aabb_field_name_);
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
        std::make_unique<FieldParams<double>>(default_node_coord_field_name_, std::topology::NODE_RANK, 3, 1));
    required_part_params->add_field_params(
        std::make_unique<FieldParams<double>>(default_radius_field_name_, std::topology::ELEMENT_RANK, 1, 1));
    required_part_params->add_field_params(
        std::make_unique<FieldParams<double>>(default_aabb_field_name_, std::topology::ELEMENT_RANK, 4, 1));
    return required_part_params;
  }

  /// \brief Get the default parameters for this class.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c ParameterList
  /// will be created. You can save the result yourself if you wish to reuse it.
  static Teuchos::ParameterList details_get_valid_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set(
        "aabb_field_name", default_aabb_field_name_,
        'Name of the element field within which the output axis-aligned boundary boxes will be written.');
    default_parameter_list.set("radius_field_name", default_radius_field_name_,
                               "Name of the element field containing the sphere radius.");
    default_parameter_list.set("node_coordinate_field_name", default_node_coord_field_name_,
                               "Name of the node field containing the coordinate of the sphere's center.");
    default_parameter_list.set("buffer_distance", default_buffer_distance_,
                               "Buffer distance to be added to the axis-aligned boundary box.");
    return default_parameter_list;
  }
  //@}

  //! \name Actions
  //@{

  void execute(const stk::mesh::Entity &element) {
    stk::mesh::Entity const *nodes = bulk_data.begin_nodes(element);
    double *coords = stk::mesh::field_data(*node_coord_field_ptr_, nodes[0]);
    double *radius = stk::mesh::field_data(*radius_field_ptr_, element);
    double *aabb = stk::mesh::field_data(*aabb_field_ptr_, element);

    aabb[0] = coords[0] - radius[0] - buffer_distance_;
    aabb[1] = coords[1] - radius[0] - buffer_distance_;
    aabb[2] = coords[2] - radius[0] - buffer_distance_;
    aabb[3] = coords[0] + radius[0] + buffer_distance_;
    aabb[4] = coords[1] + radius[0] + buffer_distance_;
    aabb[5] = coords[2] + radius[0] + buffer_distance_;
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr double default_buffer_distance_ = 0.0;
  static constexpr std::string default_aabb_field_name_ = "AABB";
  static constexpr std::string default_radius_field_name_ = "RADIUS";
  static constexpr std::string default_node_coord_field_name_ = "NODE_COORD";
  //@}

  //! \name Internal members
  //@{

  /// \brief Current parameter list with valid entries.
  Teuchos::ParameterList parameter_list_;

  /// \brief Buffer distance to be added to the axis-aligned boundary box.
  ///
  /// For example, if the original axis-aligned boundary box has left corner at [0,0,0] and right corner at [1,1,1],
  /// then a buffer distance of 2 will shift the left corner to [-2,-2,-2] and right corner to [3,3,3].
  double buffer_distance_;

  /// \brief Name of the element field within which the output axis-aligned boundary boxes will be written.
  std::string aabb_field_name_;

  /// \brief Name of the element field containing the sphere radius.
  std::string radius_field_name_;

  /// \brief Name of the node field containing the coordinate of the sphere's center.
  std::string node_coord_field_name_;

  /// \brief Element field within which the output axis-aligned boundary boxes will be written.
  stk::mesh::Field *aabb_field_ptr_;

  /// \brief Element field containing the sphere radius.
  stk::mesh::Field *radius_field_ptr_;

  /// \brief Node field containing the coordinate of the sphere's center.
  stk::mesh::Field *node_coord_field_ptr_;
  //@}
};  // ComputeAABBSphereKernel

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTEAABBSPHEREKERNEL_HPP_
