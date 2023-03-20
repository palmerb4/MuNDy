// @HEADER
// **********************************************************************************************************************
//
//                                          MuNDy: Multi-body Nonlocal Dynamics
//                                           Copyright 2023 Flatiron Institute
//                                                 Author: Bryce Palmer
//
// MuNDy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
//
// MuNDy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with MuNDy. If not, see
// <https://www.gnu.org/licenses/>.
//
// **********************************************************************************************************************
// @HEADER

#ifndef MUNDY_CORE_GROUPOFPARTICLES_HPP_
#define MUNDY_CORE_GROUPOFPARTICLES_HPP_

/// \file GroupOfParticles.hpp
/// \brief Declaration of the GroupOfParticles class

// clang-format off
#include <gtest/gtest.h>                             // for AssertHelper, etc
#include <mpi.h>                                     // for MPI_COMM_WORLD, etc
#include <stddef.h>                                  // for size_t
#include <vector>                                    // for vector, etc
#include <random>                                    // for rand
#include <memory>                                    // for shared_ptr
#include <string>                                    // for string
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

namespace particle {

/// \class GroupOfParticles
/// \brief A collection of particles, their sub-groups, and their associated fields.
///
/// \tparam ParticleTopology Topology assigned to each particle.
/// \tparam Scalar Numeric type for all default floating point fields. Defaults to <tt>double</tt>.
///
/// This class <does something>
template <stk::topology ParticleTopology, typename Scalar = double>
class GroupOfParticles : public GroupOfEntities<ParticleTopology, Scalar> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Constructor with given <tt>BulkData</tt>.
  ///
  /// Call this constructor if you have a larger <tt>BulkData</tt> containing multiple groups. The entities within this
  /// group and their associated fields will be stored within the provided <tt>BulkData</tt>. In that case, a single
  /// <tt>BulkData</tt> can be shared between each <tt>GroupOfParticles</tt>, thereby allowing a <tt>Field</tt> or
  /// <tt>Part</tt> to span multiple groups.
  ///
  /// \param bulk_data_ptr [in] Shared pointer to a larger <tt>BulkData</tt> with (potentially) multiple groups. A copy
  /// of this pointer is stored in this class until destruction.
  /// \param group_name [in] Name for the group. If the name already exists, the two groups will be merged.
  GroupOfParticles(const std::shared_ptr<stk::mesh::BulkData> &bulk_data_ptr, const std::string &group_name);
  //@}

  //@}
  //! @name Attributes
  //@{

  /// \brief Return a reference to the node coordinate field.
  FlagFieldType &get_node_coord_field();

  /// \brief Return a reference to the node orientation field.
  FlagFieldType &get_node_orientation_field();

  /// \brief Return a reference to the node force field.
  FlagFieldType &get_node_force_field();

  /// \brief Return a reference to the node torque field.
  FlagFieldType &get_node_torque_field();

  /// \brief Return a reference to the node translational velocity field.
  FlagFieldType &get_node_translational_velocity_field();

  /// \brief Return a reference to the node rotational velocity field.
  FlagFieldType &get_node_rotational_velocity_field();

  //@}

 protected:
  //! @name Default fields assigned to all group members
  //@{

  /// @brief Field containing nood spatial coordinates in the form [coord_x, coord_y, coord_z].
  FloatingPointFieldType_ &node_coord_field_;

  /// @brief Field containing nood orientation, as a quaternion in the form [quat_w, quat_x, quat_y, quat_z], where w is
  /// the scalar component and x, y, z are the vector components of the quaternion.
  FloatingPointFieldType_ &node_orientation_field_;

  /// @brief Field containing nood force in the form [force_x, force_y, force_z].
  FloatingPointFieldType_ &node_force_field_;

  /// @brief Field containing nood spatial coordinates in the form [torque_x, torque_y, torque_z].
  FloatingPointFieldType_ &node_torque_field_;

  /// @brief Field containing nood spatial coordinates in the form [trans_vel_x, trans_vel_y, trans_vel_z].
  FloatingPointFieldType_ &node_translational_velocity_field_;

  /// @brief Field containing nood spatial coordinates in the form [rot_vel_x, rot_vel_y, rot_vel_z].
  FloatingPointFieldType_ &node_rotational_velocity_field_;
  //@}

 private:
  //! \name Typedefs
  //@{

  /// \brief This groups' floating point field type.
  typedef std::mesh::Field<Scalar> FloatingPointFieldType_;
  //@}
}

}  // namespace core

}  // namespace mundy

#endif  // MUNDY_CORE_GROUPOFPARTICLES_HPP_
