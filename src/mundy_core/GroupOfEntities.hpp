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

#ifndef MUNDY_CORE_GROUPOFENTITIES_HPP_
#define MUNDY_CORE_GROUPOFENTITIES_HPP_

/// \file GroupOfEntities.hpp
/// \brief Declaration of the GroupOfEntities class

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

namespace core {

/// \class GroupOfEntities
/// \brief A collection of entities, their sub-groups, and their associated fields.
///
/// \tparam GroupTopology Topology assigned to each group member.
/// \tparam Scalar Numeric type for all default floating point fields. Defaults to <tt>double</tt>.
///
/// This class <does something>
///
/// \section A Header Title
///
/// Some userful information about this class.
///
/// \section Another Header Title
///
/// \subsection A Sub-Header Title
///
/// Some detailed informatioon
///
/// \subsection Another Sub-Header Title
///
/// More information. Maybe an example.
template <stk::topology GroupTopology, typename Scalar = double>
class GroupOfEntities {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Constructor with given <tt>BulkData</tt>.
  ///
  /// Call this constructor if you have a larger <tt>BulkData</tt> containing multiple groups. The entities within this
  /// group and their associated fields will be stored within the provided <tt>BulkData</tt>. In that case, a single
  /// <tt>BulkData</tt> can be shared between each <tt>GroupOfEntities</tt>, thereby allowing a <tt>Field</tt> or
  /// <tt>Part</tt> to span multiple groups.
  ///
  /// \param bulk_data_ptr [in] Shared pointer to a larger <tt>BulkData</tt> with (potentially) multiple groups. A copy
  /// of this pointer is stored in this class until destruction.
  /// \param group_name [in] Name for the group. If the name already exists, the two groups will be merged.
  GroupOfEntities(const std::shared_ptr<stk::mesh::BulkData> &bulk_data_ptr, const std::string &group_name);
  //@}

  //@}
  //! @name Attributes
  //@{

  /// \brief Return a reference to the group's part
  stk::mesh::Part &get_group_part();

  /// \brief Return a reference to the new entity flag field
  FlagFieldType &get_new_entity_flag_field();
  //@}

  //! @name Pre-commit setup routines
  //@{

  /// \brief Declare another <tt>GroupOfEntities</tt> as a subset of this group.
  ///
  /// By declaring <tt>subgroup</tt> as a subset of this group, all entities within <tt>subgroup</tt> will inherit this
  /// group's part and associated fields. As a result, any method that acts on this group, will also act on
  /// <tt>subgroup</tt>'s entities. To do so, <tt>subgroup</tt> must share the same topoology and scalar type as this
  /// group.
  ///
  /// For example, AnimalGroup may have LionGroup and TigerGroup as subsets, allowing all lions and tigers to be acted
  /// upon by looping over each animal in AnimalGroup.
  ///
  /// \param subgroup [in] The group to be added as a subset.
  template <stk::topology SubGroupTopology, typename SubGroupScalar>
  void declare_subgroup(const GroupOfEntities<SubGroupTopology, SubGroupScalar> &subgroup);

  /// \brief Declare a field that all entities of this group should have access to.
  ///
  /// \param field [in] The field to be added.
  /// \param field_dimension [in] The dimensionality of the field.
  /// \param init_value [in] The initial value of the field.
  template <class field_type>
  field_type &put_field_on_entire_group(const field_type &field, const unsigned int field_dimension,
                                        const typename stk::mesh::FieldTraits<field_type>::data_type *init_value);
  //@}

  //! @name Post-commit modification routines
  //@{

  /// \brief Generates new entities within this group. Optionally generate and attach nodes to these elements.
  ///
  /// The newly generated entities will have their new entity flag set to true. (Suggested) Use the
  /// get_new_entities_selector to access and fill the fields of new entitiers.
  ///
  /// \note These new entities will be accessible via this group and any of its parent groups.
  /// They will <it>not</it> be accessible by any sub-groups.
  ///
  /// \param num_new_entities [in] The number of new entities to generate.
  /// \param generate_and_attach_nodes [in] Flag specifying if the nodes of each entity should also be generated and
  /// attached. Defaults to true.
  ///
  /// \return A selector for all newly generted entities within this group.
  void generate_new_entities_in_group(const size_t num_new_entities, const bool generate_and_attach_nodes = true);
  //@}

 protected:
  //! \name Mesh information
  //@{

  /// \brief This groups' shared bulk data, which contains all entities and their fields.
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr_;

  /// \brief Default part assigned to all group members.
  stk::mesh::Part &group_part_;
  //@}

  //! @name Default fields assigned to all group members
  //@{

  /// @brief Field specifying if the entity is newly created or not.
  FlagFieldType_ &new_entity_flag_field_;
  //@}

 private:
  //! \name Typedefs
  //@{

  /// \brief This groups' flag field type
  typedef std::mesh::Field<bool> FlagFieldType_;
  //@}
}

}  // namespace core

}  // namespace mundy

//! \name template implementations
//@{
#include <GroupOfEntities.tpp>

//}
#endif  // MUNDY_CORE_GROUPOFENTITIES_HPP_
