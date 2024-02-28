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

#ifndef MUNDY_META_UTILS_MESHGENERATION_HPP_
#define MUNDY_META_UTILS_MESHGENERATION_HPP_

/// \file MeshGeneration.hpp
/// \brief A set of helper methods for generating meshes for unit tests.

// C++ core libs
#include <memory>   // for std::shared_ptr, std::unique_ptr
#include <utility>  // for std::pair, std::make_pair

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_core/throw_assert.hpp>                          // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                              // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                              // for mundy::mesh::MetaData
#include <mundy_meta/HasMeshRequirementsAndIsRegisterable.hpp>  // for mundy::meta::HasMeshRequirementsAndIsRegisterable
#include <mundy_meta/MeshRequirements.hpp>                      // for mundy::meta::MeshRequirements

namespace mundy {

namespace meta {

namespace utils {

/// \brief Helper function for generating a mesh that satisfies the requirements of a given meta class and
template <typename MetaClass>
std::pair<std::shared_ptr<typename MetaClass::PolymorphicBaseType>, std::shared_ptr<mundy::mesh::BulkData>>
generate_class_instance_and_mesh_from_meta_class_requirements(
    const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList()) {
  // Ensure that the given type has the correct static interface.
  using Checker = HasMeshRequirementsAndIsRegisterable<MetaClass>;
  static_assert(Checker::value, "The given type does not have the correct static interface for encoding requirements.");

  // Validate the fixed parameters and set defaults.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(MetaClass::get_valid_fixed_params());

  // Create a mesh that meets the requirements for MetaClass.
  auto mesh_reqs_ptr = std::make_shared<mundy::meta::MeshRequirements>(MPI_COMM_WORLD);
  mesh_reqs_ptr->set_spatial_dimension(3);
  mesh_reqs_ptr->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  mesh_reqs_ptr->merge(MetaClass::get_mesh_requirements(valid_fixed_params));
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_reqs_ptr->declare_mesh();
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();

  // At this point, we solidify the mesh structure by calling commit.
  meta_data_ptr->commit();

  // Create a new instance of MetaClass with the valid fixed params.
  // Note, we do not specify the mutable params, so the default values will be used.
  auto class_ptr = MetaClass::create_new_instance(bulk_data_ptr.get(), valid_fixed_params);

  return std::make_pair(class_ptr, bulk_data_ptr);
}

}  // namespace utils

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_UTILS_MESHGENERATION_HPP_
