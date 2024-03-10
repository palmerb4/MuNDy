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

/// \file SphereSphereLinker.cpp
/// \brief Definition of the ComputeSignedSeparationDistanceAndContactNormal's SphereSphereLinker kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data

// Mundy libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_linker/compute_signed_separation_distance_and_contact_normal/kernels/SphereSphereLinker.hpp>  // for mundy::linker::...::kernels::SphereSphereLinker
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_shape/shapes/Spheres.hpp>  // for mundy::shape::shapes::Spheres

namespace mundy {

namespace linker {

namespace compute_signed_separation_distance_and_contact_normal {

namespace kernels {

// \name Constructors and destructor
//{

SphereSphereLinker::SphereSphereLinker(mundy::mesh::BulkData *const bulk_data_ptr,
                                       const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument,
                     "SphereSphereLinker: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(SphereSphereLinker::get_valid_fixed_params());

  // Get the field pointers.
  const std::string node_coord_field_name = mundy::shape::shapes::Spheres::get_node_coord_field_name();
  const std::string element_radius_field_name = mundy::shape::shapes::Spheres::get_element_radius_field_name();
  const std::string linker_contact_normal_field_name =
      valid_fixed_params.get<std::string>("linker_contact_normal_field_name");
  const std::string linker_signed_separation_distance_field_name =
      valid_fixed_params.get<std::string>("linker_signed_separation_distance_field_name");

  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
  element_radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name);
  linker_contact_normal_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_contact_normal_field_name);
  linker_signed_separation_distance_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_signed_separation_distance_field_name);

  MUNDY_THROW_ASSERT(node_coord_field_ptr_ != nullptr, std::invalid_argument,
                     "SphereSphereLinker: node_coord_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_ASSERT(element_radius_field_ptr_ != nullptr, std::invalid_argument,
                     "SphereSphereLinker: element_radius_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_ASSERT(
      linker_contact_normal_field_ptr_ != nullptr, std::invalid_argument,
      "SphereSphereLinker: linker_contact_normal_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_ASSERT(linker_signed_separation_distance_field_ptr_ != nullptr, std::invalid_argument,
                     "SphereSphereLinker: linker_signed_separation_distance_field_ptr cannot be a nullptr. Check that "
                     "the field exists.");

  // Get the part pointers.
  Teuchos::Array<std::string> valid_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
  Teuchos::Array<std::string> valid_sphere_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_sphere_part_names");

  auto parts_from_names = [](mundy::mesh::MetaData &meta_data, const Teuchos::Array<std::string> &part_names) {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_ASSERT(
          part != nullptr, std::invalid_argument,
          "SphereSphereLinker: Part " << part_name << " cannot be a nullptr. Check that the part exists.");
      parts.push_back(part);
    }
    return parts;
  };

  valid_entity_parts_ = parts_from_names(*meta_data_ptr_, valid_entity_part_names);
  valid_sphere_parts_ = parts_from_names(*meta_data_ptr_, valid_sphere_part_names);
}
//}

// \name MetaKernel interface implementation
//{

std::vector<stk::mesh::Part *> SphereSphereLinker::get_valid_entity_parts() const {
  return valid_entity_parts_;
}

stk::topology::rank_t SphereSphereLinker::get_entity_rank() const {
  return stk::topology::CONSTRAINT_RANK;
}

void SphereSphereLinker::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  // We don't have any valid mutable params, so this seems pointless but it's useful in that it will throw if the user
  // gives us parameters. This is useful for catching user errors.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(SphereSphereLinker::get_valid_mutable_params());
}
//}

// \name Actions
//{

void SphereSphereLinker::setup() {
}

void SphereSphereLinker::execute(const stk::mesh::Entity &sphere_sphere_linker) {
  // Use references to avoid copying entities
  const stk::mesh::Entity &left_sphere_element = bulk_data_ptr_->begin_elements(sphere_sphere_linker)[0];
  const stk::mesh::Entity &right_sphere_element = bulk_data_ptr_->begin_elements(sphere_sphere_linker)[1];
  const stk::mesh::Entity &left_sphere_node = bulk_data_ptr_->begin_nodes(left_sphere_element)[0];
  const stk::mesh::Entity &right_sphere_node = bulk_data_ptr_->begin_nodes(right_sphere_element)[0];

  const double *left_coords = stk::mesh::field_data(*node_coord_field_ptr_, left_sphere_node);
  const double *right_coords = stk::mesh::field_data(*node_coord_field_ptr_, right_sphere_node);
  const double *left_radius = stk::mesh::field_data(*element_radius_field_ptr_, left_sphere_element);
  const double *right_radius = stk::mesh::field_data(*element_radius_field_ptr_, right_sphere_element);

  double *contact_normal = stk::mesh::field_data(*linker_contact_normal_field_ptr_, sphere_sphere_linker);
  double *signed_separation_distance =
      stk::mesh::field_data(*linker_signed_separation_distance_field_ptr_, sphere_sphere_linker);

  // Compute the separation distance and contact normal
  const double dx = right_coords[0] - left_coords[0];
  const double dy = right_coords[1] - left_coords[1];
  const double dz = right_coords[2] - left_coords[2];
  const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
  const double radius_sum = *left_radius + *right_radius;
  const double separation_distance = distance - radius_sum;
  const double inv_distance = 1.0 / distance;

  // Set the separation distance and contact normal
  // Notice that the contact normal points from the left sphere to the right sphere
  // It is the normal to the left sphere and negative the normal of the right sphere.
  signed_separation_distance[0] = separation_distance;
  contact_normal[0] = dx * inv_distance;
  contact_normal[1] = dy * inv_distance;
  contact_normal[2] = dz * inv_distance;
}

void SphereSphereLinker::finalize() {
}
//}

}  // namespace kernels

}  // namespace compute_signed_separation_distance_and_contact_normal

}  // namespace linker

}  // namespace mundy
