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

/// \file GridOfSpheres.cpp
/// \brief Definition of DeclareAndInitializeShapes's GridOfSpheres technique.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// External libs
#include <openrand/philox.h>

// Trilinos libs
#include <Teuchos_ParameterList.hpp>      // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>        // for stk::mesh::Field, stl::mesh::field_data
#include <stk_mesh/base/GetEntities.hpp>  // for stk::mesh::count_entities
#include <stk_search/BoundingBox.hpp>     // for stk::search::Box
#include <stk_search/CoarseSearch.hpp>    // for stk::search::coarse_search
#include <stk_search/SearchMethod.hpp>    // for stk::search::KDTREE

// Mundy libs
#include <mundy_core/throw_assert.hpp>                                                     // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                                                         // for mundy::mesh::BulkData
#include <mundy_shape/declare_and_initialize_shapes/techniques/GridCoordinateMapping.hpp>  // for mundy::shape::...::GridCoordinateMapping
#include <mundy_shape/declare_and_initialize_shapes/techniques/GridOfSpheres.hpp>  // for mundy::shape::...::GridOfSpheres
#include <mundy_shape/shapes/Spheres.hpp>                                          // for mundy::shape::shapes::Spheres

namespace mundy {

namespace shape {

namespace declare_and_initialize_shapes {

namespace techniques {

namespace {

// \name Helper functions for shuffling code
//{

/// \brief Shuffle the input index to a new index in the range [0, max_size).
/// \details The shuffle function is a one-to-one mapping from the input index to the output index.
/// Tested for max_size up to 10,000,000. This is not a perfect shuffle, but it is good enough for our purposes.
///
/// \param input The input index to be shuffled.
/// \param max_size The maximum size of the index space.
size_t shuffle(size_t input, size_t max_size) {
  return input * 833 % max_size;
}

//}

// \name Helper functions for Morton code
//{

uint32_t part1by2(uint32_t n) {
  n &= 0x000003ff;                   // Keep only the first 10 bits
  n = (n ^ (n << 16)) & 0xff0000ff;  // Move bits, spreading them apart
  n = (n ^ (n << 8)) & 0x0300f00f;   // Move bits further apart
  n = (n ^ (n << 4)) & 0x030c30c3;   // Move bits even further apart
  n = (n ^ (n << 2)) & 0x09249249;   // Final spread of bits
  return n;
}

uint64_t interleave(uint32_t i, uint32_t j, uint32_t k) {
  return static_cast<uint64_t>(part1by2(i)) | (static_cast<uint64_t>(part1by2(j)) << 1) |
         (static_cast<uint64_t>(part1by2(k)) << 2);
}

/// \brief Get the Morton code for the given i, j, k.
uint64_t get_morton_code(uint32_t i, uint32_t j, uint32_t k) {
  return interleave(i, j, k);
}
//}

}  // namespace

// \name Constructors and destructor
//{

GridOfSpheres::GridOfSpheres(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument,
                     "GridOfSpheres: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(GridOfSpheres::get_valid_fixed_params());

  // Get the field pointers.
  const std::string node_coord_field_name = mundy::shape::shapes::Spheres::get_node_coord_field_name();
  const std::string element_radius_field_name = mundy::shape::shapes::Spheres::get_element_radius_field_name();

  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
  element_radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name);

  MUNDY_THROW_ASSERT(node_coord_field_ptr_ != nullptr, std::invalid_argument,
                     "GridOfSpheres: node_coord_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_ASSERT(element_radius_field_ptr_ != nullptr, std::invalid_argument,
                     "GridOfSpheres: radius_field_ptr cannot be a nullptr. Check that the field exists.");

  // Get the part pointers.
  const Teuchos::Array<std::string> sphere_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("sphere_part_names");
  auto parts_from_names = [](mundy::mesh::MetaData &meta_data,
                             const Teuchos::Array<std::string> &part_names) -> std::vector<stk::mesh::Part *> {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      std::cout << "part_name: " << part_name << std::endl;
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_ASSERT(part != nullptr, std::invalid_argument,
                         "GridOfSpheres: Expected a part with name '" << part_name << "' but part does not exist.");
      parts.push_back(part);
    }
    return parts;
  };

  sphere_part_ptrs_ = parts_from_names(*meta_data_ptr_, sphere_part_names);
}
//}

// \name Setters
//{

void GridOfSpheres::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(GridOfSpheres::get_valid_mutable_params());

  num_spheres_x_ = valid_mutable_params.get<size_t>("num_spheres_x");
  num_spheres_y_ = valid_mutable_params.get<size_t>("num_spheres_y");
  num_spheres_z_ = valid_mutable_params.get<size_t>("num_spheres_z");
  coordinate_map_ptr_ = valid_mutable_params.get<std::shared_ptr<GridCoordinateMapping>>("coordinate_map");
  sphere_radius_lower_bound_ = valid_mutable_params.get<double>("sphere_radius_lower_bound");
  sphere_radius_upper_bound_ = valid_mutable_params.get<double>("sphere_radius_upper_bound");
  zmorton_ = valid_mutable_params.get<bool>("zmorton");
  shuffle_ = valid_mutable_params.get<bool>("shuffle");
}
//}

// \name Getters
//{

stk::mesh::EntityId GridOfSpheres::node_id(size_t i, size_t j, size_t k) const {
  // Note, the current morton code requires that i, j, and k are unsigned int or uint32_t.
  // This limits us to ints between 0 and 4,294,967,295.
  const size_t morton_code =
      get_morton_code(static_cast<uint32_t>(i), static_cast<uint32_t>(j), static_cast<uint32_t>(k));
  const size_t linearized_id =
      i * num_spheres_y_ * num_spheres_z_ + j * num_spheres_z_ + k;  // i slowest, j next, k fastest
  const size_t unshuffled_id = zmorton_ ? morton_code : linearized_id;
  return shuffle_ ? shuffle(unshuffled_id, num_spheres_x_ * num_spheres_y_ * num_spheres_z_) : unshuffled_id;
}

stk::mesh::EntityId GridOfSpheres::element_id(size_t i, size_t j, size_t k) const {
  // The entity id of the spheres is the same as that of their connected nodes.
  return node_id(i, j, k);
}

stk::mesh::Entity GridOfSpheres::node(size_t i, size_t j, size_t k) const {
  return bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, node_id(i, j, k));
}

stk::mesh::Entity GridOfSpheres::element(size_t i, size_t j, size_t k) const {
  return bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK, element_id(i, j, k));
}
//}

// \name Actions
//{

void GridOfSpheres::execute() {
  openrand::Philox rng(1, 0);
  for (size_t i = 0; i < num_spheres_x_; ++i) {
    for (size_t j = 0; j < num_spheres_y_; ++j) {
      for (size_t k = 0; k < num_spheres_z_; ++k) {
        // Create the sphere.
        stk::mesh::EntityId our_sphere_id = element_id(i, j, k);
        stk::mesh::Entity sphere = bulk_data_ptr_->declare_element(our_sphere_id);
        bulk_data_ptr_->change_entity_parts(sphere, sphere_part_ptrs_);

        // Create the node and connect it to the sphere.
        stk::mesh::EntityId our_node_id = node_id(i, j, k);
        stk::mesh::Entity node = bulk_data_ptr_->declare_node(our_node_id);
        bulk_data_ptr_->declare_relation(sphere, node, 0);

        // Set the node's coordinates using the given coordinate map.
        double *const node_coords = stk::mesh::field_data(*node_coord_field_ptr_, node);
        const auto [coord_x, coord_y, coord_z] = coordinate_map_ptr_->get_grid_coordinate({i, j, k});
        node_coords[0] = coord_x;
        node_coords[1] = coord_y;
        node_coords[2] = coord_z;

        // Set the sphere's radius.
        double *const sphere_radius = stk::mesh::field_data(*element_radius_field_ptr_, sphere);
        sphere_radius[0] =
            rng.rand<double>() * (sphere_radius_upper_bound_ - sphere_radius_lower_bound_) + sphere_radius_lower_bound_;
      }
    }
  }
}
//}

}  // namespace techniques

}  // namespace declare_and_initialize_shapes

}  // namespace shape

}  // namespace mundy