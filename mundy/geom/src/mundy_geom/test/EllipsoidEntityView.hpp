// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
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

#ifndef MUNDY_GEOM_AGGREGATES_ELLIPSOIDENTITYVIEW_HPP_
#define MUNDY_GEOM_AGGREGATES_ELLIPSOIDENTITYVIEW_HPP_

// C++ core
#include <type_traits>  // for std::conditional_t, std::false_type, std::true_type

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// STK mesh
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh

// Mundy mesh
#include <mundy_geom/aggregates/EntityView.hpp>  // for mundy::geom::EntityView and mundy::geom::create_topological_entity_view
#include <mundy_mesh/BulkData.hpp>               // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data

namespace mundy {

namespace geom {

/// @brief A view of an STK entity meant to represent a ellipsoid
///
/// Use \ref create_ellipsoid_entity_view to build an EllipsoidEntityView object with automatic template deduction.
template <typename Base, typename EllipsoidDataType>
class EllipsoidEntityView : public Base {
  static_assert(EllipsoidDataType::topology_t == NODE || EllipsoidDataType::topology_t == PARTICLE,
                "The topology of the given ellipsoid aggregate must be NODE or PARTICLE.");

 public:
  using scalar_t = typename EllipsoidDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = EllipsoidDataType::topology_t;
  static constexpr stk::topology::rank_t rank_t = EllipsoidDataType::rank_t;

  EllipsoidEntityView(const Base& base, const EllipsoidDataType& data) : Base(base), data_(data) {
  }

  stk::mesh::Entity ellipsoid_entity() const {
    return Base::entity();
  }

  stk::mesh::Entity center_node_entity() const
    requires(topology_t == stk::topology::PARTICLE)
  {
    return Base::connected_node(0);
  }

  decltype(auto) center() const {
    if constexpr (topology_t == stk::topology::NODE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.center_tag())>) {
        return mundy::mesh::vector3_field_data(data_.center_data(), ellipsoid_entity());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.center_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.center_data()[i], ellipsoid_entity());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.center_tag())>) {
        return data_.center_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.center_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.center_data()[i], ellipsoid_entity());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else if constexpr (topology_t == stk::topology::PARTICLE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.center_tag())>) {
        return mundy::mesh::vector3_field_data(data_.center_data(), center_node_entity());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.center_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(center_node_entity()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.center_data()[i], center_node_entity());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.center_tag())>) {
        return data_.center_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.center_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(center_node_entity()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.center_data()[i], center_node_entity());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid topology. This should be unreachable.");
    }
  }

  decltype(auto) orientation() const {
    if constexpr (topology_t == stk::topology::NODE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.orientation_tag())>) {
        return mundy::mesh::quaternion_field_data(data_.orientation_data(), ellipsoid_entity());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.orientation_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::quaternion_field_data(data_.orientation_data()[i], ellipsoid_entity());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.orientation_tag())>) {
        return data_.orientation_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.orientation_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::quaternion_field_data(data_.orientation_data()[i], ellipsoid_entity());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else if constexpr (topology_t == stk::topology::PARTICLE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.orientation_tag())>) {
        return mundy::mesh::quaternion_field_data(data_.orientation_data(), ellipsoid_entity());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.orientation_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::quaternion_field_data(data_.orientation_data()[i], ellipsoid_entity());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.orientation_tag())>) {
        return data_.orientation_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.orientation_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::quaternion_field_data(data_.orientation_data()[i], ellipsoid_entity());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid topology. This should be unreachable.");
    }
  }

  decltype(auto) radii() const {
    if constexpr (topology_t == stk::topology::NODE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.radii_tag())>) {
        return mundy::mesh::vector3_field_data(data_.radii_data(), ellipsoid_entity());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.radii_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.radii_data()[i], ellipsoid_entity());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.radii_tag())>) {
        return data_.radii_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.radii_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.radii_data()[i], ellipsoid_entity());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else if constexpr (topology_t == stk::topology::PARTICLE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.radii_tag())>) {
        return mundy::mesh::vector3_field_data(data_.radii_data(), ellipsoid_entity());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.radii_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.radii_data()[i], ellipsoid_entity());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.radii_tag())>) {
        return data_.radii_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.radii_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.radii_data()[i], ellipsoid_entity());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid topology. This should be unreachable.");
    }
  }

  decltype(auto) center() {
    if constexpr (topology_t == stk::topology::NODE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.center_tag())>) {
        return mundy::mesh::vector3_field_data(data_.center_data(), ellipsoid_entity());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.center_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.center_data()[i], ellipsoid_entity());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.center_tag())>) {
        return data_.center_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.center_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.center_data()[i], ellipsoid_entity());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else if constexpr (topology_t == stk::topology::PARTICLE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.center_tag())>) {
        return mundy::mesh::vector3_field_data(data_.center_data(), center_node_entity());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.center_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(center_node_entity()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.center_data()[i], center_node_entity());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.center_tag())>) {
        return data_.center_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.center_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(center_node_entity()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.center_data()[i], center_node_entity());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid topology. This should be unreachable.");
    }
  }

  decltype(auto) orientation() {
    if constexpr (topology_t == stk::topology::NODE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.orientation_tag())>) {
        return mundy::mesh::quaternion_field_data(data_.orientation_data(), ellipsoid_entity());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.orientation_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::quaternion_field_data(data_.orientation_data()[i], ellipsoid_entity());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.orientation_tag())>) {
        return data_.orientation_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.orientation_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::quaternion_field_data(data_.orientation_data()[i], ellipsoid_entity());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else if constexpr (topology_t == stk::topology::PARTICLE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.orientation_tag())>) {
        return mundy::mesh::quaternion_field_data(data_.orientation_data(), ellipsoid_entity());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.orientation_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::quaternion_field_data(data_.orientation_data()[i], ellipsoid_entity());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.orientation_tag())>) {
        return data_.orientation_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.orientation_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::quaternion_field_data(data_.orientation_data()[i], ellipsoid_entity());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid topology. This should be unreachable.");
    }
  }

  decltype(auto) radii() {
    if constexpr (topology_t == stk::topology::NODE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.radii_tag())>) {
        return mundy::mesh::vector3_field_data(data_.radii_data(), ellipsoid_entity());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.radii_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.radii_data()[i], ellipsoid_entity());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.radii_tag())>) {
        return data_.radii_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.radii_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.radii_data()[i], ellipsoid_entity());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else if constexpr (topology_t == stk::topology::PARTICLE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.radii_tag())>) {
        return mundy::mesh::vector3_field_data(data_.radii_data(), ellipsoid_entity());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.radii_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.radii_data()[i], ellipsoid_entity());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.radii_tag())>) {
        return data_.radii_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.radii_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_entity()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.radii_data()[i], ellipsoid_entity());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid topology. This should be unreachable.");
    }
  }

  /// \brief Chainable function to add augments to this entity view
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  auto augment_view(Args&&... args) const {
    return NextAugment<EllipsoidEntityView, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  const EllipsoidDataType& data_;
};  // EllipsoidEntityView

/// @brief An ngp-compatible view of an ellipsoid entity
template <typename Base, typename EllipsoidDataType>
class NgpEllipsoidEntityView : public Base {
  static_assert(EllipsoidDataType::topology_t == NODE || EllipsoidDataType::topology_t == PARTICLE,
                "The topology of the given ellipsoid aggregate must be NODE or PARTICLE.");

 public:
  using scalar_t = typename EllipsoidDataType::scalar_t;
  static constexpr stk::topology::topology_t topology_t = EllipsoidDataType::topology_t;
  static constexpr stk::topology::rank_t rank_t = EllipsoidDataType::rank_t;

  KOKKOS_INLINE_FUNCTION
  NgpEllipsoidEntityView(const Base& base, const EllipsoidDataType& data) : Base(base), data_(data) {
  }

  unsigned ellipsoid_entity_index() const {
    return Base::entity_index();
  }

  unsigned center_node_entity_index() const
    requires(topology_t == stk::topology::PARTICLE)
  {
    return Base::connected_node_index(0);
  }

  decltype(auto) center() const {
    if constexpr (topology_t == stk::topology::NODE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.center_tag())>) {
        return mundy::mesh::vector3_field_data(data_.center_data(), ellipsoid_index());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.center_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.center_data()[i], ellipsoid_index());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.center_tag())>) {
        return data_.center_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.center_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.center_data()[i], ellipsoid_index());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else if constexpr (topology_t == stk::topology::PARTICLE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.center_tag())>) {
        return mundy::mesh::vector3_field_data(data_.center_data(), center_node_index());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.center_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(center_node_index()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.center_data()[i], center_node_index());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.center_tag())>) {
        return data_.center_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.center_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(center_node_index()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.center_data()[i], center_node_index());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid topology. This should be unreachable.");
    }
  }

  decltype(auto) orientation() const {
    if constexpr (topology_t == stk::topology::NODE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.orientation_tag())>) {
        return mundy::mesh::quaternion_field_data(data_.orientation_data(), ellipsoid_index());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.orientation_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::quaternion_field_data(data_.orientation_data()[i], ellipsoid_index());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.orientation_tag())>) {
        return data_.orientation_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.orientation_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::quaternion_field_data(data_.orientation_data()[i], ellipsoid_index());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else if constexpr (topology_t == stk::topology::PARTICLE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.orientation_tag())>) {
        return mundy::mesh::quaternion_field_data(data_.orientation_data(), ellipsoid_index());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.orientation_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::quaternion_field_data(data_.orientation_data()[i], ellipsoid_index());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.orientation_tag())>) {
        return data_.orientation_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.orientation_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::quaternion_field_data(data_.orientation_data()[i], ellipsoid_index());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid topology. This should be unreachable.");
    }
  }

  decltype(auto) radii() const {
    if constexpr (topology_t == stk::topology::NODE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.radii_tag())>) {
        return mundy::mesh::vector3_field_data(data_.radii_data(), ellipsoid_index());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.radii_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.radii_data()[i], ellipsoid_index());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.radii_tag())>) {
        return data_.radii_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.radii_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.radii_data()[i], ellipsoid_index());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else if constexpr (topology_t == stk::topology::PARTICLE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.radii_tag())>) {
        return mundy::mesh::vector3_field_data(data_.radii_data(), ellipsoid_index());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.radii_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.radii_data()[i], ellipsoid_index());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.radii_tag())>) {
        return data_.radii_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.radii_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.radii_data()[i], ellipsoid_index());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid topology. This should be unreachable.");
    }
  }

  decltype(auto) center() {
    if constexpr (topology_t == stk::topology::NODE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.center_tag())>) {
        return mundy::mesh::vector3_field_data(data_.center_data(), ellipsoid_index());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.center_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.center_data()[i], ellipsoid_index());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.center_tag())>) {
        return data_.center_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.center_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.center_data()[i], ellipsoid_index());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else if constexpr (topology_t == stk::topology::PARTICLE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.center_tag())>) {
        return mundy::mesh::vector3_field_data(data_.center_data(), center_node_index());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.center_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(center_node_index()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.center_data()[i], center_node_index());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.center_tag())>) {
        return data_.center_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.center_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(center_node_index()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.center_data()[i], center_node_index());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid topology. This should be unreachable.");
    }
  }

  decltype(auto) orientation() {
    if constexpr (topology_t == stk::topology::NODE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.orientation_tag())>) {
        return mundy::mesh::quaternion_field_data(data_.orientation_data(), ellipsoid_index());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.orientation_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::quaternion_field_data(data_.orientation_data()[i], ellipsoid_index());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.orientation_tag())>) {
        return data_.orientation_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.orientation_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::quaternion_field_data(data_.orientation_data()[i], ellipsoid_index());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else if constexpr (topology_t == stk::topology::PARTICLE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.orientation_tag())>) {
        return mundy::mesh::quaternion_field_data(data_.orientation_data(), ellipsoid_index());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.orientation_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::quaternion_field_data(data_.orientation_data()[i], ellipsoid_index());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.orientation_tag())>) {
        return data_.orientation_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.orientation_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::quaternion_field_data(data_.orientation_data()[i], ellipsoid_index());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid topology. This should be unreachable.");
    }
  }

  decltype(auto) radii() {
    if constexpr (topology_t == stk::topology::NODE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.radii_tag())>) {
        return mundy::mesh::vector3_field_data(data_.radii_data(), ellipsoid_index());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.radii_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.radii_data()[i], ellipsoid_index());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.radii_tag())>) {
        return data_.radii_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.radii_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.radii_data()[i], ellipsoid_index());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else if constexpr (topology_t == stk::topology::PARTICLE) {
      if constexpr (std::is_base_of_v<tag_type::FIELD, decltype(data_.radii_tag())>) {
        return mundy::mesh::vector3_field_data(data_.radii_data(), ellipsoid_index());
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_FIELDS, decltype(data_.radii_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.radii_data()[i], ellipsoid_index());
          }
        }
      } else if constexpr (std::is_base_of_v<tag_type::SHARED, decltype(data_.radii_tag())>) {
        return data_.radii_data()();
      } else if constexpr (std::is_base_of_v<tag_type::VECTOR_OF_SHARED, decltype(data_.radii_tag())>) {
        const auto part_ptrs = data_.parts();
        unsigned num_parts = part_ptrs.size();
        for (unsigned i = 0; i < num_parts; ++i) {
          if (data_.bulk_data().bucket(ellipsoid_index()).member(*part_ptr[i])) {
            return mundy::mesh::vector3_field_data(data_.radii_data()[i], ellipsoid_index());
          }
        }
      } else {
        MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid tag type. This should be unreachable.");
      }

    } else {
      MUNDY_THROW_ASSERT(false, std::invalid_argument, "Invalid topology. This should be unreachable.");
    }
  }

  /// \brief Chainable function to add augments to this entity view
  template <template <typename, typename...> class NextAugment, typename... AugmentTemplates, typename... Args>
  KOKKOS_INLINE_FUNCTION auto augment_view(Args&&... args) const {
    return NextAugment<NgpEllipsoidEntityView, AugmentTemplates...>(*this, std::forward<Args>(args)...);
  }

 private:
  const EllipsoidDataType& data_;
};  // NgpEllipsoidEntityView

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_AGGREGATES_ELLIPSOIDENTITYVIEW_HPP_