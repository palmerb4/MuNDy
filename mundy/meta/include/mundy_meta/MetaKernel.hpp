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

#ifndef MUNDY_META_METAKERNEL_HPP_
#define MUNDY_META_METAKERNEL_HPP_

/// \file MetaKernel.hpp
/// \brief Declaration of the MetaKernel class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <tuple>        // for std::tuple
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part

// Mundy libs
#include <mundy_mesh/BulkData.hpp>          // for mundy::mesh::BulkData
#include <mundy_meta/PartRequirements.hpp>  // for mundy::meta::PartRequirements
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface

namespace mundy {

namespace meta {

// void execute(const stk::mesh::Selector &input_selector) {
//   setup();

//   auto valid_entity_parts = get_valid_entity_parts();
//   auto kernel_entity_rank = get_entity_rank();
//   stk::mesh::Selector locally_owned_intersection_with_valid_entity_parts =
//       stk::mesh::selectIntersection(valid_entity_parts) & meta_data_ptr_->locally_owned_part() & input_selector;
//   stk::mesh::for_each_entity_run(
//       *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_), kernel_entity_rank,
//       locally_owned_intersection_with_valid_entity_parts,
//       []([[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &entity) {
//         execute(entity);
//       });

//   finalize();
// }

//   void execute(const stk::mesh::Selector &input_selector) {
//     setup();

//     auto valid_entity_parts = get_valid_entity_parts();
//     auto kernel_entity_rank = get_entity_rank();
//     stk::mesh::Selector locally_owned_intersection_with_valid_entity_parts =
//         stk::mesh::Selector(meta_data_ptr_->locally_owned_part()) & input_selector;
//     for (auto *part_ptr : valid_entity_parts) {
//       locally_owned_intersection_with_valid_entity_parts &= *part_ptr;
//     }

//     stk::mesh::for_each_entity_run(
//         *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_), kernel_entity_rank,
//         locally_owned_intersection_with_valid_entity_parts,
//         []([[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &entity) {
//           execute(entity);
//         });

//     finalize();
//   }

/// \class MetaKernel
/// \brief A virtual interface that defines the core functionality of a kernel--a class that acts on a single entity.
///
/// \tparam ReturnType The return type of the execute function.
/// \tparam Args The types of the arguments to the execute function.
template <typename... Args>
class MetaKernel : public MetaMethodSubsetExecutionInterface<void, Args...> {};  // MetaKernel

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METAKERNEL_HPP_
