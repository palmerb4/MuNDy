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

#ifndef MUNDY_META_METAKWAYKERNEL_HPP_
#define MUNDY_META_METAKWAYKERNEL_HPP_

/// \file MetaKWayKernel.hpp
/// \brief Declaration of the MetaKWayKernel class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part

// Mundy libs
#include <mundy_mesh/BulkData.hpp>                              // for mundy::mesh::BulkData
#include <mundy_meta/HasMeshRequirementsAndIsRegisterable.hpp>  // for mundy::meta::HasMeshRequirementsAndIsRegisterable
#include <mundy_meta/PartRequirements.hpp>                      // for mundy::meta::PartRequirements

namespace mundy {

namespace meta {

/// \class MetaKWayKernel
/// \brief A virtual interface that defines the core functionality of a K-way kernel--a class that acts on K entities.
///
/// \tparam K The number of entities passed to execute.
/// \tparam ReturnType The return type of the execute function.
/// \tparam Args The types of the arguments to the execute function.
template <std::size_t K, typename ReturnType_t, typename... Args>
class MetaKWayKernel {
 public:
  //! \name Typedefs
  //@{

  using ReturnType = ReturnType_t;
  using ArgsTuple = std::tuple<Args...>;
  //@}

  //! \name Setters
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  virtual void set_mutable_params(const Teuchos::ParameterList &mutable_params) = 0;
  //@}

  //! \name Actions
  //@{

  /// \brief Setup the kernel's core calculations.
  /// For example, communicate information to the GPU, populate ghosts, or zero out fields.
  virtual void setup() = 0;

  /// \brief Run the kernel's core calculation.
  /// For example, calculate the force on an entity.
  /// \param entity_array The array of entities to act on.
  /// \param args The additional arguments to the kernel's core calculation.
  virtual ReturnType execute(const std::array<stk::mesh::Entity, K> &entity_array, Args... args) = 0;

  /// \brief Finalize the kernel's core calculations.
  /// For example, communicate between ghosts, perform reductions over shared entities, or swap internal variables.
  virtual void finalize() = 0;
  //@}
};  // MetaKWayKernel

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METAKWAYKERNEL_HPP_
