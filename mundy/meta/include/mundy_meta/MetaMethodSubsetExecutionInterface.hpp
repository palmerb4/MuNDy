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

#ifndef MUNDY_META_METAMETHODSUBSETEXECUTIONINTERFACE_HPP_
#define MUNDY_META_METAMETHODSUBSETEXECUTIONINTERFACE_HPP_

/// \file MetaMethodSubsetExecutionInterface.hpp
/// \brief Declaration of the MetaMethodSubsetExecutionInterface class

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

namespace mundy {

namespace meta {

/// \class MetaMethodSubsetExecutionInterface
/// \brief The execute interface for a meta method that acts on a subset of entities. The entities must reside within a
/// valid part or parts and are passed into the method as a selector.
///
/// The design of all MetaMethods was chosen such that we could easily (and at runtime) switch between different
/// techniques for implementing the core functionality of a method while allowing users to make their own custom
/// implementations. This is why we use a virtual interface where all mutable and valid parameters are passed in via
/// parameter lists. As such, while we allow MetaMethods to have any number of additional args passed to their execute
/// function, this should be used sparingly. The primary way to pass in additional arguments is through the mutable and
/// fixed parameter lists. Currently, the only difference between the API of meta methods is their execute interface.
///
/// \tparam ReturnType The return type of the execute function.
/// \tparam Args The types of the arguments to the execute function.
template <typename ReturnType_t, typename... Args>
class MetaMethodSubsetExecutionInterface {
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

  //! \name Getters
  //@{

  /// \brief Get valid entity parts for the method.
  /// By "valid entity parts," we mean the parts whose entities this method can act on.
  virtual std::vector<stk::mesh::Part *> get_valid_entity_parts() const = 0;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  /// For example, calculate the force on each entity in the given selector.
  /// \param input_selector The selector that defines the entities to act on.
  /// \param args The additional arguments to the methods's core calculation.
  virtual ReturnType execute(const stk::mesh::Selector &input_selector, Args... args) = 0;
  //@}
};  // MetaMethodSubsetExecutionInterface

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METAMETHODSUBSETEXECUTIONINTERFACE_HPP_
