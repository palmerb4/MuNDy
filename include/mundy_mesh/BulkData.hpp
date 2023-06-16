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

#ifndef MUNDY_MESH_BULKDATA_HPP_
#define MUNDY_MESH_BULKDATA_HPP_

/// \file BulkData.hpp
/// \brief Declaration of the BulkData class

// C++ core libs
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <typeindex>    // for std::type_index
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData

// Mundy libs
#include <mundy_mesh/MetaData.hpp>  // for mundy::mesh::MetaData

namespace mundy {

namespace mesh {

/// \class BulkData
/// \brief A extension of STK's BulkData, with streamlined access to Mundy's stk wrappers.
///
/// For now, this extension simply stores and returns our MetaData wrapper
class BulkData : public stk::mesh::BulkData {
 public:
  //! \name Getters
  //@{

  /// \brief Fetch the meta data manager for this bulk data manager.
  const MetaData &mesh_meta_data() const {
    return *meta_data_ptr_;
  }

  /// \brief Fetch the meta data manager for this bulk data manager.
  MetaData &mesh_meta_data() {
    return *meta_data_ptr_;
  }

  /// \brief Fetch the pointer to the meta data manager for this bulk data manager.
  const std::shared_ptr<MetaData> mesh_meta_data_ptr() const {
    return meta_data_ptr_;
  }

  /// \brief Fetch the pointer to the meta data manager for this bulk data manager.
  std::shared_ptr<MetaData> mesh_meta_data_ptr() {
    return meta_data_ptr_;
  }
  //@}

 private:
  //! \name Internal members
  //@{

  std::shared_ptr<MetaData> meta_data_ptr_;
  //@}
};  // BulkData

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_BULKDATA_HPP_
