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

/// \file MeshBuilder.cpp
/// \brief Definition of the MeshBuilder class

// C++ core libs
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs
#include <Teuchos_TestForException.hpp>            // for TEUCHOS_TEST_FOR_EXCEPTION
#include <stk_mesh/base/BulkData.hpp>              // for stk::mesh::BulkData
#include <stk_mesh/base/FieldDataManager.hpp>      // for stl::mesh::FieldDataManager
#include <stk_mesh/base/MeshBuilder.hpp>           // for stk::mesh::MeshBuilder
#include <stk_mesh/baseImpl/BucketRepository.hpp>  // stk::impl::BucketRepository
#include <stk_util/parallel/Parallel.hpp>          // for stk::ParallelMachine

// Mundy libs
#include <mundy_mesh/MeshBuilder.hpp>  // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>     // for mundy::mesh::MetaData

namespace mundy {

namespace mesh {

// \name Constructors and destructor
//{

MeshBuilder::MeshBuilder()
    : builder_(MPI_COMM_NULL),
      comm_(MPI_COMM_NULL),
      has_comm_(false),
      aura_option_(stk::mesh::BulkData::AUTO_AURA),
      field_data_manager_ptr_(nullptr),
      bucket_capacity_(stk::mesh::impl::BucketRepository::default_bucket_capacity),
      spatial_dimension_(0),
      entity_rank_names_(),
      upward_connectivity_flag_(true) {
}

MeshBuilder::MeshBuilder(stk::ParallelMachine comm)
    : builder_(comm),
      comm_(comm),
      has_comm_(true),
      aura_option_(stk::mesh::BulkData::AUTO_AURA),
      field_data_manager_ptr_(nullptr),
      bucket_capacity_(stk::mesh::impl::BucketRepository::default_bucket_capacity),
      spatial_dimension_(0),
      entity_rank_names_(),
      upward_connectivity_flag_(true) {
}
//}

// @name Setters
//{

MeshBuilder &MeshBuilder::set_spatial_dimension(const unsigned spatial_dimension) {
  spatial_dimension_ = spatial_dimension;
  builder_.set_spatial_dimension(spatial_dimension_);
  return *this;
}

MeshBuilder &MeshBuilder::set_entity_rank_names(const std::vector<std::string> &entity_rank_names) {
  entity_rank_names_ = entity_rank_names;
  builder_.set_entity_rank_names(entity_rank_names_);
  return *this;
}

MeshBuilder &MeshBuilder::set_communicator(const stk::ParallelMachine &comm) {
  comm_ = comm;
  has_comm_ = true;
  builder_.set_communicator(comm_);
  return *this;
}

MeshBuilder &MeshBuilder::set_aura_option(const stk::mesh::BulkData::AutomaticAuraOption &aura_option) {
  aura_option_ = aura_option;
  builder_.set_aura_option(aura_option_);
  return *this;
}

MeshBuilder &MeshBuilder::set_field_data_manager(stk::mesh::FieldDataManager *const field_data_manager_ptr) {
  field_data_manager_ptr_ = field_data_manager_ptr;
  builder_.set_field_data_manager(field_data_manager_ptr_);
  return *this;
}

MeshBuilder &MeshBuilder::set_bucket_capacity(const unsigned bucket_capacity) {
  bucket_capacity_ = bucket_capacity;
  builder_.set_bucket_capacity(bucket_capacity_);
  return *this;
}

MeshBuilder &MeshBuilder::set_upward_connectivity_flag(const bool enable_upward_connectivity) {
  upward_connectivity_flag_ = enable_upward_connectivity;
  builder_.set_upward_connectivity(upward_connectivity_flag_);
  return *this;
}
//}

// @name Actions
//{

std::shared_ptr<mundy::mesh::MetaData> MeshBuilder::create_meta_data() {
  return builder_.create_meta_data();
}

std::unique_ptr<stk::mesh::BulkData> MeshBuilder::create_bulk_data() {
  return this->create_bulk_data(this->create_meta_data());
}

std::unique_ptr<stk::mesh::BulkData> MeshBuilder::create_bulk_data(std::shared_ptr<mundy::mesh::MetaData> metaData) {
  TEUCHOS_TEST_FOR_EXCEPTION(has_comm_, std::logic_error,
                             "MeshBuilder: Must be given an MPI communicator before creating BulkData.");

  return builder_.create(MetaData);
}
//}

}  // namespace mesh

}  // namespace mundy
