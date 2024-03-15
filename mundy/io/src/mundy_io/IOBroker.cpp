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

/// \file IOBroker.cpp
/// \brief Definition of IOBroker functions

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>      // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>        // for stk::mesh::Field, stl::mesh::field_data
#include <stk_mesh/base/GetEntities.hpp>  // for stk::mesh::count_entities

// Mundy libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_io/IOBroker.hpp>        // for mundy::io::IOBroker
#include <mundy_mesh/BulkData.hpp>      // for mundy::mesh::BulkData

namespace mundy {

namespace io {

//! \name Setters
//@{

void IOBroker::set_coordinate_field() {
  meta_data_ptr_->set_coordinate_field_name(coordinate_field_name_);
  // If we are restarting, then the restart will take care of setting this
  if (!enable_restart_) {
    // Get the coordinate field base so we can set the IOSS role (NOTE: Chris, I do not like this, as we might change
    // variable types later)
    coordinate_field_ptr_ = meta_data_ptr_->get_field(stk::topology::NODE_RANK, coordinate_field_name_);
    // Set the role of the field
    stk::io::set_field_role(*coordinate_field_ptr_, Ioss::Field::MESH);
  }
}

void IOBroker::set_transient_fields(const Teuchos::ParameterList &valid_fixed_params) {
  // Loop over the rank names we have in the mesh
  for (const std::string &rank_name : meta_data_ptr_->entity_rank_names()) {
    // Grab the correct version of the enabled_io_fields param
    std::string enabled_rank_name_str = "enabled_io_fields_" + rank_name + "_rank";
    std::string rank_name_str = rank_name + "_RANK";
    std::transform(enabled_rank_name_str.begin(), enabled_rank_name_str.end(), enabled_rank_name_str.begin(),
                   [](unsigned char c) {
                     return std::tolower(c);
                   });

    Teuchos::Array<std::string> enabled_io_fields =
        valid_fixed_params.get<Teuchos::Array<std::string>>(enabled_rank_name_str);
    if (!enabled_io_fields.empty()) {
      for (const std::string &io_field_name : enabled_io_fields) {
        if (io_field_name != "") {
          std::ostringstream ostream;
          ostream << "Enabling Field IO (" << rank_name_str << ") " << io_field_name;
          stk::log_with_time_and_memory(bulk_data_ptr_->parallel(), ostream.str());

          // Get the field and tag as TRANSIENT
          stk::mesh::FieldBase *io_field_ptr =
              meta_data_ptr_->get_field(mundy::mesh::map_string_to_rank(rank_name_str), io_field_name);
          MUNDY_THROW_ASSERT(io_field_ptr != nullptr, std::invalid_argument,
                             "IOBroker: could not find field " + io_field_name + " with rank " + rank_name_str);
          stk::io::set_field_role(*io_field_ptr, Ioss::Field::TRANSIENT);

          enabled_io_fields_.push_back(io_field_ptr);
        }
      }
    }
  }

  // Set the transient_coordinate_field directly
  transient_coordinate_field_ptr_ =
      meta_data_ptr_->get_field(stk::topology::NODE_RANK, transient_coordinate_field_name_);
  MUNDY_THROW_ASSERT(transient_coordinate_field_ptr_ != nullptr, std::invalid_argument,
                     "IOBroker: transient coordinate field alias set incorrectly.");
  stk::io::set_field_role(*transient_coordinate_field_ptr_, Ioss::Field::TRANSIENT);
  enabled_io_fields_.push_back(transient_coordinate_field_ptr_);
}

//@}

//! \name Printers
//@{

void IOBroker::print_field_roles() {
  stk::log_with_time_and_memory(bulk_data_ptr_->parallel(), "IO roles for fields");
  // const stk::mesh::FieldVector &fields = STKioBroker_.meta_data().get_fields();
  const stk::mesh::FieldVector &fields = meta_data_ptr_->get_fields();
  for (size_t i = 0; i < fields.size(); ++i) {
    const Ioss::Field::RoleType *role = stk::io::get_field_role(*fields[i]);
    std::ostringstream ostream;
    ostream << "...field: " << *fields[i];
    if (role) {
      ostream << ", role: " << Ioss::Field::role_string(*role);
    }
    stk::log_with_time_and_memory(bulk_data_ptr_->parallel(), ostream.str());
  }
}

void IOBroker::print_io_broker() {
  stk::log_with_time_and_memory(bulk_data_ptr_->parallel(), "MuNDy IOBroker");
  stk::log_with_time_and_memory(bulk_data_ptr_->parallel(), "EXODUS database: " + exodus_database_output_filename_);
  stk::log_with_time_and_memory(bulk_data_ptr_->parallel(), "Parallel IO mode: " + parallel_io_mode_);
  stk::log_with_time_and_memory(bulk_data_ptr_->parallel(), "Coordinates field: " + coordinate_field_name_);
  stk::log_with_time_and_memory(bulk_data_ptr_->parallel(),
                                "...(Transient) Coordinates field: " + transient_coordinate_field_name_);

  if (enable_restart_) {
    stk::log_with_time_and_memory(bulk_data_ptr_->parallel(), "Enable restart: true");
    stk::log_with_time_and_memory(bulk_data_ptr_->parallel(),
                                  "EXODUS database input: " + exodus_database_input_filename_);
  }
  {
    std::ostringstream ostream;
    ostream << "Database purpose: ";
    if (database_purpose_ == stk::io::WRITE_RESULTS) {
      ostream << "WRITE_RESULTS";
    } else if (database_purpose_ == stk::io::WRITE_RESTART) {
      ostream << "WRITE_RESTART";
    } else if (database_purpose_ == stk::io::APPEND_RESULTS) {
      ostream << "APPEND_RESULTS";
    }
    stk::log_with_time_and_memory(bulk_data_ptr_->parallel(), ostream.str());
  }

  // Print the IO fields (all fields too)
  print_field_roles();
}

//@}

// \name Actions
//{

void IOBroker::finalize_io_broker() {
  // Flush the output to disk to make sure we're good
  STKioBroker_.flush_output();
  // Now close it
  STKioBroker_.close_output_mesh(io_index_);
}

void IOBroker::restart_mesh() {
  // Tell the user we are restaring the simulation/whatever
  std::ostringstream restartstream;
  restartstream << "...RESTART enabled, input database " << exodus_database_input_filename_;
  stk::log_with_time_and_memory(bulk_data_ptr_->parallel(), restartstream.str());

  // Create a local ioBroker to not interfere with the other one that we use for output
  stk::io::StkMeshIoBroker ioBroker(bulk_data_ptr_->parallel());
  // Set the bulk data in order to populate everything
  ioBroker.set_bulk_data(*bulk_data_ptr_);
  // Create the input mesh
  size_t input_index = ioBroker.add_mesh_database(exodus_database_input_filename_, "exodus", stk::io::READ_RESTART);
  // Set the mesh to active, and activate it
  ioBroker.set_active_mesh(input_index);
  ioBroker.create_input_mesh();

  // The fields should already be set to TRANSIENT, along with pointers to them. Try to populate the bulk data.
  ioBroker.populate_bulk_data();

  // Add the input fields
  for (size_t i = 0; i < enabled_io_fields_.size(); ++i) {
    ioBroker.add_input_field(enabled_io_fields_[i]);
  }
  // Read the defined input fields, looking for any that are missing
  std::vector<stk::io::MeshField> missingFields;
  ioBroker.read_defined_input_fields(ioBroker.get_max_time(), &missingFields);

  // Let the user know if there are missing fields
  if (missingFields.size() > 0) {
    for (size_t i = 0; i < missingFields.size(); ++i) {
      std::ostringstream ostream;
      ostream << "...IOBroker: Found missing FIELD " << missingFields[i].field()->name();
      stk::log_with_time_and_memory(bulk_data_ptr_->parallel(), ostream.str());
    }
  }

  // Make sure we set the coordinate_field_ptr_ (it is not set on restart earlier)
  coordinate_field_ptr_ = meta_data_ptr_->get_field(stk::topology::NODE_RANK, coordinate_field_name_);
}

void IOBroker::setup_io_broker() {
  // Create the output mesh
  io_index_ = STKioBroker_.create_output_mesh(exodus_database_output_filename_, database_purpose_);

  // Add the enabled fields to the output mesh
  for (size_t i = 0; i < enabled_io_fields_.size(); ++i) {
    STKioBroker_.add_field(io_index_, *enabled_io_fields_[i]);
  }

  // No matter what, the setup function writes an output mesh
  STKioBroker_.write_output_mesh(io_index_);
}

void IOBroker::synchronize_node_coordinates_from_transient() {
  // Get the locally owned part
  stk::mesh::Selector locally_owned = meta_data_ptr_->locally_owned_part();
  // Alias the coordinate fields
  auto &coordinate_field = *coordinate_field_ptr_;
  auto &transient_coordinate_field = *transient_coordinate_field_ptr_;
  // Check if we have the field pointers
  MUNDY_THROW_ASSERT(
      coordinate_field_ptr_ != nullptr, std::invalid_argument,
      "IOBroker::synchronize_node_coordinates_from_transient coordinate_field_ptr_ cannot be a nullptr.");
  MUNDY_THROW_ASSERT(
      transient_coordinate_field_ptr_ != nullptr, std::invalid_argument,
      "IOBroker::synchronize_node_coordinates_from_transient transient_coordinate_field_ptr_ cannot be a nullptr.");
  // This is how we loop over entities and assign them to each other
  stk::mesh::for_each_entity_run(
      *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_), stk::topology::NODE_RANK, locally_owned,
      [&coordinate_field, &transient_coordinate_field]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                                       const stk::mesh::Entity &entity) {
        double *coordinates = reinterpret_cast<double *>(stk::mesh::field_data(coordinate_field, entity));
        double *transient_coordinates =
            reinterpret_cast<double *>(stk::mesh::field_data(transient_coordinate_field, entity));
        coordinates[0] = transient_coordinates[0];
        coordinates[1] = transient_coordinates[1];
        coordinates[2] = transient_coordinates[2];
      });
}

void IOBroker::write_io_broker(double time) {
  // Before we write, synchronize the TRANSIENT coordinate field
  stk::mesh::Selector locally_owned = meta_data_ptr_->locally_owned_part();
  // Alias the coordinate fields
  auto &coordinate_field = *coordinate_field_ptr_;
  auto &transient_coordinate_field = *transient_coordinate_field_ptr_;
  // This is how we loop over entities and assign them to each other
  stk::mesh::for_each_entity_run(
      *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_), stk::topology::NODE_RANK, locally_owned,
      [&coordinate_field, &transient_coordinate_field]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                                       const stk::mesh::Entity &entity) {
        double *coordinates = reinterpret_cast<double *>(stk::mesh::field_data(coordinate_field, entity));
        double *transient_coordinates =
            reinterpret_cast<double *>(stk::mesh::field_data(transient_coordinate_field, entity));
        transient_coordinates[0] = coordinates[0];
        transient_coordinates[1] = coordinates[1];
        transient_coordinates[2] = coordinates[2];
      });

  // Save the IO
  STKioBroker_.begin_output_step(io_index_, time);
  STKioBroker_.write_defined_output_fields(io_index_);
  STKioBroker_.end_output_step(io_index_);
}

// }

}  // namespace io

}  // namespace mundy
