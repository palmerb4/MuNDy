// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2023 Flatiron Institute
//                                              Author: Shihab Shahriar Khan
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

#ifndef MUNDY_METHODS_STEPEULER_HPP_
#define MUNDY_METHODS_STEPEULER_HPP_

/// \file StepEuler.hpp
/// \brief Declaration of the StepEuler class


// clang-format off
#include <gtest/gtest.h>                             // for AssertHelper, etc
#include <mpi.h>                                     // for MPI_COMM_WORLD, etc
#include <stddef.h>                                  // for size_t
#include <vector>                                    // for vector, etc
#include <random>                                    // for rand
#include <memory>                                    // for shared_ptr
#include <string>                                    // for string
#include <type_traits>                               // for static_assert
#include <stk_math/StkVector.hpp>                    // for Vec
#include <stk_mesh/base/BulkData.hpp>                // for BulkData
#include <stk_mesh/base/MetaData.hpp>                // for MetaData
#include <stk_mesh/base/GetEntities.hpp>             // for count_selected_entities
#include <stk_mesh/base/Types.hpp>                   // for EntityVector, etc
#include <stk_mesh/base/Ghosting.hpp>                // for create_ghosting
#include <stk_io/WriteMesh.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_search/SearchMethod.hpp>               // for KDTREE
#include <stk_search/CoarseSearch.hpp>               // for coarse_search
#include <stk_search/BoundingBox.hpp>                // for Sphere, Box, tec.
#include <stk_balance/balance.hpp>                   // for balanceStkMesh
#include <stk_util/parallel/Parallel.hpp>            // for ParallelMachine
#include <stk_util/environment/WallTime.hpp>         // for wall_time
#include <stk_util/environment/perf_util.hpp>
#include <stk_unit_test_utils/BuildMesh.hpp>
#include "stk_util/util/ReportHandler.hpp"           // for ThrowAssert, etc
// clang-format on

namespace mundy {

namespace methods {

/// \class StepEuler
/// \brief Method for computing the axis aligned boundary box of different parts.
class StepEuler : public MetaMethod<StepEuler>, public MetaMethodRegistry<StepEuler> {
 public:
    //! \name Constructors and destructor
    //@{

    /// \brief No default constructor
    StepEuler() = delete;

    /// \brief Constructor
    StepEuler(const stk::mesh::BulkData *bulk_data_ptr, const std::vector<*stk::mesh::Part> &part_ptr_vector,
              const Teuchos::ParameterList &parameter_list)
      : bulk_data_ptr_(bulk_data_ptr), part_ptr_vector_(part_ptr_vector), num_parts_(part_ptr_vector_.size()) {
    // The bulk data pointer must not be null.
        TEUCHOS_TEST_FOR_EXCEPTION(bulk_data_ptr_ == nullptr, std::invalid_argument,
                                "mundy::methods::ComputeAABB: bulk_data_ptr cannot be a nullptr.");

        // The parts cannot intersect.
        for (int i = 0; i < num_parts_; i++) {
            for (int j = 0; j < num_parts_; j++) {
                fi(i==j) continue;
                const bool parts_intersect = stk::mesh::intersect(*part_ptr_vector[i], *part_ptr_vector[j]);
                TEUCHOS_TEST_FOR_EXCEPTION(parts_intersect, std::invalid_argument,
                                        "mundy::methods::ComputeAABB: Part " << part_ptr_vector[i]->name() << " and "
                                                                                << "Part " << part_ptr_vector[j]->name()
                                                                                << "intersect.");
            }
        }


        // Store the input parameters, use default parameters for any parameter not given.
        // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
        parameter_list_ = parameter_list;
        parameter_list_.validateParametersAndSetDefaults(get_valid_params());

        // Create and store the required kernels.
        const Teuchos::ParameterList &euler_kernel_parameter_list = parameter_list.sublist("kernels").sublist("step_euler");

        for (int i = 0; i < num_parts_; i++) {
            // Fetch the parameters for this part's kernel
            const std::string part_name = part_ptr_vector_[i]->name();
            const Teuchos::ParameterList &part_euler_kernel_parameter_list = euler_kernel_parameter_list.sublist(part_name);

            // Get the name of the kernel to be used on this part.
            const std::string kernel_name = part_euler_kernel_parameter_list.get<std::string>("name");
            step_euler_kernels_.push_back(MetaKernelFactory<StepEuler>::create_new_instance(
                kernel_name, bulk_data_ptr, part_euler_kernel_parameter_list));
        }
    }
    //@}

    //! MethaMethod Interface Implementation
    //@{

    /// \brief Get the requirements that this method imposes upon each particle and/or constraint.
    ///
    /// \param parameter_list [in] Optional list of parameters for setting up this class. A
    /// default parameter list is accessible via \c get_valid_params.
    ///
    /// \note This method does not cache its return value, so every time you call this method, a new \c PartParams
    /// will be created. You can save the result yourself if you wish to reuse it.
    static std::unique_ptr<PartParams> details_get_part_requirements(
        [[maybe_unused]] const Teuchos::ParameterList &parameter_list) {
        // TODO(palmerb4): we need the ability to not specify requirements for part name or even topology
        std::unique_ptr<PartParams> required_part_params = std::make_unique<PartParams>(std::topology::PARTICLE);
        int spatial_dimention = parameter_list.get<int>("spatial_dimention");

        required_part_params->add_field_params(
            std::make_unique<FieldParams<double>>(default_coord_field_name_, std::topology::NODE_RANK, spatial_dimention, 1));
        required_part_params->add_field_params(
            std::make_unique<FieldParams<double>>(default_orientation_field_name_, std::topology::ELEMENT_RANK, spatial_dimention+1, 1));
        required_part_params->add_field_params(
            std::make_unique<FieldParams<double>>(default_omega_field_name_, std::topology::NODE_RANK, spatial_dimention, 1));
        required_part_params->add_field_params(
            std::make_unique<FieldParams<double>>(default_velocity_field_name_, std::topology::NODE_RANK, spatial_dimention, 1));
        return required_part_params;
    }


    /// \brief Get the default parameters for this class.
    static Teuchos::ParameterList details_get_valid_params() {
        static Teuchos::ParameterList default_parameter_list;
        default_parameter_list.set(
            "coord_field_name_", default_coord_field_name_,
            "Coordinate field of the particle");
        default_parameter_list.set(
            "omega_field_name_", default_omega_field_name_,
            "Angular velocity field of the particle");
        
        default_parameter_list.set(
            "velocity_field_name_", default_velocity_field_name_,
            "Particle velocity field");
        
        default_parameter_list.set(
            "orientation_field_name_", default_orientation_field_name_,
            "Field to store the quaternion representing the particle's orientation");
        
        return default_parameter_list;
    }

    /// \brief Get the unique class identifier. Ideally, this should be unique and not shared by any other \c MetaMethod.
    static std::string details_get_class_identifier() const {
        return "StepEuler";
    }

    /// \brief Generate a new instance of this class.
    ///
    /// \param parameter_list [in] Optional list of parameters for setting up this class. A
    /// default parameter list is accessible via \c get_valid_params.
    static std::unique_ptr<MetaMethod> details_create_new_instance(const stk::mesh::BulkData *bulk_data_ptr,
                                                                    const std::vector<*stk::mesh::Part> &part_ptr_vector,
                                                                    const Teuchos::ParameterList &parameter_list) const {
        return std::make_unique<StepEuler>(bulk_data_ptr, part_ptr_vector, parameter_list);
    }
    //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
void execute() {
    for (int i = 0; i < num_parts_; i++) {
        const MetaKernel &step_euler_kernel = step_euler_kernels_[i];
        stk::mesh::Selector locally_owned_part = meta_mesh.locally_owned_part() && *part_ptr_vector_[i];
        
        step_euler_kernel.execute(meta_mesh, locally_owned_part.primary_entity_rank(), selector);
    }
  }
  //@}


private:
    //! \name Default parameters
    //@{
    static constexpr std::string default_omega_field_name_ = "NODE_OOMEGA";
    static constexpr std::string default_coord_field_name_ = "NODE_COORD";
    static constexpr std::string default_orientation_field_name_ = "ORIENTATION";
    static constexpr std::string default_velocity_field_name_ = "NODE_VELOCITY";

    //@}

    //! \name Internal members
    //@{

    /// \brief Number of parts that this method acts on.
    size_t num_parts_;

    /// \brief Current parameter list with valid entries.
    Teuchos::ParameterList parameter_list_;

    /// \brief Vector of pointers to the parts that this class will act upon.
    std::vector<*stk::mesh::Part> &part_ptr_vector_;

    /// \brief Kernels corresponding to each of the specified parts.
    std::vector<MetaKernel> step_euler_kernels_;
    //@}
};  // ComputeAABB

}  // namespace methods

}  // namespace mundy

#endif