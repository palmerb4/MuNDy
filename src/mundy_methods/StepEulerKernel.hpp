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

/// \file StepEulerkernel.hpp
/// \brief Declaration of the StepEulerkernel class


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

/// \class StepEulerKernel
/// \brief Concrete implementation of \c MetaKernel for computing the axis aligned boundary box of spheres.
class StepEulerKernel : public MetaKernel<StepEulerKernel>,
                        public MetaKernelRegistry<StepEulerKernel, StepEuler>{

public:
    //! \name Constructors and destructor
    //@{

    /// \brief Constructor
    explicit StepEulerKernel(const stk::mesh::BulkData *bulk_data_ptr,
                                    const Teuchos::ParameterList &parameter_list) {
    
        // Store the input parameters, use default parameters for any parameter not given.
        // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
        parameter_list_ = parameter_list;
        parameter_list_.validateParametersAndSetDefaults(get_valid_params());

        // Fill the internal members using the internal parameter list.
        buffer_distance_ = parameter_list_.get<double>("dt");
        node_coord_field_name_ = parameter_list_.get<std::string>("node_coord_field_name");
        node_omega_field_name_ = parameter_list_.get<std::string>("node_omega_field_name");
        node_velocity_field_name_ = parameter_list_.get<std::string>("node_velocity_field_name");
        particle_orientation_field_name_ = parameter_list_.get<std::string>("particle_orientation_field_name");

        // Store the input params.
        coord_field_ptr_ = bulk_data_ptr->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name_);
        velocity_field_ptr_ = bulk_data_ptr->get_field<double>(stk::topology::NODE_RANK, node_velocity_field_name_);
        omega_field_ptr_ = bulk_data_ptr->get_field<double>(stk::topology::NODE_RANK, node_omega_field_name_);
        orientation_field_ptr_ = bulk_data_ptr->get_field<double>(stk::topology::ELEM_RANK, particle_orientation_field_name_);
    }//@}

  //! \name MetaKernel interface implementation
  //@{

  /// \brief Get the requirements that this manager imposes upon each particle and/or constraint.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartParams
  /// will be created. You can save the result yourself if you wish to reuse it.
    static std::unique_ptr<PartParams> details_get_part_requirements(
        [[maybe_unused]] const Teuchos::ParameterList &parameter_list) {
        return StepEuler::details_get_part_requirements(parameter_list);
    }

  /// \brief Get the default parameters for this class.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c ParameterList
  /// will be created. You can save the result yourself if you wish to reuse it.
    static Teuchos::ParameterList details_get_valid_params() {
        static Teuchos::ParameterList default_parameter_list = StepEuler::details_get_valid_params() ;
        default_parameter_list.set("dt", default_dt_,"Time step size");
        return default_parameter_list;
    }
    //@}

    //! \name Actions
    //@{

    void execute(const BulkData &mesh,
                stk::topology::rank_t rank,
                const Selector &selector) {

        coord_field_ptr_device_ = stk::mesh::get_updated_ngp_field<double>(coord_field_ptr_device_);
        velocity_field_ptr_device_ = stk::mesh::get_updated_ngp_field<double>(velocity_field_ptr_);
        omega_field_ptr_device_ = stk::mesh::get_updated_ngp_field<double>(omega_field_ptr_);
        orientation_field_ptr_device_ = stk::mesh::get_updated_ngp_field<double>(orientation_field_ptr_);
        
        mundy::utils::for_each_entity_run(
            mesh,
            rank,
            selector,
            KOKKOS_LAMBDA(stk::mesh::FastMeshIndex index){
                coord_field_ptr_device_(index, 0) += dt*velocity_field_ptr_device_(index, 0);
                
                Quaternion quat(
                    orientation_field_ptr_device_(index, 0), orientation_field_ptr_device_(index, 1), 
                    orientation_field_ptr_device_(index, 2), orientation_field_ptr_device_(index, 3)
                );
                quat.rotate_self(omega_field_ptr_device_(index, 0), omega_field_ptr_device_(index, 1), 
                                omega_field_ptr_device_(index, 2),  omega_field_ptr_device_(index, 3), dt);
                orientation_field_ptr_device_(index, 0) = quat.w;
                orientation_field_ptr_device_(index, 1) = quat.x;
                orientation_field_ptr_device_(index, 2) = quat.y;
                orientation_field_ptr_device_(index, 3) = quat.z;

            }
        );
        coord_field_ptr_device_.sync_to_host();
        // velocity_field_ptr_device_.sync_to_host(); //Cause we haven't written to these fields.
        // omega_field_ptr_device_.sync_to_host();
        orientation_field_ptr_device_.sync_to_host();
        
    }
    //@}


private:
    //! \name Default parameters
    //@{

    static constexpr double default_dt_= 0.0001;
    static constexpr std::string default_node_coord_field_name_ = "NODE_COORD";
    static constexpr std::string default_node_omega_field_name_ = "NODE_OMAGA";
    static constexpr std::string default_node_velocity_field_name_ = "NODE_VELOCITY";
    static constexpr std::string default_particle_orientation_field_name_ = "ORIENTATION";
    //@}

    //! \name Internal members
    //@{

    /// \brief Current parameter list with valid entries.
    Teuchos::ParameterList parameter_list_;

    /// \brief Time step size.
    double dt_;
    /// \brief Name of the node field containing the coordinate of center of mass.
    std::string node_coord_field_name_;
    /// \brief Name of the node field containing
    std::string node_omega_field_name_;
    /// \brief Name of the node field containing
    std::string node_velocity_field_name_;
    /// \brief Name of the node field containing
    std::string particle_orientation_field_name_;

    stk::mesh::Field<double> *coord_field_ptr_;
    stk::mesh::NgpField<double> *coord_field_ptr_device_;

    stk::mesh::Field<double> *velocity_field_ptr_;
    stk::mesh::NgpField<double> *velocity_field_ptr_device_;

    stk::mesh::Field<double> *omega_field_ptr_;
    stk::mesh::NgpField<double> *omega_field_ptr_device_;

    stk::mesh::Field<double> *orientation_field_ptr_;
    stk::mesh::NgpField<double> *orientation_field_ptr_device_;

    //@}
};  // StepEulerKernel

}  // namespace methods

}  // namespace mundy

#endif