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

/// \file Collision.cpp
/// \brief Definition of ComputeConstraintProjection's Collision kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field, stl::mesh::field_data

// Mundy libs
#include <mundy_meta/FieldRequirements.hpp>   // for mundy::meta::FieldRequirements
#include <mundy_meta/MetaKernel.hpp>          // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaKernelFactory.hpp>   // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernelRegistry.hpp>  // for mundy::meta::MetaKernelRegistry
#include <mundy_meta/PartRequirements.hpp>    // for mundy::meta::PartRequirements
#include <mundy_methods/Collision.hpp>        // for mundy::methods::Collision
#include <mundy_methods/resolve_constraints/techniques/non_smooth_lcp/compute_constraint_projection/kernels/Collision.hpp>  // for mundy::methods::...::kernels::Collision

namespace mundy {

namespace methods {

namespace resolve_constraints {

namespace techniques {

namespace non_smooth_lcp {

namespace compute_constraint_projection {

namespace kernels {

// \name Constructors and destructor
//{

Collision::Collision(stk::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &parameter_list)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // Store the input parameters, use default parameters for any parameter not given.
  // Throws an error if a parameter is defined but not in the valid params. This helps catch misspellings.
  Teuchos::ParameterList valid_parameter_list = parameter_list;
  valid_parameter_list.validateParametersAndSetDefaults(this->get_valid_params());

  // Fill the internal members using the internal parameter list.
  minimum_allowable_separation_ = valid_parameter_list.get<double>("minimum_allowable_separation");
  node_coord_field_name_ = valid_parameter_list.get<std::string>("node_coordinate_field_name");
  node_normal_vec_field_name_ = valid_parameter_list.get<std::string>("node_normal_vector_field_name");
  lagrange_multiplier_field_name_ = valid_parameter_list.get<std::string>("lagrange_multiplier_field_name");
  constraint_violation_field_name_ = valid_parameter_list.get<std::string>("constraint_violation_field_name");

  // Store the input params.
  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name_);
  node_normal_vec_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_normal_vec_field_name_);
  lagrange_multiplier_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, lagrange_multiplier_field_name_);
  constraint_violation_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEM_RANK, constraint_violation_field_name_);
}
//}

// \name Actions
//{

void Collision::execute(const stk::mesh::Entity &element) {
  stk::mesh::Entity const *nodes = bulk_data_ptr_->begin_nodes(element);
  const double *contact_pointI = stk::mesh::field_data(*node_coord_field_ptr_, nodes[1]);
  const double *contact_pointJ = stk::mesh::field_data(*node_coord_field_ptr_, nodes[2]);
  const double *contact_normal_vecI = stk::mesh::field_data(*node_normal_vec_field_ptr_, nodes[1]);
  double *constraint_violation = stk::mesh::field_data(*constraint_violation_field_ptr_, element);

  constraint_violation[0] = contact_normal_vecI[0] * (contact_pointJ[0] - contact_pointI[0]) +
                            contact_normal_vecI[1] * (contact_pointJ[1] - contact_pointI[1]) +
                            contact_normal_vecI[2] * (contact_pointJ[2] - contact_pointI[2]) -
                            minimum_allowable_separation_;
}
//}

}  // namespace kernels

}  // namespace compute_constraint_projection

}  // namespace non_smooth_lcp

}  // namespace techniques

}  // namespace resolve_constraints

}  // namespace methods

}  // namespace mundy
