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

#ifndef MUNDY_METHODS_COMPUTEOBBSPHEREKERNEL_HPP_
#define MUNDY_METHODS_COMPUTEOBBSPHEREKERNEL_HPP_

/// \file ComputeOBBSphereKernel.hpp
/// \brief Declaration of the ComputeOBBSphereKernel class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>     // for stk::mesh::Field, stl::mesh::field_data
#include <stk_topology/topology.hpp>   // for stk::topology

// Mundy libs
#include <mundy_meta/FieldRequirements.hpp>   // for mundy::meta::FieldRequirements
#include <mundy_meta/MetaKernel.hpp>          // for mundy::meta::MetaKernel, mundy::meta::MetaKernelBase
#include <mundy_meta/MetaKernelFactory.hpp>   // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernelRegistry.hpp>  // for mundy::meta::MetaKernelRegistry
#include <mundy_meta/PartRequirements.hpp>    // for mundy::meta::PartRequirements
#include <mundy_methods/ComputeOBB.hpp>       // for mundy::methods::ComputeOBB

namespace mundy {

namespace methods {

/// \class ComputeOBBSphereKernel
/// \brief Concrete implementation of \c MetaKernel for computing the axis aligned boundary box of spheres.
class ComputeOBBSphereKernel : public mundy::meta::MetaKernel<void, ComputeOBBSphereKernel>,
                               public mundy::meta::MetaKernelRegistry<void, ComputeOBBSphereKernel, ComputeOBB> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit ComputeOBBSphereKernel(stk::mesh::BulkData *const bulk_data_ptr,
                                  const Teuchos::ParameterList &parameter_list);
  //@}

  //! \name MetaKernel interface implementation
  //@{

  /// \brief Get the requirements that this manager imposes upon each particle and/or constraint.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::PartRequirements> details_get_part_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &parameter_list) {
    std::vector<std::shared_ptr<mundy::meta::PartRequirements>> required_part_params;
    required_part_params.emplace_back(std::make_shared<mundy::meta::PartRequirements>());
    required_part_params[0]->set_part_topology(stk::topology::PARTICLE);
    required_part_params[0]->add_field_reqs(
        std::make_shared<mundy::meta::FieldRequirements<double>>("obb", stk::topology::ELEMENT_RANK, 4, 1));
    required_part_params[0]->add_field_reqs(
        std::make_shared<mundy::meta::FieldRequirements<double>>("radius", stk::topology::ELEMENT_RANK, 1, 1));
    required_part_params[0]->add_field_reqs(
        std::make_shared<mundy::meta::FieldRequirements<double>>("node_coord", stk::topology::NODE_RANK, 3, 1));
    return required_part_params;
  }

  /// \brief Get the default parameters for this class.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c ParameterList
  /// will be created. You can save the result yourself if you wish to reuse it.
  static Teuchos::ParameterList details_get_valid_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("buffer_distance", default_buffer_distance_);
    default_parameter_list.set("obb_field_name", default_obb_field_name_);
    default_parameter_list.set("radius_field_name", default_radius_field_name_);
    default_parameter_list.set("node_coordinate_field_name", default_node_coord_field_name_);
    return default_parameter_list;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Run the kernel's core calculation.
  /// \param element [in] The element acted on by the kernel.
  void execute(const stk::mesh::Entity &element) override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr double default_buffer_distance_ = 0.0;
  static constexpr std::string_view default_obb_field_name_ = "OBB";
  static constexpr std::string_view default_radius_field_name_ = "RADIUS";
  static constexpr std::string_view default_node_coord_field_name_ = "NODE_COORD";
  //@}

  //! \name Internal members
  //@{
    
  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our MetaKernelRegistry.
  static const std::string class_identifier_ = "SPHERE";
  
    /// \brief The BulkData objects this class acts upon.
  stk::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData objects this class acts upon.
  stk::mesh::MetaData *meta_data_ptr_ = nullptr;
  
  /// \brief Buffer distance to be added to the object-aligned boundary box.
  ///
  /// For example, if the original object-aligned boundary box has left corner at [0,0,0] and right corner at [1,1,1],
  /// then a buffer distance of 2 will shift the left corner to [-2,-2,-2] and right corner to [3,3,3].
  double buffer_distance_;

  /// \brief Name of the element field within which the output object-aligned boundary boxes will be written.
  std::string obb_field_name_;

  /// \brief Name of the element field containing the sphere radius.
  std::string radius_field_name_;

  /// \brief Name of the node field containing the coordinate of the sphere's center
  std::string node_coord_field_name_;

  /// \brief Element field within which the output object-aligned boundary boxes will be written.
  stk::mesh::Field<double *obb_field_ptr_;

  /// \brief Element field containing the sphere radius.
  stk::mesh::Field<double *radius_field_ptr_;

  /// \brief Node field containing the coordinate of the sphere's center
  stk::mesh::Field<double *node_coord_field_ptr_;
  //@}
};  // ComputeOBBSphereKernel

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTEOBBSPHEREKERNEL_HPP_
