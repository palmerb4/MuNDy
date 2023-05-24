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

#ifndef MUNDY_METHODS_COMPUTE_TIME_INTEGRATION_TECHNIQUES_NODE_EULER_KERNELS_LINKERSPHERE_HPP_
#define MUNDY_METHODS_COMPUTE_TIME_INTEGRATION_TECHNIQUES_NODE_EULER_KERNELS_LINKERSPHERE_HPP_

/// \file LinkerSphere.hpp
/// \brief Declaration of NodeEuler's LinkerSphere kernel.

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
#include <mundy_meta/MetaPairwiseKernel.hpp>  // for mundy::meta::MetaPairwiseKernel, mundy::meta::MetaPairwiseKernelBase
#include <mundy_meta/MetaPairwiseKernelFactory.hpp>   // for mundy::meta::MetaPairwiseKernelFactory
#include <mundy_meta/MetaPairwiseKernelRegistry.hpp>  // for mundy::meta::MetaPairwiseKernelRegistry
#include <mundy_meta/PartRequirements.hpp>            // for mundy::meta::PartRequirements
#include <mundy_methods/compute_time_integration/techniques/NodeEuler.hpp>  // for mundy::methods::...::techniques::NodeEuler

namespace mundy {

namespace methods {

namespace compute_time_integration {

namespace techniques {

namespace local_drag {

namespace kernels {

/// \class LinkerSphere
/// \brief Concrete implementation of \c MetaPairwiseKernel for computing the axis aligned boundary box of spheres.
class LinkerSphere : public mundy::meta::MetaPairwiseKernel<void, LinkerSphere>,
                     public mundy::meta::MetaPairwiseKernelRegistry<void, LinkerSphere, NodeEuler> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit LinkerSphere(stk::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &parameter_list);
  //@}

  //! \name MetaPairwiseKernel interface implementation
  //@{

  /// \brief Get the requirements that this kernel imposes upon each particle and/or constraint.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c PartRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::pair<std::shared_ptr<PartRequirements>, std::shared_ptr<PartRequirements>>
  details_static_get_part_requirements([[maybe_unused]] const Teuchos::ParameterList &parameter_list) {
    std::shared_ptr<mundy::meta::PartRequirements> required_linker_part_params =
        std::make_shared<mundy::meta::PartRequirements>();
    required_linker_part_params->set_part_rank(stk::topology::CONSTRAINT_RANK);
    required_linker_part_params->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_node_coord_field_name_), stk::topology::NODE_RANK, 3, 1));
    required_linker_part_params->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_node_velocity_field_name_), stk::topology::NODE_RANK, 3, 1));

    std::shared_ptr<mundy::meta::PartRequirements> required_sphere_part_params =
        std::make_shared<mundy::meta::PartRequirements>();
    required_sphere_part_params->set_part_topology(stk::topology::PARTICLE);
    required_sphere_part_params->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_orientation_field_name_), stk::topology::ELEMENT_RANK, 3, 1));
    required_sphere_part_params->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_node_coord_field_name_), stk::topology::NODE_RANK, 3, 1));
    required_sphere_part_params->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_node_velocity_field_name_), stk::topology::NODE_RANK, 3, 1));
    required_sphere_part_params->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
        std::string(default_node_omega_field_name_name_), stk::topology::NODE_RANK, 3, 1));

    return std::make_pair(required_linker_part_params, required_sphere_part_params);
  }

  /// \brief Get the default parameters for this class.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c ParameterList
  /// will be created. You can save the result yourself if you wish to reuse it.
  static Teuchos::ParameterList details_static_get_valid_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("time_step_size", default_time_step_size_, "The numerical timestep size.");
    default_parameter_list.set("element_orientation_field_name", std::string(default_element_orientation_field_name_),
                               "Name of the element field containing the orientation of the sphere about its center.");
    default_parameter_list.set("node_coordinate_field_name", std::string(default_node_coord_field_name_),
                               "Name of the node field containing the coordinate of the sphere's center.");
    default_parameter_list.set("node_velocity_field_name", std::string(default_node_velocity_field_name_),
                               "Name of the node field containing the translational velocity of the sphere's center.");
    default_parameter_list.set("node_omega_field_name_name", std::string(default_node_omega_field_name_name_),
                               "Name of the node field containing the rotational velocity of the sphere's center.");
    return default_parameter_list;
  }

  /// \brief Get the unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our \c MetaPairwiseKernelRegistry.
  static std::string details_static_get_class_identifier() {
    return std::string(class_identifier_);
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_params.
  static std::shared_ptr<mundy::meta::MetaPairwiseKernelBase<void>> details_static_create_new_instance(
      stk::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &parameter_list) {
    return std::make_shared<LinkerSphere>(bulk_data_ptr, parameter_list);
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Run the kernel's core calculation.
  /// \param linker [in] The linker containing the element's dynamic connectivity.
  /// \param element [in] The element acted on by the kernel.
  void execute(const stk::mesh::Entity &linker, const stk::mesh::Entity &element) override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr double default_time_step_size_ = -1;
  static constexpr std::string_view default_element_orientation_field_name_ = "ELEMENT_ORIENTATION";
  static constexpr std::string_view default_node_coord_field_name_ = "NODE_COORD";
  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
  static constexpr std::string_view default_node_omega_field_name_name_ = "NODE_OMEGA";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our \c MetaPairwiseKernelRegistry.
  static const std::string_view class_identifier_ = "SPHERE";

  /// \brief The BulkData objects this class acts upon.
  stk::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData objects this class acts upon.
  stk::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief The numerical timestep size.
  double time_step_size_;

  /// \brief Name of the element field containing the orientation of the sphere about its center.
  std::string element_orientation_field_name_;

  /// \brief Name of the node field containing the coordinate of the sphere's center.
  std::string node_coord_field_name_;

  /// \brief Name of the node field containing the translational velocity of the sphere's center.
  std::string node_velocity_field_name_;

  /// \brief Name of the node field containing the rotational velocity of the sphere's center.
  std::string node_omega_field_name_;

  /// \brief Element field containing the orientation of the sphere about its center.
  stk::mesh::Field<double> *element_orientation_field_ptr_ = nullptr;

  /// \brief Node field containing the coordinate of the sphere's center.
  stk::mesh::Field<double> *node_coord_field_ptr_ = nullptr;

  /// \brief Node field containing the translational velocity of the sphere's center.
  stk::mesh::Field<double> *node_velocity_field_ptr_ = nullptr;

  /// \brief Node field containing the rotational velocity of the sphere's center.
  stk::mesh::Field<double> *node_omega_field_ptr_ = nullptr;
  //@}
};  // LinkerSphere

}  // namespace kernels

}  // namespace local_drag

}  // namespace compute_aabb

}  // namespace methods

}  // namespace mundy

#endif  // MUNDY_METHODS_COMPUTE_TIME_INTEGRATION_TECHNIQUES_NODE_EULER_KERNELS_LINKERSPHERE_HPP_
