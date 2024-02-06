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


/* Notes:

Our goal is to simulate N Brownian diffusing spheres in a 3D domain. We will use the following free parameters:
  - N: number of particles
  - L: length of the domain
  - T: total time of the simulation
  - dt: time step
  - D: diffusion coefficient

Each timestep will consist of
  1. Compute the Brownian contribution to the velocity of each particle
  2. Update the position of each particle using a first order Euler timestep

We'll need two MetaMethods: one for computing the brownian motion and one for taking the timestep.
*/

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector
#include <stk_topology/topology.hpp>   // for stk::topology

// Mundy libs
#include <mundy_core/throw_assert.hpp>                   // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                  // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                  // for mundy::mesh::MetaData
#include <mundy_meta/MetaFactory.hpp>               // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                // for mundy::meta::MetaKernel, mundy::meta::MetaKernel
#include <mundy_meta/MetaMethod.hpp>                // for mundy::meta::MetaMethod
#include <mundy_meta/MetaRegistry.hpp>              // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/PartRequirements.hpp>          // for mundy::meta::PartRequirements

class ComputeEulerTimestep : public mundy::meta::MetaKernelDispatcher<ComputeEulerTimestep> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeEulerTimestep() = delete;

  /// \brief Constructor
  ComputeEulerTimestep(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name MetaFactory static interface implementation
  //@{

  /// \brief Get the unique registration identifier. Ideally, this should be unique and not shared by any other \c
  /// MetaMethod.
  static RegistrationType get_registration_id() {
    return registration_id_;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethod<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<ComputeEulerTimestep>(bulk_data_ptr, fixed_params);
  }
  //@}

 private:
  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other methods in our MetaMethodRegistry.
  static constexpr std::string_view registration_id_ = "COMPUTE_EULER_TIMESTEP";
  //@}
};  // ComputeEulerTimestep


/// \class NodeEulerSphere
/// \brief Concrete implementation of \c MetaKernel for computing the node euler timestep of spheres.
class NodeEulerSphere : public mundy::meta::MetaKernel<void> {
 public:
  //! \name Typedefs
  //@{

  using RegistrationType = std::string_view;
  using PolymorphicBaseType = mundy::meta::MetaKernel<void>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit NodeEulerSphere(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name MetaKernel interface implementation
  //@{

  /// \brief Get the requirements that this method imposes upon each particle and/or constraint.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c MeshRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::MeshRequirements> get_mesh_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_params) {
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

    // Fill the requirements using the given parameter list.
    std::string node_velocity_field_name = valid_fixed_params.get<std::string>("node_velocity_field_name");
    std::string associated_part_name = valid_fixed_params.get<std::string>("part_name");

    auto part_reqs = std::make_shared<mundy::meta::PartRequirements>();
    part_reqs->set_part_name(associated_part_name);
    part_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
        node_velocity_field_name, stk::topology::NODE_RANK, 3, 1));

    const std::string parent_part_name = "SPHERES";
    const std::string grandparent_part_name = "SHAPES";
    if (associated_part_name == default_part_name_) {
      // Add the requirements directly to spheres part.
      mundy::agent::AgentHierarchy::add_part_reqs(part_reqs, parent_part_name, grandparent_part_name);
    } else {
      // Add the associated part as a subset of the spheres part.
      mundy::agent::AgentHierarchy::add_subpart_reqs(part_reqs, parent_part_name, grandparent_part_name);
    }
    return mundy::agent::AgentHierarchy::get_mesh_requirements(parent_part_name, grandparent_part_name);
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const fixed_params_ptr) {
    if (fixed_params_ptr->isParameter("node_velocity_field_name")) {
      const bool valid_type =
          fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("node_velocity_field_name");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                         "NodeEulerSphere: Type error. Given a parameter with name 'node_velocity_field_name' but "
                             << "with a type other than std::string");
    } else {
      fixed_params_ptr->set(
          "node_velocity_field_name", std::string(default_node_velocity_field_name_),
          "Name of the node velocity field to be used for computing the node euler timestep of the sphere.");
    }

    if (fixed_params_ptr->isParameter("part_name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("part_name");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                         "NodeEulerSphere: Type error. Given a parameter with name 'part_name' but "
                             << "with a type other than std::string");
    } else {
      fixed_params_ptr->set("part_name", std::string(default_part_name_),
                            "Name of the part associated with this kernel.");
    }
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const mutable_params_ptr) {
    MUNDY_THROW_ASSERT(mutable_params_ptr->isParameter("time_step_size"), std::invalid_argument,
                       "NodeEulerSphere: Missing parameter 'time_step_size' in the mutable parameter list.");
    if (mutable_params_ptr->isParameter("time_step_size")) {
      const bool valid_type = mutable_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<double>("time_step_size");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                         "NodeEulerSphere: Type error. Given a parameter with name 'time_step_size' but "
                             << "with a type other than double");
    } else {
      mutable_params_ptr->set("time_step_size", default_buffer_distance_,
                              "Buffer distance to be added to the axis-aligned boundary box.");
    }
  }

  /// \brief Get the unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our \c MetaKernelRegistry.
  static RegistrationType get_registration_id() {
    return registration_id_;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaKernel<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<Sphere>(bulk_data_ptr, fixed_params);
  }

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;
  //@}

  //! \name Actions
  //@{

  /// \brief Setup the kernel's core calculations.
  /// For example, communicate information to the GPU, populate ghosts, or zero out fields.
  void setup() override;

  /// \brief Run the kernel's core calculation.
  /// \param sphere_element [in] The sphere element acted on by the kernel.
  void execute(const stk::mesh::Entity &sphere_element) override;

  /// \brief Finalize the kernel's core calculations.
  /// For example, communicate between ghosts, perform reductions over shared entities, or swap internal variables.
  void finalize() override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr double default_buffer_distance_ = 0.0;
  static constexpr std::string_view default_part_name_ = "SPHERES";
  static constexpr std::string_view default_node_velocity_field_name_ = "ELEMENT_AABB";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our \c MetaKernelRegistry.
  static constexpr std::string_view registration_id_ = "SPHERES";

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Buffer distance to be added to the axis-aligned boundary box.
  ///
  /// For example, if the original axis-aligned boundary box has left corner at [0,0,0] and right corner at [1,1,1],
  /// then a buffer distance of 2 will shift the left corner to [-2,-2,-2] and right corner to [3,3,3].
  double buffer_distance_ = default_buffer_distance_;

  /// \brief Node field containing the coordinate of the Sphere's center.
  stk::mesh::Field<double> *node_coord_field_ptr_ = nullptr;

  /// \brief Element field within which the output axis-aligned boundary boxes will be written.
  stk::mesh::Field<double> *element_aabb_field_ptr_ = nullptr;

  /// \brief Element field containing the sphere radius.
  stk::mesh::Field<double> *element_radius_field_ptr_ = nullptr;

  //@}
};  // NodeEulerSphere




class NodeEuler : public mundy::meta::MetaMethod<void> {
 public:
  //! \name Typedefs
  //@{

  using RegistrationType = std::string_view;
  using PolymorphicBaseType = mundy::meta::MetaMethod<void>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  NodeEuler() = delete;

  /// \brief Constructor
  NodeEuler(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name MetaFactory static interface implementation
  //@{

  /// \brief Get the requirements that this method imposes upon each particle and/or constraint.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c MeshRequirements
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::MeshRequirements> get_mesh_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_params) {
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

    // Fill the requirements using the given parameter list.
    // For now, we allow this method to assign these fields to all bodies.
    // TODO(palmerb4): We should allow these fields to differ from multibody type to multibody type.
    std::string node_coord_field_name = valid_fixed_params.get<std::string>("node_coord_field_name");
    std::string node_velocity_field_name = valid_fixed_params.get<std::string>("node_velocity_field_name");
    std::string node_omega_field_name_name = valid_fixed_params.get<std::string>("node_omega_field_name_name");

    auto part_reqs = std::make_shared<mundy::meta::PartRequirements>();
    part_reqs->set_part_name("BODY");
    part_reqs->set_part_rank(stk::topology::ELEMENT_RANK);
    part_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(node_coord_field_name,
                                                                                       stk::topology::NODE_RANK, 3, 1));
    part_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(node_velocity_field_name,
                                                                                       stk::topology::NODE_RANK, 3, 1));
    part_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(node_omega_field_name_name,
                                                                                       stk::topology::NODE_RANK, 3, 1));

    auto mesh_reqs = std::make_shared<mundy::meta::MeshRequirements>();
    mesh_reqs->add_part_reqs(part_reqs);

    return mesh_reqs;
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const fixed_params_ptr) {
    if (fixed_params_ptr->isParameter("node_coord_field_name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("node_coord_field_name");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                         "NodeEuler: Type error. Given a parameter with name 'node_coord_field_name' but "
                             << "with a type other than std::string");
    } else {
      fixed_params_ptr->set("node_coord_field_name", std::string(default_node_coord_field_name_),
                            "Name of the node field containing the node's spatial coordinate.");
    }

    if (fixed_params_ptr->isParameter("node_velocity_field_name")) {
      const bool valid_type =
          fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("node_velocity_field_name");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                         "NodeEuler: Type error. Given a parameter with name 'node_velocity_field_name' but "
                             << "with a type other than std::string");
    } else {
      fixed_params_ptr->set("node_velocity_field_name", std::string(default_node_velocity_field_name_),
                            "Name of the node field containing the node's translational velocity.");
    }

    if (fixed_params_ptr->isParameter("node_omega_field_name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("node_omega_field_name");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                         "NodeEuler: Type error. Given a parameter with name 'node_omega_field_name' but "
                         "with a type other than std::string");
    } else {
      fixed_params_ptr->set("node_omega_field_name", std::string(default_node_omega_field_name_),
                            "Name of the node field containing the node's rotational velocity.");
    }
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const mutable_params_ptr) {
    if (mutable_params_ptr->isParameter("time_step_size")) {
      const bool valid_type = mutable_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<double>("time_step_size");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                         "NodeEuler: Type error. Given a parameter with name 'time_step_size' but "
                             << "with a type other than double");
      const bool is_time_step_size_positive = mutable_params_ptr->get<double>("time_step_size") > 0;
      MUNDY_THROW_ASSERT(is_time_step_size_positive, std::invalid_argument,
                         "NodeEuler: Invalid parameter. Given a parameter with name 'time_step_size' but "
                             << "with a value less than or equal to zero.");
    } else {
      mutable_params_ptr->set("time_step_size", default_time_step_size_, "The numerical timestep size.");
    }
  }

  /// \brief Get the unique registration identifier. Ideally, this should be unique and not shared by any other \c
  /// MetaMethod.
  static RegistrationType get_registration_id() {
    return registration_id_;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethod<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<NodeEuler>(bulk_data_ptr, fixed_params);
  }

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  void execute(const stk::mesh::Selector &input_selector) override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr double default_time_step_size_ = 1.0;
  static constexpr std::string_view default_node_coord_field_name_ = "NODE_COORD";
  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
  static constexpr std::string_view default_node_omega_field_name_ = "NODE_OMEGA";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other methods in our MetaMethodRegistry.
  static constexpr std::string_view registration_id_ = "NODE_EULER";

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Number of parts that this method acts on.
  size_t num_parts_ = 0;

  /// \brief Vector of pointers to the parts that this class will act upon.
  std::vector<stk::mesh::Part *> part_ptr_vector_;

  /// \brief The numerical timestep size.
  double time_step_size_;

  /// \brief Name of the node field containing the node's spatial coordinate.
  std::string node_coord_field_name_;

  /// \brief Name of the node field containing the node's translational velocity.
  std::string node_velocity_field_name_;

  /// \brief Name of the node field containing the node's rotational velocity.
  std::string node_omega_field_name_;

  /// \brief Node field containing the node's spatial coordinate.
  stk::mesh::Field<double> *node_coord_field_ptr_ = nullptr;

  /// \brief Node field containing the node's translational velocity.
  stk::mesh::Field<double> *node_velocity_field_ptr_ = nullptr;

  /// \brief Node field containing the node's rotational velocity.
  stk::mesh::Field<double> *node_omega_field_ptr_ = nullptr;
  //@}
};  // NodeEuler