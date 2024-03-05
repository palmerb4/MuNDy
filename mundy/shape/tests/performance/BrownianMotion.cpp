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
  - number_of_particles
  - length_of_domain
  - total_time
  - time_step_size
  - diffusion_coeff

Each timestep will consist of
  1. Compute the Brownian contribution to the velocity of each particle
  2. Update the position of each particle using a first order Euler timestep

We'll need two MetaMethods: one for computing the brownian motion and one for taking the timestep.
*/

// External libs
#include <openrand/philox.h>

// Trilinos libs
#include <Kokkos_Core.hpp>                 // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer
#include <Teuchos_ParameterList.hpp>       // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>        // for stk::mesh::Entity
#include <stk_mesh/base/Part.hpp>          // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>      // for stk::mesh::Selector
#include <stk_topology/topology.hpp>       // for stk::topology
#include <stk_util/parallel/Parallel.hpp>  // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy libs
#include <mundy_agent/AgentHierarchy.hpp>       // for mundy::agent::AgentHierarchy
#include <mundy_core/StringLiteral.hpp>         // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_core/throw_assert.hpp>          // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>              // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>              // for mundy::mesh::MetaData
#include <mundy_meta/MetaFactory.hpp>           // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>            // for mundy::meta::MetaKernel
#include <mundy_meta/MetaKernelDispatcher.hpp>  // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter
#include <mundy_meta/PartRequirements.hpp>  // for mundy::meta::PartRequirements
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements
#include <mundy_shape/PerformRegistration.hpp>  // for mundy::shape::perform_registration
#include <mundy_shape/shapes/Spheres.hpp>       // for mundy::shape::shapes::Spheres

class NodeEuler
    : public mundy::meta::MetaKernelDispatcher<NodeEuler, mundy::meta::make_registration_string("NODE_EULER")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  NodeEuler() = delete;

  /// \brief Constructor
  NodeEuler(mundy::mesh::BulkData *const bulk_data_ptr,
            const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList())
      : mundy::meta::MetaKernelDispatcher<NodeEuler, mundy::meta::make_registration_string("NODE_EULER")>(
            bulk_data_ptr, fixed_params) {
  }
  //@}


  //! \name MetaKernelDispatcher static interface implementation
  //@{

  /// \brief Get the valid fixed parameters that we require our techniques have.
  static Teuchos::ParameterList get_valid_required_kernel_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we require our techniques have.
  static Teuchos::ParameterList get_valid_required_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid fixed parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set(
        "node_velocity_field_name", std::string(default_node_velocity_field_name_),
        "Name of the node velocity field to be used for computing the node euler timestep of the sphere.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("time_step_size", default_time_step_size_, "The timestep size.");
    return default_parameter_list;
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static inline double default_time_step_size_ = 0.0;
  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
  //@}
};  // NodeEuler

/// \class NodeEulerSphere
/// \brief Concrete implementation of \c MetaKernel for computing the node euler timestep of spheres.
class NodeEulerSphere : public mundy::meta::MetaKernel<void> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaKernel<void>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit NodeEulerSphere(mundy::mesh::BulkData *const bulk_data_ptr,
                           const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList())
      : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
    // The bulk data pointer must not be null.
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument,
                       "NodeEulerSphere: bulk_data_ptr cannot be a nullptr.");

    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(NodeEulerSphere::get_valid_fixed_params());

    // Store the valid entity parts for the kernel.
    Teuchos::Array<std::string> valid_entity_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    for (const std::string &part_name : valid_entity_part_names) {
      valid_entity_parts_.push_back(meta_data_ptr_->get_part(part_name));
      MUNDY_THROW_ASSERT(
          valid_entity_parts_.back() != nullptr, std::invalid_argument,
          "NodeEulerSphere: Part '" << part_name << "' from the valid_entity_part_names does not exist in the meta data.");
    }

    // Fetch the fields.
    const std::string node_coord_field_name = mundy::shape::shapes::Spheres::get_node_coord_field_name();
    const std::string node_velocity_field_name = valid_fixed_params.get<std::string>("node_velocity_field_name");

    node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
    node_velocity_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_velocity_field_name);
  }
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
      [[maybe_unused]] const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList()) {
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(NodeEulerSphere::get_valid_fixed_params());

    // Fill the requirements using the given parameter list.
    auto mesh_reqs_ptr = std::make_shared<mundy::meta::MeshRequirements>();
    std::string node_velocity_field_name = valid_fixed_params.get<std::string>("node_velocity_field_name");
    Teuchos::Array<std::string> valid_entity_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    const int num_parts = static_cast<int>(valid_entity_part_names.size());
    for (int i = 0; i < num_parts; i++) {
      const std::string part_name = valid_entity_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartRequirements>();
      part_reqs->set_part_name(part_name);
      part_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
          node_velocity_field_name, stk::topology::NODE_RANK, 3, 1));

      if (part_name == "SPHERES") {
        // Add the requirements directly to spheres part.
        const std::string parent_part_name = "SHAPES";
        mundy::agent::AgentHierarchy::add_part_reqs(part_reqs, part_name, parent_part_name);
        mesh_reqs_ptr->merge(mundy::agent::AgentHierarchy::get_mesh_requirements(part_name, parent_part_name));
      } else {
        // Add the associated part as a subset of the spheres part.
        const std::string parent_part_name = "SPHERES";
        mundy::agent::AgentHierarchy::add_subpart_reqs(part_reqs, part_name, parent_part_name);
        mesh_reqs_ptr->merge(mundy::agent::AgentHierarchy::get_mesh_requirements(part_name, parent_part_name));
      }
    }
    return mesh_reqs_ptr;
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set<Teuchos::Array<std::string>>(
        "valid_entity_part_names", Teuchos::tuple<std::string>(std::string(default_part_name_)),
        "Name of the parts associated with this kernel.");
    default_parameter_list.set(
        "node_velocity_field_name", std::string(default_node_velocity_field_name_),
        "Name of the node velocity field to be used for computing the node euler timestep of the sphere.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("time_step_size", default_time_step_size_, "The timestep size.");
    return default_parameter_list;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaKernel<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr,
      const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList()) {
    return std::make_shared<NodeEulerSphere>(bulk_data_ptr, fixed_params);
  }

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override {
    Teuchos::ParameterList valid_mutable_params = mutable_params;
    valid_mutable_params.validateParametersAndSetDefaults(NodeEulerSphere::get_valid_mutable_params());
    time_step_size_ = valid_mutable_params.get<double>("time_step_size");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get valid entity parts for the kernel.
  /// By "valid entity parts," we mean the parts whose entities the kernel can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override {
    return valid_entity_parts_;
  }

  /// \brief Get the entity rank that the kernel acts on.
  stk::topology::rank_t get_entity_rank() const override {
    return stk::topology::NODE_RANK;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Setup the kernel's core calculations.
  /// For example, communicate information to the GPU, populate ghosts, or zero out fields.
  void setup() override {
  }

  /// \brief Run the kernel's core calculation.
  /// \param sphere_node [in] The sphere's node acted on by the kernel.
  void execute(const stk::mesh::Entity &sphere_node) override {
    double *node_coords = stk::mesh::field_data(*node_coord_field_ptr_, sphere_node);
    double *node_velocity = stk::mesh::field_data(*node_velocity_field_ptr_, sphere_node);
    node_coords[0] += time_step_size_ * node_velocity[0];
    node_coords[1] += time_step_size_ * node_velocity[1];
    node_coords[2] += time_step_size_ * node_velocity[2];
  }

  /// \brief Finalize the kernel's core calculations.
  /// For example, communicate between ghosts, perform reductions over shared entities, or swap internal variables.
  void finalize() override {
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static inline double default_time_step_size_ = 0.0;
  static constexpr std::string_view default_part_name_ = "SPHERES";
  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
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

  /// \brief The valid entity parts for the kernel.
  std::vector<stk::mesh::Part *> valid_entity_parts_;

  /// \brief The numerical timestep size.
  double time_step_size_;

  /// \brief Node field containing the node's spatial coordinate.
  stk::mesh::Field<double> *node_coord_field_ptr_ = nullptr;

  /// \brief Node field containing the node's translational velocity.
  stk::mesh::Field<double> *node_velocity_field_ptr_ = nullptr;
  //@}
};  // NodeEulerSphere

/// @brief Register NodeEulerSphere with NodeEuler default kernels
MUNDY_REGISTER_METACLASS("SPHERE", NodeEulerSphere, NodeEuler::OurKernelFactory)

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

class ComputeBrownianVelocity
    : public mundy::meta::MetaKernelDispatcher<ComputeBrownianVelocity,
                                               mundy::meta::make_registration_string("COMPUTE_BROWNIAN_VELOCITY")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  ComputeBrownianVelocity() = delete;

  /// \brief Constructor
  ComputeBrownianVelocity(mundy::mesh::BulkData *const bulk_data_ptr,
                          const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList())
      : mundy::meta::MetaKernelDispatcher<ComputeBrownianVelocity,
                                          mundy::meta::make_registration_string("COMPUTE_BROWNIAN_VELOCITY")>(
            bulk_data_ptr, fixed_params) {
  }
  //@}


  //! \name MetaKernelDispatcher static interface implementation
  //@{

  /// \brief Get the valid fixed parameters that we require our techniques have.
  static Teuchos::ParameterList get_valid_required_kernel_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we require our techniques have.
  static Teuchos::ParameterList get_valid_required_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get the valid fixed parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("node_brownian_velocity_field_name",
                               std::string(default_node_brownian_velocity_field_name_),
                               "Name of the node broenian velocity field.");
    default_parameter_list.set("node_rng_counter_field_name", std::string(default_node_rng_counter_field_name_),
                               "Name of the node rng counter for generating new parallel streams.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we will forward to our kernels.
  static Teuchos::ParameterList get_valid_forwarded_kernel_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("time_step_size", default_time_step_size_, "The timestep size.");
    default_parameter_list.set("alpha", default_alpha_,
                               "Scale for the brownian velocity such that V = beta * V0 + alpha * Vnew.");
    default_parameter_list.set("beta", default_beta_,
                               "Scale for the brownian velocity such that V = beta * V0 + alpha * Vnew.");
    return default_parameter_list;
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static inline double default_time_step_size_ = 0.0;
  static constexpr double default_alpha_ = 1.0;
  static constexpr double default_beta_ = 0.0;
  static constexpr std::string_view default_node_brownian_velocity_field_name_ = "NODE_BROWNIAN_VELOCITY";
  static constexpr std::string_view default_node_rng_counter_field_name_ = "NODE_RNG_COUNTER";
  //@}
};  // ComputeBrownianVelocity

/// \class ComputeBrownianVelocitySphere
/// \brief Concrete implementation of \c MetaKernel for computing the node euler timestep of spheres.
class ComputeBrownianVelocitySphere : public mundy::meta::MetaKernel<void> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaKernel<void>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit ComputeBrownianVelocitySphere(mundy::mesh::BulkData *const bulk_data_ptr,
                                         const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList())
      : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
    // The bulk data pointer must not be null.
    MUNDY_THROW_ASSERT(bulk_data_ptr_ != nullptr, std::invalid_argument,
                       "ComputeBrownianVelocitySphere: bulk_data_ptr cannot be a nullptr.");

    // Validate the input params. Use default values for any parameter not given.
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(ComputeBrownianVelocitySphere::get_valid_fixed_params());

    // Store the valid entity parts for the kernel.
    Teuchos::Array<std::string> valid_entity_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    for (const std::string &part_name : valid_entity_part_names) {
      valid_entity_parts_.push_back(meta_data_ptr_->get_part(part_name));
      MUNDY_THROW_ASSERT(valid_entity_parts_.back() != nullptr, std::invalid_argument,
                         "ComputeBrownianVelocitySphere: Part '"
                             << part_name << "' from the valid_entity_part_names does not exist in the meta data.");
    }

    // Fetch the fields.
    const std::string node_brownian_velocity_field_name =
        valid_fixed_params.get<std::string>("node_brownian_velocity_field_name");
    const std::string node_rng_counter_field_name = valid_fixed_params.get<std::string>("node_rng_counter_field_name");

    node_brownian_velocity_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_brownian_velocity_field_name);
    node_rng_counter_field_ptr_ =
        meta_data_ptr_->get_field<unsigned>(stk::topology::NODE_RANK, node_rng_counter_field_name);
  }
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
      [[maybe_unused]] const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList()) {
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(ComputeBrownianVelocitySphere::get_valid_fixed_params());

    // Fill the requirements using the given parameter list.
    auto mesh_reqs_ptr = std::make_shared<mundy::meta::MeshRequirements>();
    std::string node_brownian_velocity_field_name =
        valid_fixed_params.get<std::string>("node_brownian_velocity_field_name");
    std::string node_rng_counter_field_name = valid_fixed_params.get<std::string>("node_rng_counter_field_name");
    Teuchos::Array<std::string> valid_entity_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    const int num_parts = static_cast<int>(valid_entity_part_names.size());
    for (int i = 0; i < num_parts; i++) {
      const std::string part_name = valid_entity_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartRequirements>();
      part_reqs->set_part_name(part_name);
      part_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<double>>(
          node_brownian_velocity_field_name, stk::topology::NODE_RANK, 3, 1));
      part_reqs->add_field_reqs(std::make_shared<mundy::meta::FieldRequirements<unsigned>>(
          node_rng_counter_field_name, stk::topology::NODE_RANK, 1, 1));

      if (part_name == "SPHERES") {
        // Add the requirements directly to spheres part.
        const std::string parent_part_name = "SHAPES";
        mundy::agent::AgentHierarchy::add_part_reqs(part_reqs, part_name, parent_part_name);
        mesh_reqs_ptr->merge(mundy::agent::AgentHierarchy::get_mesh_requirements(part_name, parent_part_name));
      } else {
        // Add the associated part as a subset of the spheres part.
        const std::string parent_part_name = "SPHERES";
        mundy::agent::AgentHierarchy::add_subpart_reqs(part_reqs, part_name, parent_part_name);
        mesh_reqs_ptr->merge(mundy::agent::AgentHierarchy::get_mesh_requirements(part_name, parent_part_name));
      }
    }
    return mesh_reqs_ptr;
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set<Teuchos::Array<std::string>>(
        "valid_entity_part_names", Teuchos::tuple<std::string>(std::string(default_part_name_)),
        "Name of the parts associated with this kernel.");
    default_parameter_list.set(
        "node_brownian_velocity_field_name", std::string(default_node_brownian_velocity_field_name_),
        "Name of the node velocity field to be used for computing the node euler timestep of the sphere.");
    default_parameter_list.set(
        "node_rng_counter_field_name", std::string(default_node_rng_counter_field_name_),
        "Name of the node velocity field to be used for computing the node euler timestep of the sphere.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("time_step_size", default_time_step_size_, "The timestep size.");
    default_parameter_list.set("diffusion_coeff", default_diffusion_coeff_, "The diffusion coefficient.");
    default_parameter_list.set("alpha", default_alpha_,
                               "Scale for the brownian velocity such that V = beta * V0 + alpha * Vnew.");
    default_parameter_list.set("beta", default_beta_,
                               "Scale for the brownian velocity such that V = beta * V0 + alpha * Vnew.");
    return default_parameter_list;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaKernel<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr,
      const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList()) {
    return std::make_shared<ComputeBrownianVelocitySphere>(bulk_data_ptr, fixed_params);
  }

  //! \name Setters
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override {
    Teuchos::ParameterList valid_mutable_params = mutable_params;
    valid_mutable_params.validateParametersAndSetDefaults(ComputeBrownianVelocitySphere::get_valid_mutable_params());
    time_step_size_ = valid_mutable_params.get<double>("time_step_size");
    diffusion_coeff_ = valid_mutable_params.get<double>("diffusion_coeff");
    alpha_ = valid_mutable_params.get<double>("alpha");
    beta_ = valid_mutable_params.get<double>("beta");
  }
  //@}

  //! \name Getters
  //@{

  /// \brief Get valid entity parts for the kernel.
  /// By "valid entity parts," we mean the parts whose entities the kernel can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override {
    return valid_entity_parts_;
  }

  /// \brief Get the entity rank that the kernel acts on.
  stk::topology::rank_t get_entity_rank() const override {
    return stk::topology::NODE_RANK;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Setup the kernel's core calculations.
  /// For example, communicate information to the GPU, populate ghosts, or zero out fields.
  void setup() override {
  }

  /// \brief Run the kernel's core calculation.
  /// \param sphere_node [in] The sphere's node acted on by the kernel.
  void execute(const stk::mesh::Entity &sphere_node) override {
    double *node_brownian_velocity = stk::mesh::field_data(*node_brownian_velocity_field_ptr_, sphere_node);
    const stk::mesh::EntityId sphere_node_gid = bulk_data_ptr_->identifier(sphere_node);
    unsigned *node_rng_counter = stk::mesh::field_data(*node_rng_counter_field_ptr_, sphere_node);

    openrand::Philox rng(sphere_node_gid, node_rng_counter[0]);
    node_brownian_velocity[0] = alpha_ * std::sqrt(2.0 * diffusion_coeff_ * time_step_size_) * rng.randn<double>() +
                                beta_ * node_brownian_velocity[0];
    node_brownian_velocity[1] = alpha_ * std::sqrt(2.0 * diffusion_coeff_ * time_step_size_) * rng.randn<double>() +
                                beta_ * node_brownian_velocity[1];
    node_brownian_velocity[2] = alpha_ * std::sqrt(2.0 * diffusion_coeff_ * time_step_size_) * rng.randn<double>() +
                                beta_ * node_brownian_velocity[2];
    node_rng_counter[0]++;

    // // Can't use Openrand until they fix their DEVICE define
    // node_brownian_velocity[0] =
    //     alpha_ * std::sqrt(2.0 * diffusion_coeff_ * time_step_size_) * ((double)rand() / (RAND_MAX)) +
    //     beta_ * node_brownian_velocity[0];
    // node_brownian_velocity[1] =
    //     alpha_ * std::sqrt(2.0 * diffusion_coeff_ * time_step_size_) * ((double)rand() / (RAND_MAX)) +
    //     beta_ * node_brownian_velocity[1];
    // node_brownian_velocity[2] =
    //     alpha_ * std::sqrt(2.0 * diffusion_coeff_ * time_step_size_) * ((double)rand() / (RAND_MAX)) +
    //     beta_ * node_brownian_velocity[2];
  }

  /// \brief Finalize the kernel's core calculations.
  /// For example, communicate between ghosts, perform reductions over shared entities, or swap internal variables.
  void finalize() override {
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static inline double default_time_step_size_ = 0.0;
  static inline double default_diffusion_coeff_ = 0.0;
  static constexpr double default_alpha_ = 1.0;
  static constexpr double default_beta_ = 0.0;
  static constexpr std::string_view default_part_name_ = "SPHERES";
  static constexpr std::string_view default_node_brownian_velocity_field_name_ = "NODE_BROWNIAN_VELOCITY";
  static constexpr std::string_view default_node_rng_counter_field_name_ = "NODE_RNG_COUNTER";
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

  /// \brief The valid entity parts for the kernel.
  std::vector<stk::mesh::Part *> valid_entity_parts_;

  /// \brief The numerical timestep size.
  double time_step_size_;

  /// \brief The diffusion coefficient.
  double diffusion_coeff_;

  /// \brief Scale for the brownian velocity such that V = beta * V0 + alpha * Vnew.
  double alpha_;

  /// \brief Scale for the brownian velocity such that V = beta * V0 + alpha * Vnew.
  double beta_;

  /// \brief Node field containing the node's translational velocity.
  stk::mesh::Field<double> *node_brownian_velocity_field_ptr_ = nullptr;

  /// \brief Node field containing the node's random number generator counter.
  stk::mesh::Field<unsigned> *node_rng_counter_field_ptr_ = nullptr;
  //@}
};  // ComputeBrownianVelocitySphere

/// @brief Register ComputeBrownianVelocitySphere with ComputeBrownianVelocity default kernels
MUNDY_REGISTER_METACLASS("SPHERE", ComputeBrownianVelocitySphere, ComputeBrownianVelocity::OurKernelFactory)

int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  // Parse the inputs
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " <number_of_particles> <length_of_domain> <num_time_steps> <time_step_size> <diffusion_coeff>"
              << std::endl;
    return 1;
  }

  const int number_of_particles = std::stoi(argv[1]);
  const double length_of_domain = std::stod(argv[2]);
  const int num_time_steps = std::stoi(argv[3]);
  const double time_step_size = std::stod(argv[4]);
  const double diffusion_coeff = std::stod(argv[5]);

  mundy::shape::perform_registration();  // TODO(palmerb4): I hate this. Mundy should be restructured to not require
                                         // this. Shapes should be a subpackage of agents.

  /////////////////////
  // Create the mesh //
  /////////////////////
  auto mesh_reqs_ptr = std::make_shared<mundy::meta::MeshRequirements>(MPI_COMM_WORLD);
  mesh_reqs_ptr->set_spatial_dimension(3);
  mesh_reqs_ptr->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  Teuchos::ParameterList compute_brownian_velocity_fixed_params;
  compute_brownian_velocity_fixed_params.sublist("kernels").set("count", 1);
  compute_brownian_velocity_fixed_params.sublist("kernels").sublist("kernel_0").set("name", "SPHERES");
  compute_brownian_velocity_fixed_params.sublist("kernels")
      .sublist("kernel_0")
      .set("valid_entity_part_names", Teuchos::Array<std::string>(1, "SPHERES"));
  compute_brownian_velocity_fixed_params.sublist("kernels")
      .sublist("kernel_0")
      .set("node_rng_counter_field_name", "NODE_RNG_COUNTER");
  compute_brownian_velocity_fixed_params.sublist("kernels")
      .sublist("kernel_0")
      .set("node_brownian_velocity_field_name", "NODE_BROWNIAN_VELOCITY");
  mesh_reqs_ptr->merge(ComputeBrownianVelocity::get_mesh_requirements(compute_brownian_velocity_fixed_params));

  Teuchos::ParameterList node_euler_fixed_params;
  node_euler_fixed_params.sublist("kernels").set("count", 1);
  node_euler_fixed_params.sublist("kernels").sublist("kernel_0").set("name", "SPHERES");
  node_euler_fixed_params.sublist("kernels")
      .sublist("kernel_0")
      .set("valid_entity_part_names", Teuchos::Array<std::string>(1, "SPHERES"));
  node_euler_fixed_params.sublist("kernels").sublist("kernel_0").set("node_velocity_field_name", "NODE_VELOCITY");
  mesh_reqs_ptr->merge(NodeEuler::get_mesh_requirements(node_euler_fixed_params));

  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_reqs_ptr->declare_mesh();
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  meta_data_ptr->use_simple_fields();

  meta_data_ptr->commit();

  // Create the meta classes
  auto compute_brownian_velocity_ptr =
      ComputeBrownianVelocity::create_new_instance(bulk_data_ptr.get(), compute_brownian_velocity_fixed_params);
  Teuchos::ParameterList compute_brownian_velocity_mutable_params;
  compute_brownian_velocity_mutable_params.sublist("kernels").set("count", 1);
  compute_brownian_velocity_mutable_params.sublist("kernels").sublist("kernel_0").set("name", "SPHERES");
  compute_brownian_velocity_mutable_params.sublist("kernels")
      .sublist("kernel_0")
      .set("valid_entity_part_names", Teuchos::Array<std::string>(1, "SPHERES"));
  compute_brownian_velocity_mutable_params.sublist("kernels")
      .sublist("kernel_0")
      .set("diffusion_coeff", diffusion_coeff);
  compute_brownian_velocity_mutable_params.sublist("kernels").sublist("kernel_0").set("time_step_size", time_step_size);
  compute_brownian_velocity_ptr->set_mutable_params(compute_brownian_velocity_mutable_params);

  auto node_euler_ptr = NodeEuler::create_new_instance(bulk_data_ptr.get(), node_euler_fixed_params);
  Teuchos::ParameterList node_euler_mutable_params;
  node_euler_mutable_params.sublist("kernels").set("count", 1);
  node_euler_mutable_params.sublist("kernels").sublist("kernel_0").set("name", "SPHERES");
  node_euler_mutable_params.sublist("kernels")
      .sublist("kernel_0")
      .set("valid_entity_part_names", Teuchos::Array<std::string>(1, "SPHERES"));
  node_euler_mutable_params.sublist("kernels").sublist("kernel_0").set("time_step_size", time_step_size);
  node_euler_ptr->set_mutable_params(node_euler_mutable_params);

  ////////////////////////////////////
  // Generate the spheres and nodes //
  ////////////////////////////////////
  int num_spheres_local = number_of_particles / bulk_data_ptr->parallel_size();
  const int remaining_spheres = number_of_particles - num_spheres_local * bulk_data_ptr->parallel_size();
  if (bulk_data_ptr->parallel_rank() < remaining_spheres) {
    num_spheres_local += 1;
  }
  int num_nodes_local = num_spheres_local;

  stk::mesh::Part *spheres_part_ptr = meta_data_ptr->get_part("SPHERES");
  MUNDY_THROW_ASSERT(spheres_part_ptr != nullptr, std::invalid_argument, "SPHERES part not found.");
  stk::mesh::Part &spheres_part = *spheres_part_ptr;

  bulk_data_ptr->modification_begin();
  std::vector<size_t> requests(meta_data_ptr->entity_rank_count(), 0);
  requests[stk::topology::NODE_RANK] = num_nodes_local;
  requests[stk::topology::ELEMENT_RANK] = num_spheres_local;

  std::vector<stk::mesh::Entity> requested_entities;
  bulk_data_ptr->generate_new_entities(requests, requested_entities);

  // Associate each segments with the sphere part and connect them to their nodes.
  std::vector<stk::mesh::Part *> add_spheres_part = {spheres_part_ptr};
  for (int i = 0; i < num_spheres_local; i++) {
    stk::mesh::Entity sphere_i = requested_entities[num_nodes_local + i];
    stk::mesh::Entity node_i = requested_entities[i];
    bulk_data_ptr->change_entity_parts(sphere_i, add_spheres_part);
    bulk_data_ptr->declare_relation(sphere_i, node_i, 0);
  }
  bulk_data_ptr->modification_end();

  // Start the particles at random positions with zero velocity
  // Typically, this would occur in a requirements wrapped class, but we haven't created the declare or initialize shape
  // functions.
  auto node_coordinates_field_ptr = meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDINATES");
  auto node_velocity_field_ptr = meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_VELOCITY");
  auto node_rng_counter_field_ptr = meta_data_ptr->get_field<unsigned>(stk::topology::NODE_RANK, "NODE_RNG_COUNTER");

  auto check_if_exists = [](const stk::mesh::FieldBase *const field_ptr, const std::string &name) {
    MUNDY_THROW_ASSERT(field_ptr != nullptr, std::invalid_argument,
                       name + "cannot be a nullptr. Check that the field exists.");
  };

  check_if_exists(node_coordinates_field_ptr, "NODE_COORDINATES");
  check_if_exists(node_velocity_field_ptr, "NODE_VELOCITY");
  check_if_exists(node_rng_counter_field_ptr, "NODE_RNG_COUNTER");

  openrand::Philox rng(bulk_data_ptr->parallel_rank(), 0);
  for (int i = 0; i < num_spheres_local; i++) {
    stk::mesh::Entity node_i = requested_entities[i];
    double *node_coords = stk::mesh::field_data(*node_coordinates_field_ptr, node_i);
    double *node_velocity = stk::mesh::field_data(*node_velocity_field_ptr, node_i);
    unsigned *node_rng_counter = stk::mesh::field_data(*node_rng_counter_field_ptr, node_i);
    node_coords[0] = length_of_domain * rng.rand<double>();
    node_coords[1] = length_of_domain * rng.rand<double>();
    node_coords[2] = length_of_domain * rng.rand<double>();
    // node_coords[0] = length_of_domain * ((double)rand() / (RAND_MAX));
    // node_coords[1] = length_of_domain * ((double)rand() / (RAND_MAX));
    // node_coords[2] = length_of_domain * ((double)rand() / (RAND_MAX));
    node_velocity[0] = 0.0;
    node_velocity[1] = 0.0;
    node_velocity[2] = 0.0;
    node_rng_counter[0] = 0;
  }

  // Run the simulation
  // Keep time using Kokkos::Timer
  Kokkos::Timer timer;
  for (int i = 0; i < num_time_steps; i++) {
    compute_brownian_velocity_ptr->execute(spheres_part);
    node_euler_ptr->execute(spheres_part);
  }

  // Do a synchronize to force everybody to stop here, then write the time
  stk::parallel_machine_barrier(bulk_data_ptr->parallel());

  if (bulk_data_ptr->parallel_rank() == 0) {
    double timesteps_per_second = num_time_steps / timer.seconds();
    std::cout << "Performance: " << timesteps_per_second << std::endl;
  }

  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}
