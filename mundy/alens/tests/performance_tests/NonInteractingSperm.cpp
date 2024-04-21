// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
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

The goal of this example is to simulate the swimming motion of a multiple, non-interacting long sperm.
*/

// External libs
#include <openrand/philox.h>

// Trilinos libs
#include <Kokkos_Core.hpp>                   // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer
#include <Teuchos_CommandLineProcessor.hpp>  // for Teuchos::CommandLineProcessor
#include <Teuchos_ParameterList.hpp>         // for Teuchos::ParameterList
#include <stk_balance/balance.hpp>           // for stk::balance::balanceStkMesh, stk::balance::BalanceSettings
#include <stk_io/StkMeshIoBroker.hpp>        // for stk::io::StkMeshIoBroker
#include <stk_mesh/base/CreateEdges.hpp>     // for stk::mesh::create_edges
#include <stk_mesh/base/DumpMeshInfo.hpp>    // for stk::mesh::impl::dump_all_mesh_info
#include <stk_mesh/base/Entity.hpp>          // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>   // for stk::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>            // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>        // for stk::mesh::Selector
#include <stk_topology/topology.hpp>         // for stk::topology
#include <stk_util/parallel/Parallel.hpp>    // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy libs
#include <mundy_core/MakeStringArray.hpp>  // for mundy::core::make_string_array
#include <mundy_core/throw_assert.hpp>     // for MUNDY_THROW_ASSERT
#include <mundy_math/Matrix3.hpp>          // for mundy::math::Matrix3
#include <mundy_math/Quaternion.hpp>       // for mundy::math::Quaternion
#include <mundy_math/Vector3.hpp>          // for mundy::math::Vector3
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>       // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/MetaData.hpp>         // for mundy::mesh::MetaData
#include <mundy_mesh/utils/FillFieldWithValue.hpp>  // for mundy::mesh::utils::fill_field_with_value
#include <mundy_meta/PartRequirements.hpp>          // for mundy::meta::PartRequirements

/// \brief The main function for the sperm simulation broken down into digestible chunks.
///
/// This class is in charge of setting up and executing the following tasks. In the future, this will all be handled by
/// the Configurator.
///   // Preprocess
///   - Parse user parameter
///
///   // Setup
///   - Declare the fixed and mutable params for the desired MetaMethods
///   - Construct the mesh and the method instances
///   - Declare and initialize the sperm's nodes, edges, and elements (centerline twist springs)
///    (Using BulkData's declare_node, declare_element, create_edges functions)
///
///   // Timeloop
///   - Run the timeloop for t in range(0, T):
///       // IO.
///       - If desired, write out the data for time t
///         (Using stk::io::StkMeshIoBroker)
///
///       // Prepare the current configuration.
///       - Rotate the field states
///         (Using BulkData's update_field_data_states function)
///
///       - Zero the node forces and velocities for time t + dt
///         (Using mundy's fill_field_with_value function)
///
///       // Motion from t -> t + dt:
///        - Apply velocity/acceleration constraints like no motion for particle 1
///         (By directly looping over all nodes and setting the velocity/acceleration to zero)
///
///        - Evaluate x(t + dt) = x(t) + v(t) * dt + a(t) * dt^2 / 2
///         (By looping over all nodes and updating the coordinates)
///
///       // Evaluate forces f(x(t + dt))
///       {
///         // Centerline twist rod model
///         - Compute the edge information (length, tangent, and binormal)
///          (By looping over all edges and computing the edge length, tangent, binormal)
///
///         - Compute the node curvature and rotation gradient
///          (By looping over all centerline twist spring elements and computing the curvature & rotation gradient at
///          their center node using the edge orientations)
///
///         - Compute the internal force and twist torque
///          (By looping over all centerline twist spring elements, using the curvature to compute the induced
///          benching/twisting torque and stretch force, and using them to compute the force and twist-torque on the
///          nodes. .)
///       }
///
///       // Compute velocity and acceleration
///        - Evaluate a(t + dt) = M^{-1} f(x(t + dt))
///         (By looping over all nodes and computing the node acceleration and twist acceleration from the force and
///         twist torque)
///
///        - Evaluate v(t + dt) = v(t) + (a(t) + a(t + dt)) * dt / 2
///         (By looping over all nodes and updating the node velocity and twist rate using the corresponding
///         accelerations)
///
/// The sperm themselves are modeled as a chain of rods with a centerline twist spring connecting pairs of adjacent
/// edges:
/*
/// n1       n3        n5        n7
///  \      /  \      /  \      /
///   s1   s2   s3   s4   s5   s6
///    \  /      \  /      \  /
///     n2        n4        n6
*/
/// The centerline twist springs are hard to draw with ASCII art, but they are centered at every interior node and
/// connected to the node's neighbors:
///   c1 has a center node at n2 and connects to n1 and n3.
///   c2 has a center node at n3 and connects to n2 and n4.
///   and so on.
///
/// STK EntityId-wise. Nodes are numbered sequentially from 1 to num_nodes. Centerline twist springs are numbered
/// sequentially from 1 to num_nodes-2.
class SpermSimulation {
 public:
  SpermSimulation() = default;

  void print_rank0(auto think_to_print, int indent_level = 0) {
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::string indent(indent_level * 2, ' ');
      std::cout << indent << think_to_print << std::endl;
    }
  }

  void debug_print([[maybe_unused]] auto thing_to_print, [[maybe_unused]] int indent_level = 0) {
#ifdef DEBUG
    print_rank0(thing_to_print, indent_level);
#endif
  }

  bool parse_user_inputs(int argc, char **argv) {
    debug_print("Parsing user inputs.");

    // Parse the command line options.
    Teuchos::CommandLineProcessor cmdp(false, true);

    // Optional command line arguments for controlling
    //   Sperm initialization:
    cmdp.setOption("num_sperm", &num_sperm_, "Number of sperm.");
    cmdp.setOption("num_nodes_per_sperm", &num_nodes_per_sperm_, "Number of nodes per sperm.");
    cmdp.setOption("sperm_radius", &sperm_radius_, "The radius of each sperm.");
    cmdp.setOption("sperm_initial_segment_length", &sperm_initial_segment_length_, "Initial sperm segment length.");
    cmdp.setOption("sperm_rest_segment_length", &sperm_rest_segment_length_, "Rest sperm segment length.");
    cmdp.setOption("sperm_rest_curvature_twist", &sperm_rest_curvature_twist_, "Rest curvature (twist) of the sperm.");
    cmdp.setOption("sperm_rest_curvature_bend1", &sperm_rest_curvature_bend1_,
                   "Rest curvature (bend along the first coordinate direction) of the sperm.");
    cmdp.setOption("sperm_rest_curvature_bend2", &sperm_rest_curvature_bend2_,
                   "Rest curvature (bend along the second coordinate direction) of the sperm.");

    cmdp.setOption("sperm_density", &sperm_density_, "Density of the sperm.");
    cmdp.setOption("sperm_youngs_modulus", &sperm_youngs_modulus_, "Young's modulus of the sperm.");
    cmdp.setOption("sperm_poissons_ratio", &sperm_poissons_ratio_, "Poisson's ratio of the sperm.");
    cmdp.setOption("sperm_density", &sperm_density_, "Density of the sperm.");

    //   The simulation:
    cmdp.setOption("num_time_steps", &num_time_steps_, "Number of time steps.");
    cmdp.setOption("timestep_size", &timestep_size_, "Time step size.");
    cmdp.setOption("skin_distance", &skin_distance_, "Skin distance for neighbor detection. \n As long as no particle moves more than 
    one skin distance, the current neighbor list is valid. This is used to avoid the expensive neighbor detection step.");
    cmdp.setOption("io_frequency", &io_frequency_, "Number of timesteps between writing output.");


    bool was_parse_successful = cmdp.parse(argc, argv) == Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL;
    return was_parse_successful;
  }

  void check_input_parameters() {
    debug_print("Checking input parameters.");
    MUNDY_THROW_ASSERT(num_sperm_ > 0, std::invalid_argument, "num_sperm_ must be greater than 0.");
    MUNDY_THROW_ASSERT(num_nodes_per_sperm_ > 0, std::invalid_argument, "num_nodes_per_sperm_ must be greater than 0.");
    MUNDY_THROW_ASSERT(sperm_radius_ > 0, std::invalid_argument, "sperm_radius_ must be greater than 0.");
    MUNDY_THROW_ASSERT(sperm_initial_segment_length_ > -1e-12, std::invalid_argument,
                       "sperm_initial_segment_length_ must be greater than or equal to 0.");
    MUNDY_THROW_ASSERT(sperm_rest_segment_length_ > -1e-12, std::invalid_argument,
                       "sperm_rest_segment_length_ must be greater than or equal to 0.");
    MUNDY_THROW_ASSERT(sperm_youngs_modulus_ > 0, std::invalid_argument,
                       "sperm_youngs_modulus_ must be greater than 0.");
    MUNDY_THROW_ASSERT(sperm_poissons_ratio_ > 0, std::invalid_argument,
                       "sperm_poissons_ratio_ must be greater than 0.");

    MUNDY_THROW_ASSERT(num_time_steps_ > 0, std::invalid_argument, "num_time_steps_ must be greater than 0.");
    MUNDY_THROW_ASSERT(timestep_size_ > 0, std::invalid_argument, "timestep_size_ must be greater than 0.");
    MUNDY_THROW_ASSERT(skin_distance_ > -1e-12, std::invalid_argument,
                       "skin_distance_ must be greater than or equal to 0.");
  }

  void dump_user_inputs() {
    debug_print("Dumping user inputs.");
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::cout << "##################################################" << std::endl;
      std::cout << "INPUT PARAMETERS:" << std::endl;
      std::cout << "  num_sperm_: " << num_sperm_ << std::endl;
      std::cout << "  num_nodes_per_sperm_: " << num_nodes_per_sperm_ << std::endl;
      std::cout << "  sperm_radius_: " << sperm_radius_ << std::endl;
      std::cout << "  sperm_initial_segment_length_: " << sperm_initial_segment_length_ << std::endl;
      std::cout << "  sperm_rest_segment_length_: " << sperm_rest_segment_length_ << std::endl;
      std::cout << "  sperm_rest_curvature_twist_: " << sperm_rest_curvature_twist_ << std::endl;
      std::cout << "  sperm_rest_curvature_bend1_: " << sperm_rest_curvature_bend1_ << std::endl;
      std::cout << "  sperm_rest_curvature_bend2_: " << sperm_rest_curvature_bend2_ << std::endl;
      std::cout << "  sperm_youngs_modulus_: " << sperm_youngs_modulus_ << std::endl;
      std::cout << "  sperm_poissons_ratio_: " << sperm_poissons_ratio_ << std::endl;
      std::cout << "  sperm_density_: " << sperm_density_ << std::endl;
      std::cout << "  num_time_steps_: " << num_time_steps_ << std::endl;
      std::cout << "  timestep_size_: " << timestep_size_ << std::endl;
      std::cout << "  skin_distance_: " << skin_distance_ << std::endl;
      std::cout << "##################################################" << std::endl;
    }
  }

  void build_our_mesh_and_method_instances() {
    debug_print("Building our mesh and method instances.");

    // Setup the mesh requirements.
    // First, we need to fetch the mesh requirements for each method, then we can create the class instances.
    // In the future, all of this will be done via the Configurator.
    mesh_reqs_ptr_->set_spatial_dimension(3);
    mesh_reqs_ptr_->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

    // Add custom requirements for this example. These are requirements that exceed those of the enabled methods and
    // allow us to extend the functionality offered natively by Mundy.
    //
    // We require that the centerline twist springs part exists, has a BEAM_3 topology, and has the desired
    // node/edge/element fields.
    auto clt_part_reqs = std::make_shared<mundy::meta::PartRequirements>()
                             ->set_part_name("CENTERLINE_TWIST_SPRINGS")
                             .set_part_topology(stk::topology::BEAM_3)

                             // Add the node fields
                             .add_field_reqs<double>("NODE_COORDINATES", node_rank_, 3, 2)
                             .add_field_reqs<double>("NODE_VELOCITY", node_rank_, 3, 2)
                             .add_field_reqs<double>("NODE_FORCE", node_rank_, 3, 1)
                             .add_field_reqs<double>("NODE_ACCELERATION", node_rank_, 3, 2)

                             .add_field_reqs<double>("NODE_TWIST", node_rank_, 1, 2)
                             .add_field_reqs<double>("NODE_TWIST_VELOCITY", node_rank_, 1, 2)
                             .add_field_reqs<double>("NODE_TWIST_TORQUE", node_rank_, 1, 1)
                             .add_field_reqs<double>("NODE_TWIST_ACCELERATION", node_rank_, 1, 2)

                             .add_field_reqs<double>("NODE_CURVATURE", node_rank_, 3, 1)
                             .add_field_reqs<double>("NODE_REST_CURVATURE", node_rank_, 3, 1)
                             .add_field_reqs<double>("NODE_ROTATION_GRADIENT", node_rank_, 4, 1)

                             // Add the edge fields
                             .add_field_reqs<double>("EDGE_ORIENTATION", edge_rank_, 4, 2)
                             .add_field_reqs<double>("EDGE_TANGENT", edge_rank_, 3, 2)
                             .add_field_reqs<double>("EDGE_BINORMAL", edge_rank_, 3, 1)
                             .add_field_reqs<double>("EDGE_LENGTH", edge_rank_, 1, 1)

                             // Add the element fields
                             .add_field_reqs<double>("ELEMENT_RADIUS", element_rank_, 1, 1)
                             .add_field_reqs<double>("ELEMENT_YOUNGS_MODULUS", element_rank_, 1, 1)
                             .add_field_reqs<double>("ELEMENT_POISSONS_RATIO", element_rank_, 1, 1)
                             .add_field_reqs<double>("ELEMENT_REST_LENGTH", element_rank_, 1, 1);
    mesh_reqs_ptr_->add_part_reqs(clt_part_reqs);

    // The mesh requirements are now set up, so we solidify the mesh structure.
    bulk_data_ptr_ = mesh_reqs_ptr_->declare_mesh();
    meta_data_ptr_ = bulk_data_ptr_->mesh_meta_data_ptr();
    meta_data_ptr_->set_coordinate_field_name("NODE_COORDINATES");
    meta_data_ptr_->commit();
  }

  template <typename FieldType>
  stk::mesh::Field<FieldType> *fetch_field(const std::string &field_name, stk::topology::rank_t rank) {
    auto field_ptr = meta_data_ptr_->get_field<FieldType>(rank, field_name);
    MUNDY_THROW_ASSERT(field_ptr != nullptr, std::invalid_argument,
                       "Field " << field_name << " not found in the mesh meta data.");
    return field_ptr;
  }

  stk::mesh::Part *fetch_part(const std::string &part_name) {
    auto part_ptr = meta_data_ptr_->get_part(part_name);
    MUNDY_THROW_ASSERT(part_ptr != nullptr, std::invalid_argument,
                       "Part " << part_name << " not found in the mesh meta data.");
    return part_ptr;
  }

  void fetch_fields_and_parts() {
    debug_print("Fetching fields and parts.");

    // Fetch the fields
    node_coordinates_field_ptr_ = fetch_field<double>("NODE_COORDINATES", node_rank_);
    node_velocity_field_ptr_ = fetch_field<double>("NODE_VELOCITY", node_rank_);
    node_force_field_ptr_ = fetch_field<double>("NODE_FORCE", node_rank_);
    node_acceleration_field_ptr_ = fetch_field<double>("NODE_ACCELERATION", node_rank_);
    node_twist_field_ptr_ = fetch_field<double>("NODE_TWIST", node_rank_);
    node_twist_velocity_field_ptr_ = fetch_field<double>("NODE_TWIST_VELOCITY", node_rank_);
    node_twist_torque_field_ptr_ = fetch_field<double>("NODE_TWIST_TORQUE", node_rank_);
    node_twist_acceleration_field_ptr_ = fetch_field<double>("NODE_TWIST_ACCELERATION", node_rank_);
    node_curvature_field_ptr_ = fetch_field<double>("NODE_CURVATURE", node_rank_);
    node_rest_curvature_field_ptr_ = fetch_field<double>("NODE_REST_CURVATURE", node_rank_);
    node_rotation_gradient_field_ptr_ = fetch_field<double>("NODE_ROTATION_GRADIENT", node_rank_);
    node_radius_field_ptr_ = fetch_field<double>("NODE_RADIUS", node_rank_);

    edge_orientation_field_ptr_ = fetch_field<double>("EDGE_ORIENTATION", edge_rank_);
    edge_tangent_field_ptr_ = fetch_field<double>("EDGE_TANGENT", edge_rank_);
    edge_binormal_field_ptr_ = fetch_field<double>("EDGE_BINORMAL", edge_rank_);
    edge_length_field_ptr_ = fetch_field<double>("EDGE_LENGTH", edge_rank_);

    element_radius_field_ptr_ = fetch_field<double>("ELEMENT_RADIUS", element_rank_);
    element_youngs_modulus_field_ptr_ = fetch_field<double>("ELEMENT_YOUNGS_MODULUS", element_rank_);
    element_poissons_ratio_field_ptr_ = fetch_field<double>("ELEMENT_POISSONS_RATIO", element_rank_);
    element_rest_length_field_ptr_ = fetch_field<double>("ELEMENT_REST_LENGTH", element_rank_);

    // Fetch the parts
    centerline_twist_springs_part_ptr_ = fetch_part("CENTERLINE_TWIST_SPRINGS");
  }

  void setup_io() {
    debug_print("Setting up IO.");

    // Declare each part as an IO part
    stk::io::put_io_part_attribute(*centerline_twist_springs_part_ptr_);

    // Setup the IO broker
    stk_io_broker_.use_simple_fields();
    stk_io_broker_.set_bulk_data(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr_.get()));

    output_file_index_ = stk_io_broker_.create_output_mesh("Sperm.exo", stk::io::WRITE_RESULTS);
    stk_io_broker_.add_field(output_file_index_, *node_coordinates_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_velocity_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_force_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_acceleration_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_twist_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_twist_velocity_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_twist_torque_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_twist_acceleration_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_curvature_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_rest_curvature_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_rotation_gradient_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *node_radius_field_ptr_);

    stk_io_broker_.add_field(output_file_index_, *edge_orientation_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *edge_tangent_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *edge_binormal_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *edge_length_field_ptr_);

    stk_io_broker_.add_field(output_file_index_, *element_radius_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *element_youngs_modulus_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *element_poissons_ratio_field_ptr_);
    stk_io_broker_.add_field(output_file_index_, *element_rest_length_field_ptr_);
  }

  void declare_and_initialize_sperm() {
    debug_print("Declaring and initializing the sperm.");

    // Declare N spring chains with a slight shift to each chain
    for (size_t j = 0; j < num_sperm_; j++) {
      mundy::math::Vector3<double> tail_coord(0.0, 2 * j * sperm_radius_, 0.0);
      mundy::math::Vector3<double> sperm_axis(1.0, 0.0, 0.0);

      // Because we are creating multiple sperm, we need to determine the node and element index ranges for each sperm.
      size_t start_node_id = num_nodes_per_sperm_ * j + 1u;
      size_t start_edge_id = (num_nodes_per_sperm_ - 1) * j + 1u;
      size_t start_centerline_twist_spring_id = (num_nodes_per_sperm_ - 2) * j + 1u;

      auto get_node_id = [start_node_id](const size_t &seq_node_index) {
        return start_node_id + seq_node_index;
      };

      auto get_node = [this](const size_t &seq_node_index) {
        return bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, get_node_id(seq_node_index));
      };

      auto get_edge_id = [start_edge_id](const size_t &seq_node_index) {
        return start_edge_id + seq_node_index;
      };

      auto get_edge = [this](const size_t &seq_node_index) {
        return bulk_data_ptr_->get_entity(stk::topology::EDGE_RANK, get_edge_id(seq_node_index));
      };

      auto get_centerline_twist_spring_id = [start_centerline_twist_spring_id](const size_t &seq_spring_index) {
        return start_centerline_twist_spring_id + seq_spring_index;
      };

      auto get_centerline_twist_spring = [this](const size_t &seq_spring_index) {
        return bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK,
                                          get_centerline_twist_spring_id(seq_spring_index));
      };

      // Create the springs and their connected nodes, distributing the work across the ranks.
      const size_t rank = bulk_data_ptr_->parallel_rank();
      const size_t nodes_per_rank = num_nodes_ / bulk_data_ptr_->parallel_size();
      const size_t remainder = num_nodes_ % bulk_data_ptr_->parallel_size();
      const size_t start_seq_node_index = rank * nodes_per_rank + std::min(rank, remainder);
      const size_t end_seq_node_index = start_seq_node_index + nodes_per_rank + (rank < remainder ? 1 : 0);

      bulk_data_ptr_->modification_begin();
      for (size_t i = start_seq_node_index; i < end_seq_node_index; ++i) {
        // Create the node.
        stk::mesh::EntityId our_node_id = get_node_id(i);
        stk::mesh::Entity node = bulk_data_ptr_->declare_node(our_node_id);
        bulk_data_ptr_->change_entity_parts(node, stk::mesh::PartVector{centerline_twist_springs_part_ptr_});

        // Set the node's data
        mundy::mesh::vector3_field_data(*node_coord_field_ptr_, node) =
            tail_coord + sperm_axis * i * sperm_initial_segment_length_;
        mundy::mesh::vector3_field_data(*node_velocity_field_ptr_, node) = {0.0, 0.0, 0.0};
        mundy::mesh::vector3_field_data(*node_force_field_ptr_, node) = {0.0, 0.0, 0.0};
        mundy::mesh::vector3_field_data(*node_acceleration_field_ptr_, node) = {0.0, 0.0, 0.0};
        stk::mesh::field_data(*node_twist_field_ptr_, node)[0] = 0.0;
        stk::mesh::field_data(*node_twist_velocity_field_ptr_, node)[0] = 0.0;
        stk::mesh::field_data(*node_twist_torque_field_ptr_, node)[0] = 0.0;
        stk::mesh::field_data(*node_twist_acceleration_field_ptr_, node)[0] = 0.0;
        mundy::mesh::vector3_field_data(*node_curvature_field_ptr_, node) = {0.0, 0.0, 0.0};
        mundy::mesh::vector3_field_data(*node_rest_curvature_field_ptr_, node) = {0.0, 0.0, 0.0};
        stk::mesh::field_data(*node_radius_field_ptr_, node)[0] = sperm_radius_;
        stk::mesh::field_data(*node_archlength_field_ptr_, node)[0] = i * sperm_initial_segment_length_;
      }

      // Centerline twist springs connect nodes i, i+1, and i+2. We need to start at node i=0 and end at node N - 2.
      const size_t start_element_chain_index = start_node_index;
      const size_t end_start_element_chain_index =
          (rank == bulk_data_ptr_->parallel_size() - 1) ? end_node_index - 2 : end_node_index - 1;
      for (size_t i = start_element_chain_index; i < end_start_element_chain_index; ++i) {
        // Create the centerline twist spring.
        stk::mesh::EntityId spring_id = get_centerline_twist_spring_id(i);
        stk::mesh::Entity spring = bulk_data_ptr_->declare_element(spring_id);
        bulk_data_ptr_->change_entity_parts(spring, stk::mesh::PartVector{centerline_twist_springs_part_ptr_});

        // Fetch the nodes, and edges, and connect them to the spring.
        // To map our sequential index to the node sequential index, we connect to node i, i + 1, and i + 2.
        // Our center node is node i. Note, the node ordinals for BEAM_3 are
        /* n1        n2
        //   \      /
        //    e1   e2
        //     \  /
        //      n3
        */
        stk::mesh::Entity left_node = get_node(i);
        stk::mesh::Entity center_node = get_node(i + 1);
        stk::mesh::Entity right_node = get_node(i + 2);

        stk::mesh::Entity left_edge = get_edge(i);
        stk::mesh::Entity right_edge = get_edge(i + 1);

        bulk_data_ptr_->declare_relation(spring, left_node, 0);
        bulk_data_ptr_->declare_relation(spring, right_node, 1);
        bulk_data_ptr_->declare_relation(spring, center_node, 2);

        // Populate the spring's data
        stk::mesh::field_data(*element_radius_field_ptr_, spring)[0] = sperm_radius_;
        stk::mesh::field_data(*element_youngs_modulus_field_ptr_, spring)[0] = sperm_youngs_modulus_;
        stk::mesh::field_data(*element_poissons_ratio_field_ptr_, spring)[0] = sperm_poissons_ratio_;
        stk::mesh::field_data(*element_rest_length_field_ptr_, spring)[0] = sperm_rest_segment_length_;
      }

      // Share the nodes with the neighboring ranks.
      // Note, node sharing is symmetric. If we don't own the node that we intend to share, we need to declare it before
      // marking it as shared. If we are rank 0, we share our final node with rank 1 and receive their first node. If we
      // are rank N, we share our first node with rank N - 1 and receive their final node. Otherwise, we share our first
      // and last nodes with the corresponding neighboring ranks and receive their corresponding nodes.
      if (bulk_data_ptr_->parallel_size() > 1) {
        if (rank == 0) {
          // Share the last node with rank 1.
          stk::mesh::Entity node = get_node(end_node_index - 1);
          bulk_data_ptr_->add_node_sharing(node, rank + 1);

          // Receive the first node from rank 1
          stk::mesh::EntityId received_node_id = get_node_id(end_node_index);
          stk::mesh::Entity received_node = bulk_data_ptr_->declare_node(received_node_id);
          bulk_data_ptr_->add_node_sharing(received_node, rank + 1);
        } else if (rank == bulk_data_ptr_->parallel_size() - 1) {
          // Share the first node with rank N - 1.
          stk::mesh::Entity node = get_node(start_node_index);
          bulk_data_ptr_->add_node_sharing(node, rank - 1);

          // Receive the last node from rank N - 1.
          stk::mesh::EntityId received_node_id = get_node_id(start_node_index - 1);
          stk::mesh::Entity received_node = bulk_data_ptr_->declare_node(received_node_id);
          bulk_data_ptr_->add_node_sharing(received_node, rank - 1);
        } else {
          // Share the first and last nodes with the corresponding neighboring ranks.
          stk::mesh::Entity first_node = get_node(start_node_index);
          stk::mesh::Entity last_node = get_node(end_node_index - 1);
          bulk_data_ptr_->add_node_sharing(first_node, rank - 1);
          bulk_data_ptr_->add_node_sharing(last_node, rank + 1);

          // Receive the corresponding nodes from the neighboring ranks.
          stk::mesh::EntityId received_first_node_id = get_node_id(start_node_index - 1);
          stk::mesh::EntityId received_last_node_id = get_node_id(end_node_index);
          stk::mesh::Entity received_first_node = bulk_data_ptr_->declare_node(received_first_node_id);
          stk::mesh::Entity received_last_node = bulk_data_ptr_->declare_node(received_last_node_id);
          bulk_data_ptr_->add_node_sharing(received_first_node, rank - 1);
          bulk_data_ptr_->add_node_sharing(received_last_node, rank + 1);
        }
      }
      bulk_data_ptr_->modification_end();

      // Create the edges and connect them to the nodes.
      // The canonical way to create and share edges is stk's create_edges method
      stk::mesh::create_edges(*bulk_data_ptr_);

      // Populate the edge data
      stk::mesh::Field<double> &edge_orientation_field = *edge_orientation_field_ptr_;
      stk::mesh::Field<double> &edge_tangent_field = *edge_tangent_field_ptr_;
      const double sperm_initial_segment_length = sperm_initial_segment_length_;
      stk::mesh::for_each_entity_run(
          *bulk_data_ptr_, stk::topology::EDGE_RANK, meta_data_ptr_->locally_owned_part(),
          [&edge_orientation_field, &edge_tangent_field, sperm_axis, sperm_initial_segment_length](
              const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &edge) {
            // The orientation of the edge is the identity since we are currently in the reference configuration.
            mundy::mesh::quaternion_field_data(*edge_orientation_field_ptr_, edge) =
                mundy::math::Quaternion<double>::identity();
            mundy::mesh::vector3_field_data(*edge_tangent_field_ptr_, edge) = sperm_axis;
            stk::mesh::field_data(*edge_length_field_ptr_, edge)[0] = sperm_initial_segment_length_;
          });
    }
  }

  void loadbalance() {
    debug_print("Load balancing the mesh.");
    stk::balance::balanceStkMesh(balance_settings_, *static_cast<stk::mesh::BulkData *>(bulk_data_ptr_.get()));
  }

  void rotate_field_states() {
    debug_print("Rotating the field states.");
    bulk_data_ptr_->update_field_data_states();
  }

  void zero_out_transient_node_fields() {
    debug_print("Zeroing out the transient node fields.");
    mundy::mesh::utils::fill_field_with_value<double>(*node_velocity_field_ptr_, std::array<double, 3>{0.0, 0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_force_field_ptr_, std::array<double, 3>{0.0, 0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_acceleration_field_ptr_,
                                                      std::array<double, 3>{0.0, 0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_twist_velocity_field_ptr_, std::array<double, 1>{0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_twist_torque_field_ptr_, std::array<double, 1>{0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_twist_acceleration_field_ptr_, std::array<double, 1>{0.0});
  }

  void update_position_and_twist() {
    debug_print("Updating the position and twist.");
    // x(t + dt) = x(t) + v(t) * dt + a(t) * dt^2 / 2.

    // No data to communicate since we only act on locally owned nodes.

    // Get references to internal members so we aren't passing around *this
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    stk::mesh::Part &centerline_twist_part = *centerline_twist_part_ptr_;
    stk::mesh::Field<double> &node_coordinates_field = *node_coordinates_field_ptr_;
    stk::mesh::Field<double> &node_coordinates_field_old = node_coordinates_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_velocity_field_old = node_velocity_field_ptr_->field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_acceleration_field_old =
        node_acceleration_field_ptr_->field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_twist_field = *node_twist_field_ptr_;
    stk::mesh::Field<double> &node_twist_field_old = node_twist_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_twist_velocity_field_old =
        node_twist_velocity_field_ptr_->field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_twist_acceleration_field_old =
        node_twist_acceleration_field_ptr_->field_of_state(stk::mesh::StateN);
    const time_step_size = timestep_size_;

    // Update the position and twist of the nodes
    auto locally_owned_selector = stk::mesh::Selector(centerline_twist_part) & meta_data_ptr_->locally_owned_part();
    stk::mesh::for_each_entity_run(
        bulk_data, node_rank_, locally_owned_selector,
        [&node_coordinates_field, &node_coordinates_field_old, &node_velocity_field_old, &node_acceleration_field_old,
         &node_twist_field, &node_twist_field_old, &node_twist_velocity_field_old, &node_twist_acceleration_field_old,
         time_step_size](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &node) {
          // Get the required input fields
          const auto node_coords_old = mundy::mesh::vector3_field_data(node_coordinates_field_old, node);
          const auto node_velocity_old = mundy::mesh::vector3_field_data(node_velocity_field_old, node);
          const auto node_acceleration_old = mundy::mesh::vector3_field_data(node_acceleration_field_old, node);
          const double node_twist_old = stk::mesh::field_data(node_twist_field_old, node)[0];
          const double node_twist_rate_old = stk::mesh::field_data(node_twist_rate_field_old, node)[0];
          const double node_twist_acceleration_old = stk::mesh::field_data(node_twist_acceleration_field_old, node)[0];

          // Get the output fields
          auto node_coords = mundy::mesh::vector3_field_data(node_coordinates_field, node);
          double *node_twist = stk::mesh::field_data(node_twist_field, node);

          // Update the current configuration
          node_coords = node_coords_old + node_velocity_old * time_step_size +
                        0.5 * node_acceleration_old * time_step_size * time_step_size;
          node_twist[0] = node_twist_old + node_twist_rate_old * time_step_size +
                          0.5 * node_twist_acceleration_old * time_step_size * time_step_size;
        });
  }

  void compute_edge_information() {
    debug_print("Computing the edge information.");

    // Communicate the fields of downward connected entities.
    stk::mesh::communicate_field_data(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr_),
                                      {node_coordinates_field_ptr_});

    // Get references to internal members so we aren't passing around *this
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    stk::mesh::Part &centerline_twist_part = *centerline_twist_part_ptr_;
    stk::mesh::Field<double> &node_coordinates_field = *node_coordinates_field_ptr_;
    stk::mesh::Field<double> &edge_orientation_field = *edge_orientation_field_ptr_;
    stk::mesh::Field<double> &edge_tangent_field = *edge_tangent_field_ptr_;
    stk::mesh::Field<double> &edge_tangent_field_old = edge_tangent_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &edge_binormal_field = *edge_binormal_field_ptr_;
    stk::mesh::Field<double> &edge_length_field = *edge_length_field_ptr_;

    // For each edge in the centerline twist part, compute the edge tangent, binormal, and length.
    // length^i = ||x_{i+1} - x_i||
    // edge_tangent^i = (x_{i+1} - x_i) / length
    // edge_binormal^i = (2 edge_tangent_old^i x edge_tangent^i) / (1 + edge_tangent_old^i dot edge_tangent^i)
    auto locally_owned_selector = stk::mesh::Selector(centerline_twist_part) & meta_data_ptr_->locally_owned_part();
    stk::mesh::for_each_entity_run(
        bulk_data, stk::topology::EDGE_RANK, locally_owned_selector,
        [&node_coordinates_field, &edge_orientation_field, &edge_tangent_field, &edge_tangent_field_old,
         &edge_binormal_field,
         &edge_length_field](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &edge) {
          // Get the nodes of the edge
          stk::mesh::Entity const *edge_nodes = bulk_data.begin_nodes(edge);
          stk::mesh::Entity const &node_i = edge_nodes[0];
          stk::mesh::Entity const &node_ip1 = edge_nodes[1];

          // Get the required input fields
          const auto node_i_coords = mundy::mesh::vector3_field_data(node_coordinates_field, node_i);
          const auto node_ip1_coords = mundy::mesh::vector3_field_data(node_coordinates_field, node_ip1);
          const auto edge_tangent_old = mundy::mesh::vector3_field_data(edge_tangent_field_old, edge);

          // Get the output fields
          auto edge_tangent = mundy::mesh::vector3_field_data(edge_tangent_field, edge);
          auto edge_binormal = mundy::mesh::vector3_field_data(edge_binormal_field, edge);
          double *edge_length = stk::mesh::field_data(edge_length_field, edge);

          // Compute the un-normalized edge tangent
          edge_tangent = node_ip1_coords - node_i_coords;
          edge_length[0] = mundy::math::norm(edge_tangent);
          edge_tangent /= edge_length[0];

          // Compute the edge binormal
          edge_binormal = (2.0 * mundy::math::cross(edge_tangent_old, edge_tangent)) /
                          (1.0 + mundy::math::dot(edge_tangent_old, edge_tangent));
        });
  }

  void compute_node_curvature_and_rotation_gradient() {
    debug_print("Computing the node curvature and rotation gradient.");

    // Communicate the fields of downward connected entities.
    // TODO(palmerb4): Technically, we could avoid this communication if we compute the edge information for locally
    // owned and ghosted edges. Computation is cheaper than communication.
    stk::mesh::communicate_field_data(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr_),
                                      {edge_orientation_field_ptr_});

    // Get references to internal members so we aren't passing around *this
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    stk::mesh::Part &centerline_twist_part = *centerline_twist_part_ptr_;
    stk::mesh::Field<double> &edge_orientation_field = *edge_orientation_field_ptr_;
    stk::mesh::Field<double> &node_curvature_field = *node_curvature_field_ptr_;
    stk::mesh::Field<double> &node_rotation_gradient_field = *node_rotation_gradient_field_ptr_;

    // Bug fix:
    // Originally this function acted on the locally owned elements of the centerline twist part, using them to fetch
    // the nodes/edges in the correct order and performing the computation. However, this assumes that the center node
    // of this element is locally owned as well. If this assumption fails, we'll end up writing the result to a shared
    // but not locally owned node. The corresponding locally owned node on a different process won't have its curvature
    // updated. That node is, thankfully, connected to a ghosted version of the element on this process, so we can fix
    // this issue by looping over all elements, including ghosted ones.
    //
    // We'll have to double check that this indeed works. I know that it will properly ensure that all locally owned
    // nodes are updated, but we also write to some non-locally owned nodes. I want to make sure that the values in the
    // non-locally owned nodes are updated using the locally-owned values. I think this is the case, but I want to
    // double check.

    // For each element in the centerline twist part, compute the node curvature at the center node.
    // The curvature can be computed from the edge orientations using
    //   kappa^i = q_i - conj(q_i) = 2 * vec(q_i)
    // where
    //   q_i = conj(d^{i-1}) d^i is the Lagrangian rotation gradient.
    stk::mesh::for_each_entity_run(
        bulk_data, stk::topology::ELEMENT_RANK, centerline_twist_part,
        [&edge_orientation_field, &node_curvature_field, &node_rotation_gradient_field](
            const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &element) {
          // Curvature needs to "know" about the order of edges, so it's best to loop over
          // the slt elements and not the nodes. Get the lower rank entities
          stk::mesh::Entity const *element_nodes = bulk_data.begin_nodes(element);
          stk::mesh::Entity const *element_edges = bulk_data.begin_edges(element);

          MUNDY_THROW_ASSERT(bulk_data.num_nodes(element) == 3, std::logic_error,
                             "Elements must have exactly 3 nodes.");
          MUNDY_THROW_ASSERT(bulk_data.num_edges(element) == 2, std::logic_error,
                             "Elements must have exactly 2 edges.");

          // Get the required input fields
          const auto edge_im1_orientation =
              mundy::mesh::quaternion_field_data(edge_orientation_field, element_edges[0]);
          const auto edge_i_orientation = mundy::mesh::quaternion_field_data(edge_orientation_field, element_edges[1]);

          // Get the output fields
          auto node_curvature = mundy::mesh::vector3_field_data(node_curvature_field, element_nodes[1]);
          auto node_rotation_gradient =
              mundy::mesh::quaternion_field_data(node_rotation_gradient_field, element_nodes[1]);

          // Compute the node curvature
          node_rotation_gradient = mundy::math::conj(edge_im1_orientation) * edge_i_orientation;
          node_curvature = 2.0 * node_rotation_gradient.vector();
        });
  }

  void compute_internal_force_and_twist_torque() {
    debug_print("Computing the internal force and twist torque.");

    // Communicate the fields of downward connected entities.
    // TODO(palmerb4): Technically, we could avoid this entire communication if we compute the edge information for locally
    // owned and ghosted edges. Computation is cheaper than communication.
    stk::mesh::communicate_field_data(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr_),
                                      {node_curvature_field_ptr_, node_rest_curvature_field_ptr_, node_twist_field_ptr_,
                                       node_rotation_gradient_field_ptr_, edge_tangent_field_ptr_,
                                       edge_binormal_field_ptr_, edge_length_field_ptr_, edge_orientation_field_ptr_});

    // Get references to internal members so we aren't passing around *this
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    stk::mesh::Part &centerline_twist_part = *centerline_twist_part_ptr_;
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
    stk::mesh::Field<double> &node_twist_torque_field = *node_twist_torque_field_ptr_;
    stk::mesh::Field<double> &node_curvature_field = *node_curvature_field_ptr_;
    stk::mesh::Field<double> &node_rest_curvature_field = *node_rest_curvature_field_ptr_;
    stk::mesh::Field<double> &node_rotation_gradient_field = *node_rotation_gradient_field_ptr_;
    stk::mesh::Field<double> &edge_tangent_field = *edge_tangent_field_ptr_;
    stk::mesh::Field<double> &edge_binormal_field = *edge_binormal_field_ptr_;
    stk::mesh::Field<double> &edge_length_field = *edge_length_field_ptr_;
    stk::mesh::Field<double> &edge_orientation_field = *edge_orientation_field_ptr_;

    // Compute internal force and torque induced by differences in rest and current curvature
    auto locally_owned_selector = stk::mesh::Selector(centerline_twist_part) & meta_data_ptr_->locally_owned_part();
    stk::mesh::for_each_entity_run(
        bulk_data, element_rank_, locally_owned_selector,
        [&node_force_field, &node_twist_torque_field, &node_curvature_field, &node_rest_curvature_field,
         &node_rotation_gradient_field, &edge_tangent_field, &edge_binormal_field, &edge_length_field,
         &edge_orientation_field](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &element) {
          // Ok. This is a bit involved.
          // First, we need to use the node curvature to compute the induced lagrangian torque according to the
          // Kirchhoff rod model. Then, we need to use a convoluted map to take this torque to force and torque on the
          // nodes.
          //
          // The torque induced by the curvature is
          //  T = B (kappa - kappa_rest)
          // where B is the diagonal matrix of bending moduli and kappa_rest is the rest curvature.
          //
          // To start, we set B = I

          // Get the lower rank entities
          stk::mesh::Entity const *element_nodes = bulk_data.begin_nodes(element);
          stk::mesh::Entity const *element_edges = bulk_data.begin_edges(element);
          MUNDY_THROW_ASSERT(bulk_data.num_nodes(element) == 3, std::logic_error,
                             "SLT: Elements must have exactly 3 nodes.");
          MUNDY_THROW_ASSERT(bulk_data.num_edges(element) == 2, std::logic_error,
                             "SLT: Elements must have exactly 2 edges.");
          stk::mesh::Entity &node_im1 = element_nodes[0];
          stk::mesh::Entity &node_i = element_nodes[1];
          stk::mesh::Entity &node_ip1 = element_nodes[2];
          stk::mesh::Entity &edge_im1 = element_edges[0];
          stk::mesh::Entity &edge_i = element_edges[1];

          // Get the required input fields
          const auto node_i_curvature = mundy::mesh::vector3_field_data(node_curvature_field, node_i);
          const auto node_i_rest_curvature = mundy::mesh::vector3_field_data(node_rest_curvature_field, node_i);
          const auto node_i_rotation_gradient =
              mundy::mesh::quaternion_field_data(node_rotation_gradient_field, node_i);
          const auto edge_im1_tangent = mundy::mesh::vector3_field_data(edge_tangent_field, edge_im1);
          const auto edge_i_tangent = mundy::mesh::vector3_field_data(edge_tangent_field, edge_i);
          const auto edge_im1_binormal = mundy::mesh::vector3_field_data(edge_binormal_field, edge_im1);
          const auto edge_i_binormal = mundy::mesh::vector3_field_data(edge_binormal_field, edge_i);
          const double edge_im1_length = stk::mesh::field_data(edge_length_field, edge_im1)[0];
          const double edge_i_length = stk::mesh::field_data(edge_length_field, edge_i)[0];
          const auto edge_im1_orientation = mundy::mesh::quaternion_field_data(edge_orientation_field, edge_im1);

          // Get the output fields
          double *node_im1_force = stk::mesh::field_data(node_force_field, node_im1);
          double *node_i_force = stk::mesh::field_data(node_force_field, node_i);
          double *node_ip1_force = stk::mesh::field_data(node_force_field, node_ip1);
          double *node_im1_twist_torque = stk::mesh::field_data(node_twist_torque_field, node_im1);
          double *node_i_twist_torque = stk::mesh::field_data(node_twist_torque_field, node_i);

          // Compute the torque induced by the curvature
          // To start, the torque is in the lagrangian frame.
          auto bending_modulus = mundy::math::Matrix3::identity();  // TODO(palmerb4): Replace with Kirchhoff rod model
          auto bending_torque = bending_modulus * (node_i_curvature - node_i_rest_curvature);

          // We'll use the bending torque as a placeholder for the rotated bending torque
          bending_torque =
              edge_im1_orientation * (node_i_rotation_gradient.w() * bending_torque +
                                      mundy::math::cross(node_i_rotation_gradient.vector(), bending_torque));

          // Compute the force and torque on the nodes
          const auto tmp_force_ip1 = 1.0 / edge_i_length *
                                     (mundy::math::cross(bending_torque, edge_i_tangent) +
                                      0.5 * mundy::math::dot(edge_i_tangent, bending_torque) *
                                          (mundy::math::dot(edge_i_tangent, edge_i_binormal) * edge_i_tangent) -
                                      edge_i_binormal);
          const auto tmp_force_im1 =
              1.0 / edge_im1_length *
              (mundy::math::cross(bending_torque, edge_im1_tangent) +
               0.5 * mundy::math::dot(edge_im1_tangent, bending_torque) *
                   (mundy::math::dot(edge_im1_tangent, edge_im1_binormal) * edge_im1_tangent - edge_im1_binormal));

#pragma omp atomic
          node_i_twist_torque[0] += edge_i_tangent.dot(bending_torque);
#pragma omp atomic
          node_im1_twist_torque[0] += -edge_im1_tangent.dot(bending_torque);
#pragma omp atomic
          node_ip1_force[0] += tmp_force_ip1[0];
#pragma omp atomic
          node_ip1_force[1] += tmp_force_ip1[1];
#pragma omp atomic
          node_ip1_force[2] += tmp_force_ip1[2];
#pragma omp atomic
          node_i_force[0] -= tmp_force_ip1[0] + tmp_force_im1[0];
#pragma omp atomic
          node_i_force[1] -= tmp_force_ip1[1] + tmp_force_im1[1];
#pragma omp atomic
          node_i_force[2] -= tmp_force_ip1[2] + tmp_force_im1[2];
#pragma omp atomic
          node_im1_force[0] += tmp_force_im1[0];
#pragma omp atomic
          node_im1_force[1] += tmp_force_im1[1];
#pragma omp atomic
          node_im1_force[2] += tmp_force_im1[2];
        });

    // Compute internal force induced by differences in rest and current length
    double rest_length = sperm_rest_segment_length_;
    stk::mesh::for_each_entity_run(
        bulk_data, edge_rank_, locally_owned_selector,
        [&node_force_field, &edge_tangent_field, &edge_length_field, &rest_length](const stk::mesh::BulkData &bulk_data,
                                                                                   const stk::mesh::Entity &edge) {
          // F_left = k (l - l_rest) tangent
          // F_right = -k (l - l_rest) tangent
          //
          // k can be computed using the material properties of the rod
          // For now, we set k = 1.

          // Get the lower rank entities
          stk::mesh::Entity const *edge_nodes = bulk_data.begin_nodes(edge);
          MUNDY_THROW_ASSERT(bulk_data.num_nodes(edge) == 2, std::logic_error, "Edges must have exactly 2 nodes.");
          stk::mesh::Entity &node_im1 = element_nodes[0];
          stk::mesh::Entity &node_i = element_nodes[1];

          // Get the required input fields
          const auto edge_tangent = mundy::mesh::vector3_field_data(edge_tangent_field, edge);
          const double edge_length = stk::mesh::field_data(edge_length_field, edge)[0];

          // Get the output fields
          double *node_im1_force = stk::mesh::field_data(node_force_field, node_im1);
          double *node_i_force = stk::mesh::field_data(node_force_field, node_i);

          // Compute the internal force
          const auto tmp_force = (edge_length - rest_length) * edge_tangent;
#pragma omp atomic
          node_i_force[0] += tmp_force[0];
#pragma omp atomic
          node_i_force[1] += tmp_force[1];
#pragma omp atomic
          node_i_force[2] += tmp_force[2];
#pragma omp atomic
          node_im1_force[0] -= tmp_force[0];
#pragma omp atomic
          node_im1_force[1] -= tmp_force[1];
#pragma omp atomic
          node_im1_force[2] -= tmp_force[2];
        });

    // Sum the node force and torque over shared nodes.
    stk::mesh::parallel_sum(*bulk_data_ptr_, {node_force_field_ptr_, node_twist_torque_field_ptr_});
  }


  void compute_centerline_twist_force_and_torque() {
    debug_print("Computing the centerline twist force and torque.");
    // Compute the edge information in the current timestep (length, tangent, and binormal).
    compute_edge_information();

    // Compute the node curvature and rotation gradient.
    compute_node_curvature_and_rotation_gradient();

    // Compute the internal force and twist torque.
    compute_internal_force_and_twist_torque();
  }

  void compute_velocity_and_acceleration() {
    debug_print("Computing the velocity and acceleration.");

    // Get references to internal members so we aren't passing around *this
    mundy::mesh::BulkData &bulk_data = *bulk_data_ptr_;
    stk::mesh::Part &centerline_twist_part = *centerline_twist_part_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field_ref = node_velocity_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_acceleration_field = *node_acceleration_field_ptr_;
    stk::mesh::Field<double> &node_acceleration_field_ref = node_acceleration_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;

    stk::mesh::Field<double> &node_twist_rate_field = *node_twist_rate_field_ptr_;
    stk::mesh::Field<double> &node_twist_rate_field_ref = node_twist_rate_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_twist_acceleration_field = *node_twist_acceleration_field_ptr_;
    stk::mesh::Field<double> &node_twist_acceleration_field_ref =
        node_twist_acceleration_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_twist_torque_field = *node_twist_torque_field_ptr_;
    const time_step_size = timestep_size_;

    // a(t + dt) = M^{-1} f(x(t + dt))
    stk::mesh::for_each_entity_run(
        bulk_data, node_rank_, locally_owned_selector,
        [&node_acceleration_field, &node_force_field, $node_twist_acceleration_field, &node_twist_torque_field,
         node_mass, node_moment_of_inertia](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &node) {
          // Get the required input fields
          const auto node_force = mundy::mesh::vector3_field_data(node_force_field, node);
          const double node_twist_torque = stk::mesh::field_data(node_twist_torque_field, node)[0];

          // Get the output fields
          auto node_acceleration = mundy::mesh::vector3_field_data(node_acceleration_field, node);
          double *node_twist_acceleration = stk::mesh::field_data(node_twist_acceleration_field, node);

          // Compute the acceleration
          node_acceleration = node_force / node_mass;
          node_twist_acceleration[0] = node_twist_torque / node_moment_of_inertia;
        });

    // v(t + dt) = v(t) + (a(t) + a(t + dt)) * dt / 2
    stk::mesh::for_each_entity_run(
        bulk_data, node_rank_, locally_owned_selector,
        [&node_velocity_field, &node_velocity_field_ref, &node_acceleration_field, &node_acceleration_field_ref,
         &node_twist_rate_field, &node_twist_rate_field_ref, &node_twist_acceleration_field,
         &node_twist_acceleration_field_ref,
         time_step_size](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &node) {
          // Get the required input fields
          const auto node_velocity_ref = mundy::mesh::vector3_field_data(node_velocity_field_ref, node);
          const auto node_acceleration = mundy::mesh::vector3_field_data(node_acceleration_field, node);
          const auto node_acceleration_ref = mundy::mesh::vector3_field_data(node_acceleration_field_ref, node);
          const double node_twist_rate_ref = stk::mesh::field_data(node_twist_rate_field_ref, node)[0];
          const double node_twist_acceleration = stk::mesh::field_data(node_twist_acceleration_field, node)[0];
          const double node_twist_acceleration_ref = stk::mesh::field_data(node_twist_acceleration_field_ref, node)[0];

          // Get the output fields
          auto node_velocity = mundy::mesh::vector3_field_data(node_velocity_field, node);
          double *node_twist_rate = stk::mesh::field_data(node_twist_rate_field, node);

          // Compute the velocity
          node_velocity = node_velocity_ref + 0.5 * (node_acceleration + node_acceleration_ref) * time_step_size;
          node_twist_rate[0] =
              node_twist_rate_ref + 0.5 * (node_twist_acceleration + node_twist_acceleration_ref) * time_step_size;
        });
  }

  void run(int argc, char **argv) {
    debug_print("Running the simulation.");

    // Preprocess
    if (!parse_user_inputs(argc, argv)) {
      std::cerr << "Failed to parse user inputs." << std::endl;
      return;
    }
    check_input_parameters();
    dump_user_inputs();

    // Setup
    build_our_mesh_and_method_instances();
    fetch_fields_and_parts();
    setup_io();
    declare_and_initialize_the_sperm();

    // Time loop
    if (bulk_data_ptr_->parallel_rank() == 0) {
      std::cout << "Running the simulation for " << num_time_steps_ << " time steps." << std::endl;
    }

    Kokkos::Timer timer;
    for (size_t i = 0; i < num_time_steps_; i++) {
      debug_print("Time step " << i << ".");

      // Rotate the field states.
      bulk_data.update_field_data_states();

      // IO. If desired, write out the data for time t.
      if (i % io_frequency_ == 0) {
        stk_io_broker_.begin_output_step(output_file_index_, static_cast<double>(i));
        stk_io_broker_.write_defined_output_fields(output_file_index_);
        stk_io_broker_.end_output_step(output_file_index_);
        stk_io_broker_.flush_output();
      }

      // Prepare the current configuration.
      {
        // Rotate the field states.
        rotate_field_states();

        // Motion from t -> t + dt.
        // x(t + dt) = x(t) + v(t) * dt + a(t) * dt^2 / 2.
        update_position_and_twist();

        // Zero the node velocities, accelerations, and forces/torques for time t + dt.
        zero_out_transient_node_fields();
      }

      // Evaluate forces f(x(t + dt)).
      {
        // Centerline twist rod model
        compute_centerline_twist_force_and_torque();
      }

      // Compute velocity and acceleration.
      {
        // Evaluate a(t + dt) = M^{-1} f(x(t + dt)).
        // Evaluate v(t + dt) = v(t) + (a(t) + a(t + dt)) * dt / 2.
        compute_velocity_and_acceleration();
      }
    }

    // Do a synchronize to force everybody to stop here, then write the time
    stk::parallel_machine_barrier(bulk_data_ptr_->parallel());
    if (bulk_data_ptr_->parallel_rank() == 0) {
      double avg_time_per_timestep = static_cast<double>(timer.seconds()) / static_cast<double>(num_time_steps_);
      std::cout << "Time per timestep: " << std::setprecision(15) << avg_time_per_timestep << std::endl;
    }
  }

 private:
  //! \name Useful aliases
  //@{

  constexpr auto node_rank_ = stk::topology::NODE_RANK;
  constexpr auto edge_rank_ = stk::topology::EDGE_RANK;
  constexpr auto element_rank = stk::topology::ELEMENT_RANK;
  //@}

  //! \name Internal state
  //@{

  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr_;
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr_;
  std::shared_ptr<mundy::meta::MeshRequirements> mesh_reqs_ptr_;
  stk::io::StkMeshIoBroker stk_io_broker_;
  size_t output_file_index_;
  //@}

  //! \name Fields
  //@{

  stk::mesh::Field<double> *node_coordinates_field_ptr_;
  stk::mesh::Field<double> *node_velocity_field_ptr_;
  stk::mesh::Field<double> *node_force_field_ptr_;
  stk::mesh::Field<double> *node_acceleration_field_ptr_;
  stk::mesh::Field<double> *node_twist_field_ptr_;
  stk::mesh::Field<double> *node_twist_velocity_field_ptr_;
  stk::mesh::Field<double> *node_twist_torque_field_ptr_;
  stk::mesh::Field<double> *node_twist_acceleration_field_ptr_;
  stk::mesh::Field<double> *node_curvature_field_ptr_;
  stk::mesh::Field<double> *node_rest_curvature_field_ptr_;
  stk::mesh::Field<double> *node_rotation_gradient_field_ptr_;
  stk::mesh::Field<double> *node_radius_field_ptr_;
  stk::mesh::Field<double> *node_archlength_field_ptr_;

  stk::mesh::Field<double> *edge_orientation_field_ptr_;
  stk::mesh::Field<double> *edge_tangent_field_ptr_;
  stk::mesh::Field<double> *edge_binormal_field_ptr_;
  stk::mesh::Field<double> *edge_length_field_ptr_;

  stk::mesh::Field<double> *element_radius_field_ptr_;
  stk::mesh::Field<double> *element_youngs_modulus_field_ptr_;
  stk::mesh::Field<double> *element_poissons_ratio_field_ptr_;
  stk::mesh::Field<double> *element_rest_length_field_ptr_;
  //@}

  //! \name Parts
  //@{

  stk::mesh::Part *centerline_twist_springs_part_ptr_;
  //@}

  //! \name Partitioning settings
  //@{

  class RcbSettings : public stk::balance::BalanceSettings {
   public:
    RcbSettings() {
    }
    virtual ~RcbSettings() {
    }

    virtual bool isIncrementalRebalance() const {
      return false;
    }
    virtual std::string getDecompMethod() const {
      return std::string("rcb");
    }
    virtual std::string getCoordinateFieldName() const {
      return std::string("NODE_COORDINATES");
    }
    virtual bool shouldPrintMetrics() const {
      return false;
    }
  };  // RcbSettings

  RcbSettings balance_settings_;
  //@}

  //! \name User parameters
  //@{

  size_t num_sperm_ = 1;
  size_t num_nodes_per_sperm_ = 10;
  double sperm_radius_ = 0.6;
  double sperm_initial_segment_length_ = 1.0;
  double sperm_rest_segment_length_ = 2 * sperm_radius_;
  double sperm_rest_curvature_twist_ = 0.0;
  double sperm_rest_curvature_bend1_ = 0.0;
  double sperm_rest_curvature_bend2_ = 0.0;

  double sperm_youngs_modulus_ = 1000.0;
  double sperm_poissons_ratio_ = 0.3;
  double sperm_density_ = 1.0;

  size_t num_time_steps_ = 100;
  double timestep_size_ = 0.01;
  size_t io_frequency_ = 10;
  //@}
};  // SpermSimulation

int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  // Run the simulation using the given parameters
  SpermSimulation().run(argc, argv);

  // Finalize MPI
  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}
