// C++ core libs
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector
#include <iostream>   // for std::cout, etc.
#include <sstream>    // for std::stringstream
#include <fstream>    // for std::ofstream

#include <numeric>    // for std::iota
#include <algorithm>  // for std::random_shuffle
#include <ctime>      // std::time
#include <cstdlib>    // std::rand, std::srand

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/BulkData.hpp>       // for stk::mesh::BulkData
#include <stk_mesh/base/Comm.hpp>           // for comm_mesh_counts
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/MeshBuilder.hpp>    // for stk::mesh::MeshBuilder
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>          // for stk::mesh::EntityProc, EntityVector, etc

#include <stk_mesh/base/DumpMeshInfo.hpp>   // for stk::mesh::impl::dump_all_mesh_info, dump_mesh_per_proc
#include <stk_mesh/base/Field.hpp>          // for Field

// external libs
#include <gtest/gtest.h>    // for AssertHelper, EXPECT_EQ, etc
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#define USE_NANOBENCH

// test configuration:
//
// two (2) nodes belong to processor P0.  the nodes connect n_springs beam_2 elements which can belong
// to either processor P0 or P1, if two processes are used.  only up to two processors can be used.
//
//                                spring1 (P0)
//                  node1 (P0) ------------------ node2 (P0)
//                                spring1 (P1)
//                  node1 (P0) ------------------ node2 (P0)
//                                      .
//                                      .
//                                      .
//                                n_springs (P1)
//                  node1 (P0) ------------------ node2 (P0)
//
// springs are (arbitrarily) destroyed from and added to the initial mesh configuration.

// typedefs
using stk::mesh::Entity;
using stk::mesh::MetaData;
using stk::mesh::BulkData;
typedef stk::mesh::Field<double> ScalarField;

// make these global for some reason. so that check_buckets() can access them easily
const std::string pressure_field_name = "pressure";
const double init_pressure = 2.0; // have to pass as ref

// just change modification sorting here, rather than in each call to modification_end()
#define MUNDY_MOD_END_SORT_TYPE stk::mesh::impl::MeshModification::MOD_END_NO_SORT

// the first input in a pair is the processor rank and the second input is the spring id
typedef std::vector<std::pair<unsigned int, unsigned int>> SpringProcVec;

// the calls to inspect_spring() and check_buckets() happen after each
// mesh modification cycle.  count keeps track of the number of times 
// a mesh modification has occurred for log file names
unsigned int count = 0;

// specify whether the springs created after the deletion step are
// added to the same process they were destroyed from, or to the other process.
// they can also be added randomly. see intialize_vectors()
typedef enum class Add : char { Same, Other, Random } Add;

// initialize three vectors containing <process, id> pairs. these are used for specifying which process an entity
// will belong to once created, which process an entity will be destroyed from, and which process an entity will be
// created on following the destruction cycle, respectively.
void initialize_vectors(SpringProcVec& initial, SpringProcVec& destroyed, SpringProcVec& created, Add init = Add::Same)
{
    int n_proc = stk::parallel_machine_size(MPI_COMM_WORLD);
    int n_springs = initial.size();
    // create a random ordering of spring ids for selected deletions/creations
    std::srand (unsigned(std::time(0)));
    std::vector<unsigned int> id_list(n_springs);
    std::iota(id_list.begin(), id_list.end(), 1); // starts at value 1 rather than 0 to avoid invalid id
    std::random_shuffle(id_list.begin(), id_list.end());

    // fill initial vector with processor ranks and spring ids
    for (int i = 0; i < n_springs; i++)
    {
        initial[i].first = i % n_proc;  // process rank
        initial[i].second = id_list[i]; // spring id
    }
    // fill destroyed vector with processor ranks and spring ids
    for (std::size_t i = 0; i < destroyed.size(); i++)
    {
        unsigned int random_index_into_id_list = id_list[i] - 1;        // id_list ranges: 1,...,n_springs. subtract 1 for zero-indexed
        destroyed[i].first = initial[random_index_into_id_list].first;    // process rank
        destroyed[i].second = initial[random_index_into_id_list].second;  // spring id
    }
    // fill created vector with processor ranks and spring ids
    switch (init)
    {
        case Add::Same:
            for (std::size_t i = 0; i < created.size(); i++)
            {
                created[i].first = destroyed[i].first;    // process rank
                created[i].second = destroyed[i].second;  // spring id
            }
            break;
        case Add::Other:
            for (std::size_t i = 0; i < created.size(); i++)
            {
                created[i].first = destroyed[i].first == 0 ? 1 : 0;   // process rank
                created[i].second = destroyed[i].second;              // spring id
            }
            break;
        case Add::Random:
            for (std::size_t i = 0; i < created.size(); i++)
            {
                created[i].first = std::rand() % n_proc;   // process rank
                created[i].second = destroyed[i].second;     // spring id
            }
            break;
        default:
            break;
    }
}

// checks basic assumptions on the structure of the spring elements, i.e. validity, node numbers, etc.
void inspect_springs(const BulkData &bulk_data)
{
    const int my_parallel_rank = stk::parallel_machine_rank(MPI_COMM_WORLD);

    // get locally owned element-rank entities to inspect
    stk::mesh::Part& own_part = bulk_data.mesh_meta_data().locally_owned_part();
    stk::mesh::Selector own_selector(own_part);
    std::vector<Entity> springs_vector;
    bulk_data.get_entities(stk::topology::ELEMENT_RANK, own_selector, springs_vector);
    // get nodes individually, simpler check
    Entity node1 = bulk_data.get_entity(stk::topology::NODE_RANK, 1);
    Entity node2 = bulk_data.get_entity(stk::topology::NODE_RANK, 2);
    for (Entity spring : springs_vector)
    {
        EXPECT_TRUE(bulk_data.is_valid(spring));
        EXPECT_TRUE(bulk_data.is_valid(node1));
        EXPECT_TRUE(bulk_data.is_valid(node2));
        EXPECT_EQ(bulk_data.num_nodes(spring), 2u);
        EXPECT_EQ(node1, bulk_data.begin_nodes(spring)[0]);
        EXPECT_EQ(node2, bulk_data.begin_nodes(spring)[1]);
    }
}

// checks bucket information following a mesh modification cycle. information includes
// number of buckets, bucket size, entity bucket ordinals, and memory locations, etc.
void check_buckets(const BulkData &bulk_data)
{
    const int my_parallel_rank = stk::parallel_machine_rank(MPI_COMM_WORLD);
    // setup file name for logging
    std::stringstream bucket_ordinal_file;
    bucket_ordinal_file << "check_buckets_proc" << stk::parallel_machine_rank(MPI_COMM_WORLD) << "_" << count;
    std::ofstream out_file_stream{bucket_ordinal_file.str()};

    // select buckets that are locally owned by the process
    stk::mesh::Part& own_part = bulk_data.mesh_meta_data().locally_owned_part();
    stk::mesh::Selector own_selector(own_part);
    std::vector<stk::mesh::Bucket*> owned_buckets = bulk_data.get_buckets(stk::topology::ELEMENT_RANK, own_selector);
    out_file_stream << "number of owned element buckets: " << owned_buckets.size() << "\n";
    for (const stk::mesh::Bucket *bucket_p : owned_buckets)
    {
        //auto pressure_field = bulk_data.mesh_meta_data().get_field(stk::topology::ELEMENT_RANK, pressure_field_name);
        BulkData& bucket_bulk_data = bucket_p->mesh();
        auto pressure_field = bucket_bulk_data.mesh_meta_data().get_field(stk::topology::ELEMENT_RANK, pressure_field_name);
        out_file_stream << "bucket id: " << bucket_p->bucket_id() << "\n";
        out_file_stream << "bucket size: " << bucket_p->size() << "\n";
        out_file_stream << "bucket memory size: " << bucket_p->memory_size_in_bytes() << "\n";
        out_file_stream << "bucket address: " << bucket_p << "\n";
        // loop over the entities in this bucket
        
        std::vector<Entity> springs_vector;
        bucket_bulk_data.get_entities(stk::topology::ELEMENT_RANK, own_selector, springs_vector);
        for (Entity spring : springs_vector)
        {
            EXPECT_TRUE(bucket_bulk_data.is_valid(spring));
            out_file_stream << "\tSPRING" << bucket_bulk_data.identifier(spring) << " ";
            out_file_stream << "(bucket " << bucket_bulk_data.bucket(spring).bucket_id() << "):\n";
            out_file_stream << "\t\tbucket ordinal:" << bucket_bulk_data.bucket_ordinal(spring);
            out_file_stream << "\t\tlocal offset: " << spring.local_offset();
            out_file_stream << "\t\t field data: " << stk::mesh::field_data(*pressure_field, spring) << "\n";
            out_file_stream << "\t\tis shared? " << std::boolalpha << bucket_bulk_data.in_shared(spring) << "\n";
        }
        std::vector<Entity> nodes_vector;
        bucket_bulk_data.get_entities(stk::topology::NODE_RANK, own_selector, nodes_vector);
        for (Entity node : nodes_vector)
        {
            EXPECT_TRUE(bucket_bulk_data.is_valid(node));
            out_file_stream << "\tNODE" << bucket_bulk_data.identifier(node) << " " ;
            out_file_stream << " (bucket " << bucket_bulk_data.bucket(node).bucket_id() << "):\n";
            out_file_stream << "\t\tbucket ordinal:" << bucket_bulk_data.bucket_ordinal(node);
            out_file_stream << "\t\tlocal offset: " << node.local_offset() << "\n";
            out_file_stream << "\t\tis shared? " << std::boolalpha << bucket_bulk_data.in_shared(node) << "\n";
        }
    }
    out_file_stream << "\n";
    
    std::stringstream file_name; 
    file_name << "process" << my_parallel_rank << "_elements" << count++;
    stk::mesh::impl::dump_mesh_per_proc(bulk_data, file_name.str());
}

// used to analyze the mesh following a modification cycle
void check_mesh(const BulkData& bulk_data)
{
    inspect_springs(bulk_data);
    check_buckets(bulk_data);
}

// springs (specified springs_to_destroy) are destroyed from a mesh (specified by bulk_data_p)
void destroy_springs(std::shared_ptr<BulkData>& bulk_data_p, const SpringProcVec& springs_to_destroy)
{
    const int my_parallel_rank = stk::parallel_machine_rank(MPI_COMM_WORLD);

    stk::mesh::Part* spring_part = bulk_data_p->mesh_meta_data().get_part("spring");
    Entity node1 = bulk_data_p->get_entity(stk::topology::NODE_RANK, 1);
    Entity node2 = bulk_data_p->get_entity(stk::topology::NODE_RANK, 2);

    bulk_data_p->modification_begin("creating springs");
    for (auto pair : springs_to_destroy)
    {
        unsigned int proc = pair.first;
        unsigned int id = pair.second;
        if (my_parallel_rank == proc)
        {
            Entity spring1 = bulk_data_p->get_entity(stk::topology::ELEMENT_RANK, id);
            bulk_data_p->destroy_entity(spring1);
        }
    }
    bulk_data_p->modification_end(MUNDY_MOD_END_SORT_TYPE);
}

// this is used inside create_initial_mesh() and create_springs() to create spring elements.
// it has to be called WITHIN a modification cycle, because there is no modification_begin() or 
// modification_end() inside of this function
void create_springs_no_mod(std::shared_ptr<BulkData>& bulk_data_p, const SpringProcVec& springs_to_destroy)
{
    const int my_parallel_rank = stk::parallel_machine_rank(MPI_COMM_WORLD);

    stk::mesh::Part* spring_part = bulk_data_p->mesh_meta_data().get_part("spring");
    Entity node1 = bulk_data_p->get_entity(stk::topology::NODE_RANK, 1);
    Entity node2 = bulk_data_p->get_entity(stk::topology::NODE_RANK, 2);

    for (auto pair : springs_to_destroy)
    {
        unsigned int proc = pair.first;
        unsigned int id = pair.second;
        if (my_parallel_rank == proc)
        {
            Entity spring = bulk_data_p->declare_element(id, stk::mesh::ConstPartVector{spring_part});
            bulk_data_p->declare_relation(spring, node1, 0);
            bulk_data_p->declare_relation(spring, node2, 1);
        }
    }
}

// springs (specified springs_to_create) are created in a mesh (specified by bulk_data_p)
void create_springs(std::shared_ptr<BulkData>& bulk_data_p, const SpringProcVec& springs_to_create)
{
    bulk_data_p->modification_begin("creating springs");
    create_springs_no_mod(bulk_data_p, springs_to_create);
    bulk_data_p->modification_end(MUNDY_MOD_END_SORT_TYPE);
}

// create an initial mesh of two nodes and springs (specified by springs_to_create).
// the resulting mesh is stored in bulk_data_p and returned from the function
// by storing it in bulk_data_p_out
void create_initial_mesh(std::shared_ptr<BulkData>& bulk_data_p_out, const SpringProcVec& springs_to_create)
{
    const int my_parallel_rank = stk::parallel_machine_rank(MPI_COMM_WORLD);
    const int other_parallel_rank = my_parallel_rank == 0 ? 1 : 0;

    // create an instance of a MeshBuilder to build a simple mesh
    const std::size_t spatial_dimension = 3; // set spatial dimensions
    stk::mesh::MeshBuilder mesh_builder(MPI_COMM_WORLD); // initialize the mesh builder
    mesh_builder.set_spatial_dimension(spatial_dimension); // tell the builder about the spatial dimensions
    mesh_builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"}); // tell the builder about entity ranks

    // create an instance of MetaData for the mesh
    std::shared_ptr<stk::mesh::MetaData> meta_data_p = mesh_builder.create_meta_data();
    meta_data_p->use_simple_fields();
    stk::mesh::Part& spring_part = meta_data_p->declare_part_with_topology("spring", stk::topology::BEAM_2);
    
    ScalarField& pressure_field = meta_data_p->declare_field<double>(stk::topology::ELEM_RANK, pressure_field_name);
    stk::mesh::put_field_on_entire_mesh_with_initial_value(pressure_field, &init_pressure);
    meta_data_p->commit();

    // create an instance of BulkData given the meta data
    std::shared_ptr<stk::mesh::BulkData> bulk_data_p = mesh_builder.create(meta_data_p);

    bulk_data_p->modification_begin("create initial mesh");
    
    Entity node1 = bulk_data_p->declare_node(1);
    Entity node2 = bulk_data_p->declare_node(2);
    bulk_data_p->add_node_sharing(node1, other_parallel_rank);
    bulk_data_p->add_node_sharing(node2, other_parallel_rank);
    create_springs_no_mod(bulk_data_p, springs_to_create);
    
    bulk_data_p->modification_end(MUNDY_MOD_END_SORT_TYPE);

    // pass bulk data out of here to use with destroy_one_spring() and create_one_spring()
    bulk_data_p_out = bulk_data_p;
}

// wrapper for finding the big O complexity of a given function `func`
template<typename T>
auto complexity(const std::string& name, const int& size, ankerl::nanobench::Bench& bench, T&& func)
{
    auto decorated_function = [func = std::forward<T>(func), &name, &size, &bench](auto&&... args)->void
    {
        bench.minEpochIterations(25).complexityN(size).run(name, [&]{
            func(std::forward<decltype(args)>(args)...);
        });
    };
    return decorated_function;
}

// full program with benchmarking
void run_benchmarks()
{
    const std::vector<unsigned int> n_springs_vec = {5000U, 5000U, 5000U, 5000U};
    const std::vector<unsigned int> n_springs_modified_vec = {1000U, 2000U, 3000U, 4000U};
    assert(n_springs_vec.size() == n_springs_modified_vec.size());

    // each benchmark needs its own instance of nanobench::Bench
    ankerl::nanobench::Bench create_mesh_bench;
    ankerl::nanobench::Bench destroy_springs_bench;
    ankerl::nanobench::Bench create_springs_bench;
    for (std::size_t i = 0; i < n_springs_vec.size(); i++)
    {
        const unsigned int n_springs = n_springs_vec[i];
        const unsigned int n_springs_to_destroy = n_springs_modified_vec[i];
        const unsigned int n_springs_to_create = n_springs_modified_vec[i];
        const unsigned int size = n_springs_to_destroy;

        std::shared_ptr<BulkData> bulk_data_p;

        SpringProcVec initial_springs_to_create(n_springs);
        SpringProcVec springs_to_destroy(n_springs_to_destroy);
        SpringProcVec springs_to_create(n_springs_to_create);
        initialize_vectors(initial_springs_to_create, springs_to_destroy, springs_to_create, Add::Same);
        complexity("create mesh", size, create_mesh_bench, create_initial_mesh)(bulk_data_p, initial_springs_to_create);
        complexity("destroy springs", size, destroy_springs_bench, destroy_springs)(bulk_data_p, springs_to_destroy);
        complexity("create springs", size, create_springs_bench, create_springs)(bulk_data_p, springs_to_create);
    } 
    std::cout << "\n";
    std::cout << "create mesh complexity: \n" << create_mesh_bench.complexityBigO() << "\n"; 
    std::cout << "destroy springs complexity: \n" << destroy_springs_bench.complexityBigO() << "\n"; 
    std::cout << "create springs complexity: \n" << create_springs_bench.complexityBigO() << "\n"; 
}

// full program without benchmarking
void run_no_benchmarks()
{
    const unsigned int n_springs = 5;
    const unsigned int n_springs_modified = 2;
    const unsigned int n_springs_to_destroy = n_springs_modified;
    const unsigned int n_springs_to_create = n_springs_modified;

    std::shared_ptr<BulkData> bulk_data_p;

    SpringProcVec initial_springs_to_create(n_springs);
    SpringProcVec springs_to_destroy(n_springs_to_destroy);
    SpringProcVec springs_to_create(n_springs_to_create);
    initialize_vectors(initial_springs_to_create, springs_to_destroy, springs_to_create, Add::Other);

    create_initial_mesh(bulk_data_p, initial_springs_to_create);
    check_mesh(*bulk_data_p);

    destroy_springs(bulk_data_p, springs_to_destroy);
    check_mesh(*bulk_data_p);
    
    create_springs(bulk_data_p, springs_to_create);
    check_mesh(*bulk_data_p);
}

int main(int argc, char **argv)
{
    stk::parallel_machine_init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    const int n_proc = stk::parallel_machine_size(MPI_COMM_WORLD);

    #ifdef USE_NANOBENCH
        assert(n_proc == 1);
        run_benchmarks();
    #else
        assert(n_proc == 1 || n_proc == 2);
        run_no_benchmarks();
    #endif

    Kokkos::finalize();
    stk::parallel_machine_finalize();

    return 0;
}