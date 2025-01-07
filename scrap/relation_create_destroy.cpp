// C++ core libs
#include <memory>           // for std::shared_ptr, std::unique_ptr
#include <stdexcept>        // for std::logic_error, std::invalid_argument
#include <string>           // for std::string
#include <vector>           // for std::vector
#include <iostream>         // for std::cout, etc.
#include <sstream>          // for std::stringstream
#include <fstream>          // for std::ofstream

#include <numeric>          // for std::iota
#include <algorithm>        // for std::random_shuffle
#include <ctime>            // for std::time
#include <cstdlib>          // for std::rand, std::srand
#include <type_traits>      // for std::invoke_result_t
#include <unordered_map>    // for std::unordered_map
#include <array>            // for std::array

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

//#define USE_NANOBENCH

// test configuration:
//
//
//                         Rank              Entity tree
//                      constraint.................O 
//                                                / \ 
//                                               /   \ 
//                                              /     \ 
//                                             /       \ 
//                                            /         \ 
//                                           /           \ 
//                      elem................O.............O
//                                         /|\           /|\ 
//                                        / | \         / | \ 
//                      edge............./..O..\......./..O  \ 
//                                      /  / \  \     /  / \  \ 
//                                     / /     \ \   / /     \ \ 
//                                    //         \\ //         \\ 
//                      node..........O............O............O 
// 
// determine:
//      1) time difference between fixed relation creation/deletion vs dynamic creation/deleted
//      2) what purpose of BulkData::reserve_relations()
//      3) time difference creating relations one at a time vs multiple for both fixed and dynamic cases
//      4) time difference in fixed relation creation/deletion vs dynamic creation/deleted for cases
//         with and without topology specified during entity creation

// typedefs
using stk::mesh::Entity;
using stk::mesh::EntityRank;
using stk::mesh::EntityId;
using stk::topology::NODE_RANK;
using stk::topology::EDGE_RANK;
using stk::topology::ELEMENT_RANK;
using stk::topology::CONSTRAINT_RANK;
using stk::mesh::MetaData;
using stk::mesh::BulkData;

// just change modification sorting here, rather than in each call to modification_end()
#define MUNDY_MOD_END_SORT_TYPE stk::mesh::impl::MeshModification::MOD_END_SORT

// enumeration of different relation types to test
typedef enum Relation : std::size_t 
{ 
    EdgeNode = 0, 
    ElementNode = 1, ElementEdge = 2, 
    ConstraintNode = 3, ConstraintEdge = 4, ConstraintElement = 5,
    
    Total
} Relation;

// the relations we are interested in measuring. used for identifying the RelationReqs
std::vector<Relation> relations = { 
                                    Relation::EdgeNode, 
                                    Relation::ElementNode, Relation::ElementEdge, 
                                    Relation::ConstraintNode, Relation::ConstraintEdge, Relation::ConstraintElement
                                };

// nanobench uses string names to differentiate benchmarks
std::unordered_map<Relation, std::string> relation_name_map =   {
                                                                    {Relation::EdgeNode, "edge to node"},
                                                                    {Relation::ElementNode, "element to node"},
                                                                    {Relation::ElementEdge, "element to edge"},
                                                                    {Relation::ConstraintNode, "constraint to node"},
                                                                    {Relation::ConstraintEdge, "constraint to edge"},
                                                                    {Relation::ConstraintElement, "constraint to element"}
                                                                };

// for creating relations. the RelationReqs struct below contains information for creating a relation.
// this map is just used to reduce the constructor parameters for that struct
typedef std::pair<EntityRank, EntityRank> FromToPair;
std::unordered_map<Relation, FromToPair> relation_rank_map =    {
                                                                    {Relation::EdgeNode, {EDGE_RANK, NODE_RANK}},
                                                                    {Relation::ElementNode, {ELEMENT_RANK, NODE_RANK}},
                                                                    {Relation::ElementEdge, {ELEMENT_RANK, EDGE_RANK}},
                                                                    {Relation::ConstraintNode, {CONSTRAINT_RANK, NODE_RANK}},
                                                                    {Relation::ConstraintEdge, {CONSTRAINT_RANK, EDGE_RANK}},
                                                                    {Relation::ConstraintElement, {CONSTRAINT_RANK, ELEMENT_RANK}}
                                                                };

// contains information required for declaring or destroying a relation
typedef struct RelationReqs
{
    // for benchmarking
    std::string name;
    Relation relation;
    // for creating/deleting the relation
    EntityRank from_rank;
    EntityId from_id;
    EntityRank to_rank;
    EntityId to_id;

    RelationReqs() = default;
    RelationReqs(Relation relation, EntityId from_id, EntityId to_id)
    :   name{relation_name_map[relation]}, 
        relation{relation},
        from_rank{relation_rank_map[relation].first}, 
        from_id{from_id}, 
        to_rank{relation_rank_map[relation].second}, 
        to_id{to_id}
    {}
    RelationReqs(const RelationReqs& reqs) = default;
    RelationReqs& operator=(const RelationReqs& reqs) = default;
    RelationReqs(RelationReqs&& reqs) = default;
    RelationReqs& operator=(RelationReqs&& reqs) = default;
} RelationReqs;

int main(int argc, char **argv)
{
    stk::parallel_machine_init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    const int n_proc = stk::parallel_machine_size(MPI_COMM_WORLD);
    assert(n_proc == 1);

    // these have to be outside of the for loop
    ankerl::nanobench::Bench create_bench;
    ankerl::nanobench::Bench destroy_bench;
    // number of iterations for big O measurement
    std::vector<int> n_relations_to_modify = {5, 50, 500, 5000};
    // loop over all relations
    for (auto relation : relations)
    {
        for (auto size : n_relations_to_modify)
        {
            // create an instance of a MeshBuilder to build a simple mesh
            const std::size_t spatial_dimension = 3; // set spatial dimensions
            stk::mesh::MeshBuilder mesh_builder(MPI_COMM_WORLD); // initialize the mesh builder
            mesh_builder.set_spatial_dimension(spatial_dimension); // tell the builder about the spatial dimensions
            mesh_builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"}); // tell the builder about entity ranks

            // create an instance of MetaData for the mesh
            std::shared_ptr<stk::mesh::MetaData> meta_data_p = mesh_builder.create_meta_data();
            meta_data_p->use_simple_fields();
            stk::mesh::Part& element_part = meta_data_p->declare_part("element", stk::topology::ELEMENT_RANK);
            stk::mesh::Part& edge_part = meta_data_p->declare_part("edge", stk::topology::EDGE_RANK);
            stk::mesh::Part& node_part = meta_data_p->declare_part("node", stk::topology::NODE_RANK);
            stk::mesh::Part& constraint_part = meta_data_p->declare_part("constraint", stk::topology::CONSTRAINT_RANK);
            meta_data_p->commit();

            // create an instance of BulkData given the meta data
            std::shared_ptr<BulkData> bulk_data_p = mesh_builder.create(meta_data_p);

            bulk_data_p->modification_begin("create initial mesh");

            int j = 0; // count for non-node entities
            // you don't need all of these entities for each iteration of the outer for loop, for (auto relation : relations),
            // but it's not that big of a deal to build them and ignore what you don't need. then there is less tweaking
            // and recompiling for different tests without introducing additional enums, maps, etc.
            for (int i = 1; i < 2*size + 1; i += 2)
            {
                j += 1;
                // nodes
                bulk_data_p->declare_node(i, stk::mesh::ConstPartVector{&node_part});
                bulk_data_p->declare_node(i + 1, stk::mesh::ConstPartVector{&node_part});
                // non-nodes
                bulk_data_p->declare_edge(j, stk::mesh::ConstPartVector{&edge_part});
                bulk_data_p->declare_element(j, stk::mesh::ConstPartVector{&element_part});
                bulk_data_p->declare_constraint(j, stk::mesh::ConstPartVector{&constraint_part});
            }
            bulk_data_p->modification_end(MUNDY_MOD_END_SORT_TYPE);

            // fill in the required information for each relation to be created and then destroyed
            std::vector<EntityId> from_ids(size);
            std::iota(from_ids.begin(), from_ids.end(), 1);
            std::vector<EntityId> to_ids(size);
            std::iota(to_ids.begin(), to_ids.end(), 1);
            std::vector<RelationReqs> requirements(size);
            for (int i = 0; i < size; i++)
            {
                EntityId from_id = from_ids[i];
                EntityId to_id = to_ids[i];
                requirements[i] = RelationReqs{relation, from_id, to_id};
            }

            // time each of the relations being created
            create_bench.minEpochIterations(1000).complexityN(size).run("create relation", [&]{
                // create relations
                int relation_id = 0; // referred to as an "attribute" in the document
                bulk_data_p->modification_begin("create relation");
                for (auto& req : requirements)
                {
                    relation_id += 1;
                    Entity from_entity = bulk_data_p->get_entity(req.from_rank, req.from_id);
                    Entity to_entity = bulk_data_p->get_entity(req.to_rank, req.to_id);
                    bulk_data_p->declare_relation(from_entity, to_entity, relation_id);
                }
                bulk_data_p->modification_end(MUNDY_MOD_END_SORT_TYPE);
            });

            // time each of the relations being destroyed
            destroy_bench.minEpochIterations(1000).complexityN(size).run("destroy relation", [&]{
                int relation_id = 0; // referred to as an "attribute" in the document
                bulk_data_p->modification_begin("destroy relation");
                for (auto& req : requirements)
                {
                    relation_id += 1;
                    Entity from_entity = bulk_data_p->get_entity(req.from_rank, req.from_id);
                    Entity to_entity = bulk_data_p->get_entity(req.to_rank, req.to_id);
                    bool success = bulk_data_p->destroy_relation(from_entity, to_entity, relation_id);  
                }
                bulk_data_p->modification_end(MUNDY_MOD_END_SORT_TYPE); 
            });

        }
        std::cout << "\n";
        std::cout << "create " << relation_name_map[relation] << " relation big O: " << create_bench.complexityBigO() << "\n";
        std::cout << "destroy " << relation_name_map[relation] << " relation big O: " << destroy_bench.complexityBigO() << "\n";
    }
    
    Kokkos::finalize();
    stk::parallel_machine_finalize();

    return 0;
}