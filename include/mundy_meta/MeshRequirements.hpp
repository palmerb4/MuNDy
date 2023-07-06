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

#ifndef MUNDY_META_MESHREQUIREMENTS_HPP_
#define MUNDY_META_MESHREQUIREMENTS_HPP_

/// \file MeshRequirements.hpp
/// \brief Declaration of the MeshRequirements class

// C++ core libs
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>       // for Teuchos::ParameterList
#include <stk_mesh/base/Bucket.hpp>        // for stk::mesh::get_default_bucket_capacity
#include <stk_mesh/base/Part.hpp>          // for stk::mesh::Part
#include <stk_topology/topology.hpp>       // for stk::topology
#include <stk_util/parallel/Parallel.hpp>  // for stk::ParallelMachine

// Mundy libs
#include <mundy_mesh/BulkData.hpp>           // for mundy::mesh::BulkData
#include <mundy_meta/FieldRequirements.hpp>  // for mundy::meta::FieldRequirements, mundy::meta::FieldRequirementsBase
#include <mundy_meta/PartRequirements.hpp>   // for mundy::meta::PartRequirements

namespace mundy {

namespace meta {

/// \class MeshRequirements
/// \brief A set requirements imposed upon a MetaMesh, its Parts, and its Fields.
class MeshRequirements {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default construction is allowed
  /// Default construction corresponds to having no requirements.
  MeshRequirements() = default;

  /// \brief Construct a fully specified set of mesh requirements.
  ///
  /// \param comm [in] The MPI communicator.
  explicit MeshRequirements(const stk::ParallelMachine &comm);
  
  /// \brief Construct from a parameter list.
  ///
  /// \param parameter_list [in] Optional list of parameters for specifying the mesh requirements. The set of valid
  /// parameters is accessible via \c get_valid_params.
  explicit MeshRequirements(const Teuchos::ParameterList &parameter_list);
  //@}

  //! \name Setters
  //@{

  /// \brief Set the spatial dimension of the mash.
  /// \param spatial_dimension [in] The dimension of the space within which the parts and entities reside.
  void set_spatial_dimension(const unsigned spatial_dimension);

  /// \brief Set the names assigned to each rank.
  /// \param entity_rank_names [in] The names assigned to each rank.
  void set_entity_rank_names(const std::vector<std::string> &entity_rank_names);

  /// \brief Set the MPI communicator to be used by STK.
  /// \param comm [in] The MPI communicator.
  void set_communicator(const stk::ParallelMachine &comm);

  /// \brief Set the chosen Aura option. For example, mundy::mesh::BulkData::AUTO_AURA.
  /// \param aura_option [in] The chosen Aura option.
  void set_aura_option(const mundy::mesh::BulkData::AutomaticAuraOption &aura_option);

  /// \brief Set the field data manager.
  /// \param field_data_manager_ptr [in] Pointer to an existing field data manager.
  void set_field_data_manager(stk::mesh::FieldDataManager *const field_data_manager_ptr);

  /// \brief Set the upper bound on the number of mesh entities that may be associated with a single bucket.
  ///
  /// Although subject to change, the maximum bucket capacity is currently 1024 and the default capacity is 512.
  ///
  /// \param bucket_capacity [in] The bucket capacity.
  void set_bucket_capacity(const unsigned bucket_capacity);

  /// \brief Set the flag specifying if upward connectivity will be enabled or not.
  /// \param enable_upward_connectivity [in] A flag specifying if upward connectivity will be enabled or not.
  void set_upward_connectivity_flag(const bool enable_upward_connectivity);

  /// \brief Delete the spatial dimension constraint (if it exists).
  void delete_spatial_dimension();

  /// \brief Delete the entity rank names constraint (if it exists).
  void delete_entity_rank_names();

  /// \brief Delete the communicator constraint (if it exists).
  void delete_communicator();

  /// \brief Delete the aura option constraint (if it exists).
  void delete_aura_option();

  /// \brief Delete the field data manager constraint (if it exists).
  void delete_field_data_manager();

  /// \brief Delete the bucket capacity constraint (if it exists).
  void delete_bucket_capacity();

  /// \brief Delete the upward connectivity flag constraint (if it exists).
  void delete_upward_connectivity_flag();

  /// \brief Add the provided field to the part, given that it is valid and does not conflict with existing fields.
  ///
  /// \param field_req_ptr [in] Pointer to the field parameters to add to the part.
  void add_field_reqs(std::shared_ptr<FieldRequirementsBase> field_req_ptr);

  /// \brief Add the provided part to the mesh, given that it is valid.
  ///
  /// TODO(palmerb4): Are there any restrictions on what can and cannot be a part? If so, encode them here.
  ///
  /// \param part_req_ptr [in] Pointer to the part requirements to add to the mesh.
  void add_part_reqs(std::shared_ptr<PartRequirements> part_req_ptr);

  /// \brief Store a copy of an attribute on the mesh.
  ///
  /// Attributes are fetched from an mundy::mesh::MetaData via the get_attribute<T> routine. As a result, the
  /// identifying feature of an attribute is its type. If you attempt to add a new attribute requirement when an
  /// attribute of that type already exists, then the contents of the two attributes must match.
  ///
  /// Note, in all-too-common case where one knows the type of the desired attribute but wants to specify the value
  /// post-mesh construction, we suggest that you set store a void shared or unique pointer inside of some_attribute.
  ///
  /// \param some_attribute Any attribute that you wish to store on the mesh.
  void add_mesh_attribute(const std::any &some_attribute);

  /// \brief Store an attribute on the mesh.
  ///
  /// Attributes are fetched from an mundy::mesh::MetaData via the get_attribute<T> routine. As a result, the
  /// identifying feature of an attribute is its type. If you attempt to add a new attribute requirement when an
  /// attribute of that type already exists, then the contents of the two attributes must match.
  ///
  /// Note, in all-too-common case where one knows the type of the desired attribute but wants to specify the value
  /// post-mesh construction, we suggest that you set store a void shared or unique pointer inside of some_attribute.
  ///
  /// \param some_attribute Any attribute that you wish to store on the mesh.
  void add_mesh_attribute(std::any &&some_attribute);
  //@}

  //! \name Getters
  //@{

  /// \brief Get if the spatial dimension is constrained or not.
  bool constrains_spatial_dimension() const;

  /// \brief Get if the entity rank names are constrained or not.
  bool constrains_entity_rank_names() const;

  /// \brief Get if the communicator is constrained or not.
  bool constrains_communicator() const;

  /// \brief Get if the aura option is constrained or not.
  bool constrains_aura_option() const;

  /// \brief Get if the field data manager is constrained or not.
  bool constrains_field_data_manager() const;

  /// \brief Get if the bucket capacity is constrained or not.
  bool constrains_bucket_capacity() const;

  /// \brief Get if the upward connectivity flag is constrained or not.
  bool constrains_upward_connectivity_flag() const;

  /// @brief Get if the mesh is fully specified.
  bool is_fully_specified() const;

  /// \brief Return the dimension of the space within which the parts and entities reside.
  unsigned get_spatial_dimension() const;

  /// \brief Return the names assigned to each rank.
  std::vector<std::string> get_entity_rank_names() const;

  /// \brief Return the MPI communicator to be used by STK.
  stk::ParallelMachine get_communicator() const;

  /// \brief Return the chosen Aura option. For example, mundy::mesh::BulkData::AUTO_AURA.
  mundy::mesh::BulkData::AutomaticAuraOption get_aura_option() const;

  /// \brief Return the field data manager.
  stk::mesh::FieldDataManager *get_field_data_manager() const;

  /// \brief Return the upper bound on the number of mesh entities that may be associated with a single bucket.
  unsigned get_bucket_capacity() const;

  /// \brief Return if upward connectivity will be enabled or not.
  /// \param enable_upward_connectivity [in] A flag specifying if upward connectivity will be enabled or not.
  bool get_upward_connectivity_flag() const;

  /// \brief Return the mesh field map for each rank.
  std::vector<std::map<std::string, std::shared_ptr<FieldRequirementsBase>>> get_mesh_ranked_field_map();

  /// \brief Return the mesh part map.
  std::map<std::string, std::shared_ptr<PartRequirements>> get_mesh_part_map();

  /// \brief Return the mesh attribute map.
  std::map<std::type_index, std::any> get_mesh_attributes_map();

  /// \brief Validate the given parameters and set the default values if not provided.
  static void validate_parameters_and_set_defaults(Teuchos::ParameterList *parameter_list_ptr) {
    if (parameter_list_ptr->isParameter("spatial_dimension")) {
      const bool valid_type = parameter_list_ptr->INVALID_TEMPLATE_QUALIFIER isType<unsigned>("spatial_dimension");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                                 "MeshRequirements: Type error. Given a parameter with name 'spatial_dimension' but "
                                 "with a type other than unsigned");
    } else {
      parameter_list_ptr->set("spatial_dimension", default_spatial_dimension_,
                              "Dimension of the space within which the parts and entities reside.");
    }

    if (parameter_list_ptr->isParameter("entity_rank_names")) {
      const bool valid_type =
          parameter_list_ptr->INVALID_TEMPLATE_QUALIFIER isType<Teuchos::Array<std::string>>("entity_rank_names");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                                 "MeshRequirements: Type error. Given a parameter with name 'entity_rank_names' but "
                                 "with a type other than Teuchos::Array<std::string>");
    } else {
      parameter_list_ptr->set("entity_rank_names", default_entity_rank_names_,
                              "Vector of names assigned to each rank.");
    }

    // if (parameter_list_ptr->isParameter("communicator")) {
    //   const bool valid_type =
    //       parameter_list_ptr->INVALID_TEMPLATE_QUALIFIER isType<stk::ParallelMachine>("communicator");
    //   MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
    //                              "MeshRequirements: Type error. Given a parameter with name 'communicator' but with a
    //                              " "type other than stk::ParallelMachine");
    // } else {
    //   parameter_list_ptr->set("communicator", default_communicator_, "MPI communicator to be used by STK.");
    // }

    if (parameter_list_ptr->isParameter("aura_option")) {
      const bool valid_type =
          parameter_list_ptr->INVALID_TEMPLATE_QUALIFIER isType<mundy::mesh::BulkData::AutomaticAuraOption>(
              "aura_option");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                                 "MeshRequirements: Type error. Given a parameter with name 'aura_option' but with a "
                                 "type other than mundy::mesh::BulkData::AutomaticAuraOption");
    } else {
      parameter_list_ptr->set("aura_option", default_aura_option_, "The chosen Aura option.");
    }

    if (parameter_list_ptr->isParameter("field_data_manager_ptr")) {
      const bool valid_type = parameter_list_ptr->INVALID_TEMPLATE_QUALIFIER isType<stk::mesh::FieldDataManager *>(
          "field_data_manager_ptr");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                                 "MeshRequirements: Type error. Given a parameter with name 'field_data_manager_ptr' "
                                 "but with a type other than stk::mesh::FieldDataManager *");
    } else {
      parameter_list_ptr->set("field_data_manager_ptr", default_field_data_manager_ptr_,
                              "A pointer to a preexisting field data manager.");
    }

    if (parameter_list_ptr->isParameter("bucket_capacity")) {
      const bool valid_type = parameter_list_ptr->INVALID_TEMPLATE_QUALIFIER isType<unsigned>("bucket_capacity");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                                 "MeshRequirements: Type error. Given a parameter with name 'bucket_capacity' but with "
                                 "a type other than unsigned");
    } else {
      parameter_list_ptr->set(
          "bucket_capacity", default_bucket_capacity_,
          "Upper bound on the number of mesh entities that may be associated with a single bucket.");
    }

    if (parameter_list_ptr->isParameter("upward_connectivity_flag")) {
      const bool valid_type = parameter_list_ptr->INVALID_TEMPLATE_QUALIFIER isType<bool>("upward_connectivity_flag");
      MUNDY_THROW_ASSERT(valid_type, std::invalid_argument,
                                 "MeshRequirements: Type error. Given a parameter with name 'upward_connectivity_flag' "
                                 "but with a type other than bool");
    } else {
      parameter_list_ptr->set("upward_connectivity_flag", default_upward_connectivity_flag_,
                              "Flag specifying if upward connectivity will be enabled or not.");
    }

    if (parameter_list_ptr->isSublist("fields")) {
      Teuchos::ParameterList &fields_sublist = parameter_list_ptr->sublist("fields");
      const unsigned num_fields = fields_sublist.get<unsigned>("count");
      for (unsigned i = 0; i < num_fields; i++) {
        Teuchos::ParameterList &field_i_sublist = parameter_list_ptr->sublist("field_" + std::to_string(i));
        FieldRequirementsBase::validate_parameters_and_set_defaults(&field_i_sublist);
      }
    }

    if (parameter_list_ptr->isSublist("parts")) {
      Teuchos::ParameterList &fields_sublist = parameter_list_ptr->sublist("parts");
      const unsigned num_fields = fields_sublist.get<unsigned>("count");
      for (unsigned i = 0; i < num_fields; i++) {
        Teuchos::ParameterList &part_i_sublist = parameter_list_ptr->sublist("parts_" + std::to_string(i));
        PartRequirements::validate_parameters_and_set_defaults(&part_i_sublist);
      }
    }
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Declare the mesh that this class defines including any of its fields, parts, and their
  /// fields/parts/subparts.
  ///
  /// The only setting that must be specified before declaring the mesh is the MPI communicator; all other settings have
  /// default options which will be used if not set.
  std::shared_ptr<mundy::mesh::BulkData> declare_mesh() const;

  /// \brief Ensure that the current set of parameters is valid.
  ///
  /// Here, valid means:
  ///   - TODO(palmerb4): What are the mesh invariants set by STK?
  void check_if_valid() const;

  /// \brief Merge the current requirements with another \c MeshRequirements.
  ///
  /// \param part_req_ptr [in] An \c MeshRequirements object to merge with the current object.
  void merge(const std::shared_ptr<MeshRequirements> &mesh_req_ptr);

  /// \brief Merge the current requirements with any number of other \c MeshRequirements.
  ///
  /// \param vector_of_part_req_ptrs [in] A vector of pointers to other \c MeshRequirements objects to merge with the
  /// current object.
  void merge(const std::vector<std::shared_ptr<MeshRequirements>> &vector_of_mesh_req_ptrs);
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr unsigned default_spatial_dimension_ = 3;
  static const inline Teuchos::Array<std::string> default_entity_rank_names_ = Teuchos::Array<std::string>();
  static constexpr stk::ParallelMachine default_communicator_ = MPI_COMM_NULL;
  static constexpr mundy::mesh::BulkData::AutomaticAuraOption default_aura_option_ = mundy::mesh::BulkData::AUTO_AURA;
  static constexpr stk::mesh::FieldDataManager *default_field_data_manager_ptr_ = nullptr;
  static const unsigned default_bucket_capacity_;  // Unlike the others, this parameter cannot be filled inline.
  static constexpr bool default_upward_connectivity_flag_ = true;
  //@}

  /// @brief The dimension of the space within which the parts and entities reside.
  unsigned spatial_dimension_;

  /// @brief The names assigned to each rank.
  std::vector<std::string> entity_rank_names_;

  /// @brief The MPI communicator to be used by STK.
  stk::ParallelMachine communicator_;

  /// @brief The chosen Aura option. For example, mundy::mesh::BulkData::AUTO_AURA.
  mundy::mesh::BulkData::AutomaticAuraOption aura_option_;

  /// @brief Pointer to an existing field data manager.
  stk::mesh::FieldDataManager *field_data_manager_ptr_;

  /// @brief Upper bound on the number of mesh entities that may be associated with a single bucket.
  unsigned bucket_capacity_;

  /// @brief A flag specifying if upward connectivity will be enabled or not.
  bool upward_connectivity_flag_;

  /// \brief If the spacial dimension is set or not.
  bool spatial_dimension_is_set_ = false;

  /// \brief If the names of each rank are set or not.
  bool entity_rank_names_is_set_ = false;

  /// \brief If the MPI communicator is set or not.
  bool communicator_is_set_ = false;

  /// \brief If the aura option is set or not.
  bool aura_option_is_set_ = false;

  /// \brief If the field manager is set or not.
  bool field_data_manager_ptr_is_set_ = false;

  /// \brief If the bucket capacity is set or not.
  bool bucket_capacity_is_set_ = false;

  /// \brief If the upward connectivity flag is set or not.
  bool upward_connectivity_flag_is_set_ = false;

  /// \brief A set of maps from field name to field params for each rank.
  std::vector<std::map<std::string, std::shared_ptr<FieldRequirementsBase>>> mesh_ranked_field_maps_{
      stk::topology::NUM_RANKS};

  /// \brief A map from part name to the part params for that part.
  std::map<std::string, std::shared_ptr<PartRequirements>> mesh_part_map_;

  /// \brief A map from attribute type to this field's attributes.
  std::map<std::type_index, std::any> mesh_attributes_map_;
};  // MeshRequirements

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_MESHREQUIREMENTS_HPP_
