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

#ifndef MUNDY_LINKER_GENERATENEIGHBORLINKERS_HPP_
#define MUNDY_LINKER_GENERATENEIGHBORLINKERS_HPP_

/// \file GenerateNeighborLinkers.hpp
/// \brief Declaration of the GenerateNeighborLinkers class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_core/StringLiteral.hpp>                                     // for mundy::core::make_string_literal
#include <mundy_linker/generate_neighbor_linkers/techniques/STKSearch.hpp>  // for mundy::linker::generate_neighbor_linkers::techniques::STKSearch
#include <mundy_mesh/BulkData.hpp>                                          // for mundy::mesh::BulkData
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::GlobalMetaMethodRegistry
#include <mundy_meta/MetaTechniqueDispatcher.hpp>  // for mundy::meta::MetaMethodPairwiseSubsetExecutionDispatcher

namespace mundy {

namespace linker {

/// \class GenerateNeighborLinkers
/// \brief Method for generating neighbor linkers between source-target entity pairs.
class GenerateNeighborLinkers
    : public mundy::meta::MetaMethodPairwiseSubsetExecutionDispatcher<
          GenerateNeighborLinkers, void, mundy::meta::make_registration_string("GENERATE_NEIGHBOR_LINKERS"),
          mundy::meta::make_registration_string("STK_SEARCH")> {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  GenerateNeighborLinkers() = delete;

  /// \brief Constructor
  GenerateNeighborLinkers(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
      : mundy::meta::MetaMethodPairwiseSubsetExecutionDispatcher<
            GenerateNeighborLinkers, void, mundy::meta::make_registration_string("GENERATE_NEIGHBOR_LINKERS"),
            mundy::meta::make_registration_string("STK_SEARCH")>(bulk_data_ptr, fixed_params) {
  }
  //@}

  //! \name MetaMethodPairwiseSubsetExecutionDispatcher static interface implementation
  //@{

  /// \brief Get the valid fixed parameters that we require all techniques registered with our technique factory to
  /// have.
  static Teuchos::ParameterList get_valid_required_technique_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;

    default_parameter_list.set("neighbor_linkers_part_name", default_neighbor_linkers_part_name_,
                               "The part name to which we will add the neighbor linkers.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters that we require all techniques registered with our technique factory to
  /// have.
  static Teuchos::ParameterList get_valid_required_technique_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_neighbor_linkers_part_name_ = "NEIGHBOR_LINKERS";
  //@}
};  // GenerateNeighborLinkers

}  // namespace linker

}  // namespace mundy

//! \name Registration
//@{

/// @brief Register our default techniques
MUNDY_REGISTER_METACLASS("STK_SEARCH", mundy::linker::generate_neighbor_linkers::techniques::STKSearch,
                         mundy::linker::GenerateNeighborLinkers::OurTechniqueFactory)

//@}

#endif  // MUNDY_LINKER_GENERATENEIGHBORLINKERS_HPP_
