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

#ifndef MUNDY_MULTIBODY_MULTIBODYREGISTRY_HPP_
#define MUNDY_MULTIBODY_MULTIBODYREGISTRY_HPP_

/// \file Registry.hpp
/// \brief Declaration of the Registry class

// C++ core libs
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <utility>      // for std::pair

// Mundy libs
#include <mundy_multibody/MultibodyFactory.hpp>  // for mundy::multibody::MultibodyFactory

namespace mundy {

namespace multibody {

/// \class MultibodyRegistry
/// \brief A class for registering \c Multibodys with \c MultibodyFactory.
///
/// Most users shouldn't directly interface with this registry; instead, registration is performed using the provided
/// MUNDY_REGISTER_MULTIBODYTYPE macro. See the documentation for that macro for more information.
///
/// \tparam ClassToRegister A class derived from \c Multibody that we wish to register.
template <class ClassToRegister>
struct MultibodyRegistry {
  //! \name Member variable definitions
  //@{

  /// @brief A flag for if the given type has been registered with \c MultibodyFactory or not.
  inline static const bool is_registered = false;
  //@}
};  // MultibodyRegistry

}  // namespace multibody

}  // namespace mundy

/// \brief A helper macro for checking if a \c Multibody has been registered with \c MultibodyFactory.
///
/// This macro is used to check if a \c Multibody has been registered with the \c MultibodyFactory. The macro should
/// be used in the following way:
///
/// \code{.cpp}
/// MUNDY_IS_MULTIBODYTYPE_REGISTERED(ClassToCheck)
/// \endcode
///
/// \note This macro used a lambda function to check if the class has been registered. This ensures that each use of
/// \c MUNDY_IS_MULTIBODYTYPE_REGISTERED does not create a new definition of \c is_registered, thereby avoiding multiple
/// definition errors.
///
/// \param ClassToCheck A class derived from \c Multibody that we wish to check if it has been registered.
#define MUNDY_IS_MULTIBODYTYPE_REGISTERED(ClassToCheck)                      \
  ([]() -> bool {                                                            \
    return mundy::multibody::MultibodyRegistry<ClassToCheck>::is_registered; \
  }())

/// @brief A helper macro for registering a \c Multibody with \c MultibodyFactory.
///
/// This macro is used to register a \c Multibody with \c MultibodyFactory. The macro should be
/// used in the following way:
///
/// \code{.cpp}
/// MUNDY_REGISTER_MULTIBODYTYPE(ClassToRegister)
/// \endcode
///
/// There are some important notes about proper use of this macro:
///
/// - Registration in Global Scope: The registration should typically be done in the global scope, not inside any
/// function (including main()). This is because if the registration is done inside a function, it will not happen until
/// that function is called, which could be after main() starts.
///
/// - Registration in Source Files: Best practice is to perform registration in a source file, not a header file. This
/// is because if the registration is done in a header file, it will be registered in every translation unit that
/// includes that header file. This can lead to multiple registrations of the same class, which will cause an error.
/// Using header guards will not prevent this issue.
///
/// - No Dependency on Other Static Variables in Registration: Since C++ doesn't guarantee an order of initialization
/// for static variables across different translation units, you should make sure that the registration of a multibody
/// type does not depend on the initialization of other static variables. If it does, those variables might not be
/// initialized at the time the multibody type is registered.
///
/// - No Duplicated Names: Each multibody type must be registered (to a certain MetaFactory) with a unique
/// identifier. If two classes are registered with the same identifier, only the first one will actually be registered,
/// and subsequent registrations will throw an error. While allowing overwriting of existing registrations is possible,
/// we chose to not implement such functionality because it could lead to unexpected behavior. Specifically, when using
/// static initialization across multiple translation units, there is no guaranteed order of initialization. This means
/// that if two classes are registered with the same identifier, it is not guaranteed which one will be registered
/// first.
///
/// As long as these points are followed, the registration of multibody type subclasses should occur before main()
/// starts.
///
/// \note To anyone reading this code, it may seem like the is_registered variable is unused. However, this is not the
/// case. When the program is started, one of the first steps is to initialize static objects. Even if is_registered
/// appears to be unused, static storage duration guarantees that this variable won’t be optimized away.
///
/// \param ClassToRegister A class derived from \c Multibody that we wish to register.
#define MUNDY_REGISTER_MULTIBODYTYPE(ClassToRegister)                   \
  namespace mundy {                                                     \
  namespace multibody {                                                 \
  template <>                                                           \
  struct MultibodyRegistry<ClassToRegister> {                           \
    inline static const bool is_registered = ([]() -> bool {            \
      MultibodyFactory::template register_new_class<ClassToRegister>(); \
      return true;                                                      \
    })();                                                               \
  };                                                                    \
  }                                                                     \
  }

#endif  // MUNDY_MULTIBODY_MULTIBODYREGISTRY_HPP_
