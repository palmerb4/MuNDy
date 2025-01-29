
// How to check for subsets:
#include <type_traits>
#include <iostream>

// 1. Wrap types into a parameter-pack container (a type_list)
template <typename... Ts>
struct type_list {};

// 2. contains<T, Pack...> checks if T is one of the Pack types
template <typename T, typename... Pack>
struct contains : std::bool_constant<(std::is_same_v<T, Pack> || ...)> {};

// 3. is_subset_of<type_list<A...>, type_list<B...>> checks if all A... are in B...
template <typename, typename>
struct is_subset_of;

template <typename... As, typename... Bs>
struct is_subset_of<type_list<As...>, type_list<Bs...>>
  : std::bool_constant<(contains<As, Bs...>::value && ...)> {};

// Helper variable templates for clarity
template <typename T, typename... Pack>
inline constexpr bool contains_v = contains<T, Pack...>::value;

template <typename AList, typename BList>
inline constexpr bool is_subset_of_v = is_subset_of<AList, BList>::value;

// Demo
int main()
{
    using SetA = type_list<int, double>;
    using SetB = type_list<int, float, double, char>;

    static_assert( is_subset_of_v<SetA, SetB>, "SetA should be a subset of SetB" );
    std::cout << "is_subset_of<SetA, SetB> = " 
              << std::boolalpha 
              << is_subset_of_v<SetA, SetB>
              << std::endl;  // prints true

    using SetC = type_list<int, bool>;
    static_assert( !is_subset_of_v<SetC, SetB>, "SetC should NOT be a subset of SetB" );
    std::cout << "is_subset_of<SetC, SetB> = "
              << std::boolalpha
              << is_subset_of_v<SetC, SetB>
              << std::endl;  // prints false
}
