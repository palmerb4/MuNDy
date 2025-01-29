#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <map>

struct Invalid {
  static inline std::string name = "INVALID";
  static constexpr unsigned value = 0;
};

struct A {
  static inline std::string name = "A";
  static constexpr unsigned value = 333;
};

struct B {
  static inline std::string name = "B";
  static constexpr unsigned value = 222;
};

template <unsigned Value>
struct value_to_type {
    using type = Invalid;
};

template <unsigned Value>
using value_to_type_t = typename value_to_type<Value>::type;

template<>
struct value_to_type<0> {
    using type = Invalid;
};

template<>
struct value_to_type<333> {
    using type = A;
};

template<>
struct value_to_type<222> {
    using type = B;
};

/// Static functor wrapper
template <typename Functor, unsigned Value>
struct FunctorWrapper {
  static void apply(const Functor &functor) {
    using type = value_to_type_t<Value>;
    functor(type{});
  }
};

template <typename Functor, unsigned... Values>
const auto get_functor_jump_table_impl(std::integer_sequence<unsigned, Values...> /* int_seq */)
{
  static constexpr void (*jump_table[])(const Functor &functor) = {
      FunctorWrapper<Functor, Values>::apply...};
  return jump_table;
}

template <typename Functor>
const auto get_functor_jump_table()
{
 return get_functor_jump_table_impl<Functor>(
        std::make_integer_sequence<unsigned, static_cast<unsigned>(1000)>{});
}

template <typename Functor>
void run(unsigned runtime_value, const Functor &functor) {
    auto jump_table = get_functor_jump_table<Functor>();
    jump_table[runtime_value](functor);
}

int main() {
   run(333, [](auto thing){
    std::cout << thing.name << std::endl;
   });

   std::cout << "did it work?" << std::endl;

  return 0;
}



// template<typename Tag>
// struct Reg {
//   template<typename FunctorType>
//   static void run(const FunctorType& functor) {
//     functor(Tag{});
//   }
// };

// struct Factory {
//   template<typename RegType>
//   static void reg() {
//     jump_table[RegType::name] = RegType{};
//   }

//   template<typename FunctorType>
//   static void run(const std::string name, const FunctorType& functor) {   
//     jump_table[name]->run(functor);
//   }

//   template<typename RegType, typename FunctorType>
//   static void run(const FunctorType& functor) {
//     RegType::run(functor);
//   }

//   static std::map<std::string, std::function<void()>> string_to_reg_map;
// };


// struct Factory {
//   template<typename RegType>
//   static void reg() {
//     jump_table[RegType::name] = RegType{};
//   }

//   template<typename FunctorType>
//   static void run(const std::string name, const FunctorType& functor) {
//     RegInv<FunctorType>::run<Tag>(functor);

    
//     jump_table[name]->run(functor);
//   }

//   template<typename RegType, typename FunctorType>
//   static void run(const FunctorType& functor) {
//     RegType::run(functor);
//   }

//   static std::map<std::string, std::function<void()>> jump_table;
// };

// template <typename Functor>
// void run(unsigned runtime_value, const Functor &functor) {
//     auto jump_table = get_functor_jump_table<Functor>();
//     jump_table[runtime_value](functor);
// }

// int main() {

//    Reg<B>::run([](auto thing){
//     std::cout << thing.name << std::endl;
//    });

//    Factory::run<Reg<A>>([](auto thing){
//     std::cout << thing.name << std::endl;
//    });

//   return 0;
// }