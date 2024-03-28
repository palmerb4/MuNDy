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

// External libs
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

// C++ core libs
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move
#include <vector>       // for std::vector

// Mundy libs
#include <mundy_math/Array.hpp>    // for mundy::math::Array
#include <mundy_math/Matrix3.hpp>  // for mundy::math::Matrix3
#include <mundy_math/Vector3.hpp>  // for mundy::math::Vector3

// Note, these tests are meant to look like real use cases for the Vector3 class. As a result, we use implicit type
// conversions rather than being explicit about types. This is to ensure that the Vector3 class can be used in a
// natural way. This choice means that compiling this test with -Wdouble-promotion or -Wconversion will result in many
// warnings. We will not however, locally disable these warnings.

namespace mundy {

namespace math {

namespace {

//! \name Helper functions
//@{

/// \brief Test that two algebraic types are close
/// \param[in] a The first algebraic type
/// \param[in] b The second algebraic type
/// \param[in] message_if_fail The message to print if the test fails
template <typename U, typename T>
void is_close_debug(const U& a, const T& b, const std::string& message_if_fail = "")
  requires std::is_arithmetic_v<T> && std::is_arithmetic_v<U>
{
  using TU = decltype(U() - T());

  bool is_close;
  if constexpr (std::is_floating_point_v<TU>) {
    // For floating-point types, compare with a tolerance determined by the type
    const auto tol = get_default_tolerance<TU>();
    is_close = (std::abs(a - b) < tol);
  } else {
    // For integral types, compare with exact equality
    is_close = (a == b);
  }

  if (!is_close) {
    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;
    std::cout << "diff = " << a - b << std::endl;
  }

  EXPECT_TRUE(is_close) << message_if_fail;
}

/// \brief Test that two Matrix3s are close
/// \param[in] m1 The first Matrix3
/// \param[in] m2 The second Matrix3
/// \param[in] message_if_fail The message to print if the test fails
template <typename U, typename OtherAccessor, typename T, typename Accessor>
void is_close_debug(const Matrix3<U, OtherAccessor>& m1, const Matrix3<T, Accessor>& m2,
                    const std::string& message_if_fail = "") {
  if (!is_close(m1, m2)) {
    std::cout << "m1 = " << m1 << std::endl;
    std::cout << "m2 = " << m2 << std::endl;
  }
  EXPECT_TRUE(is_close(m1, m2)) << message_if_fail;
}

/// \brief Test that two Vector3s are close
/// \param[in] v1 The first Vector3
/// \param[in] v2 The second Vector3
/// \param[in] message_if_fail The message to print if the test fails
template <typename U, typename OtherAccessor, typename T, typename Accessor>
void is_close_debug(const Vector3<U, OtherAccessor>& v1, const Vector3<T, Accessor>& v2,
                    const std::string& message_if_fail = "") {
  if (!is_close(v1, v2)) {
    std::cout << "v1 = " << v1 << std::endl;
    std::cout << "v2 = " << v2 << std::endl;
  }
  EXPECT_TRUE(is_close(v1, v2)) << message_if_fail;
}

/// \brief Test that two Matrix3s are different
/// \param[in] m1 The first Matrix3
/// \param[in] m2 The second Matrix3
/// \param[in] message_if_fail The message to print if the test fails
template <typename U, typename OtherAccessor, typename T, typename Accessor>
void is_different_debug(const Matrix3<U, OtherAccessor>& m1, const Matrix3<T, Accessor>& m2,
                        const std::string& message_if_fail = "") {
  if (is_close(m1, m2)) {
    std::cout << "m1 = " << m1 << std::endl;
    std::cout << "m2 = " << m2 << std::endl;
  }
  EXPECT_TRUE(!is_close(m1, m2)) << message_if_fail;
}

/// \brief Test that two Vector3s are different
/// \param[in] v1 The first Vector3
/// \param[in] v2 The second Vector3
/// \param[in] message_if_fail The message to print if the test fails
template <typename U, typename OtherAccessor, typename T, typename Accessor>
void is_different_debug(const Vector3<U, OtherAccessor>& v1, const Vector3<T, Accessor>& v2,
                        const std::string& message_if_fail = "") {
  if (is_close(v1, v2)) {
    std::cout << "v1 = " << v1 << std::endl;
    std::cout << "v2 = " << v2 << std::endl;
  }
  EXPECT_TRUE(!is_close(v1, v2)) << message_if_fail;
}
//@}

//! \name GTEST typed test fixtures
//@{

/// \brief GTEST typed test fixture so we can run tests on multiple types
/// \tparam U The type to run the tests on
template <typename U>
class Vector3SingleTypeTest : public ::testing::Test {
  using T = U;
};  // Vector3SingleTypeTest

/// \brief List of types to run the tests on
using MyTypes = ::testing::Types<int, float, double>;

/// \brief Tell GTEST to run the tests on the types in MyTypes
TYPED_TEST_SUITE(Vector3SingleTypeTest, MyTypes);

/// \brief A helper class for a pair of types
/// \tparam U1 The first type
/// \tparam U2 The second type
template <typename U1, typename U2>
struct TypePair {
  using T1 = U1;
  using T2 = U2;
};

/// \brief GTETS typed test fixture so we can run tests on multiple pairs of types
/// \tparam Pair The pair of types to run the tests on
template <typename Pair>
class Vector3PairwiseTypeTest : public ::testing::Test {};  // Vector3PairwiseTypeTest

/// \brief List of pairs of types to run the tests on
using MyTypePairs = ::testing::Types<TypePair<int, float>, TypePair<int, double>, TypePair<float, double>,
                                     TypePair<int, int>, TypePair<float, float>, TypePair<double, double>>;

/// \brief Tell GTEST to run the tests on the types in MyTypePairs
TYPED_TEST_SUITE(Vector3PairwiseTypeTest, MyTypePairs);
//@}

//! \name Vector3 Constructors and Destructor
//@{

TYPED_TEST(Vector3SingleTypeTest, DefaultConstructor) {
  ASSERT_NO_THROW(Vector3<TypeParam>());
}

TYPED_TEST(Vector3SingleTypeTest, ConstructorFromThreeScalars) {
  ASSERT_NO_THROW(Vector3<TypeParam>(1, 2, 3));
  Vector3<TypeParam> v(1, 2, 3);
  is_close_debug(v.x(), 1);
  is_close_debug(v.y(), 2);
  is_close_debug(v.z(), 3);
}

TYPED_TEST(Vector3SingleTypeTest, Comparison) {
  Vector3<TypeParam> v1{1, 2, 3};
  Vector3<TypeParam> v2{4, 5, 6};
  EXPECT_TRUE(is_close(v1, v1));
  EXPECT_FALSE(is_close(v1, v2));
  is_close_debug(v1, v1);
  is_close_debug(v1, Vector3<TypeParam>{1, 2, 3});
}

TYPED_TEST(Vector3SingleTypeTest, CopyConstructor) {
  Vector3<TypeParam> v1{1, 2, 3};
  Vector3<TypeParam> v2(v1);
  is_close_debug(v1, v2, "Copy constructor failed.");

  // The copy owns its own data since v1 is not a view
  v1.set(4, 5, 6);
  is_different_debug(v1, v2, "Copy constructor failed.");
}

TYPED_TEST(Vector3SingleTypeTest, MoveConstructor) {
  Vector3<TypeParam> v1{1, 2, 3};
  Vector3<TypeParam> v2(std::move(v1));
  is_close_debug(v2, Vector3<TypeParam>{1, 2, 3}, "Move constructor failed.");
}

TYPED_TEST(Vector3SingleTypeTest, CopyAssignment) {
  Vector3<TypeParam> v1{1, 2, 3};
  Vector3<TypeParam> v2{4, 5, 6};
  ASSERT_NO_THROW(v2 = v1);
  is_close_debug(v1, v2, "Copy assignment failed.");
}

TYPED_TEST(Vector3SingleTypeTest, MoveAssignment) {
  Vector3<TypeParam> v1{1, 2, 3};
  Vector3<TypeParam> v2{4, 5, 6};
  ASSERT_NO_THROW(v2 = std::move(v1));
  is_close_debug(v2, Vector3<TypeParam>{1, 2, 3}, "Move assignment failed.");
}

TYPED_TEST(Vector3SingleTypeTest, Destructor) {
  ASSERT_NO_THROW(Vector3<TypeParam>());
}
//@}

//! \name Vector3 Accessors
//@{

TYPED_TEST(Vector3SingleTypeTest, Accessors) {
  Vector3<TypeParam> v(1, 2, 3);
  is_close_debug(v.x(), 1);
  is_close_debug(v.y(), 2);
  is_close_debug(v.z(), 3);

  v.x() = 4;
  v.y() = 5;
  v.z() = 6;
  is_close_debug(v.x(), 4);
  is_close_debug(v.y(), 5);
  is_close_debug(v.z(), 6);

  v[0] = 7;
  v[1] = 8;
  v[2] = 9;
  is_close_debug(v.x(), 7);
  is_close_debug(v.y(), 8);
  is_close_debug(v.z(), 9);

  is_close_debug(v[0], 7);
  is_close_debug(v[1], 8);
  is_close_debug(v[2], 9);

  is_close_debug(v[0], v.x());
  is_close_debug(v[1], v.y());
  is_close_debug(v[2], v.z());
}
//@}

//! \name Vector3 Setters
//@{

TYPED_TEST(Vector3SingleTypeTest, Setters) {
  Vector3<TypeParam> v;
  v.set(1, 2, 3);
  is_close_debug(v, Vector3<TypeParam>{1, 2, 3}, "Set by three scalars failed.");

  v.set(Vector3<TypeParam>{4, 5, 6});
  is_close_debug(v, Vector3<TypeParam>{4, 5, 6}, "Set by vector failed.");

  v.fill(7);
  is_close_debug(v, Vector3<TypeParam>{7, 7, 7}, "Fill failed.");
}
//@}

//! \name Vector3 Special vectors
//@{

TYPED_TEST(Vector3SingleTypeTest, SpecialVectors) {
  ASSERT_NO_THROW(Vector3<TypeParam>::zeros());
  ASSERT_NO_THROW(Vector3<TypeParam>::ones());
  auto ones = Vector3<TypeParam>::ones();
  auto zeros = Vector3<TypeParam>::zeros();

  is_close_debug(ones, Vector3<TypeParam>{1, 1, 1}, "Ones failed.");
  is_close_debug(zeros, Vector3<TypeParam>{0, 0, 0}, "Zeros failed.");
}
//@}

//! \name Vector3 Addition and subtraction
//@{

TYPED_TEST(Vector3PairwiseTypeTest, AdditionAndSubtractionWithVector3) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Vector3<T1> v1(1, 2, 3);
  Vector3<T2> v2(4, 5, 6);
  auto v3 = v1 + v2;
  using T3 = decltype(v3)::value_type;
  is_close_debug(v3, Vector3<T3>{5, 7, 9}, "Vector-vector addition failed.");

  v1 += v2;
  is_close_debug(v1, Vector3<T1>{5, 7, 9}, "Vector-vector addition assignment failed.");

  v3 = v1 - v2;
  is_close_debug(v3, Vector3<T3>{1, 2, 3}, "Vector-vector subtraction failed.");

  v1 -= v2;
  is_close_debug(v1, Vector3<T1>{1, 2, 3}, "Vector-vector subtraction assignment failed.");
}

TYPED_TEST(Vector3PairwiseTypeTest, AdditionAndSubtractionWithScalars) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Vector3<T1> v1(1, 2, 3);
  auto v2 = v1 + T2(1);
  using T3 = decltype(v2)::value_type;
  is_close_debug(v2, Vector3<T3>{2, 3, 4}, "Vector-scalar addition failed.");

  v2 = T2(1) + v1;
  is_close_debug(v2, Vector3<T3>{2, 3, 4}, "Scalar-vector addition failed.");

  v2 = v1 - T2(1);
  is_close_debug(v2, Vector3<T3>{0, 1, 2}, "Vector-scalar subtraction failed.");

  v2 = T2(1) - v1;
  is_close_debug(v2, Vector3<T3>{0, -1, -2}, "Scalar-vector subtraction failed.");
}

TYPED_TEST(Vector3PairwiseTypeTest, AdditionAndSubtractionEdgeCases) {
  // Test that the addition and subtraction operators work with rvalues
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Vector3<T1> v1(1, 2, 3);
  auto v2 = v1 + Vector3<T2>(4, 5, 6);
  using T3 = decltype(v2)::value_type;
  is_close_debug(v2, Vector3<T3>{5, 7, 9}, "Vector-vector addition failed.");

  v2 = Vector3<T2>(4, 5, 6) + v1;
  is_close_debug(v2, Vector3<T3>{5, 7, 9}, "Vector-vector addition failed.");

  v2 = v1 - Vector3<T2>(4, 5, 6);
  is_close_debug(v2, Vector3<T3>{-3, -3, -3}, "Vector-vector subtraction failed.");

  v2 = Vector3<T2>(4, 5, 6) - v1;
  is_close_debug(v2, Vector3<T3>{3, 3, 3}, "Vector-vector subtraction failed.");
}
//@}

//! \name Vector3 Multiplication and division
//@{

TYPED_TEST(Vector3PairwiseTypeTest, MultiplicationAndDivisionWithScalars) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Vector3<T1> v1(1, 2, 3);
  auto v2 = v1 * T2(2);
  using T3 = decltype(v2)::value_type;
  is_close_debug(v2, Vector3<T3>{2, 4, 6}, "Vector-scalar multiplication failed.");

  v2 = T2(2) * v1;
  is_close_debug(v2, Vector3<T3>{2, 4, 6}, "Scalar-vector multiplication failed.");

  v2 = v2 / T2(2);
  is_close_debug(v2, Vector3<T3>{1, 2, 3}, "Vector-scalar division failed.");
}
//@}

//! \name Vector3 Basic arithmetic reduction operations
//@{

TYPED_TEST(Vector3SingleTypeTest, BasicArithmeticReductionOperations) {
  Vector3<TypeParam> v1(1, 2, 3);
  is_close_debug(sum(v1), 6, "Sum failed.");
  is_close_debug(product(v1), 6, "Product failed.");
  is_close_debug(min(v1), 1, "Min failed.");
  is_close_debug(max(v1), 3, "Max failed.");
  is_close_debug(mean(v1), 2, "Mean failed.");
  is_close_debug(variance(v1), 2.0 / 3.0, "Variance failed.");
  is_close_debug(stddev(v1), std::sqrt(2.0 / 3.0), "Stddev failed.");

  Vector3<TypeParam> v2(-1, 2, -3);
  is_close_debug(sum(v2), -2, "Sum failed.");
  is_close_debug(product(v2), 6, "Product failed.");
  is_close_debug(min(v2), -3, "Min failed.");
  is_close_debug(max(v2), 2, "Max failed.");
  is_close_debug(mean(v2), -2.0 / 3.0, "Mean failed.");
  is_close_debug(variance(v2), 38.0 / 9.0, "Variance failed.");
  is_close_debug(stddev(v2), std::sqrt(38.0 / 9.0), "Stddev failed.");
}
//@}

//! \name Vector3 Special vector operations
//@{

TYPED_TEST(Vector3PairwiseTypeTest, SpecialOperations) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;
  using T3 = decltype(std::declval<T1>() * std::declval<T2>());

  Vector3<T1> v1(1, 2, 3);
  Vector3<T2> v2(4, 5, 6);
  is_close_debug(dot(v1, v2), 32.0, "Dot product failed.");
  is_close_debug(cross(v1, v2), Vector3<double>{-3.0, 6.0, -3.0}, "Cross product failed.");
  is_close_debug(norm(v1), std::sqrt(14.0), "Norm failed.");
  is_close_debug(norm_squared(v1), 14.0, "Norm squared failed.");
  is_close_debug(infinity_norm(v1), 3.0, "Infinity norm failed.");
  is_close_debug(one_norm(v1), 6.0, "One norm failed.");
  is_close_debug(two_norm(v1), std::sqrt(14.0), "Two norm failed.");
  is_close_debug(two_norm_squared(v1), 14.0, "Two norm squared failed.");
  is_close_debug(minor_angle(v1, v2), std::acos(32.0 / (std::sqrt(14.0) * std::sqrt(77.0))), "Minor angle failed.");
  is_close_debug(major_angle(v1, v2), M_PI - std::acos(32.0 / (std::sqrt(14.0) * std::sqrt(77.0))),
                 "Major angle failed.");

  auto outer = outer_product(v1, v2);
  is_close_debug(outer, Matrix3<T3>{4, 5, 6, 8, 10, 12, 12, 15, 18}, "Outer product failed.");
}

TYPED_TEST(Vector3PairwiseTypeTest, SpecialOperationsEdgeCases) {
  // Test that the special vector operations work with rvalues
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Vector3<T2> v2(4, 5, 6);
  is_close_debug(dot(Vector3<T1>{1, 2, 3}, v2), 32.0, "Dot product failed.");
  is_close_debug(cross(Vector3<T1>{1, 2, 3}, v2), Vector3<double>{-3.0, 6.0, -3.0}, "Cross product failed.");
  is_close_debug(norm(Vector3<T1>{1, 2, 3}), std::sqrt(14.0), "Norm failed.");
  is_close_debug(norm_squared(Vector3<T1>{1, 2, 3}), 14.0, "Norm squared failed.");
  is_close_debug(infinity_norm(Vector3<T1>{1, 2, 3}), 3.0, "Infinity norm failed.");
  is_close_debug(one_norm(Vector3<T1>{1, 2, 3}), 6.0, "One norm failed.");
  is_close_debug(two_norm(Vector3<T1>{1, 2, 3}), std::sqrt(14.0), "Two norm failed.");
  is_close_debug(two_norm_squared(Vector3<T1>{1, 2, 3}), 14.0, "Two norm squared failed.");
  is_close_debug(minor_angle(Vector3<T1>{1, 2, 3}, v2), std::acos(32.0 / (std::sqrt(14.0) * std::sqrt(77.0))),
                 "Minor angle failed.");
  is_close_debug(major_angle(Vector3<T1>{1, 2, 3}, v2), M_PI - std::acos(32 / (std::sqrt(14.0) * std::sqrt(77.0))),
                 "Major angle failed.");

  auto outer = outer_product(Vector3<T1>{1, 2, 3}, v2);
  using T3 = decltype(outer)::value_type;
  is_close_debug(outer, Matrix3<T3>{4, 5, 6, 8, 10, 12, 12, 15, 18}, "Outer product failed.");
}
//@}

//! \name Vector3 Views
//@{

TYPED_TEST(Vector3SingleTypeTest, Views) {
  // Create a view from a subset of an std::vector<TypeParam>
  std::vector<TypeParam> v1{0, 0, 1, 2, 3, 0, 0};
  {
    auto v2 = get_vector3_view<TypeParam>(v1.data() + 2);
    is_close_debug(v2.x(), 1);
    is_close_debug(v2.y(), 2);
    is_close_debug(v2.z(), 3);
    auto ptr_before = v1.data();
    v1 = {2, 3, 4, 5, 6, 7, 8};
    auto ptr_after = v1.data();
    ASSERT_EQ(ptr_before, ptr_after);
    is_close_debug(v2.x(), 4);
    is_close_debug(v2.y(), 5);
    is_close_debug(v2.z(), 6);
  }

  // Create a view from a TypeParam*
  TypeParam v3[3] = {4, 5, 6};
  {
    auto v4 = get_vector3_view<TypeParam>(&v3[0]);
    is_close_debug(v4.x(), 4);
    is_close_debug(v4.y(), 5);
    is_close_debug(v4.z(), 6);
    v3[0] = 7;
    v3[1] = 8;
    v3[2] = 9;
    is_close_debug(v4.x(), 7);
    is_close_debug(v4.y(), 8);
    is_close_debug(v4.z(), 9);
  }

  // Create a const view from an std::vector<TypeParam>
  const std::vector<TypeParam> v5{7, 8, 9};
  {
    auto v6 = get_vector3_view<TypeParam>(v5.data());
    is_close_debug(v6, Vector3<TypeParam>(7, 8, 9), "Const view isn't const");
  }
}
//@}

}  // namespace

}  // namespace math

}  // namespace mundy
