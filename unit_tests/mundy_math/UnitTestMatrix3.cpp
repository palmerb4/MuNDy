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
#include <mundy_math/Matrix3.hpp>  // for mundy::math::Matrix3
#include <mundy_math/Vector3.hpp>  // for mundy::math::Vector3

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
void is_close_debug(const Matrix3<U, OtherAccessor>& m1, const Matrix3<T, Accessor>& m2, const std::string& message_if_fail = "") {
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
void is_close_debug(const Vector3<U, OtherAccessor>& v1, const Vector3<T, Accessor>& v2, const std::string& message_if_fail = "") {
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
void is_different_debug(const Matrix3<U, OtherAccessor>& m1, const Matrix3<T, Accessor>& m2, const std::string& message_if_fail = "") {
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
void is_different_debug(const Vector3<U, OtherAccessor>& v1, const Vector3<T, Accessor>& v2, const std::string& message_if_fail = "") {
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
class Matrix3SingleTypeTest : public ::testing::Test {
  using T = U;
};  // Matrix3SingleTypeTest

/// \brief List of types to run the tests on
using MyTypes = ::testing::Types<int, float, double>;

/// \brief Tell GTEST to run the tests on the types in MyTypes
TYPED_TEST_SUITE(Matrix3SingleTypeTest, MyTypes);

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
class Matrix3PairwiseTypeTest : public ::testing::Test {};  // Matrix3PairwiseTypeTest

/// \brief List of pairs of types to run the tests on
using MyTypePairs = ::testing::Types<TypePair<int, float>, TypePair<int, double>, TypePair<float, double>,
                                     TypePair<int, int>, TypePair<float, float>, TypePair<double, double>>;

/// \brief Tell GTEST to run the tests on the types in MyTypePairs
TYPED_TEST_SUITE(Matrix3PairwiseTypeTest, MyTypePairs);
//@}

//! \name Matrix3 Constructors and Destructor
//@{

TYPED_TEST(Matrix3SingleTypeTest, DefaultConstructor) {
  ASSERT_NO_THROW(Matrix3<TypeParam>());
}

TYPED_TEST(Matrix3SingleTypeTest, ConstructorFromNineScalars) {
  ASSERT_NO_THROW(Matrix3<TypeParam>(1, 2, 3, 4, 5, 6, -7, -8, -9));
  Matrix3<TypeParam> m(1, 2, 3, 4, 5, 6, -7, -8, -9);
  is_close_debug(m(0), 1);
  is_close_debug(m(1), 2);
  is_close_debug(m(2), 3);
  is_close_debug(m(3), 4);
  is_close_debug(m(4), 5);
  is_close_debug(m(5), 6);
  is_close_debug(m(6), -7);
  is_close_debug(m(7), -8);
  is_close_debug(m(8), -9);
}

TYPED_TEST(Matrix3SingleTypeTest, Comparison) {
  Matrix3<TypeParam> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  Matrix3<TypeParam> m2(4, 5, 6, -7, -8, -9, 10, 11, 12);
  EXPECT_TRUE(is_close(m1, m1));
  EXPECT_FALSE(is_close(m1, m2));

  is_close_debug(m1, m1);
  is_close_debug(m1, Matrix3<TypeParam>{1, 2, 3, 4, 5, 6, -7, -8, -9});
}

TYPED_TEST(Matrix3SingleTypeTest, CopyConstructor) {
  Matrix3<TypeParam> m1{1, 2, 3, 4, 5, 6, -7, -8, -9};
  ASSERT_NO_THROW(Matrix3<TypeParam>(m1));
  Matrix3<TypeParam> m2(m1);
  is_close_debug(m1, m2, "Copy constructor failed.");

  // The copy owns its own data since m1 is not a view
  m1 = {4, 5, 6, -7, -8, -9, 10, 11, 12};
  is_different_debug(m1, m2, "Copy constructor failed.");
}

TYPED_TEST(Matrix3SingleTypeTest, MoveConstructor) {
  Matrix3<TypeParam> m1{1, 2, 3, 4, 5, 6, -7, -8, -9};
  Matrix3<TypeParam> m2(std::move(m1));
  is_close_debug(m2, Matrix3<TypeParam>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Move constructor failed.");
}

TYPED_TEST(Matrix3SingleTypeTest, CopyAssignment) {
  Matrix3<TypeParam> m1{1, 2, 3, 4, 5, 6, -7, -8, -9};
  Matrix3<TypeParam> m2{4, 5, 6, -7, -8, -9, 10, 11, 12};
  ASSERT_NO_THROW(m2 = m1);
  is_close_debug(m1, m2, "Copy assignment failed.");
}

TYPED_TEST(Matrix3SingleTypeTest, MoveAssignment) {
  Matrix3<TypeParam> m1{1, 2, 3, 4, 5, 6, -7, -8, -9};
  Matrix3<TypeParam> m2{4, 5, 6, -7, -8, -9, 10, 11, 12};
  ASSERT_NO_THROW(m2 = std::move(m1));
  is_close_debug(m2, Matrix3<TypeParam>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Move assignment failed.");
}

TYPED_TEST(Matrix3SingleTypeTest, Destructor) {
  ASSERT_NO_THROW(Matrix3<TypeParam>());
}
//@}

//! \name Matrix3 Accessors
//@{

TYPED_TEST(Matrix3SingleTypeTest, Accessors) {
  Matrix3<TypeParam> m(1, 2, 3, 4, 5, 6, -7, -8, -9);

  // By row-major flattened index
  is_close_debug(m(0), 1);
  is_close_debug(m(1), 2);
  is_close_debug(m(2), 3);
  is_close_debug(m(3), 4);
  is_close_debug(m(4), 5);
  is_close_debug(m(5), 6);
  is_close_debug(m(6), -7);
  is_close_debug(m(7), -8);
  is_close_debug(m(8), -9);

  // By row and column indices, version 1
  is_close_debug(m(0, 0), 1);
  is_close_debug(m(0, 1), 2);
  is_close_debug(m(0, 2), 3);
  is_close_debug(m(1, 0), 4);
  is_close_debug(m(1, 1), 5);
  is_close_debug(m(1, 2), 6);
  is_close_debug(m(2, 0), -7);
  is_close_debug(m(2, 1), -8);
  is_close_debug(m(2, 2), -9);

  // Fetch a row (deep copy)
  Vector3<TypeParam> row0 = m.get_row(0);
  is_close_debug(row0.x(), 1);
  is_close_debug(row0.y(), 2);
  is_close_debug(row0.z(), 3);
  row0 = {4, 5, 6};
  is_close_debug(m(0, 0), 1);
  is_close_debug(m(0, 1), 2);
  is_close_debug(m(0, 2), 3);

  // Fetch a column (deep copy)
  Vector3<TypeParam> col0 = m.get_column(0);
  is_close_debug(col0.x(), 1);
  is_close_debug(col0.y(), 4);
  is_close_debug(col0.z(), -7);
  col0 = {4, 5, 6};
  is_close_debug(m(0, 0), 1);
  is_close_debug(m(1, 0), 4);
  is_close_debug(m(2, 0), -7);
}
//@}

//! \name Matrix3 Setters
//@{

TYPED_TEST(Matrix3SingleTypeTest, Setters) {
  // Set entire matrix by row-major flattened index
  Matrix3<TypeParam> m;
  m.set(1, 2, 3, 4, 5, 6, -7, -8, -9);
  is_close_debug(m, Matrix3<TypeParam>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Set by row-major flattened index failed.");

  // Set entire matrix by another Matrix3
  m.set(Matrix3<TypeParam>{1, 2, 3, 4, 5, 6, -7, -8, -9});
  is_close_debug(m, Matrix3<TypeParam>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Set by Matrix3 failed.");

  // Set a column by a vector
  m.set_column(0, Vector3<TypeParam>{-1, -2, -3});
  is_close_debug(m.get_column(0), Vector3<TypeParam>{-1, -2, -3}, "Set column by vector failed.");

  // Set a column by three scalars
  m.set_column(1, -4, -5, -6);
  is_close_debug(m.get_column(1), Vector3<TypeParam>{-4, -5, -6}, "Set column by scalars failed.");

  // Set a row by a vector
  m.set_row(0, Vector3<TypeParam>{-4, -5, -6});
  is_close_debug(m.get_row(0), Vector3<TypeParam>{-4, -5, -6}, "Set row by vector failed.");

  // Set a row by three scalars
  m.set_row(1, -1, -2, -3);
  is_close_debug(m.get_row(1), Vector3<TypeParam>{-1, -2, -3}, "Set row by scalars failed.");

  // Fill entire matrix with a scalar
  m.fill(7);
  is_close_debug(m, Matrix3<TypeParam>{7, 7, 7, 7, 7, 7, 7, 7, 7}, "Fill failed.");
}
//@}

//! \name Matrix3 Special vectors
//@{

TYPED_TEST(Matrix3SingleTypeTest, SpecialVectors) {
  ASSERT_NO_THROW(Matrix3<TypeParam>::identity());
  ASSERT_NO_THROW(Matrix3<TypeParam>::ones());
  ASSERT_NO_THROW(Matrix3<TypeParam>::zeros());

  auto identity = Matrix3<TypeParam>::identity();
  auto ones = Matrix3<TypeParam>::ones();
  auto zeros = Matrix3<TypeParam>::zeros();

  is_close_debug(identity, Matrix3<TypeParam>{1, 0, 0, 0, 1, 0, 0, 0, 1}, "Identity failed.");
  is_close_debug(ones, Matrix3<TypeParam>{1, 1, 1, 1, 1, 1, 1, 1, 1}, "Ones failed.");
  is_close_debug(zeros, Matrix3<TypeParam>{0, 0, 0, 0, 0, 0, 0, 0, 0}, "Zeros failed.");
}
//@}

//! \name Matrix3 Addition and subtraction
//@{

TYPED_TEST(Matrix3PairwiseTypeTest, AdditionAndSubtractionWithMatrix3) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Matrix3<T1> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  Matrix3<T2> m2(4, 5, 6, -7, -8, -9, 10, 11, 12);
  auto m3 = m1 + m2;
  using T3 = decltype(m3)::value_type;
  is_close_debug(m3, Matrix3<T3>{5, 7, 9, -3, -3, -3, 3, 3, 3}, "Addition failed.");

  m1 += m2;
  is_close_debug(m1, Matrix3<T1>{5, 7, 9, -3, -3, -3, 3, 3, 3}, "Addition assignment failed.");

  m3 = m1 - m2;
  is_close_debug(m3, Matrix3<T3>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Subtraction failed.");

  m1 -= m2;
  is_close_debug(m1, Matrix3<T1>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Subtraction assignment failed.");
}

TYPED_TEST(Matrix3PairwiseTypeTest, AdditionAndSubtractionWithScalars) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Matrix3<T1> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  auto m2 = m1 + T2(1);
  using T3 = decltype(m2)::value_type;
  is_close_debug(m2, Matrix3<T3>{2, 3, 4, 5, 6, 7, -6, -7, -8}, "Right addition failed.");

  m2 = T2(1) + m1;
  is_close_debug(m2, Matrix3<T3>{2, 3, 4, 5, 6, 7, -6, -7, -8}, "Left addition failed.");

  m2 = m1 - T2(1);
  is_close_debug(m2, Matrix3<T3>{0, 1, 2, 3, 4, 5, -8, -9, -10}, "Right subtraction failed.");

  m2 = T2(1) - m1;
  is_close_debug(m2, Matrix3<T3>{0, -1, -2, -3, -4, -5, 8, 9, 10}, "Left subtraction failed.");
}

TYPED_TEST(Matrix3PairwiseTypeTest, AdditionAndSubtractionEdgeCases) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Matrix3<T1> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  auto m3 = m1 + Matrix3<T2>{4, 5, 6, -7, -8, -9, 10, 11, 12};
  using T3 = decltype(m3)::value_type;
  is_close_debug(m3, Matrix3<T3>{5, 7, 9, -3, -3, -3, 3, 3, 3}, "Addition failed.");

  m1 += Matrix3<T2>{4, 5, 6, -7, -8, -9, 10, 11, 12};
  is_close_debug(m1, Matrix3<T1>{5, 7, 9, -3, -3, -3, 3, 3, 3}, "Addition assignment failed.");

  m3 = m1 - Matrix3<T2>{4, 5, 6, -7, -8, -9, 10, 11, 12};
  is_close_debug(m3, Matrix3<T3>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Subtraction failed.");

  m1 -= Matrix3<T2>{4, 5, 6, -7, -8, -9, 10, 11, 12};
  is_close_debug(m1, Matrix3<T1>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Subtraction assignment failed.");
}
//@}

//! \name Matrix3 Multiplication and division
//@{

TYPED_TEST(Matrix3PairwiseTypeTest, MultiplicationAndDivisionWithMatrix3) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Matrix3<T1> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  Matrix3<T2> m2(4, 5, 6, -7, -8, -9, 10, 11, 12);
  auto m3 = m1 * m2;
  using T3 = decltype(m3)::value_type;
  is_close_debug(m3, Matrix3<T3>{20, 22, 24, 41, 46, 51, -62, -70, -78}, "Multiplication failed.");

  m1 *= m2;
  is_close_debug(m1, Matrix3<T1>{20, 22, 24, 41, 46, 51, -62, -70, -78}, "Multiplication assignment failed.");
}

TYPED_TEST(Matrix3PairwiseTypeTest, MultiplicationAndDivisionWithVector3) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Matrix3<T1> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  Vector3<T2> v1(4, 5, 6);
  auto v2 = m1 * v1;
  using T3 = decltype(v2)::value_type;
  is_close_debug(v2, Vector3<T3>{32, 77, -122}, "Multiplication failed.");

  // Left multiplication by a vector: v^T * m
  auto v3 = v1 * m1;
  using T4 = decltype(v3)::value_type;
  is_close_debug(v3, Vector3<T4>{-18, -15, -12}, "Left multiplication failed.");
}

TYPED_TEST(Matrix3PairwiseTypeTest, MultiplicationAndDivisionWithScalars) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Matrix3<T1> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  auto m2 = m1 * T2(2);
  using T3 = decltype(m2)::value_type;
  is_close_debug(m2, Matrix3<T3>{2, 4, 6, 8, 10, 12, -14, -16, -18}, "Right multiplication failed.");

  m2 /= T2(2);
  is_close_debug(m2, Matrix3<T3>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Division assignment failed.");

  m2 = T2(2) * m1;
  is_close_debug(m2, Matrix3<T3>{2, 4, 6, 8, 10, 12, -14, -16, -18}, "Left multiplication failed.");

  m2 = m2 / T2(2);
  is_close_debug(m2, Matrix3<T3>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Right division failed.");

  m2 *= T2(2);
  is_close_debug(m2, Matrix3<T3>{2, 4, 6, 8, 10, 12, -14, -16, -18}, "Multiplication assignment failed.");
}

TYPED_TEST(Matrix3PairwiseTypeTest, MultiplicationAndDivisionEdgeCases) {
  // Test that the multiplication and division operators work with rvalues
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Matrix3<T1> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  auto m2 = m1 * Matrix3<T2>(4, 5, 6, -7, -8, -9, 10, 11, 12);
  using T3 = decltype(m2)::value_type;
  is_close_debug(m2, Matrix3<T3>{20, 22, 24, 41, 46, 51, -62, -70, -78}, "Right rvalue multiplication failed.");

  m1 *= Matrix3<T2>(4, 5, 6, -7, -8, -9, 10, 11, 12);
  is_close_debug(m2, Matrix3<T3>{20, 22, 24, 41, 46, 51, -62, -70, -78}, "Left rvalue multiplication failed.");
}
//@}

//! \name Matrix3 Basic arithmetic reduction operations
//@{

TYPED_TEST(Matrix3SingleTypeTest, BasicArithmeticReductionOperations) {
  Matrix3<TypeParam> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  is_close_debug(determinant(m1), 0.0, "Determinant failed.");
  is_close_debug(trace(m1), -3.0, "Trace failed.");
  is_close_debug(sum(m1), -3.0, "Sum failed.");
  is_close_debug(product(m1), -362880.0, "Product failed.");
  is_close_debug(min(m1), -9.0, "Min failed.");
  is_close_debug(max(m1), 6.0, "Max failed.");
  is_close_debug(mean(m1), -1.0 / 3.0, "Mean failed.");
  is_close_debug(variance(m1), 284.0 / 9.0, "Variance failed.");
}
//@}

//! \name Matrix3 Special matrix operations
//@{

TYPED_TEST(Matrix3SingleTypeTest, SpecialOperations) {
  // Test the transpose
  Matrix3<TypeParam> m1(1, 2, 3, 4, 5, 6, -7, -8, -9);
  auto m2 = transpose(m1);
  using T3 = decltype(m2)::value_type;
  is_close_debug(m2, Matrix3<T3>{1, 4, -7, 2, 5, -8, 3, 6, -9}, "Transpose failed.");

  // Test the inverse of a singular matrix
  EXPECT_ANY_THROW(inverse(m1));

  // Test the inverse of a non-singular matrix
  m1 = {1, 2, 3, 0, 1, 4, 5, 6, 0};
  auto m3 = inverse(m1);
  using T4 = decltype(m3)::value_type;
  is_close_debug(m3, Matrix3<T4>{-24, 18, 5, 20, -15, -4, -5, 4, 1}, "Inverse failed.");

  // Test the frobenius_inner_product
  m1 = {1, 2, 3, 2, 5, 6, 3, 6, 9};
  m2 = {4, 5, 6, 7, 8, 9, -10, -11, -12};
  is_close_debug(frobenius_inner_product(m1, m2), -64, "Frobenius inner product failed.");
  }

TYPED_TEST(Matrix3SingleTypeTest, SpecialOperationsEdgeCases) {
  // Test that the special vector operations work with rvalues
  Matrix3<TypeParam> m2 = transpose(Matrix3<TypeParam>(1, 2, 3, 4, 5, 6, -7, -8, -9));
  is_close_debug(m2, Matrix3<TypeParam>{1, 4, -7, 2, 5, -8, 3, 6, -9}, "Transpose failed.");

  // Test the inverse of a singular matrix
  EXPECT_ANY_THROW(inverse(Matrix3<TypeParam>(1, 2, 3, 2, 4, 6, 3, 6, 9)));

  // Test the inverse of a non-singular matrix
  m2 = inverse(Matrix3<TypeParam>(1, 2, 3, 0, 1, 4, 5, 6, 0));
  is_close_debug(m2, Matrix3<TypeParam>{-24, 18, 5, 20, -15, -4, -5, 4, 1}, "Inverse failed.");

  // Test the frobenius_inner_product
  is_close_debug(frobenius_inner_product(Matrix3<TypeParam>(1, 2, 3, 2, 5, 6, 3, 6, 9),
                                         Matrix3<TypeParam>(4, 5, 6, 7, 8, 9, -10, -11, -12)),
                 -64, "Frobenius inner product failed.");
}
//@}

//! \name Matrix3 Views
//@{

TYPED_TEST(Matrix3SingleTypeTest, Views) {
  // Create a view from a subset of an std::vector<TypeParam>
  std::vector<TypeParam> m1{0, 0, 1, 2, 3, 4, 5, 6, -7, -8, -9, 0, 0};
  auto m2 = get_matrix3_view<TypeParam>(m1.data() + 2);
  is_close_debug(m2, Matrix3<TypeParam>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "View failed.");
  auto ptr_before = m1.data();
  m1 = {1, 2, 4, 5, 6, -7, -8, -9, 10, 11, 12, 13, 14};
  auto ptr_after = m1.data();
  ASSERT_EQ(ptr_before, ptr_after);
  is_close_debug(m2, Matrix3<TypeParam>{4, 5, 6, -7, -8, -9, 10, 11, 12}, "View isn't shallow.");

  // Create a view from a TypeParam*
  TypeParam m3[9] = {1, 2, 3, 4, 5, 6, -7, -8, -9};
  auto m4 = get_matrix3_view<TypeParam>(&m3[0]);
  is_close_debug(m4, Matrix3<TypeParam>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "View failed.");
  m3[0] = 4;
  m3[1] = 5;
  m3[2] = 6;
  m3[3] = -7;
  m3[4] = -8;
  m3[5] = -9;
  m3[6] = 10;
  m3[7] = 11;
  m3[8] = 12;
  is_close_debug(m4, Matrix3<TypeParam>{4, 5, 6, -7, -8, -9, 10, 11, 12}, "View isn't shallow.");

  // Create a const view from an std::vector<TypeParam>
  const std::vector<TypeParam> m5{1, 2, 3, 4, 5, 6, -7, -8, -9};
  auto m6 = get_matrix3_view<TypeParam>(m5.data());
  is_close_debug(m6, Matrix3<TypeParam>{1, 2, 3, 4, 5, 6, -7, -8, -9}, "Const view failed.");
}
//@}

}  // namespace

}  // namespace math

}  // namespace mundy
