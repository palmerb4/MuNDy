#ifndef MUNDY_ALENS_COMPUTE_MOBILITY_PERIPHERY_UTILS_UTILS_HPP_
#define MUNDY_ALENS_COMPUTE_MOBILITY_PERIPHERY_UTILS_UTILS_HPP_

#include "Util/EigenDef.hpp"

namespace mundy {

namespace alens {

namespace compute_mobility {

namespace periphery_utils {

namespace cnpy {
struct NpyArray;
using npz_t = std::map<std::string, NpyArray>;
}  // namespace cnpy

namespace utils {

Eigen::MatrixXd barycentric_matrix(const Eigen::Ref<const Eigen::ArrayXd> &x, const Eigen::Ref<const Eigen::ArrayXd> &y);
Eigen::MatrixXd finite_diff(const Eigen::Ref<const Eigen::ArrayXd> &s, int M, int n_s);
Eigen::VectorXd collect_into_global(const Eigen::Ref<const Eigen::VectorXd> &local_vec);

Eigen::MatrixXd load_mat(cnpy::npz_t &npz, const char *var);
Eigen::VectorXd load_vec(cnpy::npz_t &npz, const char *var);

template <typename DerivedA, typename DerivedB>
bool allclose(
    const Eigen::DenseBase<DerivedA> &a, const Eigen::DenseBase<DerivedB> &b,
    const typename DerivedA::RealScalar &rtol = Eigen::NumTraits<typename DerivedA::RealScalar>::dummy_precision(),
    const typename DerivedA::RealScalar &atol = Eigen::NumTraits<typename DerivedA::RealScalar>::epsilon()) {
  return ((a.derived() - b.derived()).array().abs() <= (atol + rtol * b.derived().array().abs())).all();
}

}  // namespace utils

}  // namespace periphery_utils

}  // namespace compute_mobility

}  // namespace alens

}  // namespace mundy

#endif  // MUNDY_ALENS_COMPUTE_MOBILITY_PERIPHERY_UTILS_UTILS_HPP_
