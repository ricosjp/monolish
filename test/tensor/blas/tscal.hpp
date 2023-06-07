#include "../../test_utils.hpp"

template <typename T>
void ans_tscal(const double alpha, monolish::tensor::tensor_Dense<T> &A) {

  for (int i = 0; i < A.get_nnz(); i++)
    A.begin()[i] = alpha * A.begin()[i];
}

template <typename MAT, typename T>
bool test_send_tscal_core(const size_t M, const size_t N, const size_t L,
                          double tol, monolish::tensor::tensor_Dense<T> &AA,
                          MAT &A) {
  T alpha = 123.0;
  ans_tscal(alpha, AA);
  monolish::tensor::tensor_COO<T> ansA(AA);

  A.send();
  monolish::blas::tscal(alpha, A);
  A.recv();
  monolish::tensor::tensor_COO<T> resultA(A);

  return ans_check<T>(__func__, A.type(), resultA.begin(), ansA.begin(),
                      ansA.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_send_tscal(const size_t M, const size_t N, const size_t L,
                     double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);

  MAT A(seedA); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);

  return test_send_tscal_core(M, N, L, tol, AA, A);
}

template <typename T, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_send_tscal_view_core1(const size_t M, const size_t N, const size_t L,
                           double tol, monolish::tensor::tensor_Dense<T> &AA,
                           std::tuple<Tp...> &As) {
  return true;
}

template <typename T, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type test_send_tscal_view_core1(
        const size_t M, const size_t N, const size_t L, double tol,
        monolish::tensor::tensor_Dense<T> &AA, std::tuple<Tp...> &As) {
  std::get<I>(As) = AA;
  if (!test_send_tscal_core(M, N, L, tol, AA, std::get<I>(As))) {
    return false;
  }
  return test_send_tscal_view_core1<T, I + 1, Tp...>(M, N, L, tol, AA, As);
}

template <typename T>
bool test_send_tscal_view(const size_t M, const size_t N, const size_t L,
                          double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);

  monolish::tensor::tensor_Dense<T> AA(seedA);

  using T1 = monolish::tensor::tensor_Dense<T>;
  using T2_ = monolish::vector<T>;
  using T2 = monolish::view_tensor_Dense<T2_, T>;
  using T3_ = monolish::matrix::Dense<T>;
  using T3 = monolish::view_tensor_Dense<T3_, T>;
  using T4_ = monolish::tensor::tensor_Dense<T>;
  using T4 = monolish::view_tensor_Dense<T4_, T>;
  T1 x1({M, N, L});
  T2_ x2_(2 * M * N * L, 0.0, 1.0);
  T2 x2(x2_, L / 2, {M, N, L});
  T3_ x3_(2 * M * N * L, 1, 0.0, 1.0);
  T3 x3(x3_, L / 2, {M, N, L});
  T4_ x4_({2 * M * N * L, 1, 1}, 0.0, 1.0);
  T4 x4(x4_, L / 2, {M, N, L});

  auto As = std::make_tuple(x1, x2, x3, x4);

  return test_send_tscal_view_core1(M, N, L, tol, AA, As);
}

template <typename MAT, typename T>
bool test_tscal_core(const size_t M, const size_t N, const size_t L, double tol,
                     monolish::tensor::tensor_Dense<T> &AA, MAT &A) {
  T alpha = 123.0;
  ans_tscal(alpha, AA);
  monolish::tensor::tensor_COO<T> ansA(AA);

  monolish::blas::tscal(alpha, A);
  monolish::tensor::tensor_COO<T> resultA(A);

  return ans_check<T>(__func__, A.type(), resultA.begin(), ansA.begin(),
                      ansA.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_tscal(const size_t M, const size_t N, const size_t L, double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);

  MAT A(seedA); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);

  return test_tscal_core(M, N, L, tol, AA, A);
}

template <typename T, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_tscal_view_core1(const size_t M, const size_t N, const size_t L,
                      double tol, monolish::tensor::tensor_Dense<T> &AA,
                      std::tuple<Tp...> &As) {
  return true;
}

template <typename T, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type
    test_tscal_view_core1(const size_t M, const size_t N, const size_t L,
                          double tol, monolish::tensor::tensor_Dense<T> &AA,
                          std::tuple<Tp...> &As) {
  std::get<I>(As) = AA;
  if (!test_tscal_core(M, N, L, tol, AA, std::get<I>(As))) {
    return false;
  }
  return test_tscal_view_core1<T, I + 1, Tp...>(M, N, L, tol, AA, As);
}

template <typename T>
bool test_tscal_view(const size_t M, const size_t N, const size_t L,
                     double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);

  monolish::tensor::tensor_Dense<T> AA(seedA);

  using T1 = monolish::tensor::tensor_Dense<T>;
  using T2_ = monolish::vector<T>;
  using T2 = monolish::view_tensor_Dense<T2_, T>;
  using T3_ = monolish::matrix::Dense<T>;
  using T3 = monolish::view_tensor_Dense<T3_, T>;
  using T4_ = monolish::tensor::tensor_Dense<T>;
  using T4 = monolish::view_tensor_Dense<T4_, T>;
  T1 x1({M, N, L});
  T2_ x2_(2 * M * N * L, 0.0, 1.0);
  T2 x2(x2_, L / 2, {M, N, L});
  T3_ x3_(2 * M * N * L, 1, 0.0, 1.0);
  T3 x3(x3_, L / 2, {M, N, L});
  T4_ x4_({2 * M * N * L, 1, 1}, 0.0, 1.0);
  T4 x4(x4_, L / 2, {M, N, L});

  auto As = std::make_tuple(x1, x2, x3, x4);

  return test_tscal_view_core1(M, N, L, tol, AA, As);
}
