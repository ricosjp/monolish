#include "../../test_utils.hpp"
#include "monolish_blas.hpp"

template <typename T> T ans_min(monolish::tensor::tensor_Dense<T> &A) {
  return *(std::min_element(A.begin(), A.begin() + A.get_nnz()));
}

template <typename T, typename MAT1>
bool test_send_tmin_core(const size_t M, const size_t N, const size_t L,
                         double tol, monolish::tensor::tensor_Dense<T> &AA,
                         MAT1 &A) {
  T ans = ans_min(AA);

  monolish::util::send(A);
  T result = monolish::vml::min(A);

  return ans_check<T>(__func__, result, ans, tol);
}

template <typename MAT, typename T>
bool test_send_tmin(const size_t M, const size_t N, const size_t L,
                    double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);

  MAT A(seedA); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);

  return test_send_tmin_core<T, MAT>(M, N, L, tol, AA, A);
}

template <typename T, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_send_tmin_view_core1(const size_t M, const size_t N, const size_t L,
                          double tol, monolish::tensor::tensor_Dense<T> &AA,
                          std::tuple<Tp...> &As) {
  return true;
}

template <typename T, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type
    test_send_tmin_view_core1(const size_t M, const size_t N, const size_t L,
                              double tol, monolish::tensor::tensor_Dense<T> &AA,
                              std::tuple<Tp...> &As) {
  std::get<I>(As) = AA;
  if (!test_send_tmin_core(M, N, L, tol, AA, std::get<I>(As))) {
    return false;
  }
  return test_send_tmin_view_core1<T, I + 1, Tp...>(M, N, L, tol, AA, As);
}

template <typename T>
bool test_send_tmin_view(const size_t M, const size_t N, const size_t L,
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
  T2 x2(x2_, M / 2, {M, N, L});
  T3_ x3_(2 * M * N * L, 1, 0.0, 1.0);
  T3 x3(x3_, M / 2, {M, N, L});
  T4_ x4_({2 * M * N * L, 1, 1}, 0.0, 1.0);
  T4 x4(x4_, M / 2, {M, N, L});

  auto As = std::make_tuple(x1, x2, x3, x4);

  return test_send_tmin_view_core1<T, 0, T1, T2, T3, T4>(M, N, L, tol, AA, As);
}

template <typename T, typename MAT1>
bool test_tmin_core(const size_t M, const size_t N, const size_t L, double tol,
                    monolish::tensor::tensor_Dense<T> &AA, MAT1 &A) {
  T ans = ans_min(AA);

  T result = monolish::vml::min(A);

  return ans_check<T>(__func__, result, ans, tol);
}

template <typename MAT, typename T>
bool test_tmin(const size_t M, const size_t N, const size_t L, double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);

  MAT A(seedA); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);

  return test_tmin_core<T, MAT>(M, N, L, tol, AA, A);
}

template <typename T, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_tmin_view_core1(const size_t M, const size_t N, const size_t L, double tol,
                     monolish::tensor::tensor_Dense<T> &AA,
                     std::tuple<Tp...> &As) {
  return true;
}

template <typename T, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type
    test_tmin_view_core1(const size_t M, const size_t N, const size_t L,
                         double tol, monolish::tensor::tensor_Dense<T> &AA,
                         std::tuple<Tp...> &As) {
  std::get<I>(As) = AA;
  if (!test_tmin_core(M, N, L, tol, AA, std::get<I>(As))) {
    return false;
  }
  return test_tmin_view_core1<T, I + 1, Tp...>(M, N, L, tol, AA, As);
}

template <typename T>
bool test_tmin_view(const size_t M, const size_t N, const size_t L,
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
  T2 x2(x2_, M / 2, {M, N, L});
  T3_ x3_(2 * M * N * L, 1, 0.0, 1.0);
  T3 x3(x3_, M / 2, {M, N, L});
  T4_ x4_({2 * M * N * L, 1, 1}, 0.0, 1.0);
  T4 x4(x4_, M / 2, {M, N, L});

  auto As = std::make_tuple(x1, x2, x3, x4);

  return test_tmin_view_core1<T, 0, T1, T2, T3, T4>(M, N, L, tol, AA, As);
}
