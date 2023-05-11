#include "../../test_utils.hpp"
#include "monolish_blas.hpp"

template <typename T> T ans_min(monolish::matrix::Dense<T> &A) {
  return *(std::min_element(A.begin(), A.begin() + A.get_nnz()));
}

template <typename T, typename MAT1>
bool test_send_mmin_core(const size_t M, const size_t N, double tol,
                         monolish::matrix::Dense<T> &AA, MAT1 &A) {
  T ans = ans_min(AA);

  monolish::util::send(A);
  T result = monolish::vml::min(A);
  monolish::matrix::Dense<T> resultA(A);

  return ans_check<T>(__func__, result, ans, tol);
}

template <typename MAT, typename T>
bool test_send_mmin(const size_t M, const size_t N, double tol) {
  monolish::matrix::COO<T> seedA = get_seed_matrix<T>(M, N);
  MAT A(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);

  return test_send_mmin_core(M, N, tol, AA, A);
}

template <typename T, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_send_mmin_view_core1(const size_t M, const size_t N, double tol,
                          monolish::matrix::Dense<T> &AA,
                          std::tuple<Tp...> &As) {
  return true;
}

template <typename T, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type
    test_send_mmin_view_core1(const size_t M, const size_t N, double tol,
                              monolish::matrix::Dense<T> &AA,
                              std::tuple<Tp...> &As) {
  std::get<I>(As) = AA;
  if (!test_send_mmin_core(M, N, tol, AA, std::get<I>(As))) {
    return false;
  }
  return test_send_mmin_view_core1<T, I + 1, Tp...>(M, N, tol, AA, As);
}

template <typename T>
bool test_send_mmin_view(const size_t M, const size_t N, double tol) {
  monolish::matrix::COO<T> seedA = get_seed_matrix<T>(M, N);
  monolish::matrix::Dense<T> AA(seedA);

  using T1 = monolish::matrix::Dense<T>;
  using T2_ = monolish::vector<T>;
  using T2 = monolish::view_Dense<T2_, T>;
  using T3_ = monolish::matrix::Dense<T>;
  using T3 = monolish::view_Dense<T3_, T>;
  using T4_ = monolish::tensor::tensor_Dense<T>;
  using T4 = monolish::view_Dense<T4_, T>;
  T1 x1(M, N);
  T2_ x2_(2 * M * N, 0.0, 1.0);
  T2 x2(x2_, M / 2, M, N);
  T3_ x3_(2 * M * N, 1, 0.0, 1.0);
  T3 x3(x3_, M / 2, M, N);
  T4_ x4_({2 * M * N, 1, 1}, 0.0, 1.0);
  T4 x4(x4_, M / 2, M, N);

  auto As = std::make_tuple(x1, x2, x3, x4);

  return test_send_mmin_view_core1<T, 0, T1, T2, T3, T4>(M, N, tol, AA, As);
}

template <typename T, typename MAT1>
bool test_mmin_core(const size_t M, const size_t N, double tol,
                    monolish::matrix::Dense<T> &AA, MAT1 &A) {
  T ans = ans_min(AA);

  T result = monolish::vml::min(A);
  monolish::matrix::Dense<T> resultA(A);

  return ans_check<T>(__func__, result, ans, tol);
}

template <typename MAT, typename T>
bool test_mmin(const size_t M, const size_t N, double tol) {
  monolish::matrix::COO<T> seedA = get_seed_matrix<T>(M, N);
  MAT A(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);

  return test_mmin_core(M, N, tol, AA, A);
}

template <typename T, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_mmin_view_core1(const size_t M, const size_t N, double tol,
                     monolish::matrix::Dense<T> &AA, std::tuple<Tp...> &As) {
  return true;
}

template <typename T, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type
    test_mmin_view_core1(const size_t M, const size_t N, double tol,
                         monolish::matrix::Dense<T> &AA,
                         std::tuple<Tp...> &As) {
  std::get<I>(As) = AA;
  if (!test_mmin_core(M, N, tol, AA, std::get<I>(As))) {
    return false;
  }
  return test_mmin_view_core1<T, I + 1, Tp...>(M, N, tol, AA, As);
}

template <typename T>
bool test_mmin_view(const size_t M, const size_t N, double tol) {
  monolish::matrix::COO<T> seedA = get_seed_matrix<T>(M, N);
  monolish::matrix::Dense<T> AA(seedA);

  using T1 = monolish::matrix::Dense<T>;
  using T2_ = monolish::vector<T>;
  using T2 = monolish::view_Dense<T2_, T>;
  using T3_ = monolish::matrix::Dense<T>;
  using T3 = monolish::view_Dense<T3_, T>;
  using T4_ = monolish::tensor::tensor_Dense<T>;
  using T4 = monolish::view_Dense<T4_, T>;
  T1 x1(M, N);
  T2_ x2_(2 * M * N, 0.0, 1.0);
  T2 x2(x2_, M / 2, M, N);
  T3_ x3_(2 * M * N, 1, 0.0, 1.0);
  T3 x3(x3_, M / 2, M, N);
  T4_ x4_({2 * M * N, 1, 1}, 0.0, 1.0);
  T4 x4(x4_, M / 2, M, N);

  auto As = std::make_tuple(x1, x2, x3, x4);

  return test_mmin_view_core1<T, 0, T1, T2, T3, T4>(M, N, tol, AA, As);
}
