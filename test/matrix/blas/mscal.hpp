#include "../../test_utils.hpp"

template <typename T>
void ans_mscal(const double alpha, monolish::matrix::Dense<T> &A) {

  for (int i = 0; i < A.get_nnz(); i++)
    A.begin()[i] = alpha * A.begin()[i];
}

template <typename MAT, typename T>
bool test_send_mscal_core(const size_t M, const size_t N, double tol,
                          monolish::matrix::Dense<T> &AA, MAT &A) {
  T alpha = 123.0;
  ans_mscal(alpha, AA);
  monolish::matrix::COO<T> ansA(AA);

  A.send();
  monolish::blas::mscal(alpha, A);
  A.recv();
  monolish::matrix::COO<T> resultA(A);

  return ans_check<T>(__func__, A.type(), resultA.begin(), ansA.begin(),
                      ansA.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_send_mscal(const size_t M, const size_t N, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);

  MAT A(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);

  return test_send_mscal_core(M, N, tol, AA, A);
}

template <typename T, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_send_mscal_view_core1(const size_t M, const size_t N, double tol,
                           monolish::matrix::Dense<T> &AA,
                           std::tuple<Tp...> &As) {
  return true;
}

template <typename T, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type
    test_send_mscal_view_core1(const size_t M, const size_t N, double tol,
                               monolish::matrix::Dense<T> &AA,
                               std::tuple<Tp...> &As) {
  std::get<I>(As) = AA;
  if (!test_send_mscal_core(M, N, tol, AA, std::get<I>(As))) {
    return false;
  }
  return test_send_mscal_view_core1<T, I + 1, Tp...>(M, N, tol, AA, As);
}

template <typename T>
bool test_send_mscal_view(const size_t M, const size_t N, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);

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

  return test_send_mscal_view_core1<T, 0, T1, T2, T3, T4>(M, N, tol, AA, As);
}

template <typename MAT, typename T>
bool test_mscal_core(const size_t M, const size_t N, double tol,
                     monolish::matrix::Dense<T> &AA, MAT &A) {
  T alpha = 123.0;
  ans_mscal(alpha, AA);
  monolish::matrix::COO<T> ansA(AA);

  monolish::blas::mscal(alpha, A);
  monolish::matrix::COO<T> resultA(A);

  return ans_check<T>(__func__, A.type(), resultA.begin(), ansA.begin(),
                      ansA.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_mscal(const size_t M, const size_t N, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);

  MAT A(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);

  return test_mscal_core(M, N, tol, AA, A);
}

template <typename T, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_mscal_view_core1(const size_t M, const size_t N, double tol,
                      monolish::matrix::Dense<T> &AA, std::tuple<Tp...> &As) {
  return true;
}

template <typename T, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type
    test_mscal_view_core1(const size_t M, const size_t N, double tol,
                          monolish::matrix::Dense<T> &AA,
                          std::tuple<Tp...> &As) {
  std::get<I>(As) = AA;
  if (!test_mscal_core(M, N, tol, AA, std::get<I>(As))) {
    return false;
  }
  return test_mscal_view_core1<T, I + 1, Tp...>(M, N, tol, AA, As);
}

template <typename T>
bool test_mscal_view(const size_t M, const size_t N, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);

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

  return test_mscal_view_core1<T, 0, T1, T2, T3, T4>(M, N, tol, AA, As);
}
