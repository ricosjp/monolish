#include "../../test_utils.hpp"
#include "monolish_blas.hpp"

template <typename MAT> void ans_sqrt(MAT &A) {
  for (size_t i = 0; i < A.get_nnz(); i++) {
    A.begin()[i] = std::sqrt(A.begin()[i]);
  }
}

template <typename T, typename MAT1, typename MAT2>
bool test_send_msqrt_core(const size_t M, const size_t N, double tol,
                          monolish::matrix::Dense<T> &AA, MAT1 &A, MAT2 &B) {
  ans_sqrt(AA);

  monolish::util::send(A, B);
  monolish::vml::sqrt(A, B);
  monolish::util::recv(A, B);
  monolish::matrix::Dense<T> resultA(B);

  return ans_check<T>(__func__, A.type() + "=" + B.type(), resultA.begin(),
                      AA.begin(), AA.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_send_msqrt(const size_t M, const size_t N, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);
  MAT A(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);

  return test_send_msqrt_core<T, MAT, MAT>(M, N, tol, AA, A, A);
}

template <typename T, typename MAT1, std::size_t J = 0, typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_send_msqrt_view_core2(const size_t M, const size_t N, double tol,
                           monolish::matrix::Dense<T> &AA, MAT1 &A,
                           std::tuple<Tq...> &Bs) {
  return true;
}

template <typename T, typename MAT1, std::size_t J = 0, typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type
    test_send_msqrt_view_core2(const size_t M, const size_t N, double tol,
                               monolish::matrix::Dense<T> &AA, MAT1 &A,
                               std::tuple<Tq...> &Bs) {
  A = AA;
  if (!test_send_msqrt_core<T, MAT1, decltype(std::get<J>(Bs))>(
          M, N, tol, AA, A, std::get<J>(Bs))) {
    return false;
  }
  return test_send_msqrt_view_core2<T, MAT1, J + 1, Tq...>(M, N, tol, AA, A,
                                                           Bs);
}

template <typename T, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_send_msqrt_view_core1(const size_t M, const size_t N, double tol,
                           monolish::matrix::Dense<T> &AA,
                           std::tuple<Tp...> &As, std::tuple<Tp...> &Bs) {
  return true;
}

template <typename T, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type
    test_send_msqrt_view_core1(const size_t M, const size_t N, double tol,
                               monolish::matrix::Dense<T> &AA,
                               std::tuple<Tp...> &As, std::tuple<Tp...> &Bs) {
  if (!test_send_msqrt_view_core2<T, decltype(std::get<I>(As)), 0, Tp...>(
          M, N, tol, AA, std::get<I>(As), Bs)) {
    return false;
  }
  return test_send_msqrt_view_core1<T, I + 1, Tp...>(M, N, tol, AA, As, Bs);
}

template <typename T>
bool test_send_msqrt_view(const size_t M, const size_t N, double tol) {
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
  auto Bs = std::make_tuple(x1, x2, x3, x4);

  return test_send_msqrt_view_core1<T, 0, T1, T2, T3, T4>(M, N, tol, AA, As,
                                                          Bs);
}

template <typename T, typename MAT1, typename MAT2>
bool test_msqrt_core(const size_t M, const size_t N, double tol,
                     monolish::matrix::Dense<T> &AA, MAT1 &A, MAT2 &B) {
  ans_sqrt(AA);
  monolish::vml::sqrt(A, B);
  monolish::matrix::Dense<T> resultA(B);

  return ans_check<T>(__func__, A.type() + "=" + B.type(), resultA.begin(),
                      AA.begin(), AA.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_msqrt(const size_t M, const size_t N, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);
  MAT A(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);

  return test_msqrt_core<T, MAT, MAT>(M, N, tol, AA, A, A);
}

template <typename T, typename MAT1, std::size_t J = 0, typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_msqrt_view_core2(const size_t M, const size_t N, double tol,
                      monolish::matrix::Dense<T> &AA, MAT1 &A,
                      std::tuple<Tq...> &Bs) {
  return true;
}

template <typename T, typename MAT1, std::size_t J = 0, typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type
    test_msqrt_view_core2(const size_t M, const size_t N, double tol,
                          monolish::matrix::Dense<T> &AA, MAT1 &A,
                          std::tuple<Tq...> &Bs) {
  A = AA;
  if (!test_msqrt_core<T, MAT1, decltype(std::get<J>(Bs))>(M, N, tol, AA, A,
                                                           std::get<J>(Bs))) {
    return false;
  }
  return test_msqrt_view_core2<T, MAT1, J + 1, Tq...>(M, N, tol, AA, A, Bs);
}

template <typename T, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_msqrt_view_core1(const size_t M, const size_t N, double tol,
                      monolish::matrix::Dense<T> &AA, std::tuple<Tp...> &As,
                      std::tuple<Tp...> &Bs) {
  return true;
}

template <typename T, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type
    test_msqrt_view_core1(const size_t M, const size_t N, double tol,
                          monolish::matrix::Dense<T> &AA, std::tuple<Tp...> &As,
                          std::tuple<Tp...> &Bs) {
  if (!test_msqrt_view_core2<T, decltype(std::get<I>(As)), 0, Tp...>(
          M, N, tol, AA, std::get<I>(As), Bs)) {
    return false;
  }
  return test_msqrt_view_core1<T, I + 1, Tp...>(M, N, tol, AA, As, Bs);
}

template <typename T>
bool test_msqrt_view(const size_t M, const size_t N, double tol) {
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
  auto Bs = std::make_tuple(x1, x2, x3, x4);

  return test_msqrt_view_core1<T, 0, T1, T2, T3, T4>(M, N, tol, AA, As, Bs);
}
