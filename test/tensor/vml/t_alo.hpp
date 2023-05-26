#include "../../test_utils.hpp"
#include "monolish_blas.hpp"

template <typename T>
void ans_alo(monolish::tensor::tensor_Dense<T> &A, const T alpha,
             const T beta) {

  for (int i = 0; i < A.get_nnz(); i++) {
    if (A.begin()[i] > 0) {
      A.begin()[i] = alpha * A.begin()[i];
    } else {
      A.begin()[i] = beta * A.begin()[i];
    }
  }
}

template <typename T, typename MAT1, typename MAT2>
bool test_send_talo_core(const size_t M, const size_t N, const size_t L,
                         double tol, monolish::tensor::tensor_Dense<T> &AA,
                         MAT1 &A, MAT2 &B) {
  T alpha = 1.5;
  T beta = 0.5;
  ans_alo(AA, alpha, beta);

  monolish::util::send(A, B);
  monolish::vml::alo(A, alpha, beta, B);
  monolish::util::recv(A, B);
  monolish::tensor::tensor_Dense<T> resultA(B);

  return ans_check<T>(__func__, A.type() + "=" + B.type(), resultA.begin(),
                      AA.begin(), AA.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_send_talo(const size_t M, const size_t N, const size_t L,
                    double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);

  MAT A(seedA); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);

  return test_send_talo_core<T, MAT, MAT>(M, N, L, tol, AA, A, A);
}

template <typename T, typename MAT1, std::size_t J = 0, typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_send_talo_view_core2(const size_t M, const size_t N, const size_t L,
                          double tol, monolish::tensor::tensor_Dense<T> &AA,
                          MAT1 &A, std::tuple<Tq...> &Bs) {
  return true;
}

template <typename T, typename MAT1, std::size_t J = 0, typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type
    test_send_talo_view_core2(const size_t M, const size_t N, const size_t L,
                              double tol, monolish::tensor::tensor_Dense<T> &AA,
                              MAT1 &A, std::tuple<Tq...> &Bs) {
  A = AA;
  if (!test_send_talo_core<T, MAT1, decltype(std::get<J>(Bs))>(
          M, N, L, tol, AA, A, std::get<J>(Bs))) {
    return false;
  }
  return test_send_talo_view_core2<T, MAT1, J + 1, Tq...>(M, N, L, tol, AA, A,
                                                          Bs);
}

template <typename T, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_send_talo_view_core1(const size_t M, const size_t N, const size_t L,
                          double tol, monolish::tensor::tensor_Dense<T> &AA,
                          std::tuple<Tp...> &As, std::tuple<Tp...> &Bs) {
  return true;
}

template <typename T, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type
    test_send_talo_view_core1(const size_t M, const size_t N, const size_t L,
                              double tol, monolish::tensor::tensor_Dense<T> &AA,
                              std::tuple<Tp...> &As, std::tuple<Tp...> &Bs) {
  if (!test_send_talo_view_core2<T, decltype(std::get<I>(As)), 0, Tp...>(
          M, N, L, tol, AA, std::get<I>(As), Bs)) {
    return false;
  }
  return test_send_talo_view_core1<T, I + 1, Tp...>(M, N, L, tol, AA, As, Bs);
}

template <typename T>
bool test_send_talo_view(const size_t M, const size_t N, const size_t L,
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
  auto Bs = std::make_tuple(x1, x2, x3, x4);

  return test_send_talo_view_core1<T, 0, T1, T2, T3, T4>(M, N, L, tol, AA, As,
                                                         Bs);
}

template <typename T, typename MAT1, typename MAT2>
bool test_talo_core(const size_t M, const size_t N, const size_t L, double tol,
                    monolish::tensor::tensor_Dense<T> &AA, MAT1 &A, MAT2 &B) {
  T alpha = 1.5;
  T beta = 0.5;
  ans_alo(AA, alpha, beta);

  monolish::vml::alo(A, alpha, beta, B);
  monolish::tensor::tensor_Dense<T> resultA(B);

  return ans_check<T>(__func__, A.type() + "=" + B.type(), resultA.begin(),
                      AA.begin(), AA.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_talo(const size_t M, const size_t N, const size_t L, double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);

  MAT A(seedA); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);

  return test_talo_core<T, MAT, MAT>(M, N, L, tol, AA, A, A);
}

template <typename T, typename MAT1, std::size_t J = 0, typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_talo_view_core2(const size_t M, const size_t N, const size_t L, double tol,
                     monolish::tensor::tensor_Dense<T> &AA, MAT1 &A,
                     std::tuple<Tq...> &Bs) {
  return true;
}

template <typename T, typename MAT1, std::size_t J = 0, typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type
    test_talo_view_core2(const size_t M, const size_t N, const size_t L,
                         double tol, monolish::tensor::tensor_Dense<T> &AA,
                         MAT1 &A, std::tuple<Tq...> &Bs) {
  A = AA;
  if (!test_talo_core<T, MAT1, decltype(std::get<J>(Bs))>(M, N, L, tol, AA, A,
                                                          std::get<J>(Bs))) {
    return false;
  }
  return test_talo_view_core2<T, MAT1, J + 1, Tq...>(M, N, L, tol, AA, A, Bs);
}

template <typename T, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_talo_view_core1(const size_t M, const size_t N, const size_t L, double tol,
                     monolish::tensor::tensor_Dense<T> &AA,
                     std::tuple<Tp...> &As, std::tuple<Tp...> &Bs) {
  return true;
}

template <typename T, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type
    test_talo_view_core1(const size_t M, const size_t N, const size_t L,
                         double tol, monolish::tensor::tensor_Dense<T> &AA,
                         std::tuple<Tp...> &As, std::tuple<Tp...> &Bs) {
  if (!test_talo_view_core2<T, decltype(std::get<I>(As)), 0, Tp...>(
          M, N, L, tol, AA, std::get<I>(As), Bs)) {
    return false;
  }
  return test_talo_view_core1<T, I + 1, Tp...>(M, N, L, tol, AA, As, Bs);
}

template <typename T>
bool test_talo_view(const size_t M, const size_t N, const size_t L,
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
  auto Bs = std::make_tuple(x1, x2, x3, x4);

  return test_talo_view_core1<T, 0, T1, T2, T3, T4>(M, N, L, tol, AA, As, Bs);
}
