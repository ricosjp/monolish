#include "../../test_utils.hpp"

template <typename T>
void ans_st_mul(const monolish::tensor::tensor_Dense<T> &A, const T alpha,
                monolish::tensor::tensor_Dense<T> &C) {

  for (int i = 0; i < A.get_nnz(); i++) {
    C.begin()[i] = A.begin()[i] * alpha;
  }
}

template <typename T, typename MAT_A, typename MAT_C>
bool test_send_st_mul_core(const size_t M, const size_t N, const size_t L,
                           double tol, monolish::tensor::tensor_Dense<T> &AA,
                           monolish::tensor::tensor_Dense<T> &CC, MAT_A &A,
                           MAT_C &C) {
  T alpha = 123.0;
  ans_st_mul(AA, alpha, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, C);
  monolish::vml::mul(A, alpha, C);
  monolish::util::recv(A, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__,
                      A.type() + "+" + typeid(T).name() + "=" + C.type(),
                      resultC.begin(), ansC.begin(), ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_C, typename T>
bool test_send_st_mul(const size_t M, const size_t N, const size_t L,
                      double tol) {
  monolish::tensor::tensor_Dense<T> seed1({M, N, L}, 1.0, 2.0);
  monolish::tensor::tensor_COO<T> seedA(seed1);

  MAT_A A(seedA);
  MAT_C C(seedA);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedA);

  return test_send_st_mul_core(M, N, L, tol, AA, CC, A, C);
}

template <typename T, typename MAT1, std::size_t J = 0, typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_send_st_mul_view_core2(const size_t M, const size_t N, const size_t L,
                            double tol, monolish::tensor::tensor_Dense<T> &AA,
                            monolish::tensor::tensor_Dense<T> &CC, MAT1 &A,
                            std::tuple<Tq...> &Bs) {
  return true;
}

template <typename T, typename MAT1, std::size_t J = 0, typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type test_send_st_mul_view_core2(
        const size_t M, const size_t N, const size_t L, double tol,
        monolish::tensor::tensor_Dense<T> &AA,
        monolish::tensor::tensor_Dense<T> &CC, MAT1 &A, std::tuple<Tq...> &Bs) {
  A = AA;
  if (!test_send_st_mul_core<T, MAT1, decltype(std::get<J>(Bs))>(
          M, N, L, tol, AA, CC, A, std::get<J>(Bs))) {
    return false;
  }
  return test_send_st_mul_view_core2<T, MAT1, J + 1, Tq...>(M, N, L, tol, AA,
                                                            CC, A, Bs);
}

template <typename T, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_send_st_mul_view_core1(const size_t M, const size_t N, const size_t L,
                            double tol, monolish::tensor::tensor_Dense<T> &AA,
                            monolish::tensor::tensor_Dense<T> &CC,
                            std::tuple<Tp...> &As, std::tuple<Tp...> &Bs) {
  return true;
}

template <typename T, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type
    test_send_st_mul_view_core1(const size_t M, const size_t N, const size_t L,
                                double tol,
                                monolish::tensor::tensor_Dense<T> &AA,
                                monolish::tensor::tensor_Dense<T> &CC,
                                std::tuple<Tp...> &As, std::tuple<Tp...> &Bs) {
  if (!test_send_st_mul_view_core2<T, decltype(std::get<I>(As)), 0, Tp...>(
          M, N, L, tol, AA, CC, std::get<I>(As), Bs)) {
    return false;
  }
  return test_send_st_mul_view_core1<T, I + 1, Tp...>(M, N, L, tol, AA, CC, As,
                                                      Bs);
}

template <typename T>
bool test_send_st_mul_view(const size_t M, const size_t N, const size_t L,
                           double tol) {
  monolish::tensor::tensor_Dense<T> seed1({M, N, L}, 1.0, 2.0);
  monolish::tensor::tensor_COO<T> seedA(seed1);
  monolish::tensor::tensor_Dense<T> AA(seedA), CC(seedA);

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

  return test_send_st_mul_view_core1<T, 0, T1, T2, T3, T4>(M, N, L, tol, AA, CC,
                                                           As, Bs);
}

template <typename T, typename MAT_A, typename MAT_C>
bool test_st_mul_core(const size_t M, const size_t N, const size_t L,
                      double tol, monolish::tensor::tensor_Dense<T> &AA,
                      monolish::tensor::tensor_Dense<T> &CC, MAT_A &A,
                      MAT_C &C) {
  T alpha = 123.0;
  ans_st_mul(AA, alpha, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::vml::mul(A, alpha, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__,
                      A.type() + "+" + typeid(T).name() + "=" + C.type(),
                      resultC.begin(), ansC.begin(), ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_C, typename T>
bool test_st_mul(const size_t M, const size_t N, const size_t L, double tol) {
  monolish::tensor::tensor_Dense<T> seed1({M, N, L}, 1.0, 2.0);
  monolish::tensor::tensor_COO<T> seedA(seed1);

  MAT_A A(seedA);
  MAT_C C(seedA);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedA);

  return test_st_mul_core(M, N, L, tol, AA, CC, A, C);
}

template <typename T, typename MAT1, std::size_t J = 0, typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_st_mul_view_core2(const size_t M, const size_t N, const size_t L,
                       double tol, monolish::tensor::tensor_Dense<T> &AA,
                       monolish::tensor::tensor_Dense<T> &CC, MAT1 &A,
                       std::tuple<Tq...> &Bs) {
  return true;
}

template <typename T, typename MAT1, std::size_t J = 0, typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type
    test_st_mul_view_core2(const size_t M, const size_t N, const size_t L,
                           double tol, monolish::tensor::tensor_Dense<T> &AA,
                           monolish::tensor::tensor_Dense<T> &CC, MAT1 &A,
                           std::tuple<Tq...> &Bs) {
  A = AA;
  if (!test_st_mul_core<T, MAT1, decltype(std::get<J>(Bs))>(
          M, N, L, tol, AA, CC, A, std::get<J>(Bs))) {
    return false;
  }
  return test_st_mul_view_core2<T, MAT1, J + 1, Tq...>(M, N, L, tol, AA, CC, A,
                                                       Bs);
}

template <typename T, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_st_mul_view_core1(const size_t M, const size_t N, const size_t L,
                       double tol, monolish::tensor::tensor_Dense<T> &AA,
                       monolish::tensor::tensor_Dense<T> &CC,
                       std::tuple<Tp...> &As, std::tuple<Tp...> &Bs) {
  return true;
}

template <typename T, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type
    test_st_mul_view_core1(const size_t M, const size_t N, const size_t L,
                           double tol, monolish::tensor::tensor_Dense<T> &AA,
                           monolish::tensor::tensor_Dense<T> &CC,
                           std::tuple<Tp...> &As, std::tuple<Tp...> &Bs) {
  if (!test_st_mul_view_core2<T, decltype(std::get<I>(As)), 0, Tp...>(
          M, N, L, tol, AA, CC, std::get<I>(As), Bs)) {
    return false;
  }
  return test_st_mul_view_core1<T, I + 1, Tp...>(M, N, L, tol, AA, CC, As, Bs);
}

template <typename T>
bool test_st_mul_view(const size_t M, const size_t N, const size_t L,
                      double tol) {
  monolish::tensor::tensor_Dense<T> seed1({M, N, L}, 1.0, 2.0);
  monolish::tensor::tensor_COO<T> seedA(seed1);
  monolish::tensor::tensor_Dense<T> AA(seedA), CC(seedA);

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

  return test_st_mul_view_core1<T, 0, T1, T2, T3, T4>(M, N, L, tol, AA, CC, As,
                                                      Bs);
}
