#include "../../test_utils.hpp"

template <typename T>
void ans_scalar_times(const T alpha, const monolish::tensor::tensor_Dense<T> &A,
                      monolish::tensor::tensor_Dense<T> &C) {

  if (A.get_shape() != C.get_shape()) {
    std::runtime_error("A.shape != C.shape");
  }

  for (int i = 0; i < A.get_nnz(); i++) {
    C.begin()[i] = alpha * A.begin()[i];
  }
}

template <typename T, typename MAT_A, typename MAT_C>
bool test_send_scalar_times_core(const size_t M, const size_t N, const size_t L,
                                 double tol,
                                 monolish::tensor::tensor_Dense<T> &AA,
                                 monolish::tensor::tensor_Dense<T> &CC,
                                 MAT_A &A, MAT_C &C) {
  T alpha = 123.0;

  ans_scalar_times(alpha, AA, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, C);
  monolish::blas::times(alpha, A, C);
  monolish::util::recv(A, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_C, typename T>
bool test_send_scalar_times(const size_t M, const size_t N, const size_t L,
                            double tol) {

  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);

  MAT_A A(seedA);
  MAT_C C(seedA);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedA);

  return test_send_scalar_times_core(M, N, L, tol, AA, CC, A, C);
}

template <typename T, typename MAT_A, std::size_t J = 0, typename... Tp>
inline typename std::enable_if<J == sizeof...(Tp), bool>::type
test_send_scalar_times_view_core2(const size_t M, const size_t N,
                                  const size_t L, double tol,
                                  monolish::tensor::tensor_Dense<T> &AA,
                                  monolish::tensor::tensor_Dense<T> &CC,
                                  MAT_A &A, std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, typename MAT_A, std::size_t J = 0, typename... Tp>
    inline typename std::enable_if <
    J<sizeof...(Tp), bool>::type
    test_send_scalar_times_view_core2(const size_t M, const size_t N,
                                      const size_t L, double tol,
                                      monolish::tensor::tensor_Dense<T> &AA,
                                      monolish::tensor::tensor_Dense<T> &CC,
                                      MAT_A &A, std::tuple<Tp...> &Cs) {
  A = AA;
  if (!test_send_scalar_times_core(M, N, L, tol, AA, CC, A, std::get<J>(Cs))) {
    return false;
  }
  return test_send_scalar_times_view_core2<T, MAT_A, J + 1, Tp...>(
      M, N, L, tol, AA, CC, A, Cs);
}

template <typename T, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_send_scalar_times_view_core1(const size_t M, const size_t N,
                                  const size_t L, double tol,
                                  monolish::tensor::tensor_Dense<T> &AA,
                                  monolish::tensor::tensor_Dense<T> &CC,
                                  std::tuple<Tp...> &A, std::tuple<Tp...> &C) {
  return true;
}

template <typename T, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type test_send_scalar_times_view_core1(
        const size_t M, const size_t N, const size_t L, double tol,
        monolish::tensor::tensor_Dense<T> &AA,
        monolish::tensor::tensor_Dense<T> &CC, std::tuple<Tp...> &As,
        std::tuple<Tp...> &Cs) {
  if (!test_send_scalar_times_view_core2(M, N, L, tol, AA, CC, std::get<I>(As),
                                         Cs)) {
    return false;
  }
  return test_send_scalar_times_view_core1<T, I + 1, Tp...>(M, N, L, tol, AA,
                                                            CC, As, Cs);
}

template <typename T>
bool test_send_scalar_times_view(const size_t M, const size_t N, const size_t L,
                                 double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedA);

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

  T1 y1({M, N, L});
  T2_ y2_(2 * M * N * L, 0.0, 1.0);
  T2 y2(y2_, L / 2, {M, N, L});
  T3_ y3_(2 * M * N * L, 1, 0.0, 1.0);
  T3 y3(y3_, L / 2, {M, N, L});
  T4_ y4_({2 * M * N * L, 1, 1}, 0.0, 1.0);
  T4 y4(y4_, L / 2, {M, N, L});

  auto As = std::make_tuple(x1, x2, x3, x4);
  auto Cs = std::make_tuple(y1, y2, y3, y4);

  return test_send_scalar_times_view_core1(M, N, L, tol, AA, CC, As, Cs);
}

template <typename T, typename MAT_A, typename MAT_C>
bool test_scalar_times_core(const size_t M, const size_t N, const size_t L,
                            double tol, monolish::tensor::tensor_Dense<T> &AA,
                            monolish::tensor::tensor_Dense<T> &CC, MAT_A &A,
                            MAT_C &C) {
  T alpha = 123.0;

  ans_scalar_times(alpha, AA, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::blas::times(alpha, A, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_C, typename T>
bool test_scalar_times(const size_t M, const size_t N, const size_t L,
                       double tol) {

  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);

  MAT_A A(seedA);
  MAT_C C(seedA);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedA);

  return test_scalar_times_core(M, N, L, tol, AA, CC, A, C);
}

template <typename T, typename MAT_A, std::size_t J = 0, typename... Tp>
inline typename std::enable_if<J == sizeof...(Tp), bool>::type
test_scalar_times_view_core2(const size_t M, const size_t N, const size_t L,
                             double tol, monolish::tensor::tensor_Dense<T> &AA,
                             monolish::tensor::tensor_Dense<T> &CC, MAT_A &A,
                             std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, typename MAT_A, std::size_t J = 0, typename... Tp>
    inline typename std::enable_if <
    J<sizeof...(Tp), bool>::type
    test_scalar_times_view_core2(const size_t M, const size_t N, const size_t L,
                                 double tol,
                                 monolish::tensor::tensor_Dense<T> &AA,
                                 monolish::tensor::tensor_Dense<T> &CC,
                                 MAT_A &A, std::tuple<Tp...> &Cs) {
  A = AA;
  if (!test_scalar_times_core(M, N, L, tol, AA, CC, A, std::get<J>(Cs))) {
    return false;
  }
  return test_scalar_times_view_core2<T, MAT_A, J + 1, Tp...>(M, N, L, tol, AA,
                                                              CC, A, Cs);
}

template <typename T, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_scalar_times_view_core1(const size_t M, const size_t N, const size_t L,
                             double tol, monolish::tensor::tensor_Dense<T> &AA,
                             monolish::tensor::tensor_Dense<T> &CC,
                             std::tuple<Tp...> &A, std::tuple<Tp...> &C) {
  return true;
}

template <typename T, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type
    test_scalar_times_view_core1(const size_t M, const size_t N, const size_t L,
                                 double tol,
                                 monolish::tensor::tensor_Dense<T> &AA,
                                 monolish::tensor::tensor_Dense<T> &CC,
                                 std::tuple<Tp...> &As, std::tuple<Tp...> &Cs) {
  if (!test_scalar_times_view_core2(M, N, L, tol, AA, CC, std::get<I>(As),
                                    Cs)) {
    return false;
  }
  return test_scalar_times_view_core1<T, I + 1, Tp...>(M, N, L, tol, AA, CC, As,
                                                       Cs);
}

template <typename T>
bool test_scalar_times_view(const size_t M, const size_t N, const size_t L,
                            double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedA);

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

  T1 y1({M, N, L});
  T2_ y2_(2 * M * N * L, 0.0, 1.0);
  T2 y2(y2_, L / 2, {M, N, L});
  T3_ y3_(2 * M * N * L, 1, 0.0, 1.0);
  T3 y3(y3_, L / 2, {M, N, L});
  T4_ y4_({2 * M * N * L, 1, 1}, 0.0, 1.0);
  T4 y4(y4_, L / 2, {M, N, L});

  auto As = std::make_tuple(x1, x2, x3, x4);
  auto Cs = std::make_tuple(y1, y2, y3, y4);

  return test_scalar_times_view_core1(M, N, L, tol, AA, CC, As, Cs);
}
