#include "../../test_utils.hpp"

template <typename T, typename VEC>
void ans_adds_row_line(const monolish::tensor::tensor_Dense<T> &A, size_t num,
                       const VEC &mx, monolish::tensor::tensor_Dense<T> &C) {
  if (A.get_shape()[2] != mx.size()) {
    std::runtime_error("A.col != y.size");
  }

  const T *x = mx.begin();
  int M = A.get_shape()[0] * A.get_shape()[1];
  int N = A.get_shape()[2];

  for (int j = 0; j < N; j++) {
    C.begin()[num * N + j] = A.begin()[num * N + j] + x[j];
  }
}

template <typename T, typename TENS_A, typename TENS_C, typename VEC>
bool test_send_adds_row_line_core(const size_t M, const size_t N,
                                  const size_t L, double tol,
                                  monolish::tensor::tensor_Dense<T> &AA,
                                  monolish::vector<T> &vec,
                                  monolish::tensor::tensor_Dense<T> &CC,
                                  TENS_A &A, VEC &x, TENS_C &C) {
  size_t line = M * N - 1;

  ans_adds_row_line(AA, line, vec, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, x, C);
  monolish::blas::adds_row(A, line, x, C);
  monolish::util::recv(A, x, C);
  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_send_adds_row_line(const size_t M, const size_t N, const size_t L,
                             double tol) {
  monolish::vector<T> x(L, 0.0, 1.0, test_random_engine());
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);

  MAT A(seedA); // M*N tensor
  MAT C(seedA); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedA);

  return test_send_adds_row_line_core(M, N, L, tol, AA, x, CC, A, x, C);
}

template <typename T, typename TENS_A, typename VEC, std::size_t K = 0,
          typename... Tp>
inline typename std::enable_if<K == sizeof...(Tp), bool>::type
test_send_adds_row_line_view_core3(const size_t M, const size_t N,
                                   const size_t L, double tol,
                                   monolish::tensor::tensor_Dense<T> &AA,
                                   monolish::vector<T> &vec,
                                   monolish::tensor::tensor_Dense<T> &CC,
                                   TENS_A &A, VEC &x, std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, typename TENS_A, typename VEC, std::size_t K = 0,
          typename... Tp>
    inline typename std::enable_if <
    K<sizeof...(Tp), bool>::type test_send_adds_row_line_view_core3(
        const size_t M, const size_t N, const size_t L, double tol,
        monolish::tensor::tensor_Dense<T> &AA, monolish::vector<T> &vec,
        monolish::tensor::tensor_Dense<T> &CC, TENS_A &A, VEC &x,
        std::tuple<Tp...> &Cs) {
  A = AA;
  x = vec;
  std::get<K>(Cs) = CC;
  if (!test_send_adds_row_line_core(M, N, L, tol, AA, vec, CC, A, x,
                                    std::get<K>(Cs))) {
    return false;
  }
  return test_send_adds_row_line_view_core3<T, TENS_A, VEC, K + 1, Tp...>(
      M, N, L, tol, AA, vec, CC, A, x, Cs);
}

template <typename T, typename TENS_A, std::size_t J = 0, typename TENSES_C,
          typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_send_adds_row_line_view_core2(const size_t M, const size_t N,
                                   const size_t L, double tol,
                                   monolish::tensor::tensor_Dense<T> &AA,
                                   monolish::vector<T> &vec,
                                   monolish::tensor::tensor_Dense<T> &CC,
                                   TENS_A &A, std::tuple<Tq...> &x,
                                   TENSES_C &Cs) {
  return true;
}

template <typename T, typename TENS_A, std::size_t J = 0, typename TENSES_C,
          typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type test_send_adds_row_line_view_core2(
        const size_t M, const size_t N, const size_t L, double tol,
        monolish::tensor::tensor_Dense<T> &AA, monolish::vector<T> &vec,
        monolish::tensor::tensor_Dense<T> &CC, TENS_A &A, std::tuple<Tq...> &xs,
        TENSES_C &Cs) {
  if (!test_send_adds_row_line_view_core3(M, N, L, tol, AA, vec, CC, A,
                                          std::get<J>(xs), Cs)) {
    return false;
  }
  return test_send_adds_row_line_view_core2<T, TENS_A, J + 1, TENSES_C, Tq...>(
      M, N, L, tol, AA, vec, CC, A, xs, Cs);
}

template <typename T, std::size_t I = 0, typename VECS, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_send_adds_row_line_view_core1(const size_t M, const size_t N,
                                   const size_t L, double tol,
                                   monolish::tensor::tensor_Dense<T> &AA,
                                   monolish::vector<T> &vec,
                                   monolish::tensor::tensor_Dense<T> &CC,
                                   std::tuple<Tp...> &As, VECS &xs,
                                   std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, std::size_t I = 0, typename VECS, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type test_send_adds_row_line_view_core1(
        const size_t M, const size_t N, const size_t L, double tol,
        monolish::tensor::tensor_Dense<T> &AA, monolish::vector<T> &vec,
        monolish::tensor::tensor_Dense<T> &CC, std::tuple<Tp...> &As, VECS &xs,
        std::tuple<Tp...> &Cs) {
  if (!test_send_adds_row_line_view_core2(M, N, L, tol, AA, vec, CC,
                                          std::get<I>(As), xs, Cs)) {
    return false;
  }
  return test_send_adds_row_line_view_core1<T, I + 1, VECS, Tp...>(
      M, N, L, tol, AA, vec, CC, As, xs, Cs);
}

template <typename T>
bool test_send_adds_row_line_view(const size_t M, const size_t N,
                                  const size_t L, double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);
  monolish::tensor::tensor_Dense<T> AA(seedA), CC(seedA);
  monolish::vector<T> xx(AA.get_shape()[2], 0.0, 1.0, test_random_engine());

  using T1_1 = monolish::tensor::tensor_Dense<T>;
  using T1_2_ = monolish::vector<T>;
  using T1_2 = monolish::view_tensor_Dense<T1_2_, T>;
  using T1_3_ = monolish::matrix::Dense<T>;
  using T1_3 = monolish::view_tensor_Dense<T1_3_, T>;
  using T1_4_ = monolish::tensor::tensor_Dense<T>;
  using T1_4 = monolish::view_tensor_Dense<T1_4_, T>;
  T1_1 x1({M, N, L}, 0.0, 1.0);
  T1_2_ x2_(2 * M * N * L, 0.0, 1.0);
  T1_2 x2(x2_, M / 2, {M, N, L});
  T1_3_ x3_(2 * M * N * L, 1, 0.0, 1.0);
  T1_3 x3(x3_, M / 2, {M, N, L});
  T1_4_ x4_({2 * M * N * L, 1, 1}, 0.0, 1.0);
  T1_4 x4(x4_, M / 2, {M, N, L});

  auto As = std::make_tuple(x1, x2, x3, x4);

  using T2_1 = monolish::vector<T>;
  using T2_2 = monolish::view1D<T1_2_, T>;
  using T2_3 = monolish::view1D<T1_3_, T>;
  using T2_4 = monolish::view1D<T1_4_, T>;
  T2_1 y1(L, 0.0, 1.0);
  T1_2_ y2_(2 * L, 0.0, 1.0);
  T2_2 y2(y2_, L / 2, L);
  T1_3_ y3_(2 * L, 1, 0.0, 1.0);
  T2_3 y3(y3_, L / 2, L);
  T1_4_ y4_({2 * L, 1, 1}, 0.0, 1.0);
  T2_4 y4(y4_, L / 2, L);

  auto Bs = std::make_tuple(y1, y2, y3, y4);

  T1_1 z1({M, N, L}, 0.0, 1.0);
  T1_2_ z2_(2 * M * N * L, 0.0, 1.0);
  T1_2 z2(z2_, M / 2, {M, N, L});
  T1_3_ z3_(2 * M * N * L, 1, 0.0, 1.0);
  T1_3 z3(z3_, M / 2, {M, N, L});
  T1_4_ z4_({2 * M * N * L, 1, 1}, 0.0, 1.0);
  T1_4 z4(z4_, M / 2, {M, N, L});

  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_send_adds_row_line_view_core1(M, N, L, tol, AA, xx, CC, As, Bs,
                                            Cs);
}

template <typename T, typename TENS_A, typename TENS_C, typename VEC>
bool test_adds_row_line_core(const size_t M, const size_t N, const size_t L,
                             double tol, monolish::tensor::tensor_Dense<T> &AA,
                             monolish::vector<T> &vec,
                             monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                             VEC &x, TENS_C &C) {
  size_t line = M * N - 1;

  ans_adds_row_line(AA, line, vec, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::blas::adds_row(A, line, x, C);
  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_adds_row_line(const size_t M, const size_t N, const size_t L,
                        double tol) {
  monolish::vector<T> x(L, 0.0, 1.0, test_random_engine());
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);

  MAT A(seedA); // M*N tensor
  MAT C(seedA); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedA);

  return test_adds_row_line_core(M, N, L, tol, AA, x, CC, A, x, C);
}

template <typename T, typename TENS_A, typename VEC, std::size_t K = 0,
          typename... Tp>
inline typename std::enable_if<K == sizeof...(Tp), bool>::type
test_adds_row_line_view_core3(const size_t M, const size_t N, const size_t L,
                              double tol, monolish::tensor::tensor_Dense<T> &AA,
                              monolish::vector<T> &vec,
                              monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                              VEC &x, std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, typename TENS_A, typename VEC, std::size_t K = 0,
          typename... Tp>
    inline typename std::enable_if <
    K<sizeof...(Tp), bool>::type test_adds_row_line_view_core3(
        const size_t M, const size_t N, const size_t L, double tol,
        monolish::tensor::tensor_Dense<T> &AA, monolish::vector<T> &vec,
        monolish::tensor::tensor_Dense<T> &CC, TENS_A &A, VEC &x,
        std::tuple<Tp...> &Cs) {
  A = AA;
  x = vec;
  std::get<K>(Cs) = CC;
  if (!test_adds_row_line_core(M, N, L, tol, AA, vec, CC, A, x,
                               std::get<K>(Cs))) {
    return false;
  }
  return test_adds_row_line_view_core3<T, TENS_A, VEC, K + 1, Tp...>(
      M, N, L, tol, AA, vec, CC, A, x, Cs);
}

template <typename T, typename TENS_A, std::size_t J = 0, typename TENSES_C,
          typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_adds_row_line_view_core2(const size_t M, const size_t N, const size_t L,
                              double tol, monolish::tensor::tensor_Dense<T> &AA,
                              monolish::vector<T> &vec,
                              monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                              std::tuple<Tq...> &x, TENSES_C &Cs) {
  return true;
}

template <typename T, typename TENS_A, std::size_t J = 0, typename TENSES_C,
          typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type test_adds_row_line_view_core2(
        const size_t M, const size_t N, const size_t L, double tol,
        monolish::tensor::tensor_Dense<T> &AA, monolish::vector<T> &vec,
        monolish::tensor::tensor_Dense<T> &CC, TENS_A &A, std::tuple<Tq...> &xs,
        TENSES_C &Cs) {
  if (!test_adds_row_line_view_core3(M, N, L, tol, AA, vec, CC, A,
                                     std::get<J>(xs), Cs)) {
    return false;
  }
  return test_adds_row_line_view_core2<T, TENS_A, J + 1, TENSES_C, Tq...>(
      M, N, L, tol, AA, vec, CC, A, xs, Cs);
}

template <typename T, std::size_t I = 0, typename VECS, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_adds_row_line_view_core1(const size_t M, const size_t N, const size_t L,
                              double tol, monolish::tensor::tensor_Dense<T> &AA,
                              monolish::vector<T> &vec,
                              monolish::tensor::tensor_Dense<T> &CC,
                              std::tuple<Tp...> &As, VECS &xs,
                              std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, std::size_t I = 0, typename VECS, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type test_adds_row_line_view_core1(
        const size_t M, const size_t N, const size_t L, double tol,
        monolish::tensor::tensor_Dense<T> &AA, monolish::vector<T> &vec,
        monolish::tensor::tensor_Dense<T> &CC, std::tuple<Tp...> &As, VECS &xs,
        std::tuple<Tp...> &Cs) {
  if (!test_adds_row_line_view_core2(M, N, L, tol, AA, vec, CC, std::get<I>(As),
                                     xs, Cs)) {
    return false;
  }
  return test_adds_row_line_view_core1<T, I + 1, VECS, Tp...>(
      M, N, L, tol, AA, vec, CC, As, xs, Cs);
}

template <typename T>
bool test_adds_row_line_view(const size_t M, const size_t N, const size_t L,
                             double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);
  monolish::tensor::tensor_Dense<T> AA(seedA), CC(seedA);
  monolish::vector<T> xx(AA.get_shape()[2], 0.0, 1.0, test_random_engine());

  using T1_1 = monolish::tensor::tensor_Dense<T>;
  using T1_2_ = monolish::vector<T>;
  using T1_2 = monolish::view_tensor_Dense<T1_2_, T>;
  using T1_3_ = monolish::matrix::Dense<T>;
  using T1_3 = monolish::view_tensor_Dense<T1_3_, T>;
  using T1_4_ = monolish::tensor::tensor_Dense<T>;
  using T1_4 = monolish::view_tensor_Dense<T1_4_, T>;
  T1_1 x1({M, N, L}, 0.0, 1.0);
  T1_2_ x2_(2 * M * N * L, 0.0, 1.0);
  T1_2 x2(x2_, M / 2, {M, N, L});
  T1_3_ x3_(2 * M * N * L, 1, 0.0, 1.0);
  T1_3 x3(x3_, M / 2, {M, N, L});
  T1_4_ x4_({2 * M * N * L, 1, 1}, 0.0, 1.0);
  T1_4 x4(x4_, M / 2, {M, N, L});

  auto As = std::make_tuple(x1, x2, x3, x4);

  using T2_1 = monolish::vector<T>;
  using T2_2 = monolish::view1D<T1_2_, T>;
  using T2_3 = monolish::view1D<T1_3_, T>;
  using T2_4 = monolish::view1D<T1_4_, T>;
  T2_1 y1(L, 0.0, 1.0);
  T1_2_ y2_(2 * L, 0.0, 1.0);
  T2_2 y2(y2_, L / 2, L);
  T1_3_ y3_(2 * L, 1, 0.0, 1.0);
  T2_3 y3(y3_, L / 2, L);
  T1_4_ y4_({2 * L, 1, 1}, 0.0, 1.0);
  T2_4 y4(y4_, L / 2, L);

  auto Bs = std::make_tuple(y1, y2, y3, y4);

  T1_1 z1({M, N, L}, 0.0, 1.0);
  T1_2_ z2_(2 * M * N * L, 0.0, 1.0);
  T1_2 z2(z2_, M / 2, {M, N, L});
  T1_3_ z3_(2 * M * N * L, 1, 0.0, 1.0);
  T1_3 z3(z3_, M / 2, {M, N, L});
  T1_4_ z4_({2 * M * N * L, 1, 1}, 0.0, 1.0);
  T1_4 z4(z4_, M / 2, {M, N, L});

  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_adds_row_line_view_core1(M, N, L, tol, AA, xx, CC, As, Bs, Cs);
}
