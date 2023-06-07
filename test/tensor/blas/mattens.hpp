#include "../../test_utils.hpp"

template <typename T>
void ans_mattens(monolish::matrix::Dense<T> &A,
                 monolish::tensor::tensor_Dense<T> &B,
                 monolish::tensor::tensor_Dense<T> &C) {

  if (A.get_col() != B.get_shape()[0]) {
    std::runtime_error("A.col != x.size");
  }
  if (A.get_row() != C.get_shape()[0]) {
    std::runtime_error("A.row != C.row");
  }
  if (B.get_shape()[1] != C.get_shape()[1] ||
      B.get_shape()[2] != C.get_shape()[2]) {
    std::runtime_error("B.col != C.col");
  }

  T *y = C.begin();
  int M = A.get_row();
  int N = B.get_shape()[1] * B.get_shape()[2];
  int K = A.get_col();

  for (int i = 0; i < C.get_nnz(); i++)
    y[i] = 0;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        y[i * N + j] += A.begin()[i * K + k] * B.begin()[k * N + j];
      }
    }
  }
}

template <typename T>
void ans_mattens(const T &a, monolish::matrix::Dense<T> &A,
                 monolish::tensor::tensor_Dense<T> &B, const T &b,
                 monolish::tensor::tensor_Dense<T> &C) {

  if (A.get_col() != B.get_shape()[0]) {
    std::runtime_error("A.col != x.size");
  }
  if (A.get_row() != C.get_shape()[0]) {
    std::runtime_error("A.row != C.row");
  }
  if (B.get_shape()[1] != C.get_shape()[1] ||
      B.get_shape()[2] != C.get_shape()[2]) {
    std::runtime_error("B.col != C.col");
  }

  T *y = C.begin();
  int M = A.get_row();
  int N = B.get_shape()[1] * B.get_shape()[2];
  int K = A.get_col();

  for (int i = 0; i < C.get_nnz(); i++)
    y[i] = b * y[i];

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        y[i * N + j] += a * A.begin()[i * K + k] * B.begin()[k * N + j];
      }
    }
  }
}

template <typename T, typename MAT, typename TENS_B, typename TENS_C>
bool test_send_mattens1_core(const size_t M, const size_t N, const size_t L,
                             const size_t K, double tol,
                             monolish::matrix::Dense<T> &AA,
                             monolish::tensor::tensor_Dense<T> &BB,
                             monolish::tensor::tensor_Dense<T> &CC, MAT &A,
                             TENS_B &B, TENS_C &C) {
  B = BB;
  C = CC;

  ans_mattens(AA, BB, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::mattens(A, B, C);
  monolish::util::recv(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename T, typename MAT, typename TENS_B, typename TENS_C>
bool test_send_mattens2_core(const size_t M, const size_t N, const size_t L,
                             const size_t K, const T a, const T b, double tol,
                             monolish::matrix::Dense<T> &AA,
                             monolish::tensor::tensor_Dense<T> &BB,
                             monolish::tensor::tensor_Dense<T> &CC, MAT &A,
                             TENS_B &B, TENS_C &C) {
  B = BB;
  C = CC;

  ans_mattens(a, AA, BB, b, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::mattens(a, A, B, b, C);
  monolish::util::recv(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename T, typename MAT, typename TENS_B, typename TENS_C>
bool test_send_mattens_core(const size_t M, const size_t N, const size_t L,
                            const size_t K, const T a, const T b, double tol,
                            monolish::matrix::Dense<T> &AA,
                            monolish::tensor::tensor_Dense<T> &BB,
                            monolish::tensor::tensor_Dense<T> &CC, MAT &A,
                            TENS_B &B, TENS_C &C) {
  return test_send_mattens1_core(M, N, L, K, tol, AA, BB, CC, A, B, C) &&
         test_send_mattens2_core(M, N, L, K, a, b, tol, AA, BB, CC, A, B, C);
}

template <typename MAT_A, typename TENS_B, typename TENS_C, typename T>
bool test_send_mattens(const size_t M, const size_t N, const size_t L,
                       const size_t K, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, L);
  monolish::tensor::tensor_COO<T> seedB =
      get_random_structure_tensor<T>(L, N, K);
  monolish::tensor::tensor_COO<T> seedC =
      get_random_structure_tensor<T>(M, N, K);

  const T a = 3;
  const T b = 2;

  MAT_A A(seedA); // M*N tensor
  TENS_B B(seedB);
  TENS_C C(seedC); // M*N tensor

  monolish::matrix::Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  return test_send_mattens_core(M, N, L, K, a, b, tol, AA, BB, CC, A, B, C);
}

template <typename T, typename MAT, typename TENS_B, std::size_t KK = 0,
          typename... Tq>
inline typename std::enable_if<KK == sizeof...(Tq), bool>::type
test_send_mattens_view_core3_dense(const size_t M, const size_t N,
                                   const size_t L, const size_t K, double tol,
                                   monolish::matrix::Dense<T> &AA,
                                   monolish::tensor::tensor_Dense<T> &BB,
                                   monolish::tensor::tensor_Dense<T> &CC,
                                   MAT &A, TENS_B &B, std::tuple<Tq...> &Cs) {
  return true;
}

template <typename T, typename MAT, typename TENS_B, std::size_t KK = 0,
          typename... Tq>
    inline typename std::enable_if <
    KK<sizeof...(Tq), bool>::type test_send_mattens_view_core3_dense(
        const size_t M, const size_t N, const size_t L, const size_t K,
        double tol, monolish::matrix::Dense<T> &AA,
        monolish::tensor::tensor_Dense<T> &BB,
        monolish::tensor::tensor_Dense<T> &CC, MAT &A, TENS_B &B,
        std::tuple<Tq...> &Cs) {
  A = AA;
  const T a = 3;
  const T b = 2;
  if (!test_send_mattens_core(M, N, L, K, a, b, tol, AA, BB, CC, A, B,
                              std::get<KK>(Cs))) {
    return false;
  }
  return test_send_mattens_view_core3_dense<T, MAT, TENS_B, KK + 1, Tq...>(
      M, N, L, K, tol, AA, BB, CC, A, B, Cs);
}

template <typename T, typename MAT, std::size_t J = 0, typename TENSES_C,
          typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_send_mattens_view_core2_dense(const size_t M, const size_t N,
                                   const size_t L, const size_t K, double tol,
                                   monolish::matrix::Dense<T> &AA,
                                   monolish::tensor::tensor_Dense<T> &BB,
                                   monolish::tensor::tensor_Dense<T> &CC,
                                   MAT &A, std::tuple<Tq...> &B, TENSES_C &Cs) {
  return true;
}

template <typename T, typename MAT, std::size_t J = 0, typename TENSES_C,
          typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type test_send_mattens_view_core2_dense(
        const size_t M, const size_t N, const size_t L, const size_t K,
        double tol, monolish::matrix::Dense<T> &AA,
        monolish::tensor::tensor_Dense<T> &BB,
        monolish::tensor::tensor_Dense<T> &CC, MAT &A, std::tuple<Tq...> &Bs,
        TENSES_C &Cs) {
  if (!test_send_mattens_view_core3_dense(M, N, L, K, tol, AA, BB, CC, A,
                                          std::get<J>(Bs), Cs)) {
    return false;
  }
  return test_send_mattens_view_core2_dense<T, MAT, J + 1, TENSES_C, Tq...>(
      M, N, L, K, tol, AA, BB, CC, A, Bs, Cs);
}

template <typename T, std::size_t I = 0, typename TENSES_B, typename TENSES_C,
          typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_send_mattens_view_core1_dense(const size_t M, const size_t N,
                                   const size_t L, const size_t K, double tol,
                                   monolish::matrix::Dense<T> &AA,
                                   monolish::tensor::tensor_Dense<T> &BB,
                                   monolish::tensor::tensor_Dense<T> &CC,
                                   std::tuple<Tp...> &As, TENSES_B &B,
                                   TENSES_C &Cs) {
  return true;
}

template <typename T, std::size_t I = 0, typename TENSES_B, typename TENSES_C,
          typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type test_send_mattens_view_core1_dense(
        const size_t M, const size_t N, const size_t L, const size_t K,
        double tol, monolish::matrix::Dense<T> &AA,
        monolish::tensor::tensor_Dense<T> &BB,
        monolish::tensor::tensor_Dense<T> &CC, std::tuple<Tp...> &As,
        TENSES_B &Bs, TENSES_C &Cs) {
  if (!test_send_mattens_view_core2_dense(M, N, L, K, tol, AA, BB, CC,
                                          std::get<I>(As), Bs, Cs)) {
    return false;
  }
  return test_send_mattens_view_core1_dense<T, I + 1, TENSES_B, TENSES_C,
                                            Tp...>(M, N, L, K, tol, AA, BB, CC,
                                                   As, Bs, Cs);
}

template <typename T>
bool test_send_mattens_view_dense(const size_t M, const size_t N,
                                  const size_t L, const size_t K, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, L);
  monolish::tensor::tensor_COO<T> seedB =
      get_random_structure_tensor<T>(L, N, K);
  monolish::tensor::tensor_COO<T> seedC =
      get_random_structure_tensor<T>(M, N, K);
  monolish::matrix::Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  using T2_1 = monolish::matrix::Dense<T>;
  using T2_2_ = monolish::vector<T>;
  using T2_2 = monolish::view_Dense<T2_2_, T>;
  using T2_3_ = monolish::matrix::Dense<T>;
  using T2_3 = monolish::view_Dense<T2_3_, T>;
  using T2_4_ = monolish::tensor::tensor_Dense<T>;
  using T2_4 = monolish::view_Dense<T2_4_, T>;
  T2_1 x1(M, L, 0.0, 1.0);
  T2_2_ x2_(2 * M * L, 0.0, 1.0);
  T2_2 x2(x2_, L / 2, M, L);
  T2_3_ x3_(2 * M * L, 1, 0.0, 1.0);
  T2_3 x3(x3_, L / 2, M, L);
  T2_4_ x4_({2 * M * L, 1, 1}, 0.0, 1.0);
  T2_4 x4(x4_, L / 2, M, L);

  auto As = std::make_tuple(x1, x2, x3, x4);

  using T1_1 = monolish::tensor::tensor_Dense<T>;
  using T1_2_ = monolish::vector<T>;
  using T1_2 = monolish::view_tensor_Dense<T1_2_, T>;
  using T1_3_ = monolish::matrix::Dense<T>;
  using T1_3 = monolish::view_tensor_Dense<T1_3_, T>;
  using T1_4_ = monolish::tensor::tensor_Dense<T>;
  using T1_4 = monolish::view_tensor_Dense<T1_4_, T>;
  T1_1 y1({L, N, K}, 0.0, 1.0);
  T1_2_ y2_(2 * L * N * K, 0.0, 1.0);
  T1_2 y2(y2_, K / 2, {L, N, K});
  T1_3_ y3_(2 * L * N * K, 1, 0.0, 1.0);
  T1_3 y3(y3_, K / 2, {L, N, K});
  T1_4_ y4_({2 * L * N * K, 1, 1}, 0.0, 1.0);
  T1_4 y4(y4_, K / 2, {L, N, K});

  auto Bs = std::make_tuple(y1, y2, y3, y4);

  T1_1 z1({M, N, K}, 0.0, 1.0);
  T1_2_ z2_(2 * M * N * K, 0.0, 1.0);
  T1_2 z2(z2_, K / 2, {M, N, K});
  T1_3_ z3_(2 * M * N * K, 1, 0.0, 1.0);
  T1_3 z3(z3_, K / 2, {M, N, K});
  T1_4_ z4_({2 * M * N * K, 1, 1}, 0.0, 1.0);
  T1_4 z4(z4_, K / 2, {M, N, K});

  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_send_mattens_view_core1_dense(M, N, L, K, tol, AA, BB, CC, As, Bs,
                                            Cs);
}

template <typename T, typename TENS_B, std::size_t KK = 0, typename... Tq>
inline typename std::enable_if<KK == sizeof...(Tq), bool>::type
test_send_mattens_view_core2_crs(const size_t M, const size_t N, const size_t L,
                                 const size_t K, double tol,
                                 monolish::matrix::Dense<T> &AA,
                                 monolish::tensor::tensor_Dense<T> &BB,
                                 monolish::tensor::tensor_Dense<T> &CC,
                                 monolish::matrix::COO<T> &seedA, TENS_B &B,
                                 std::tuple<Tq...> &Cs) {
  return true;
}

template <typename T, typename TENS_B, std::size_t KK = 0, typename... Tq>
    inline typename std::enable_if <
    KK<sizeof...(Tq), bool>::type test_send_mattens_view_core2_crs(
        const size_t M, const size_t N, const size_t L, const size_t K,
        double tol, monolish::matrix::Dense<T> &AA,
        monolish::tensor::tensor_Dense<T> &BB,
        monolish::tensor::tensor_Dense<T> &CC, monolish::matrix::COO<T> &seedA,
        TENS_B &B, std::tuple<Tq...> &Cs) {
  monolish::matrix::CRS<T> A(seedA);
  const T a = 3;
  const T b = 2;
  if (!test_send_mattens_core(M, N, L, K, a, b, tol, AA, BB, CC, A, B,
                              std::get<KK>(Cs))) {
    return false;
  }
  return test_send_mattens_view_core2_crs<T, TENS_B, KK + 1, Tq...>(
      M, N, L, K, tol, AA, BB, CC, seedA, B, Cs);
}

template <typename T, std::size_t J = 0, typename TENSES_C, typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_send_mattens_view_core1_crs(const size_t M, const size_t N, const size_t L,
                                 const size_t K, double tol,
                                 monolish::matrix::Dense<T> &AA,
                                 monolish::tensor::tensor_Dense<T> &BB,
                                 monolish::tensor::tensor_Dense<T> &CC,
                                 monolish::matrix::COO<T> &seedA,
                                 std::tuple<Tq...> &B, TENSES_C &Cs) {
  return true;
}

template <typename T, std::size_t J = 0, typename TENSES_C, typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type test_send_mattens_view_core1_crs(
        const size_t M, const size_t N, const size_t L, const size_t K,
        double tol, monolish::matrix::Dense<T> &AA,
        monolish::tensor::tensor_Dense<T> &BB,
        monolish::tensor::tensor_Dense<T> &CC, monolish::matrix::COO<T> &seedA,
        std::tuple<Tq...> &Bs, TENSES_C &Cs) {
  if (!test_send_mattens_view_core2_crs(M, N, L, K, tol, AA, BB, CC, seedA,
                                        std::get<J>(Bs), Cs)) {
    return false;
  }
  return test_send_mattens_view_core1_crs<T, J + 1, TENSES_C, Tq...>(
      M, N, L, K, tol, AA, BB, CC, seedA, Bs, Cs);
}

template <typename T>
bool test_send_mattens_view_crs(const size_t M, const size_t N, const size_t L,
                                const size_t K, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, L);
  monolish::tensor::tensor_COO<T> seedB =
      get_random_structure_tensor<T>(L, N, K);
  monolish::tensor::tensor_COO<T> seedC =
      get_random_structure_tensor<T>(M, N, K);
  monolish::matrix::Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  using T1_1 = monolish::tensor::tensor_Dense<T>;
  using T1_2_ = monolish::vector<T>;
  using T1_2 = monolish::view_tensor_Dense<T1_2_, T>;
  using T1_3_ = monolish::matrix::Dense<T>;
  using T1_3 = monolish::view_tensor_Dense<T1_3_, T>;
  using T1_4_ = monolish::tensor::tensor_Dense<T>;
  using T1_4 = monolish::view_tensor_Dense<T1_4_, T>;
  T1_1 y1({L, N, K}, 0.0, 1.0);
  T1_2_ y2_(2 * L * N * K, 0.0, 1.0);
  T1_2 y2(y2_, K / 2, {L, N, K});
  T1_3_ y3_(2 * L * N * K, 1, 0.0, 1.0);
  T1_3 y3(y3_, K / 2, {L, N, K});
  T1_4_ y4_({2 * L * N * K, 1, 1}, 0.0, 1.0);
  T1_4 y4(y4_, K / 2, {L, N, K});

  auto Bs = std::make_tuple(y1, y2, y3, y4);

  T1_1 z1({M, N, K}, 0.0, 1.0);
  T1_2_ z2_(2 * M * N * K, 0.0, 1.0);
  T1_2 z2(z2_, K / 2, {M, N, K});
  T1_3_ z3_(2 * M * N * K, 1, 0.0, 1.0);
  T1_3 z3(z3_, K / 2, {M, N, K});
  T1_4_ z4_({2 * M * N * K, 1, 1}, 0.0, 1.0);
  T1_4 z4(z4_, K / 2, {M, N, K});

  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_send_mattens_view_core1_crs(M, N, L, K, tol, AA, BB, CC, seedA,
                                          Bs, Cs);
}

template <typename T, typename MAT, typename TENS_B, typename TENS_C>
bool test_mattens1_core(const size_t M, const size_t N, const size_t L,
                        const size_t K, double tol,
                        monolish::matrix::Dense<T> &AA,
                        monolish::tensor::tensor_Dense<T> &BB,
                        monolish::tensor::tensor_Dense<T> &CC, MAT &A,
                        TENS_B &B, TENS_C &C) {
  B = BB;
  C = CC;

  ans_mattens(AA, BB, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::blas::mattens(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename T, typename MAT, typename TENS_B, typename TENS_C>
bool test_mattens2_core(const size_t M, const size_t N, const size_t L,
                        const size_t K, const T a, const T b, double tol,
                        monolish::matrix::Dense<T> &AA,
                        monolish::tensor::tensor_Dense<T> &BB,
                        monolish::tensor::tensor_Dense<T> &CC, MAT &A,
                        TENS_B &B, TENS_C &C) {
  B = BB;
  C = CC;

  ans_mattens(a, AA, BB, b, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::mattens(a, A, B, b, C);
  monolish::util::recv(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename T, typename MAT, typename TENS_B, typename TENS_C>
bool test_mattens_core(const size_t M, const size_t N, const size_t L,
                       const size_t K, const T a, const T b, double tol,
                       monolish::matrix::Dense<T> &AA,
                       monolish::tensor::tensor_Dense<T> &BB,
                       monolish::tensor::tensor_Dense<T> &CC, MAT &A, TENS_B &B,
                       TENS_C &C) {
  return test_mattens1_core(M, N, L, K, tol, AA, BB, CC, A, B, C) &&
         test_mattens2_core(M, N, L, K, a, b, tol, AA, BB, CC, A, B, C);
}

template <typename MAT_A, typename TENS_B, typename TENS_C, typename T>
bool test_mattens(const size_t M, const size_t N, const size_t L,
                  const size_t K, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, L);
  monolish::tensor::tensor_COO<T> seedB =
      get_random_structure_tensor<T>(L, N, K);
  monolish::tensor::tensor_COO<T> seedC =
      get_random_structure_tensor<T>(M, N, K);

  const T a = 3;
  const T b = 2;

  MAT_A A(seedA); // M*N tensor
  TENS_B B(seedB);
  TENS_C C(seedC); // M*N tensor

  monolish::matrix::Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  return test_mattens_core(M, N, L, K, a, b, tol, AA, BB, CC, A, B, C);
}

template <typename T, typename MAT, typename TENS_B, std::size_t KK = 0,
          typename... Tq>
inline typename std::enable_if<KK == sizeof...(Tq), bool>::type
test_mattens_view_core3_dense(const size_t M, const size_t N, const size_t L,
                              const size_t K, double tol,
                              monolish::matrix::Dense<T> &AA,
                              monolish::tensor::tensor_Dense<T> &BB,
                              monolish::tensor::tensor_Dense<T> &CC, MAT &A,
                              TENS_B &B, std::tuple<Tq...> &Cs) {
  return true;
}

template <typename T, typename MAT, typename TENS_B, std::size_t KK = 0,
          typename... Tq>
    inline typename std::enable_if <
    KK<sizeof...(Tq), bool>::type
    test_mattens_view_core3_dense(const size_t M, const size_t N,
                                  const size_t L, const size_t K, double tol,
                                  monolish::matrix::Dense<T> &AA,
                                  monolish::tensor::tensor_Dense<T> &BB,
                                  monolish::tensor::tensor_Dense<T> &CC, MAT &A,
                                  TENS_B &B, std::tuple<Tq...> &Cs) {
  A = AA;
  const T a = 3;
  const T b = 2;
  if (!test_mattens_core(M, N, L, K, a, b, tol, AA, BB, CC, A, B,
                         std::get<KK>(Cs))) {
    return false;
  }
  return test_mattens_view_core3_dense<T, MAT, TENS_B, KK + 1, Tq...>(
      M, N, L, K, tol, AA, BB, CC, A, B, Cs);
}

template <typename T, typename MAT, std::size_t J = 0, typename TENSES_C,
          typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_mattens_view_core2_dense(const size_t M, const size_t N, const size_t L,
                              const size_t K, double tol,
                              monolish::matrix::Dense<T> &AA,
                              monolish::tensor::tensor_Dense<T> &BB,
                              monolish::tensor::tensor_Dense<T> &CC, MAT &A,
                              std::tuple<Tq...> &B, TENSES_C &Cs) {
  return true;
}

template <typename T, typename MAT, std::size_t J = 0, typename TENSES_C,
          typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type
    test_mattens_view_core2_dense(const size_t M, const size_t N,
                                  const size_t L, const size_t K, double tol,
                                  monolish::matrix::Dense<T> &AA,
                                  monolish::tensor::tensor_Dense<T> &BB,
                                  monolish::tensor::tensor_Dense<T> &CC, MAT &A,
                                  std::tuple<Tq...> &Bs, TENSES_C &Cs) {
  if (!test_mattens_view_core3_dense(M, N, L, K, tol, AA, BB, CC, A,
                                     std::get<J>(Bs), Cs)) {
    return false;
  }
  return test_mattens_view_core2_dense<T, MAT, J + 1, TENSES_C, Tq...>(
      M, N, L, K, tol, AA, BB, CC, A, Bs, Cs);
}

template <typename T, std::size_t I = 0, typename TENSES_B, typename TENSES_C,
          typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_mattens_view_core1_dense(const size_t M, const size_t N, const size_t L,
                              const size_t K, double tol,
                              monolish::matrix::Dense<T> &AA,
                              monolish::tensor::tensor_Dense<T> &BB,
                              monolish::tensor::tensor_Dense<T> &CC,
                              std::tuple<Tp...> &As, TENSES_B &B,
                              TENSES_C &Cs) {
  return true;
}

template <typename T, std::size_t I = 0, typename TENSES_B, typename TENSES_C,
          typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type test_mattens_view_core1_dense(
        const size_t M, const size_t N, const size_t L, const size_t K,
        double tol, monolish::matrix::Dense<T> &AA,
        monolish::tensor::tensor_Dense<T> &BB,
        monolish::tensor::tensor_Dense<T> &CC, std::tuple<Tp...> &As,
        TENSES_B &Bs, TENSES_C &Cs) {
  if (!test_mattens_view_core2_dense(M, N, L, K, tol, AA, BB, CC,
                                     std::get<I>(As), Bs, Cs)) {
    return false;
  }
  return test_mattens_view_core1_dense<T, I + 1, TENSES_B, TENSES_C, Tp...>(
      M, N, L, K, tol, AA, BB, CC, As, Bs, Cs);
}

template <typename T>
bool test_mattens_view_dense(const size_t M, const size_t N, const size_t L,
                             const size_t K, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, L);
  monolish::tensor::tensor_COO<T> seedB =
      get_random_structure_tensor<T>(L, N, K);
  monolish::tensor::tensor_COO<T> seedC =
      get_random_structure_tensor<T>(M, N, K);
  monolish::matrix::Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  using T2_1 = monolish::matrix::Dense<T>;
  using T2_2_ = monolish::vector<T>;
  using T2_2 = monolish::view_Dense<T2_2_, T>;
  using T2_3_ = monolish::matrix::Dense<T>;
  using T2_3 = monolish::view_Dense<T2_3_, T>;
  using T2_4_ = monolish::tensor::tensor_Dense<T>;
  using T2_4 = monolish::view_Dense<T2_4_, T>;
  T2_1 x1(M, L, 0.0, 1.0);
  T2_2_ x2_(2 * M * L, 0.0, 1.0);
  T2_2 x2(x2_, L / 2, M, L);
  T2_3_ x3_(2 * M * L, 1, 0.0, 1.0);
  T2_3 x3(x3_, L / 2, M, L);
  T2_4_ x4_({2 * M * L, 1, 1}, 0.0, 1.0);
  T2_4 x4(x4_, L / 2, M, L);

  auto As = std::make_tuple(x1, x2, x3, x4);

  using T1_1 = monolish::tensor::tensor_Dense<T>;
  using T1_2_ = monolish::vector<T>;
  using T1_2 = monolish::view_tensor_Dense<T1_2_, T>;
  using T1_3_ = monolish::matrix::Dense<T>;
  using T1_3 = monolish::view_tensor_Dense<T1_3_, T>;
  using T1_4_ = monolish::tensor::tensor_Dense<T>;
  using T1_4 = monolish::view_tensor_Dense<T1_4_, T>;
  T1_1 y1({L, N, K}, 0.0, 1.0);
  T1_2_ y2_(2 * L * N * K, 0.0, 1.0);
  T1_2 y2(y2_, K / 2, {L, N, K});
  T1_3_ y3_(2 * L * N * K, 1, 0.0, 1.0);
  T1_3 y3(y3_, K / 2, {L, N, K});
  T1_4_ y4_({2 * L * N * K, 1, 1}, 0.0, 1.0);
  T1_4 y4(y4_, K / 2, {L, N, K});

  auto Bs = std::make_tuple(y1, y2, y3, y4);

  T1_1 z1({M, N, K}, 0.0, 1.0);
  T1_2_ z2_(2 * M * N * K, 0.0, 1.0);
  T1_2 z2(z2_, K / 2, {M, N, K});
  T1_3_ z3_(2 * M * N * K, 1, 0.0, 1.0);
  T1_3 z3(z3_, K / 2, {M, N, K});
  T1_4_ z4_({2 * M * N * K, 1, 1}, 0.0, 1.0);
  T1_4 z4(z4_, K / 2, {M, N, K});

  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_mattens_view_core1_dense(M, N, L, K, tol, AA, BB, CC, As, Bs, Cs);
}

template <typename T, typename TENS_B, std::size_t KK = 0, typename... Tq>
inline typename std::enable_if<KK == sizeof...(Tq), bool>::type
test_mattens_view_core2_crs(const size_t M, const size_t N, const size_t L,
                            const size_t K, double tol,
                            monolish::matrix::Dense<T> &AA,
                            monolish::tensor::tensor_Dense<T> &BB,
                            monolish::tensor::tensor_Dense<T> &CC,
                            monolish::matrix::COO<T> &seedA, TENS_B &B,
                            std::tuple<Tq...> &Cs) {
  return true;
}

template <typename T, typename TENS_B, std::size_t KK = 0, typename... Tq>
    inline typename std::enable_if <
    KK<sizeof...(Tq), bool>::type test_mattens_view_core2_crs(
        const size_t M, const size_t N, const size_t L, const size_t K,
        double tol, monolish::matrix::Dense<T> &AA,
        monolish::tensor::tensor_Dense<T> &BB,
        monolish::tensor::tensor_Dense<T> &CC, monolish::matrix::COO<T> &seedA,
        TENS_B &B, std::tuple<Tq...> &Cs) {
  monolish::matrix::CRS<T> A(seedA);
  const T a = 3;
  const T b = 2;
  if (!test_mattens_core(M, N, L, K, a, b, tol, AA, BB, CC, A, B,
                         std::get<KK>(Cs))) {
    return false;
  }
  return test_mattens_view_core2_crs<T, TENS_B, KK + 1, Tq...>(
      M, N, L, K, tol, AA, BB, CC, seedA, B, Cs);
}

template <typename T, std::size_t J = 0, typename TENSES_C, typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_mattens_view_core1_crs(const size_t M, const size_t N, const size_t L,
                            const size_t K, double tol,
                            monolish::matrix::Dense<T> &AA,
                            monolish::tensor::tensor_Dense<T> &BB,
                            monolish::tensor::tensor_Dense<T> &CC,
                            monolish::matrix::COO<T> &seedA,
                            std::tuple<Tq...> &B, TENSES_C &Cs) {
  return true;
}

template <typename T, std::size_t J = 0, typename TENSES_C, typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type test_mattens_view_core1_crs(
        const size_t M, const size_t N, const size_t L, const size_t K,
        double tol, monolish::matrix::Dense<T> &AA,
        monolish::tensor::tensor_Dense<T> &BB,
        monolish::tensor::tensor_Dense<T> &CC, monolish::matrix::COO<T> &seedA,
        std::tuple<Tq...> &Bs, TENSES_C &Cs) {
  if (!test_mattens_view_core2_crs(M, N, L, K, tol, AA, BB, CC, seedA,
                                   std::get<J>(Bs), Cs)) {
    return false;
  }
  return test_mattens_view_core1_crs<T, J + 1, TENSES_C, Tq...>(
      M, N, L, K, tol, AA, BB, CC, seedA, Bs, Cs);
}

template <typename T>
bool test_mattens_view_crs(const size_t M, const size_t N, const size_t L,
                           const size_t K, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, L);
  monolish::tensor::tensor_COO<T> seedB =
      get_random_structure_tensor<T>(L, N, K);
  monolish::tensor::tensor_COO<T> seedC =
      get_random_structure_tensor<T>(M, N, K);
  monolish::matrix::Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  using T1_1 = monolish::tensor::tensor_Dense<T>;
  using T1_2_ = monolish::vector<T>;
  using T1_2 = monolish::view_tensor_Dense<T1_2_, T>;
  using T1_3_ = monolish::matrix::Dense<T>;
  using T1_3 = monolish::view_tensor_Dense<T1_3_, T>;
  using T1_4_ = monolish::tensor::tensor_Dense<T>;
  using T1_4 = monolish::view_tensor_Dense<T1_4_, T>;
  T1_1 y1({L, N, K}, 0.0, 1.0);
  T1_2_ y2_(2 * L * N * K, 0.0, 1.0);
  T1_2 y2(y2_, K / 2, {L, N, K});
  T1_3_ y3_(2 * L * N * K, 1, 0.0, 1.0);
  T1_3 y3(y3_, K / 2, {L, N, K});
  T1_4_ y4_({2 * L * N * K, 1, 1}, 0.0, 1.0);
  T1_4 y4(y4_, K / 2, {L, N, K});

  auto Bs = std::make_tuple(y1, y2, y3, y4);

  T1_1 z1({M, N, K}, 0.0, 1.0);
  T1_2_ z2_(2 * M * N * K, 0.0, 1.0);
  T1_2 z2(z2_, K / 2, {M, N, K});
  T1_3_ z3_(2 * M * N * K, 1, 0.0, 1.0);
  T1_3 z3(z3_, K / 2, {M, N, K});
  T1_4_ z4_({2 * M * N * K, 1, 1}, 0.0, 1.0);
  T1_4 z4(z4_, K / 2, {M, N, K});

  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_mattens_view_core1_crs(M, N, L, K, tol, AA, BB, CC, seedA, Bs,
                                     Cs);
}
