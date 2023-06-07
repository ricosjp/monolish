#include "../../test_utils.hpp"

template <typename T>
void ans_tensmat(monolish::tensor::tensor_Dense<T> &A,
                 monolish::matrix::Dense<T> &B,
                 monolish::tensor::tensor_Dense<T> &C) {

  if (A.get_shape()[2] != B.get_row()) {
    std::runtime_error("A.col != x.size");
  }
  if (A.get_shape()[0] != C.get_shape()[0] ||
      A.get_shape()[1] != C.get_shape()[1]) {
    std::runtime_error("A.row != C.row");
  }
  if (B.get_col() != C.get_shape()[2]) {
    std::runtime_error("B.col != C.col");
  }

  T *y = C.begin();
  int M = A.get_shape()[0] * A.get_shape()[1];
  int N = B.get_col();
  int K = A.get_shape()[2];

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
void ans_tensmat(const T &a, monolish::tensor::tensor_Dense<T> &A,
                 monolish::matrix::Dense<T> &B, const T &b,
                 monolish::tensor::tensor_Dense<T> &C) {

  if (A.get_shape()[2] != B.get_row()) {
    std::runtime_error("A.col != x.size");
  }
  if (A.get_shape()[0] != C.get_shape()[0] ||
      A.get_shape()[1] != C.get_shape()[1]) {
    std::runtime_error("A.row != C.row");
  }
  if (B.get_col() != C.get_shape()[2]) {
    std::runtime_error("B.col != C.col");
  }

  T *y = C.begin();
  int M = A.get_shape()[0] * A.get_shape()[1];
  int N = B.get_col();
  int K = A.get_shape()[2];

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

template <typename T, typename TENS_A, typename MAT, typename TENS_C>
bool test_send_tensmat1_core(const size_t M, const size_t N, const size_t L,
                             const size_t K, double tol,
                             monolish::tensor::tensor_Dense<T> &AA,
                             monolish::matrix::Dense<T> &BB,
                             monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                             MAT &B, TENS_C &C) {
  A = AA;
  B = BB;
  C = CC;

  ans_tensmat(AA, BB, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::tensmat(A, B, C);
  monolish::util::recv(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename T, typename TENS_A, typename MAT, typename TENS_C>
bool test_send_tensmat2_core(const size_t M, const size_t N, const size_t L,
                             const size_t K, const T a, const T b, double tol,
                             monolish::tensor::tensor_Dense<T> &AA,
                             monolish::matrix::Dense<T> &BB,
                             monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                             MAT &B, TENS_C &C) {
  A = AA;
  B = BB;
  C = CC;

  ans_tensmat(a, AA, BB, b, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::tensmat(a, A, B, b, C);
  monolish::util::recv(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename T, typename TENS_A, typename MAT, typename TENS_C>
bool test_send_tensmat_core(const size_t M, const size_t N, const size_t L,
                            const size_t K, const T a, const T b, double tol,
                            monolish::tensor::tensor_Dense<T> &AA,
                            monolish::matrix::Dense<T> &BB,
                            monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                            MAT &B, TENS_C &C) {
  return test_send_tensmat1_core(M, N, L, K, tol, AA, BB, CC, A, B, C) &&
         test_send_tensmat2_core(M, N, L, K, a, b, tol, AA, BB, CC, A, B, C);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_send_tensmat(const size_t M, const size_t N, const size_t L,
                       const size_t K, double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);
  monolish::matrix::COO<T> seedB = get_random_structure_matrix<T>(L, K);
  monolish::tensor::tensor_COO<T> seedC =
      get_random_structure_tensor<T>(M, N, K);

  const T a = 3;
  const T b = 2;

  MAT_A A(seedA); // M*N tensor
  MAT_B B(seedB);
  MAT_C C(seedC); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  return test_send_tensmat_core(M, N, L, K, a, b, tol, AA, BB, CC, A, B, C);
}

template <typename T, typename TENS_A, typename MAT, std::size_t KK = 0,
          typename... Tp>
inline typename std::enable_if<KK == sizeof...(Tp), bool>::type
test_send_tensmat_view_core3(const size_t M, const size_t N, const size_t L,
                             const size_t K, double tol,
                             monolish::tensor::tensor_Dense<T> &AA,
                             monolish::matrix::Dense<T> &BB,
                             monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                             MAT &B, std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, typename TENS_A, typename MAT, std::size_t KK = 0,
          typename... Tp>
    inline typename std::enable_if <
    KK<sizeof...(Tp), bool>::type test_send_tensmat_view_core3(
        const size_t M, const size_t N, const size_t L, const size_t K,
        double tol, monolish::tensor::tensor_Dense<T> &AA,
        monolish::matrix::Dense<T> &BB, monolish::tensor::tensor_Dense<T> &CC,
        TENS_A &A, MAT &B, std::tuple<Tp...> &Cs) {
  const T a = 3;
  const T b = 2;
  if (!test_send_tensmat_core(M, N, L, K, a, b, tol, AA, BB, CC, A, B,
                              std::get<KK>(Cs))) {
    return false;
  }
  return test_send_tensmat_view_core3<T, TENS_A, MAT, KK + 1, Tp...>(
      M, N, L, K, tol, AA, BB, CC, A, B, Cs);
}

template <typename T, typename TENS_A, std::size_t J = 0, typename TENSES_C,
          typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_send_tensmat_view_core2(const size_t M, const size_t N, const size_t L,
                             const size_t K, double tol,
                             monolish::tensor::tensor_Dense<T> &AA,
                             monolish::matrix::Dense<T> &BB,
                             monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                             std::tuple<Tq...> &Bs, TENSES_C &Cs) {
  return true;
}

template <typename T, typename TENS_A, std::size_t J = 0, typename TENSES_C,
          typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type test_send_tensmat_view_core2(
        const size_t M, const size_t N, const size_t L, const size_t K,
        double tol, monolish::tensor::tensor_Dense<T> &AA,
        monolish::matrix::Dense<T> &BB, monolish::tensor::tensor_Dense<T> &CC,
        TENS_A &A, std::tuple<Tq...> &Bs, TENSES_C &Cs) {
  if (!test_send_tensmat_view_core3(M, N, L, K, tol, AA, BB, CC, A,
                                    std::get<J>(Bs), Cs)) {
    return false;
  }
  return test_send_tensmat_view_core2<T, TENS_A, J + 1, TENSES_C, Tq...>(
      M, N, L, K, tol, AA, BB, CC, A, Bs, Cs);
}

template <typename T, std::size_t I = 0, typename MATS, typename TENSES_C,
          typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_send_tensmat_view_core1(const size_t M, const size_t N, const size_t L,
                             const size_t K, double tol,
                             monolish::tensor::tensor_Dense<T> &AA,
                             monolish::matrix::Dense<T> &BB,
                             monolish::tensor::tensor_Dense<T> &CC,
                             std::tuple<Tp...> &As, MATS &Bs, TENSES_C &Cs) {
  return true;
}

template <typename T, std::size_t I = 0, typename MATS, typename TENSES_C,
          typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type test_send_tensmat_view_core1(
        const size_t M, const size_t N, const size_t L, const size_t K,
        double tol, monolish::tensor::tensor_Dense<T> &AA,
        monolish::matrix::Dense<T> &BB, monolish::tensor::tensor_Dense<T> &CC,
        std::tuple<Tp...> &As, MATS &Bs, TENSES_C &Cs) {
  if (!test_send_tensmat_view_core2(M, N, L, K, tol, AA, BB, CC,
                                    std::get<I>(As), Bs, Cs)) {
    return false;
  }
  return test_send_tensmat_view_core1<T, I + 1, MATS, TENSES_C, Tp...>(
      M, N, L, K, tol, AA, BB, CC, As, Bs, Cs);
}

template <typename T>
bool test_send_tensmat_view(const size_t M, const size_t N, const size_t L,
                            const size_t K, double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);
  monolish::matrix::COO<T> seedB = get_random_structure_matrix<T>(L, K);
  monolish::tensor::tensor_COO<T> seedC =
      get_random_structure_tensor<T>(M, N, K);
  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  using T1_1 = monolish::tensor::tensor_Dense<T>;
  using T1_2_ = monolish::vector<T>;
  using T1_2 = monolish::view_tensor_Dense<T1_2_, T>;
  using T1_3_ = monolish::matrix::Dense<T>;
  using T1_3 = monolish::view_tensor_Dense<T1_3_, T>;
  using T1_4_ = monolish::tensor::tensor_Dense<T>;
  using T1_4 = monolish::view_tensor_Dense<T1_4_, T>;
  T1_1 y1({M, N, L}, 0.0, 1.0);
  T1_2_ y2_(2 * M * N * L, 0.0, 1.0);
  T1_2 y2(y2_, L / 2, {M, N, L});
  T1_3_ y3_(2 * M * N * L, 1, 0.0, 1.0);
  T1_3 y3(y3_, L / 2, {M, N, L});
  T1_4_ y4_({2 * M * N * L, 1, 1}, 0.0, 1.0);
  T1_4 y4(y4_, L / 2, {M, N, L});

  auto As = std::make_tuple(y1, y2, y3, y4);

  using T2_1 = monolish::matrix::Dense<T>;
  using T2_2_ = monolish::vector<T>;
  using T2_2 = monolish::view_Dense<T2_2_, T>;
  using T2_3_ = monolish::matrix::Dense<T>;
  using T2_3 = monolish::view_Dense<T2_3_, T>;
  using T2_4_ = monolish::tensor::tensor_Dense<T>;
  using T2_4 = monolish::view_Dense<T2_4_, T>;
  T2_1 x1(L, K, 0.0, 1.0);
  T2_2_ x2_(2 * L * K, 0.0, 1.0);
  T2_2 x2(x2_, K / 2, L, K);
  T2_3_ x3_(2 * L * K, 1, 0.0, 1.0);
  T2_3 x3(x3_, K / 2, L, K);
  T2_4_ x4_({2 * L * K, 1, 1}, 0.0, 1.0);
  T2_4 x4(x4_, K / 2, L, K);

  auto Bs = std::make_tuple(x1, x2, x3, x4);

  T1_1 z1({M, N, K}, 0.0, 1.0);
  T1_2_ z2_(2 * M * N * K, 0.0, 1.0);
  T1_2 z2(z2_, K / 2, {M, N, K});
  T1_3_ z3_(2 * M * N * K, 1, 0.0, 1.0);
  T1_3 z3(z3_, K / 2, {M, N, K});
  T1_4_ z4_({2 * M * N * K, 1, 1}, 0.0, 1.0);
  T1_4 z4(z4_, K / 2, {M, N, K});

  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_send_tensmat_view_core1(M, N, L, K, tol, AA, BB, CC, As, Bs, Cs);
}

template <typename T, typename TENS_A, typename MAT, typename TENS_C>
bool test_tensmat1_core(const size_t M, const size_t N, const size_t L,
                        const size_t K, double tol,
                        monolish::tensor::tensor_Dense<T> &AA,
                        monolish::matrix::Dense<T> &BB,
                        monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                        MAT &B, TENS_C &C) {
  A = AA;
  B = BB;
  C = CC;

  ans_tensmat(AA, BB, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::blas::tensmat(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename T, typename TENS_A, typename MAT, typename TENS_C>
bool test_tensmat2_core(const size_t M, const size_t N, const size_t L,
                        const size_t K, const T a, const T b, double tol,
                        monolish::tensor::tensor_Dense<T> &AA,
                        monolish::matrix::Dense<T> &BB,
                        monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                        MAT &B, TENS_C &C) {
  A = AA;
  B = BB;
  C = CC;

  ans_tensmat(a, AA, BB, b, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::tensmat(a, A, B, b, C);
  monolish::util::recv(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename T, typename TENS_A, typename MAT, typename TENS_C>
bool test_tensmat_core(const size_t M, const size_t N, const size_t L,
                       const size_t K, const T a, const T b, double tol,
                       monolish::tensor::tensor_Dense<T> &AA,
                       monolish::matrix::Dense<T> &BB,
                       monolish::tensor::tensor_Dense<T> &CC, TENS_A &A, MAT &B,
                       TENS_C &C) {
  return test_tensmat1_core(M, N, L, K, tol, AA, BB, CC, A, B, C) &&
         test_tensmat2_core(M, N, L, K, a, b, tol, AA, BB, CC, A, B, C);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_tensmat(const size_t M, const size_t N, const size_t L,
                  const size_t K, double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);
  monolish::matrix::COO<T> seedB = get_random_structure_matrix<T>(L, K);
  monolish::tensor::tensor_COO<T> seedC =
      get_random_structure_tensor<T>(M, N, K);

  const T a = 3;
  const T b = 2;

  MAT_A A(seedA); // M*N tensor
  MAT_B B(seedB);
  MAT_C C(seedC); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  return test_tensmat_core(M, N, L, K, a, b, tol, AA, BB, CC, A, B, C);
}

template <typename T, typename TENS_A, typename MAT, std::size_t KK = 0,
          typename... Tp>
inline typename std::enable_if<KK == sizeof...(Tp), bool>::type
test_tensmat_view_core3(const size_t M, const size_t N, const size_t L,
                        const size_t K, double tol,
                        monolish::tensor::tensor_Dense<T> &AA,
                        monolish::matrix::Dense<T> &BB,
                        monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                        MAT &B, std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, typename TENS_A, typename MAT, std::size_t KK = 0,
          typename... Tp>
    inline typename std::enable_if <
    KK<sizeof...(Tp), bool>::type test_tensmat_view_core3(
        const size_t M, const size_t N, const size_t L, const size_t K,
        double tol, monolish::tensor::tensor_Dense<T> &AA,
        monolish::matrix::Dense<T> &BB, monolish::tensor::tensor_Dense<T> &CC,
        TENS_A &A, MAT &B, std::tuple<Tp...> &Cs) {
  const T a = 3;
  const T b = 2;
  if (!test_tensmat_core(M, N, L, K, a, b, tol, AA, BB, CC, A, B,
                         std::get<KK>(Cs))) {
    return false;
  }
  return test_tensmat_view_core3<T, TENS_A, MAT, KK + 1, Tp...>(
      M, N, L, K, tol, AA, BB, CC, A, B, Cs);
}

template <typename T, typename TENS_A, std::size_t J = 0, typename TENSES_C,
          typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_tensmat_view_core2(const size_t M, const size_t N, const size_t L,
                        const size_t K, double tol,
                        monolish::tensor::tensor_Dense<T> &AA,
                        monolish::matrix::Dense<T> &BB,
                        monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                        std::tuple<Tq...> &Bs, TENSES_C &Cs) {
  return true;
}

template <typename T, typename TENS_A, std::size_t J = 0, typename TENSES_C,
          typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type test_tensmat_view_core2(
        const size_t M, const size_t N, const size_t L, const size_t K,
        double tol, monolish::tensor::tensor_Dense<T> &AA,
        monolish::matrix::Dense<T> &BB, monolish::tensor::tensor_Dense<T> &CC,
        TENS_A &A, std::tuple<Tq...> &Bs, TENSES_C &Cs) {
  if (!test_tensmat_view_core3(M, N, L, K, tol, AA, BB, CC, A, std::get<J>(Bs),
                               Cs)) {
    return false;
  }
  return test_tensmat_view_core2<T, TENS_A, J + 1, TENSES_C, Tq...>(
      M, N, L, K, tol, AA, BB, CC, A, Bs, Cs);
}

template <typename T, std::size_t I = 0, typename MATS, typename TENSES_C,
          typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_tensmat_view_core1(const size_t M, const size_t N, const size_t L,
                        const size_t K, double tol,
                        monolish::tensor::tensor_Dense<T> &AA,
                        monolish::matrix::Dense<T> &BB,
                        monolish::tensor::tensor_Dense<T> &CC,
                        std::tuple<Tp...> &As, MATS &Bs, TENSES_C &Cs) {
  return true;
}

template <typename T, std::size_t I = 0, typename MATS, typename TENSES_C,
          typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type test_tensmat_view_core1(
        const size_t M, const size_t N, const size_t L, const size_t K,
        double tol, monolish::tensor::tensor_Dense<T> &AA,
        monolish::matrix::Dense<T> &BB, monolish::tensor::tensor_Dense<T> &CC,
        std::tuple<Tp...> &As, MATS &Bs, TENSES_C &Cs) {
  if (!test_tensmat_view_core2(M, N, L, K, tol, AA, BB, CC, std::get<I>(As), Bs,
                               Cs)) {
    return false;
  }
  return test_tensmat_view_core1<T, I + 1, MATS, TENSES_C, Tp...>(
      M, N, L, K, tol, AA, BB, CC, As, Bs, Cs);
}

template <typename T>
bool test_tensmat_view(const size_t M, const size_t N, const size_t L,
                       const size_t K, double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);
  monolish::matrix::COO<T> seedB = get_random_structure_matrix<T>(L, K);
  monolish::tensor::tensor_COO<T> seedC =
      get_random_structure_tensor<T>(M, N, K);
  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  using T1_1 = monolish::tensor::tensor_Dense<T>;
  using T1_2_ = monolish::vector<T>;
  using T1_2 = monolish::view_tensor_Dense<T1_2_, T>;
  using T1_3_ = monolish::matrix::Dense<T>;
  using T1_3 = monolish::view_tensor_Dense<T1_3_, T>;
  using T1_4_ = monolish::tensor::tensor_Dense<T>;
  using T1_4 = monolish::view_tensor_Dense<T1_4_, T>;
  T1_1 y1({M, N, L}, 0.0, 1.0);
  T1_2_ y2_(2 * M * N * L, 0.0, 1.0);
  T1_2 y2(y2_, L / 2, {M, N, L});
  T1_3_ y3_(2 * M * N * L, 1, 0.0, 1.0);
  T1_3 y3(y3_, L / 2, {M, N, L});
  T1_4_ y4_({2 * M * N * L, 1, 1}, 0.0, 1.0);
  T1_4 y4(y4_, L / 2, {M, N, L});

  auto As = std::make_tuple(y1, y2, y3, y4);

  using T2_1 = monolish::matrix::Dense<T>;
  using T2_2_ = monolish::vector<T>;
  using T2_2 = monolish::view_Dense<T2_2_, T>;
  using T2_3_ = monolish::matrix::Dense<T>;
  using T2_3 = monolish::view_Dense<T2_3_, T>;
  using T2_4_ = monolish::tensor::tensor_Dense<T>;
  using T2_4 = monolish::view_Dense<T2_4_, T>;
  T2_1 x1(L, K, 0.0, 1.0);
  T2_2_ x2_(2 * L * K, 0.0, 1.0);
  T2_2 x2(x2_, K / 2, L, K);
  T2_3_ x3_(2 * L * K, 1, 0.0, 1.0);
  T2_3 x3(x3_, K / 2, L, K);
  T2_4_ x4_({2 * L * K, 1, 1}, 0.0, 1.0);
  T2_4 x4(x4_, K / 2, L, K);

  auto Bs = std::make_tuple(x1, x2, x3, x4);

  T1_1 z1({M, N, K}, 0.0, 1.0);
  T1_2_ z2_(2 * M * N * K, 0.0, 1.0);
  T1_2 z2(z2_, K / 2, {M, N, K});
  T1_3_ z3_(2 * M * N * K, 1, 0.0, 1.0);
  T1_3 z3(z3_, K / 2, {M, N, K});
  T1_4_ z4_({2 * M * N * K, 1, 1}, 0.0, 1.0);
  T1_4 z4(z4_, K / 2, {M, N, K});

  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_tensmat_view_core1(M, N, L, K, tol, AA, BB, CC, As, Bs, Cs);
}
