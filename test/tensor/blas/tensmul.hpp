#include "../../test_utils.hpp"

template <typename T>
void ans_tensmul(const monolish::tensor::tensor_Dense<T> &A,
                 const monolish::tensor::tensor_Dense<T> &B,
                 monolish::tensor::tensor_Dense<T> &C) {

  if (A.get_shape()[2] != B.get_shape()[0]) {
    std::cout << A.get_shape()[2] << B.get_shape()[0] << std::endl;
    std::runtime_error("test: A.shape[2] != B.shape[0]");
  }
  if (A.get_shape()[0] != C.get_shape()[0] ||
      A.get_shape()[1] != C.get_shape()[1]) {
    std::runtime_error("test: A.row != C.row");
  }
  if (C.get_shape()[2] != B.get_shape()[1] ||
      C.get_shape()[3] != B.get_shape()[2]) {
    std::runtime_error("test: C.col != B.col");
  }

  // MLJN=MLK*KJN
  int M = A.get_shape()[0];
  int L = A.get_shape()[1];
  int K = A.get_shape()[2];
  int J = B.get_shape()[1];
  int N = B.get_shape()[2];

  for (int i = 0; i < C.get_nnz(); i++) {
    C.begin()[i] = 0;
  }

  for (int ml = 0; ml < M * L; ml++) {
    for (int jn = 0; jn < J * N; jn++) {
      for (int k = 0; k < K; k++) {
        C.begin()[ml * J * N + jn] +=
            A.begin()[ml * K + k] * B.begin()[k * J * N + jn];
      }
    }
  }
}

template <typename T>
void ans_tensmul(const T &a, const monolish::tensor::tensor_Dense<T> &A,
                 const monolish::tensor::tensor_Dense<T> &B, const T &b,
                 monolish::tensor::tensor_Dense<T> &C) {

  if (A.get_shape()[2] != B.get_shape()[0]) {
    std::cout << A.get_shape()[2] << B.get_shape()[0] << std::endl;
    std::runtime_error("test: A.shape[2] != B.shape[0]");
  }
  if (A.get_shape()[0] != C.get_shape()[0] ||
      A.get_shape()[1] != C.get_shape()[1]) {
    std::runtime_error("test: A.row != C.row");
  }
  if (C.get_shape()[2] != B.get_shape()[1] ||
      C.get_shape()[3] != B.get_shape()[2]) {
    std::runtime_error("test: C.col != B.col");
  }

  // MLJN=MLK*KJN
  int M = A.get_shape()[0];
  int L = A.get_shape()[1];
  int K = A.get_shape()[2];
  int J = B.get_shape()[1];
  int N = B.get_shape()[2];

  for (int i = 0; i < C.get_nnz(); i++) {
    C.begin()[i] = b * C.begin()[i];
  }

  for (int ml = 0; ml < M * L; ml++) {
    for (int jn = 0; jn < J * N; jn++) {
      for (int k = 0; k < K; k++) {
        C.begin()[ml * J * N + jn] +=
            a * A.begin()[ml * K + k] * B.begin()[k * J * N + jn];
      }
    }
  }
}

template <typename T, typename TENS_B, typename TENS_C>
bool test_send_tensmul1_core(const size_t M, const size_t N, const size_t L,
                             const size_t J, const size_t K, double tol,
                             monolish::tensor::tensor_Dense<T> &AA,
                             monolish::tensor::tensor_Dense<T> &BB,
                             monolish::tensor::tensor_Dense<T> &CC,
                             monolish::tensor::tensor_CRS<T> &A, TENS_B &B,
                             TENS_C &C) {
  B = BB;
  C = CC;

  ans_tensmul(AA, BB, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::tensmul(A, B, C);
  monolish::util::recv(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename T, typename TENS_A, typename TENS_B, typename TENS_C>
bool test_send_tensmul1_core(const size_t M, const size_t N, const size_t L,
                             const size_t J, const size_t K, double tol,
                             monolish::tensor::tensor_Dense<T> &AA,
                             monolish::tensor::tensor_Dense<T> &BB,
                             monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                             TENS_B &B, TENS_C &C) {
  A = AA;
  B = BB;
  C = CC;

  ans_tensmul(AA, BB, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::tensmul(A, B, C);
  monolish::util::recv(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename T, typename TENS_B, typename TENS_C>
bool test_send_tensmul2_core(const size_t M, const size_t N, const size_t L,
                             const size_t J, const size_t K, const T a,
                             const T b, double tol,
                             monolish::tensor::tensor_Dense<T> &AA,
                             monolish::tensor::tensor_Dense<T> &BB,
                             monolish::tensor::tensor_Dense<T> &CC,
                             monolish::tensor::tensor_CRS<T> &A, TENS_B &B,
                             TENS_C &C) {
  B = BB;
  C = CC;

  ans_tensmul(a, AA, BB, b, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::tensmul(a, A, B, b, C);
  monolish::util::recv(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename T, typename TENS_A, typename TENS_B, typename TENS_C>
bool test_send_tensmul2_core(const size_t M, const size_t N, const size_t L,
                             const size_t J, const size_t K, const T a,
                             const T b, double tol,
                             monolish::tensor::tensor_Dense<T> &AA,
                             monolish::tensor::tensor_Dense<T> &BB,
                             monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                             TENS_B &B, TENS_C &C) {
  A = AA;
  B = BB;
  C = CC;

  ans_tensmul(a, AA, BB, b, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::tensmul(a, A, B, b, C);
  monolish::util::recv(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename T, typename TENS_A, typename TENS_B, typename TENS_C>
bool test_send_tensmul_core(const size_t M, const size_t N, const size_t L,
                            const size_t J, const size_t K, const T a,
                            const T b, double tol,
                            monolish::tensor::tensor_Dense<T> &AA,
                            monolish::tensor::tensor_Dense<T> &BB,
                            monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                            TENS_B &B, TENS_C &C) {
  return test_send_tensmul1_core(M, N, L, J, K, tol, AA, BB, CC, A, B, C) &&
         test_send_tensmul2_core(M, N, L, J, K, a, b, tol, AA, BB, CC, A, B, C);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_send_tensmul(const size_t M, const size_t N, const size_t L,
                       const size_t J, const size_t K, double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, L, K);
  monolish::tensor::tensor_COO<T> seedB =
      get_random_structure_tensor<T>(K, J, N);
  monolish::tensor::tensor_COO<T> seedC =
      get_random_structure_tensor<T>(M, L, J, N);

  const T a = 3;
  const T b = 2;

  MAT_A A(seedA);
  MAT_B B(seedB);
  MAT_C C(seedC);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  return test_send_tensmul_core(M, N, L, J, K, a, b, tol, AA, BB, CC, A, B, C);
}

template <typename T, typename TENS_A, typename TENS_B, std::size_t KK = 0,
          typename... Tp>
inline typename std::enable_if<KK == sizeof...(Tp), bool>::type
test_send_tensmul_view_core3(const size_t M, const size_t N, const size_t L,
                             const size_t J, const size_t K, double tol,
                             monolish::tensor::tensor_Dense<T> &AA,
                             monolish::tensor::tensor_Dense<T> &BB,
                             monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                             TENS_B &B, std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, typename TENS_A, typename TENS_B, std::size_t KK = 0,
          typename... Tp>
    inline typename std::enable_if <
    KK<sizeof...(Tp), bool>::type
    test_send_tensmul_view_core3(const size_t M, const size_t N, const size_t L,
                                 const size_t J, const size_t K, double tol,
                                 monolish::tensor::tensor_Dense<T> &AA,
                                 monolish::tensor::tensor_Dense<T> &BB,
                                 monolish::tensor::tensor_Dense<T> &CC,
                                 TENS_A &A, TENS_B &B, std::tuple<Tp...> &Cs) {
  const T a = 3;
  const T b = 2;
  if (!test_send_tensmul_core(M, N, L, J, K, a, b, tol, AA, BB, CC, A, B,
                              std::get<KK>(Cs))) {
    return false;
  }
  return test_send_tensmul_view_core3<T, TENS_A, TENS_B, KK + 1, Tp...>(
      M, N, L, J, K, tol, AA, BB, CC, A, B, Cs);
}

template <typename T, typename TENS_A, std::size_t JJ = 0, typename TENSES_C,
          typename... Tq>
inline typename std::enable_if<JJ == sizeof...(Tq), bool>::type
test_send_tensmul_view_core2(const size_t M, const size_t N, const size_t L,
                             const size_t J, const size_t K, double tol,
                             monolish::tensor::tensor_Dense<T> &AA,
                             monolish::tensor::tensor_Dense<T> &BB,
                             monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                             std::tuple<Tq...> &Bs, TENSES_C &Cs) {
  return true;
}

template <typename T, typename TENS_A, std::size_t JJ = 0, typename TENSES_C,
          typename... Tq>
    inline typename std::enable_if <
    JJ<sizeof...(Tq), bool>::type test_send_tensmul_view_core2(
        const size_t M, const size_t N, const size_t L, const size_t J,
        const size_t K, double tol, monolish::tensor::tensor_Dense<T> &AA,
        monolish::tensor::tensor_Dense<T> &BB,
        monolish::tensor::tensor_Dense<T> &CC, TENS_A &A, std::tuple<Tq...> &Bs,
        TENSES_C &Cs) {
  if (!test_send_tensmul_view_core3(M, N, L, J, K, tol, AA, BB, CC, A,
                                    std::get<JJ>(Bs), Cs)) {
    return false;
  }
  return test_send_tensmul_view_core2<T, TENS_A, JJ + 1, TENSES_C, Tq...>(
      M, N, L, J, K, tol, AA, BB, CC, A, Bs, Cs);
}

template <typename T, std::size_t I = 0, typename TENSES_B, typename TENSES_C,
          typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_send_tensmul_view_core1(const size_t M, const size_t N, const size_t L,
                             const size_t J, const size_t K, double tol,
                             monolish::tensor::tensor_Dense<T> &AA,
                             monolish::tensor::tensor_Dense<T> &BB,
                             monolish::tensor::tensor_Dense<T> &CC,
                             std::tuple<Tp...> &As, TENSES_B &Bs,
                             TENSES_C &Cs) {
  return true;
}

template <typename T, std::size_t I = 0, typename TENSES_B, typename TENSES_C,
          typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type test_send_tensmul_view_core1(
        const size_t M, const size_t N, const size_t L, const size_t J,
        const size_t K, double tol, monolish::tensor::tensor_Dense<T> &AA,
        monolish::tensor::tensor_Dense<T> &BB,
        monolish::tensor::tensor_Dense<T> &CC, std::tuple<Tp...> &As,
        TENSES_B &Bs, TENSES_C &Cs) {
  if (!test_send_tensmul_view_core2(M, N, L, J, K, tol, AA, BB, CC,
                                    std::get<I>(As), Bs, Cs)) {
    return false;
  }
  return test_send_tensmul_view_core1<T, I + 1, TENSES_B, TENSES_C, Tp...>(
      M, N, L, J, K, tol, AA, BB, CC, As, Bs, Cs);
}

template <typename T>
bool test_send_tensmul_view(const size_t M, const size_t N, const size_t L,
                            const size_t J, const size_t K, double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, L, K);
  monolish::tensor::tensor_COO<T> seedB =
      get_random_structure_matrix<T>(K, J, N);
  monolish::tensor::tensor_COO<T> seedC =
      get_random_structure_tensor<T>(M, L, J, N);
  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  using T1_1 = monolish::tensor::tensor_Dense<T>;
  using T1_2_ = monolish::vector<T>;
  using T1_2 = monolish::view_tensor_Dense<T1_2_, T>;
  using T1_3_ = monolish::matrix::Dense<T>;
  using T1_3 = monolish::view_tensor_Dense<T1_3_, T>;
  using T1_4_ = monolish::tensor::tensor_Dense<T>;
  using T1_4 = monolish::view_tensor_Dense<T1_4_, T>;
  T1_1 x1({M, L, K}, 0.0, 1.0);
  T1_2_ x2_(2 * M * L * K, 0.0, 1.0);
  T1_2 x2(x2_, K / 2, {M, L, K});
  T1_3_ x3_(2 * M * L * K, 1, 0.0, 1.0);
  T1_3 x3(x3_, K / 2, {M, L, K});
  T1_4_ x4_({2 * M * L * K, 1, 1}, 0.0, 1.0);
  T1_4 x4(x4_, K / 2, {M, L, K});

  auto As = std::make_tuple(x1, x2, x3, x4);

  T1_1 y1({K, J, N}, 0.0, 1.0);
  T1_2_ y2_(2 * K * J * N, 0.0, 1.0);
  T1_2 y2(y2_, N / 2, {K, J, N});
  T1_3_ y3_(2 * K * J * N, 1, 0.0, 1.0);
  T1_3 y3(y3_, N / 2, {K, J, N});
  T1_4_ y4_({2 * K * J * N, 1, 1}, 0.0, 1.0);
  T1_4 y4(y4_, N / 2, {K, J, N});

  auto Bs = std::make_tuple(y1, y2, y3, y4);

  T1_1 z1({M, L, J, N}, 0.0, 1.0);
  T1_2_ z2_(2 * M * L * J * N, 0.0, 1.0);
  T1_2 z2(z2_, N / 2, {M, L, J, N});
  T1_3_ z3_(2 * M * L * J * N, 1, 0.0, 1.0);
  T1_3 z3(z3_, N / 2, {M, L, J, N});
  T1_4_ z4_({2 * M * L * J * N, 1, 1}, 0.0, 1.0);
  T1_4 z4(z4_, N / 2, {M, L, J, N});

  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_send_tensmul_view_core1(M, N, L, J, K, tol, AA, BB, CC, As, Bs,
                                      Cs);
}

template <typename T, typename TENS_B, typename TENS_C>
bool test_tensmul1_core(const size_t M, const size_t N, const size_t L,
                        const size_t J, const size_t K, double tol,
                        monolish::tensor::tensor_Dense<T> &AA,
                        monolish::tensor::tensor_Dense<T> &BB,
                        monolish::tensor::tensor_Dense<T> &CC,
                        monolish::tensor::tensor_CRS<T> &A, TENS_B &B,
                        TENS_C &C) {
  B = BB;
  C = CC;

  ans_tensmul(AA, BB, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::blas::tensmul(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename T, typename TENS_A, typename TENS_B, typename TENS_C>
bool test_tensmul1_core(const size_t M, const size_t N, const size_t L,
                        const size_t J, const size_t K, double tol,
                        monolish::tensor::tensor_Dense<T> &AA,
                        monolish::tensor::tensor_Dense<T> &BB,
                        monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                        TENS_B &B, TENS_C &C) {
  A = AA;
  B = BB;
  C = CC;

  ans_tensmul(AA, BB, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::blas::tensmul(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename T, typename TENS_B, typename TENS_C>
bool test_tensmul2_core(const size_t M, const size_t N, const size_t L,
                        const size_t J, const size_t K, const T a, const T b,
                        double tol, monolish::tensor::tensor_Dense<T> &AA,
                        monolish::tensor::tensor_Dense<T> &BB,
                        monolish::tensor::tensor_Dense<T> &CC,
                        monolish::tensor::tensor_CRS<T> &A, TENS_B &B,
                        TENS_C &C) {
  B = BB;
  C = CC;

  ans_tensmul(a, AA, BB, b, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::tensmul(a, A, B, b, C);
  monolish::util::recv(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename T, typename TENS_A, typename TENS_B, typename TENS_C>
bool test_tensmul2_core(const size_t M, const size_t N, const size_t L,
                        const size_t J, const size_t K, const T a, const T b,
                        double tol, monolish::tensor::tensor_Dense<T> &AA,
                        monolish::tensor::tensor_Dense<T> &BB,
                        monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                        TENS_B &B, TENS_C &C) {
  A = AA;
  B = BB;
  C = CC;

  ans_tensmul(a, AA, BB, b, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::tensmul(a, A, B, b, C);
  monolish::util::recv(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename T, typename TENS_A, typename TENS_B, typename TENS_C>
bool test_tensmul_core(const size_t M, const size_t N, const size_t L,
                       const size_t J, const size_t K, const T a, const T b,
                       double tol, monolish::tensor::tensor_Dense<T> &AA,
                       monolish::tensor::tensor_Dense<T> &BB,
                       monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                       TENS_B &B, TENS_C &C) {
  return test_tensmul1_core(M, N, L, J, K, tol, AA, BB, CC, A, B, C) &&
         test_tensmul2_core(M, N, L, J, K, a, b, tol, AA, BB, CC, A, B, C);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_tensmul(const size_t M, const size_t N, const size_t L,
                  const size_t J, const size_t K, double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, L, K);
  monolish::tensor::tensor_COO<T> seedB =
      get_random_structure_tensor<T>(K, J, N);
  monolish::tensor::tensor_COO<T> seedC =
      get_random_structure_tensor<T>(M, L, J, N);

  const T a = 3;
  const T b = 2;

  MAT_A A(seedA);
  MAT_B B(seedB);
  MAT_C C(seedC);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  return test_tensmul_core(M, N, L, J, K, a, b, tol, AA, BB, CC, A, B, C);
}

template <typename T, typename TENS_A, typename TENS_B, std::size_t KK = 0,
          typename... Tp>
inline typename std::enable_if<KK == sizeof...(Tp), bool>::type
test_tensmul_view_core3(const size_t M, const size_t N, const size_t L,
                        const size_t J, const size_t K, double tol,
                        monolish::tensor::tensor_Dense<T> &AA,
                        monolish::tensor::tensor_Dense<T> &BB,
                        monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                        TENS_B &B, std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, typename TENS_A, typename TENS_B, std::size_t KK = 0,
          typename... Tp>
    inline typename std::enable_if <
    KK<sizeof...(Tp), bool>::type
    test_tensmul_view_core3(const size_t M, const size_t N, const size_t L,
                            const size_t J, const size_t K, double tol,
                            monolish::tensor::tensor_Dense<T> &AA,
                            monolish::tensor::tensor_Dense<T> &BB,
                            monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                            TENS_B &B, std::tuple<Tp...> &Cs) {
  const T a = 3;
  const T b = 2;
  if (!test_tensmul_core(M, N, L, J, K, a, b, tol, AA, BB, CC, A, B,
                         std::get<KK>(Cs))) {
    return false;
  }
  return test_tensmul_view_core3<T, TENS_A, TENS_B, KK + 1, Tp...>(
      M, N, L, J, K, tol, AA, BB, CC, A, B, Cs);
}

template <typename T, typename TENS_A, std::size_t JJ = 0, typename TENSES_C,
          typename... Tq>
inline typename std::enable_if<JJ == sizeof...(Tq), bool>::type
test_tensmul_view_core2(const size_t M, const size_t N, const size_t L,
                        const size_t J, const size_t K, double tol,
                        monolish::tensor::tensor_Dense<T> &AA,
                        monolish::tensor::tensor_Dense<T> &BB,
                        monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                        std::tuple<Tq...> &Bs, TENSES_C &Cs) {
  return true;
}

template <typename T, typename TENS_A, std::size_t JJ = 0, typename TENSES_C,
          typename... Tq>
    inline typename std::enable_if <
    JJ<sizeof...(Tq), bool>::type
    test_tensmul_view_core2(const size_t M, const size_t N, const size_t L,
                            const size_t J, const size_t K, double tol,
                            monolish::tensor::tensor_Dense<T> &AA,
                            monolish::tensor::tensor_Dense<T> &BB,
                            monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                            std::tuple<Tq...> &Bs, TENSES_C &Cs) {
  if (!test_tensmul_view_core3(M, N, L, J, K, tol, AA, BB, CC, A,
                               std::get<JJ>(Bs), Cs)) {
    return false;
  }
  return test_tensmul_view_core2<T, TENS_A, JJ + 1, TENSES_C, Tq...>(
      M, N, L, J, K, tol, AA, BB, CC, A, Bs, Cs);
}

template <typename T, std::size_t I = 0, typename TENSES_B, typename TENSES_C,
          typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_tensmul_view_core1(const size_t M, const size_t N, const size_t L,
                        const size_t J, const size_t K, double tol,
                        monolish::tensor::tensor_Dense<T> &AA,
                        monolish::tensor::tensor_Dense<T> &BB,
                        monolish::tensor::tensor_Dense<T> &CC,
                        std::tuple<Tp...> &As, TENSES_B &Bs, TENSES_C &Cs) {
  return true;
}

template <typename T, std::size_t I = 0, typename TENSES_B, typename TENSES_C,
          typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type
    test_tensmul_view_core1(const size_t M, const size_t N, const size_t L,
                            const size_t J, const size_t K, double tol,
                            monolish::tensor::tensor_Dense<T> &AA,
                            monolish::tensor::tensor_Dense<T> &BB,
                            monolish::tensor::tensor_Dense<T> &CC,
                            std::tuple<Tp...> &As, TENSES_B &Bs, TENSES_C &Cs) {
  if (!test_tensmul_view_core2(M, N, L, J, K, tol, AA, BB, CC, std::get<I>(As),
                               Bs, Cs)) {
    return false;
  }
  return test_tensmul_view_core1<T, I + 1, TENSES_B, TENSES_C, Tp...>(
      M, N, L, J, K, tol, AA, BB, CC, As, Bs, Cs);
}

template <typename T>
bool test_tensmul_view(const size_t M, const size_t N, const size_t L,
                       const size_t J, const size_t K, double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, L, K);
  monolish::tensor::tensor_COO<T> seedB =
      get_random_structure_matrix<T>(K, J, N);
  monolish::tensor::tensor_COO<T> seedC =
      get_random_structure_tensor<T>(M, L, J, N);
  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  using T1_1 = monolish::tensor::tensor_Dense<T>;
  using T1_2_ = monolish::vector<T>;
  using T1_2 = monolish::view_tensor_Dense<T1_2_, T>;
  using T1_3_ = monolish::matrix::Dense<T>;
  using T1_3 = monolish::view_tensor_Dense<T1_3_, T>;
  using T1_4_ = monolish::tensor::tensor_Dense<T>;
  using T1_4 = monolish::view_tensor_Dense<T1_4_, T>;
  T1_1 x1({M, L, K}, 0.0, 1.0);
  T1_2_ x2_(2 * M * L * K, 0.0, 1.0);
  T1_2 x2(x2_, K / 2, {M, L, K});
  T1_3_ x3_(2 * M * L * K, 1, 0.0, 1.0);
  T1_3 x3(x3_, K / 2, {M, L, K});
  T1_4_ x4_({2 * M * L * K, 1, 1}, 0.0, 1.0);
  T1_4 x4(x4_, K / 2, {M, L, K});

  auto As = std::make_tuple(x1, x2, x3, x4);

  T1_1 y1({K, J, N}, 0.0, 1.0);
  T1_2_ y2_(2 * K * J * N, 0.0, 1.0);
  T1_2 y2(y2_, N / 2, {K, J, N});
  T1_3_ y3_(2 * K * J * N, 1, 0.0, 1.0);
  T1_3 y3(y3_, N / 2, {K, J, N});
  T1_4_ y4_({2 * K * J * N, 1, 1}, 0.0, 1.0);
  T1_4 y4(y4_, N / 2, {K, J, N});

  auto Bs = std::make_tuple(y1, y2, y3, y4);

  T1_1 z1({M, L, J, N}, 0.0, 1.0);
  T1_2_ z2_(2 * M * L * J * N, 0.0, 1.0);
  T1_2 z2(z2_, N / 2, {M, L, J, N});
  T1_3_ z3_(2 * M * L * J * N, 1, 0.0, 1.0);
  T1_3 z3(z3_, N / 2, {M, L, J, N});
  T1_4_ z4_({2 * M * L * J * N, 1, 1}, 0.0, 1.0);
  T1_4 z4(z4_, N / 2, {M, L, J, N});

  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_tensmul_view_core1(M, N, L, J, K, tol, AA, BB, CC, As, Bs, Cs);
}
