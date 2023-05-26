#include "../../test_utils.hpp"

template <typename T>
void ans_matmul(const monolish::matrix::Dense<T> &A,
                const monolish::matrix::Dense<T> &B,
                monolish::matrix::Dense<T> &C) {

  if (A.get_col() != B.get_row()) {
    std::cout << A.get_col() << B.get_row() << std::endl;
    std::runtime_error("test: A.col != B.row");
  }
  if (A.get_row() != C.get_row()) {
    std::runtime_error("test: A.row != C.row");
  }
  if (C.get_col() != B.get_col()) {
    std::runtime_error("test: C.col != B.col");
  }

  // MN=MK*KN
  int M = A.get_row();
  int N = B.get_col();
  int K = A.get_col();

  for (int i = 0; i < C.get_nnz(); i++) {
    C.begin()[i] = 0;
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        C.begin()[i * N + j] += A.begin()[i * K + k] * B.begin()[k * N + j];
      }
    }
  }
}

template <typename T>
void ans_matmul(const T &a, const monolish::matrix::Dense<T> &A,
                const monolish::matrix::Dense<T> &B, const T &b,
                monolish::matrix::Dense<T> &C) {

  if (A.get_col() != B.get_row()) {
    std::cout << A.get_col() << B.get_row() << std::endl;
    std::runtime_error("test: A.col != B.row");
  }
  if (A.get_row() != C.get_row()) {
    std::runtime_error("test: A.row != C.row");
  }
  if (C.get_col() != B.get_col()) {
    std::runtime_error("test: C.col != B.col");
  }

  // MN=MK*KN
  int M = A.get_row();
  int N = B.get_col();
  int K = A.get_col();

  for (int i = 0; i < C.get_nnz(); i++) {
    C.begin()[i] = b * C.begin()[i];
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        C.begin()[i * N + j] += a * A.begin()[i * K + k] * B.begin()[k * N + j];
      }
    }
  }
}

template <typename T, typename MAT_A, typename MAT_B, typename MAT_C>
bool test_send_matmul_core(const size_t M, const size_t N, const size_t K,
                           double tol, monolish::matrix::Dense<T> &AA,
                           monolish::matrix::Dense<T> &BB,
                           monolish::matrix::Dense<T> &CC, MAT_A &A, MAT_B &B,
                           MAT_C &C) {
  ans_matmul(AA, BB, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::matmul(A, B, C);
  monolish::util::recv(A, B, C);
  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__,
                      A.type() + "+" + B.type() + "=" + C.type() + "(C=AB)",
                      resultC.begin(), ansC.begin(), ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_send_matmul(const size_t M, const size_t N, const size_t K,
                      double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, K);
  monolish::matrix::COO<T> seedB = get_random_structure_matrix<T>(K, N);
  monolish::matrix::COO<T> seedC = get_random_structure_matrix<T>(M, N);

  MAT_A A(seedA);
  MAT_B B(seedB);
  MAT_C C(seedC);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedB);
  monolish::matrix::Dense<T> CC(seedC);

  return test_send_matmul_core(M, N, K, tol, AA, BB, CC, A, B, C);
}

template <typename T, typename MAT_A, typename MAT_B, std::size_t KK = 0,
          typename... Tr>
inline typename std::enable_if<KK == sizeof...(Tr), bool>::type
test_send_matmul_view_dense_core3(const size_t M, const size_t N,
                                  const size_t K, double tol,
                                  monolish::matrix::Dense<T> &AA,
                                  monolish::matrix::Dense<T> &BB,
                                  monolish::matrix::Dense<T> &CC, MAT_A &A,
                                  MAT_B &B, std::tuple<Tr...> &Cs) {
  return true;
}

template <typename T, typename MAT_A, typename MAT_B, std::size_t KK = 0,
          typename... Tr>
    inline typename std::enable_if <
    KK<sizeof...(Tr), bool>::type test_send_matmul_view_dense_core3(
        const size_t M, const size_t N, const size_t K, double tol,
        monolish::matrix::Dense<T> &AA, monolish::matrix::Dense<T> &BB,
        monolish::matrix::Dense<T> &CC, MAT_A &A, MAT_B &B,
        std::tuple<Tr...> &Cs) {
  A = AA;
  B = BB;
  if (!test_send_matmul_core(M, N, K, tol, AA, BB, CC, A, B,
                             std::get<KK>(Cs))) {
    return false;
  }
  return test_send_matmul_view_dense_core3<T, MAT_A, MAT_B, KK + 1, Tr...>(
      M, N, K, tol, AA, BB, CC, A, B, Cs);
}

template <typename T, typename MAT_A, std::size_t J = 0, typename MATS_C,
          typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_send_matmul_view_dense_core2(const size_t M, const size_t N,
                                  const size_t K, double tol,
                                  monolish::matrix::Dense<T> &AA,
                                  monolish::matrix::Dense<T> &BB,
                                  monolish::matrix::Dense<T> &CC, MAT_A &A,
                                  std::tuple<Tq...> &Bs, MATS_C &Cs) {
  return true;
}

template <typename T, typename MAT_A, std::size_t J = 0, typename MATS_C,
          typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type test_send_matmul_view_dense_core2(
        const size_t M, const size_t N, const size_t K, double tol,
        monolish::matrix::Dense<T> &AA, monolish::matrix::Dense<T> &BB,
        monolish::matrix::Dense<T> &CC, MAT_A &A, std::tuple<Tq...> &Bs,
        MATS_C &Cs) {
  if (!test_send_matmul_view_dense_core3(M, N, K, tol, AA, BB, CC, A,
                                         std::get<J>(Bs), Cs)) {
    return false;
  }
  return test_send_matmul_view_dense_core2<T, MAT_A, J + 1, MATS_C, Tq...>(
      M, N, K, tol, AA, BB, CC, A, Bs, Cs);
}

template <typename T, std::size_t I = 0, typename MATS_B, typename MATS_C,
          typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_send_matmul_view_dense_core1(const size_t M, const size_t N,
                                  const size_t K, double tol,
                                  monolish::matrix::Dense<T> &AA,
                                  monolish::matrix::Dense<T> &BB,
                                  monolish::matrix::Dense<T> &CC,
                                  std::tuple<Tp...> &As, MATS_B &Bs,
                                  MATS_C &Cs) {
  return true;
}

template <typename T, std::size_t I = 0, typename MATS_B, typename MATS_C,
          typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type test_send_matmul_view_dense_core1(
        const size_t M, const size_t N, const size_t K, double tol,
        monolish::matrix::Dense<T> &AA, monolish::matrix::Dense<T> &BB,
        monolish::matrix::Dense<T> &CC, std::tuple<Tp...> &As, MATS_B &Bs,
        MATS_C &Cs) {
  if (!test_send_matmul_view_dense_core2(M, N, K, tol, AA, BB, CC,
                                         std::get<I>(As), Bs, Cs)) {
    return false;
  }
  return test_send_matmul_view_dense_core1<T, I + 1, MATS_B, MATS_C, Tp...>(
      M, N, K, tol, AA, BB, CC, As, Bs, Cs);
}

template <typename T>
bool test_send_matmul_view_dense(const size_t M, const size_t N, const size_t K,
                                 double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, K);
  monolish::matrix::COO<T> seedB = get_random_structure_matrix<T>(K, N);
  monolish::matrix::COO<T> seedC = get_random_structure_matrix<T>(M, N);
  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedB);
  monolish::matrix::Dense<T> CC(seedC);

  using T1_1 = monolish::matrix::Dense<T>;
  using T1_2_ = monolish::vector<T>;
  using T1_2 = monolish::view_Dense<T1_2_, T>;
  using T1_3_ = monolish::matrix::Dense<T>;
  using T1_3 = monolish::view_Dense<T1_3_, T>;
  using T1_4_ = monolish::tensor::tensor_Dense<T>;
  using T1_4 = monolish::view_Dense<T1_4_, T>;
  T1_1 x1(M, K, 0.0, 1.0);
  T1_2_ x2_(2 * M * K, 0.0, 1.0);
  T1_2 x2(x2_, M / 2, M, K);
  T1_3_ x3_(2 * M * K, 1, 0.0, 1.0);
  T1_3 x3(x3_, M / 2, M, K);
  T1_4_ x4_({2 * M * K, 1, 1}, 0.0, 1.0);
  T1_4 x4(x4_, M / 2, M, K);

  auto As = std::make_tuple(x1, x2, x3, x4);

  T1_1 y1(K, N, 0.0, 1.0);
  T1_2_ y2_(2 * K * N, 0.0, 1.0);
  T1_2 y2(y2_, K / 2, K, N);
  T1_3_ y3_(2 * K * N, 1, 0.0, 1.0);
  T1_3 y3(y3_, K / 2, K, N);
  T1_4_ y4_({2 * K * N, 1, 1}, 0.0, 1.0);
  T1_4 y4(y4_, K / 2, K, N);

  auto Bs = std::make_tuple(y1, y2, y3, y4);

  T1_1 z1(M, N, 0.0, 1.0);
  T1_2_ z2_(2 * M * N, 0.0, 1.0);
  T1_2 z2(z2_, M / 2, M, N);
  T1_3_ z3_(2 * M * N, 1, 0.0, 1.0);
  T1_3 z3(z3_, M / 2, M, N);
  T1_4_ z4_({2 * M * N, 1, 1}, 0.0, 1.0);
  T1_4 z4(z4_, M / 2, M, N);

  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_send_matmul_view_dense_core1(M, N, K, tol, AA, BB, CC, As, Bs,
                                           Cs);
}

template <typename T, typename MAT_A, typename MAT_B, typename MAT_C>
bool test_send_matmul_core(const size_t M, const size_t N, const size_t K,
                           const T a, const T b, double tol,
                           monolish::matrix::Dense<T> &AA,
                           monolish::matrix::Dense<T> &BB,
                           monolish::matrix::Dense<T> &CC, MAT_A &A, MAT_B &B,
                           MAT_C &C) {
  ans_matmul(a, AA, BB, b, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::matmul(a, A, B, b, C);
  monolish::util::recv(A, B, C);
  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__,
                      A.type() + "+" + B.type() + "=" + C.type() + "(C=aAB+bC)",
                      resultC.begin(), ansC.begin(), ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_send_matmul(const size_t M, const size_t N, const size_t K, const T a,
                      const T b, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, K);
  monolish::matrix::COO<T> seedB = get_random_structure_matrix<T>(K, N);
  monolish::matrix::COO<T> seedC = get_random_structure_matrix<T>(M, N);

  MAT_A A(seedA);
  MAT_B B(seedB);
  MAT_C C(seedC);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedB);
  monolish::matrix::Dense<T> CC(seedC);

  return test_send_matmul_core(M, N, K, a, b, tol, AA, BB, CC, A, B, C);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_send_matmul_linearoperator_only(const size_t M, const size_t N,
                                          const size_t K, double tol) {

  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, K);
  monolish::matrix::COO<T> seedB = get_random_structure_matrix<T>(K, N);
  monolish::matrix::COO<T> seedC = get_random_structure_matrix<T>(M, N);

  monolish::matrix::CRS<T> A1(seedA);
  monolish::matrix::CRS<T> B1(seedB);
  monolish::matrix::CRS<T> C1(seedC);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedB);
  monolish::matrix::Dense<T> CC(seedC);

  ans_matmul(AA, BB, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::util::send(A1, B1, C1);
  MAT_A A(A1);
  MAT_B B(B1);
  MAT_C C(C1);
  monolish::blas::matmul(A, B, C);
  C.recv();

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_send_matmul_linearoperator(const size_t M, const size_t N,
                                     const size_t K, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, K);
  monolish::matrix::COO<T> seedB = get_random_structure_matrix<T>(K, N);
  monolish::matrix::COO<T> seedC = get_random_structure_matrix<T>(M, N);

  monolish::matrix::CRS<T> A1(seedA);
  MAT_B B(seedB);
  MAT_C C(seedC);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedB);
  monolish::matrix::Dense<T> CC(seedC);

  ans_matmul(AA, BB, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::util::send(A1, B, C);
  MAT_A A(A1);
  monolish::blas::matmul(A, B, C);
  C.recv();

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_matmul(const size_t M, const size_t N, const size_t K, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, K);
  monolish::matrix::COO<T> seedB = get_random_structure_matrix<T>(K, N);
  monolish::matrix::COO<T> seedC = get_random_structure_matrix<T>(M, N);

  MAT_A A(seedA);
  MAT_B B(seedB);
  MAT_C C(seedC);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedB);
  monolish::matrix::Dense<T> CC(seedC);

  ans_matmul(AA, BB, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::blas::matmul(A, B, C);

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_matmul(const size_t M, const size_t N, const size_t K, const T a,
                 const T b, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, K);
  monolish::matrix::COO<T> seedB = get_random_structure_matrix<T>(K, N);
  monolish::matrix::COO<T> seedC = get_random_structure_matrix<T>(M, N);

  MAT_A A(seedA);
  MAT_B B(seedB);
  MAT_C C(seedC);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedB);
  monolish::matrix::Dense<T> CC(seedC);

  ans_matmul(a, AA, BB, b, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::blas::matmul(a, A, B, b, C);

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}
