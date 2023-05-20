#include "../../test_utils.hpp"

template <typename T, typename VEC>
void ans_adds_col(const monolish::matrix::Dense<T> &A, const VEC &mx,
                  monolish::matrix::Dense<T> &C) {
  if (A.get_row() != mx.size()) {
    std::runtime_error("A.row != y.size");
  }

  const T *x = mx.begin();
  int M = A.get_row();
  int N = A.get_col();

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C.begin()[i * N + j] = A.begin()[i * N + j] + x[i];
    }
  }
}

template <typename T, typename MAT_A, typename MAT_C, typename VEC>
bool test_send_adds_col_core(const size_t M, const size_t N, double tol,
                             monolish::matrix::Dense<T> &AA,
                             monolish::vector<T> &vec,
                             monolish::matrix::Dense<T> &CC, MAT_A &A, VEC &x,
                             MAT_C &C) {
  ans_adds_col(AA, vec, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::util::send(A, x, C);
  monolish::blas::adds_col(A, x, C);
  monolish::util::recv(A, x, C);
  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type() + "+" + x.type(), resultC.begin(),
                      ansC.begin(), ansC.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_send_adds_col(const size_t M, const size_t N, double tol) {
  monolish::vector<T> x(M, 0.0, 1.0, test_random_engine());
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);

  MAT A(seedA); // M*N matrix
  MAT C(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);

  return test_send_adds_col_core(M, N, tol, AA, x, CC, A, x, C);
}

template <typename T, typename MAT_A, typename VEC, std::size_t K = 0,
          typename... Tp>
inline typename std::enable_if<K == sizeof...(Tp), bool>::type
test_send_adds_col_view_core3(const size_t M, const size_t N, double tol,
                              monolish::matrix::Dense<T> &AA,
                              monolish::vector<T> &vec,
                              monolish::matrix::Dense<T> &CC, MAT_A &A, VEC &x,
                              std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, typename MAT_A, typename VEC, std::size_t K = 0,
          typename... Tp>
    inline typename std::enable_if <
    K<sizeof...(Tp), bool>::type
    test_send_adds_col_view_core3(const size_t M, const size_t N, double tol,
                                  monolish::matrix::Dense<T> &AA,
                                  monolish::vector<T> &vec,
                                  monolish::matrix::Dense<T> &CC, MAT_A &A,
                                  VEC &x, std::tuple<Tp...> &Cs) {
  A = AA;
  x = vec;
  if (!test_send_adds_col_core(M, N, tol, AA, vec, CC, A, x, std::get<K>(Cs))) {
    return false;
  }
  return test_send_adds_col_view_core3<T, MAT_A, VEC, K + 1, Tp...>(
      M, N, tol, AA, vec, CC, A, x, Cs);
}

template <typename T, typename MAT_A, std::size_t J = 0, typename MATS_C,
          typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_send_adds_col_view_core2(const size_t M, const size_t N, double tol,
                              monolish::matrix::Dense<T> &AA,
                              monolish::vector<T> &vec,
                              monolish::matrix::Dense<T> &CC, MAT_A &A,
                              std::tuple<Tq...> &x, MATS_C &Cs) {
  return true;
}

template <typename T, typename MAT_A, std::size_t J = 0, typename MATS_C,
          typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type
    test_send_adds_col_view_core2(const size_t M, const size_t N, double tol,
                                  monolish::matrix::Dense<T> &AA,
                                  monolish::vector<T> &vec,
                                  monolish::matrix::Dense<T> &CC, MAT_A &A,
                                  std::tuple<Tq...> &xs, MATS_C &Cs) {
  if (!test_send_adds_col_view_core3(M, N, tol, AA, vec, CC, A, std::get<J>(xs),
                                     Cs)) {
    return false;
  }
  return test_send_adds_col_view_core2<T, MAT_A, J + 1, MATS_C, Tq...>(
      M, N, tol, AA, vec, CC, A, xs, Cs);
}

template <typename T, std::size_t I = 0, typename VECS, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_send_adds_col_view_core1(const size_t M, const size_t N, double tol,
                              monolish::matrix::Dense<T> &AA,
                              monolish::vector<T> &vec,
                              monolish::matrix::Dense<T> &CC,
                              std::tuple<Tp...> &As, VECS &xs,
                              std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, std::size_t I = 0, typename VECS, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type test_send_adds_col_view_core1(
        const size_t M, const size_t N, double tol,
        monolish::matrix::Dense<T> &AA, monolish::vector<T> &vec,
        monolish::matrix::Dense<T> &CC, std::tuple<Tp...> &As, VECS &xs,
        std::tuple<Tp...> &Cs) {
  if (!test_send_adds_col_view_core2(M, N, tol, AA, vec, CC, std::get<I>(As),
                                     xs, Cs)) {
    return false;
  }
  return test_send_adds_col_view_core1<T, I + 1, VECS, Tp...>(
      M, N, tol, AA, vec, CC, As, xs, Cs);
}

template <typename T>
bool test_send_adds_col_view(const size_t M, const size_t N, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);
  monolish::matrix::Dense<T> AA(seedA), CC(seedA);
  monolish::vector<T> xx(AA.get_row(), 0.0, 1.0, test_random_engine());

  using T1_1 = monolish::matrix::Dense<T>;
  using T1_2_ = monolish::vector<T>;
  using T1_2 = monolish::view_Dense<T1_2_, T>;
  using T1_3_ = monolish::matrix::Dense<T>;
  using T1_3 = monolish::view_Dense<T1_3_, T>;
  using T1_4_ = monolish::tensor::tensor_Dense<T>;
  using T1_4 = monolish::view_Dense<T1_4_, T>;
  T1_1 x1(M, N, 0.0, 1.0);
  T1_2_ x2_(2 * M * N, 0.0, 1.0);
  T1_2 x2(x2_, M / 2, M, N);
  T1_3_ x3_(2 * M * N, 1, 0.0, 1.0);
  T1_3 x3(x3_, M / 2, M, N);
  T1_4_ x4_({2 * M * N, 1, 1}, 0.0, 1.0);
  T1_4 x4(x4_, M / 2, M, N);

  auto As = std::make_tuple(x1, x2, x3, x4);

  using T2_1 = monolish::vector<T>;
  using T2_2 = monolish::view1D<T1_2_, T>;
  using T2_3 = monolish::view1D<T1_3_, T>;
  using T2_4 = monolish::view1D<T1_4_, T>;
  T2_1 y1(M, 0.0, 1.0);
  T1_2_ y2_(2 * M, 0.0, 1.0);
  T2_2 y2(y2_, M / 2, M);
  T1_3_ y3_(2 * M, 1, 0.0, 1.0);
  T2_3 y3(y3_, M / 2, M);
  T1_4_ y4_({2 * M, 1, 1}, 0.0, 1.0);
  T2_4 y4(y4_, M / 2, M);

  auto Bs = std::make_tuple(y1, y2, y3, y4);

  T1_1 z1(M, N, 0.0, 1.0);
  T1_2_ z2_(2 * M * N, 0.0, 1.0);
  T1_2 z2(z2_, M / 2, M, N);
  T1_3_ z3_(2 * M * N, 1, 0.0, 1.0);
  T1_3 z3(z3_, M / 2, M, N);
  T1_4_ z4_({2 * M * N, 1, 1}, 0.0, 1.0);
  T1_4 z4(z4_, M / 2, M, N);

  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_send_adds_col_view_core1(M, N, tol, AA, xx, CC, As, Bs, Cs);
}

template <typename T, typename MAT_A, typename MAT_C, typename VEC>
bool test_adds_col_core(const size_t M, const size_t N, double tol,
                        monolish::matrix::Dense<T> &AA,
                        monolish::vector<T> &vec,
                        monolish::matrix::Dense<T> &CC, MAT_A &A, VEC &x,
                        MAT_C &C) {
  ans_adds_col(AA, vec, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::blas::adds_col(A, x, C);
  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type() + "+" + x.type(), resultC.begin(),
                      ansC.begin(), ansC.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_adds_col(const size_t M, const size_t N, double tol) {
  monolish::vector<T> x(M, 0.0, 1.0, test_random_engine());
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);

  MAT A(seedA); // M*N matrix
  MAT C(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);

  return test_adds_col_core(M, N, tol, AA, x, CC, A, x, C);
}

template <typename T, typename MAT_A, typename VEC, std::size_t K = 0,
          typename... Tp>
inline typename std::enable_if<K == sizeof...(Tp), bool>::type
test_adds_col_view_core3(const size_t M, const size_t N, double tol,
                         monolish::matrix::Dense<T> &AA,
                         monolish::vector<T> &vec,
                         monolish::matrix::Dense<T> &CC, MAT_A &A, VEC &x,
                         std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, typename MAT_A, typename VEC, std::size_t K = 0,
          typename... Tp>
    inline typename std::enable_if <
    K<sizeof...(Tp), bool>::type
    test_adds_col_view_core3(const size_t M, const size_t N, double tol,
                             monolish::matrix::Dense<T> &AA,
                             monolish::vector<T> &vec,
                             monolish::matrix::Dense<T> &CC, MAT_A &A, VEC &x,
                             std::tuple<Tp...> &Cs) {
  A = AA;
  x = vec;
  if (!test_adds_col_core(M, N, tol, AA, vec, CC, A, x, std::get<K>(Cs))) {
    return false;
  }
  return test_adds_col_view_core3<T, MAT_A, VEC, K + 1, Tp...>(
      M, N, tol, AA, vec, CC, A, x, Cs);
}

template <typename T, typename MAT_A, std::size_t J = 0, typename MATS_C,
          typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_adds_col_view_core2(const size_t M, const size_t N, double tol,
                         monolish::matrix::Dense<T> &AA,
                         monolish::vector<T> &vec,
                         monolish::matrix::Dense<T> &CC, MAT_A &A,
                         std::tuple<Tq...> &x, MATS_C &Cs) {
  return true;
}

template <typename T, typename MAT_A, std::size_t J = 0, typename MATS_C,
          typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type
    test_adds_col_view_core2(const size_t M, const size_t N, double tol,
                             monolish::matrix::Dense<T> &AA,
                             monolish::vector<T> &vec,
                             monolish::matrix::Dense<T> &CC, MAT_A &A,
                             std::tuple<Tq...> &xs, MATS_C &Cs) {
  if (!test_adds_col_view_core3(M, N, tol, AA, vec, CC, A, std::get<J>(xs),
                                Cs)) {
    return false;
  }
  return test_adds_col_view_core2<T, MAT_A, J + 1, MATS_C, Tq...>(
      M, N, tol, AA, vec, CC, A, xs, Cs);
}

template <typename T, std::size_t I = 0, typename VECS, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_adds_col_view_core1(const size_t M, const size_t N, double tol,
                         monolish::matrix::Dense<T> &AA,
                         monolish::vector<T> &vec,
                         monolish::matrix::Dense<T> &CC, std::tuple<Tp...> &As,
                         VECS &xs, std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, std::size_t I = 0, typename VECS, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type test_adds_col_view_core1(
        const size_t M, const size_t N, double tol,
        monolish::matrix::Dense<T> &AA, monolish::vector<T> &vec,
        monolish::matrix::Dense<T> &CC, std::tuple<Tp...> &As, VECS &xs,
        std::tuple<Tp...> &Cs) {
  if (!test_adds_col_view_core2(M, N, tol, AA, vec, CC, std::get<I>(As), xs,
                                Cs)) {
    return false;
  }
  return test_adds_col_view_core1<T, I + 1, VECS, Tp...>(M, N, tol, AA, vec, CC,
                                                         As, xs, Cs);
}

template <typename T>
bool test_adds_col_view(const size_t M, const size_t N, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);
  monolish::matrix::Dense<T> AA(seedA), CC(seedA);
  monolish::vector<T> xx(AA.get_row(), 0.0, 1.0, test_random_engine());

  using T1_1 = monolish::matrix::Dense<T>;
  using T1_2_ = monolish::vector<T>;
  using T1_2 = monolish::view_Dense<T1_2_, T>;
  using T1_3_ = monolish::matrix::Dense<T>;
  using T1_3 = monolish::view_Dense<T1_3_, T>;
  using T1_4_ = monolish::tensor::tensor_Dense<T>;
  using T1_4 = monolish::view_Dense<T1_4_, T>;
  T1_1 x1(M, N, 0.0, 1.0);
  T1_2_ x2_(2 * M * N, 0.0, 1.0);
  T1_2 x2(x2_, M / 2, M, N);
  T1_3_ x3_(2 * M * N, 1, 0.0, 1.0);
  T1_3 x3(x3_, M / 2, M, N);
  T1_4_ x4_({2 * M * N, 1, 1}, 0.0, 1.0);
  T1_4 x4(x4_, M / 2, M, N);

  auto As = std::make_tuple(x1, x2, x3, x4);

  using T2_1 = monolish::vector<T>;
  using T2_2 = monolish::view1D<T1_2_, T>;
  using T2_3 = monolish::view1D<T1_3_, T>;
  using T2_4 = monolish::view1D<T1_4_, T>;
  T2_1 y1(M, 0.0, 1.0);
  T1_2_ y2_(2 * M, 0.0, 1.0);
  T2_2 y2(y2_, M / 2, M);
  T1_3_ y3_(2 * M, 1, 0.0, 1.0);
  T2_3 y3(y3_, M / 2, M);
  T1_4_ y4_({2 * M, 1, 1}, 0.0, 1.0);
  T2_4 y4(y4_, M / 2, M);

  auto Bs = std::make_tuple(y1, y2, y3, y4);

  T1_1 z1(M, N, 0.0, 1.0);
  T1_2_ z2_(2 * M * N, 0.0, 1.0);
  T1_2 z2(z2_, M / 2, M, N);
  T1_3_ z3_(2 * M * N, 1, 0.0, 1.0);
  T1_3 z3(z3_, M / 2, M, N);
  T1_4_ z4_({2 * M * N, 1, 1}, 0.0, 1.0);
  T1_4 z4(z4_, M / 2, M, N);

  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_adds_col_view_core1(M, N, tol, AA, xx, CC, As, Bs, Cs);
}
