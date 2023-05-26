#include "../../test_utils.hpp"

template <typename T>
void ans_matvec_T(monolish::matrix::Dense<T> &A, monolish::vector<T> &mx,
                  monolish::vector<T> &my) {

  if (A.get_row() != mx.size()) {
    std::runtime_error("A.col != x.size");
  }
  if (A.get_col() != my.size()) {
    std::runtime_error("A.row != y.size");
  }

  T *x = mx.begin();
  T *y = my.begin();
  int M = A.get_row();
  int N = A.get_col();

  for (int i = 0; i < my.size(); i++)
    y[i] = 0;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      y[j] += A.begin()[N * i + j] * x[i];
    }
  }
}

template <typename T, typename MAT, typename VEC_A, typename VEC_B>
bool test_send_matvec_T_core(const size_t M, const size_t N, double tol,
                             monolish::matrix::Dense<T> &AA,
                             monolish::vector<T> &xx, MAT &A, VEC_A &x,
                             VEC_B &y) {
  monolish::vector<T> ansy(A.get_col());
  ans_matvec_T(AA, xx, ansy);

  monolish::util::send(A, x, y);
  monolish::blas::matvec_T(A, x, y);
  monolish::util::recv(A, x, y);

  return ans_check<T>(__func__, A.type() + "+" + x.type() + "=" + y.type(),
                      y.begin(), ansy.begin(), y.size(), tol);
}

template <typename MAT, typename T>
bool test_send_matvec_T(const size_t M, const size_t N, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);

  monolish::matrix::Dense<T> AA(seedA);
  MAT A(seedA); // M*N matrix

  monolish::vector<T> x(A.get_row(), 0.0, 1.0, test_random_engine());
  monolish::vector<T> y(A.get_col(), 0.0, 1.0, test_random_engine());

  monolish::vector<T> xx(A.get_row(), 0.0, 1.0, test_random_engine());
  xx = x;

  return test_send_matvec_T_core(M, N, tol, AA, xx, A, x, y);
}

template <typename T, typename MAT, typename VEC_A, std::size_t K = 0,
          typename... Tq>
inline typename std::enable_if<K == sizeof...(Tq), bool>::type
test_send_matvec_T_view_dense_core3(const size_t M, const size_t N, double tol,
                                    monolish::matrix::Dense<T> &AA,
                                    monolish::vector<T> &xx, MAT &A, VEC_A &x,
                                    std::tuple<Tq...> &Cs) {
  return true;
}

template <typename T, typename MAT, typename VEC_A, std::size_t K = 0,
          typename... Tq>
    inline typename std::enable_if <
    K<sizeof...(Tq), bool>::type test_send_matvec_T_view_dense_core3(
        const size_t M, const size_t N, double tol,
        monolish::matrix::Dense<T> &AA, monolish::vector<T> &xx, MAT &A,
        VEC_A &x, std::tuple<Tq...> &Cs) {
  A = AA;
  x = xx;
  if (!test_send_matvec_T_core(M, N, tol, AA, xx, A, x, std::get<K>(Cs))) {
    return false;
  }
  return test_send_matvec_T_view_dense_core3<T, MAT, VEC_A, K + 1, Tq...>(
      M, N, tol, AA, xx, A, x, Cs);
}

template <typename T, typename MAT, std::size_t J = 0, typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_send_matvec_T_view_dense_core2(const size_t M, const size_t N, double tol,
                                    monolish::matrix::Dense<T> &AA,
                                    monolish::vector<T> &xx, MAT &A,
                                    std::tuple<Tq...> &Bs,
                                    std::tuple<Tq...> &Cs) {
  return true;
}

template <typename T, typename MAT, std::size_t J = 0, typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type test_send_matvec_T_view_dense_core2(
        const size_t M, const size_t N, double tol,
        monolish::matrix::Dense<T> &AA, monolish::vector<T> &xx, MAT &A,
        std::tuple<Tq...> &Bs, std::tuple<Tq...> &Cs) {
  if (!test_send_matvec_T_view_dense_core3(M, N, tol, AA, xx, A,
                                           std::get<J>(Bs), Cs)) {
    return false;
  }
  return test_send_matvec_T_view_dense_core2<T, MAT, J + 1, Tq...>(
      M, N, tol, AA, xx, A, Bs, Cs);
}

template <typename T, std::size_t I = 0, typename VECS, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_send_matvec_T_view_dense_core1(const size_t M, const size_t N, double tol,
                                    monolish::matrix::Dense<T> &AA,
                                    monolish::vector<T> &xx,
                                    std::tuple<Tp...> &As, VECS &Bs, VECS &Cs) {
  return true;
}

template <typename T, std::size_t I = 0, typename VECS, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type test_send_matvec_T_view_dense_core1(
        const size_t M, const size_t N, double tol,
        monolish::matrix::Dense<T> &AA, monolish::vector<T> &xx,
        std::tuple<Tp...> &As, VECS &Bs, VECS &Cs) {
  if (!test_send_matvec_T_view_dense_core2(M, N, tol, AA, xx, std::get<I>(As),
                                           Bs, Cs)) {
    return false;
  }
  return test_send_matvec_T_view_dense_core1<T, I + 1, VECS, Tp...>(
      M, N, tol, AA, xx, As, Bs, Cs);
}

template <typename T>
bool test_send_matvec_T_view_dense(const size_t M, const size_t N, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);
  monolish::matrix::Dense<T> AA(seedA);
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

  T2_1 z1(N, 0.0, 1.0);
  T1_2_ z2_(2 * N, 0.0, 1.0);
  T2_2 z2(z2_, N / 2, N);
  T1_3_ z3_(2 * N, 1, 0.0, 1.0);
  T2_3 z3(z3_, N / 2, N);
  T1_4_ z4_({2 * N, 1, 1}, 0.0, 1.0);
  T2_4 z4(z4_, N / 2, N);

  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_send_matvec_T_view_dense_core1(M, N, tol, AA, xx, As, Bs, Cs);
}

template <typename T, typename VEC_A, std::size_t K = 0, typename... Tq>
inline typename std::enable_if<K == sizeof...(Tq), bool>::type
test_send_matvec_T_view_crs_core2(const size_t M, const size_t N, double tol,
                                  monolish::matrix::COO<T> &seedA,
                                  monolish::vector<T> &xx, VEC_A &x,
                                  std::tuple<Tq...> &Cs) {
  return true;
}

template <typename T, typename VEC_A, std::size_t K = 0, typename... Tq>
    inline typename std::enable_if <
    K<sizeof...(Tq), bool>::type test_send_matvec_T_view_crs_core2(
        const size_t M, const size_t N, double tol,
        monolish::matrix::COO<T> &seedA, monolish::vector<T> &xx, VEC_A &x,
        std::tuple<Tq...> &Cs) {
  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::CRS<T> A(seedA);
  x = xx;
  if (!test_send_matvec_T_core(M, N, tol, AA, xx, A, x, std::get<K>(Cs))) {
    return false;
  }
  return test_send_matvec_T_view_crs_core2<T, VEC_A, K + 1, Tq...>(
      M, N, tol, seedA, xx, x, Cs);
}

template <typename T, std::size_t J = 0, typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_send_matvec_T_view_crs_core1(const size_t M, const size_t N, double tol,
                                  monolish::matrix::COO<T> &seedA,
                                  monolish::vector<T> &xx,
                                  std::tuple<Tq...> &Bs,
                                  std::tuple<Tq...> &Cs) {
  return true;
}

template <typename T, std::size_t J = 0, typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type test_send_matvec_T_view_crs_core1(
        const size_t M, const size_t N, double tol,
        monolish::matrix::COO<T> &seedA, monolish::vector<T> &xx,
        std::tuple<Tq...> &Bs, std::tuple<Tq...> &Cs) {
  if (!test_send_matvec_T_view_crs_core2(M, N, tol, seedA, xx, std::get<J>(Bs),
                                         Cs)) {
    return false;
  }
  return test_send_matvec_T_view_crs_core1<T, J + 1, Tq...>(M, N, tol, seedA,
                                                            xx, Bs, Cs);
}

template <typename T>
bool test_send_matvec_T_view_crs(const size_t M, const size_t N, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);
  monolish::vector<T> xx(seedA.get_row(), 0.0, 1.0, test_random_engine());

  using T1_2_ = monolish::vector<T>;
  using T1_3_ = monolish::matrix::Dense<T>;
  using T1_4_ = monolish::tensor::tensor_Dense<T>;

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

  T2_1 z1(N, 0.0, 1.0);
  T1_2_ z2_(2 * N, 0.0, 1.0);
  T2_2 z2(z2_, N / 2, N);
  T1_3_ z3_(2 * N, 1, 0.0, 1.0);
  T2_3 z3(z3_, N / 2, N);
  T1_4_ z4_({2 * N, 1, 1}, 0.0, 1.0);
  T2_4 z4(z4_, N / 2, N);

  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_send_matvec_T_view_crs_core1(M, N, tol, seedA, xx, Bs, Cs);
}

template <typename MAT, typename T>
bool test_send_matvec_T_linearoperator(const size_t M, const size_t N,
                                       double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);
  monolish::matrix::CRS<T> A1(seedA);

  monolish::vector<T> x(A1.get_row(), 0.0, 1.0, test_random_engine());
  monolish::vector<T> y(A1.get_col(), 0.0, 1.0, test_random_engine());

  monolish::vector<T> ansy(A1.get_col());
  ansy = y;

  monolish::matrix::Dense<T> AA(seedA);
  ans_matvec_T(AA, x, ansy);

  monolish::util::send(A1, x, y);
  MAT A2(A1); // M*N matrix
  monolish::blas::matvec_T(A2, x, y);
  monolish::util::recv(A1, A2, x, y);

  return ans_check<T>(__func__, A2.type(), y.begin(), ansy.begin(), y.size(),
                      tol);
}

template <typename T, typename MAT, typename VEC_A, typename VEC_B>
bool test_matvec_T_core(const size_t M, const size_t N, double tol,
                        monolish::matrix::Dense<T> &AA, monolish::vector<T> &xx,
                        MAT &A, VEC_A &x, VEC_B &y) {
  monolish::vector<T> ansy(A.get_col());
  ans_matvec_T(AA, xx, ansy);

  monolish::blas::matvec_T(A, x, y);

  return ans_check<T>(__func__, A.type() + "+" + x.type() + "=" + y.type(),
                      y.begin(), ansy.begin(), y.size(), tol);
}

template <typename MAT, typename T>
bool test_matvec_T(const size_t M, const size_t N, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);

  monolish::matrix::Dense<T> AA(seedA);
  MAT A(seedA); // M*N matrix

  monolish::vector<T> x(A.get_row(), 0.0, 1.0, test_random_engine());
  monolish::vector<T> y(A.get_col(), 0.0, 1.0, test_random_engine());

  return test_matvec_T_core(M, N, tol, AA, x, A, x, y);
}

template <typename T, typename MAT, typename VEC_A, std::size_t K = 0,
          typename... Tq>
inline typename std::enable_if<K == sizeof...(Tq), bool>::type
test_matvec_T_view_dense_core3(const size_t M, const size_t N, double tol,
                               monolish::matrix::Dense<T> &AA,
                               monolish::vector<T> &xx, MAT &A, VEC_A &x,
                               std::tuple<Tq...> &Cs) {
  return true;
}

template <typename T, typename MAT, typename VEC_A, std::size_t K = 0,
          typename... Tq>
    inline typename std::enable_if <
    K<sizeof...(Tq), bool>::type
    test_matvec_T_view_dense_core3(const size_t M, const size_t N, double tol,
                                   monolish::matrix::Dense<T> &AA,
                                   monolish::vector<T> &xx, MAT &A, VEC_A &x,
                                   std::tuple<Tq...> &Cs) {
  A = AA;
  x = xx;
  if (!test_matvec_T_core(M, N, tol, AA, xx, A, x, std::get<K>(Cs))) {
    return false;
  }
  return test_matvec_T_view_dense_core3<T, MAT, VEC_A, K + 1, Tq...>(
      M, N, tol, AA, xx, A, x, Cs);
}

template <typename T, typename MAT, std::size_t J = 0, typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_matvec_T_view_dense_core2(const size_t M, const size_t N, double tol,
                               monolish::matrix::Dense<T> &AA,
                               monolish::vector<T> &xx, MAT &A,
                               std::tuple<Tq...> &Bs, std::tuple<Tq...> &Cs) {
  return true;
}

template <typename T, typename MAT, std::size_t J = 0, typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type test_matvec_T_view_dense_core2(
        const size_t M, const size_t N, double tol,
        monolish::matrix::Dense<T> &AA, monolish::vector<T> &xx, MAT &A,
        std::tuple<Tq...> &Bs, std::tuple<Tq...> &Cs) {
  if (!test_matvec_T_view_dense_core3(M, N, tol, AA, xx, A, std::get<J>(Bs),
                                      Cs)) {
    return false;
  }
  return test_matvec_T_view_dense_core2<T, MAT, J + 1, Tq...>(M, N, tol, AA, xx,
                                                              A, Bs, Cs);
}

template <typename T, std::size_t I = 0, typename VECS, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_matvec_T_view_dense_core1(const size_t M, const size_t N, double tol,
                               monolish::matrix::Dense<T> &AA,
                               monolish::vector<T> &xx, std::tuple<Tp...> &As,
                               VECS &Bs, VECS &Cs) {
  return true;
}

template <typename T, std::size_t I = 0, typename VECS, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type
    test_matvec_T_view_dense_core1(const size_t M, const size_t N, double tol,
                                   monolish::matrix::Dense<T> &AA,
                                   monolish::vector<T> &xx,
                                   std::tuple<Tp...> &As, VECS &Bs, VECS &Cs) {
  if (!test_matvec_T_view_dense_core2(M, N, tol, AA, xx, std::get<I>(As), Bs,
                                      Cs)) {
    return false;
  }
  return test_matvec_T_view_dense_core1<T, I + 1, VECS, Tp...>(M, N, tol, AA,
                                                               xx, As, Bs, Cs);
}

template <typename T>
bool test_matvec_T_view_dense(const size_t M, const size_t N, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);
  monolish::matrix::Dense<T> AA(seedA);
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

  T2_1 z1(N, 0.0, 1.0);
  T1_2_ z2_(2 * N, 0.0, 1.0);
  T2_2 z2(z2_, N / 2, N);
  T1_3_ z3_(2 * N, 1, 0.0, 1.0);
  T2_3 z3(z3_, N / 2, N);
  T1_4_ z4_({2 * N, 1, 1}, 0.0, 1.0);
  T2_4 z4(z4_, N / 2, N);

  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_matvec_T_view_dense_core1(M, N, tol, AA, xx, As, Bs, Cs);
}

template <typename T, typename VEC_A, std::size_t K = 0, typename... Tq>
inline typename std::enable_if<K == sizeof...(Tq), bool>::type
test_matvec_T_view_crs_core2(const size_t M, const size_t N, double tol,
                             monolish::matrix::COO<T> &seedA,
                             monolish::vector<T> &xx, VEC_A &x,
                             std::tuple<Tq...> &Cs) {
  return true;
}

template <typename T, typename VEC_A, std::size_t K = 0, typename... Tq>
    inline typename std::enable_if <
    K<sizeof...(Tq), bool>::type
    test_matvec_T_view_crs_core2(const size_t M, const size_t N, double tol,
                                 monolish::matrix::COO<T> &seedA,
                                 monolish::vector<T> &xx, VEC_A &x,
                                 std::tuple<Tq...> &Cs) {
  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::CRS<T> A(seedA);
  x = xx;
  if (!test_matvec_T_core(M, N, tol, AA, xx, A, x, std::get<K>(Cs))) {
    return false;
  }
  return test_matvec_T_view_crs_core2<T, VEC_A, K + 1, Tq...>(M, N, tol, seedA,
                                                              xx, x, Cs);
}

template <typename T, std::size_t J = 0, typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_matvec_T_view_crs_core1(const size_t M, const size_t N, double tol,
                             monolish::matrix::COO<T> &seedA,
                             monolish::vector<T> &xx, std::tuple<Tq...> &Bs,
                             std::tuple<Tq...> &Cs) {
  return true;
}

template <typename T, std::size_t J = 0, typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type
    test_matvec_T_view_crs_core1(const size_t M, const size_t N, double tol,
                                 monolish::matrix::COO<T> &seedA,
                                 monolish::vector<T> &xx, std::tuple<Tq...> &Bs,
                                 std::tuple<Tq...> &Cs) {
  if (!test_matvec_T_view_crs_core2(M, N, tol, seedA, xx, std::get<J>(Bs),
                                    Cs)) {
    return false;
  }
  return test_matvec_T_view_crs_core1<T, J + 1, Tq...>(M, N, tol, seedA, xx, Bs,
                                                       Cs);
}

template <typename T>
bool test_matvec_T_view_crs(const size_t M, const size_t N, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);
  monolish::vector<T> xx(seedA.get_row(), 0.0, 1.0, test_random_engine());

  using T1_2_ = monolish::vector<T>;
  using T1_3_ = monolish::matrix::Dense<T>;
  using T1_4_ = monolish::tensor::tensor_Dense<T>;

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

  T2_1 z1(N, 0.0, 1.0);
  T1_2_ z2_(2 * N, 0.0, 1.0);
  T2_2 z2(z2_, N / 2, N);
  T1_3_ z3_(2 * N, 1, 0.0, 1.0);
  T2_3 z3(z3_, N / 2, N);
  T1_4_ z4_({2 * N, 1, 1}, 0.0, 1.0);
  T2_4 z4(z4_, N / 2, N);

  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_matvec_T_view_crs_core1(M, N, tol, seedA, xx, Bs, Cs);
}

template <typename MAT, typename T>
bool test_matvec_T_linearoperator(const size_t M, const size_t N, double tol) {
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);

  monolish::matrix::CRS<T> A1(seedA);

  monolish::vector<T> x(A1.get_row(), 0.0, 1.0, test_random_engine());
  monolish::vector<T> y(A1.get_col(), 0.0, 1.0, test_random_engine());

  monolish::vector<T> ansy(A1.get_col());
  ansy = y;

  monolish::matrix::Dense<T> AA(seedA);
  ans_matvec_T(AA, x, ansy);

  MAT A2(A1); // M*N matrix
  monolish::blas::matvec_T(A2, x, y);

  return ans_check<T>(__func__, A2.type(), y.begin(), ansy.begin(), y.size(),
                      tol);
}
