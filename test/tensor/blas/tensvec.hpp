#include "../../test_utils.hpp"

template <typename T, typename VEC>
void ans_tensvec(monolish::tensor::tensor_Dense<T> &A, VEC &mx,
                 monolish::tensor::tensor_Dense<T> &C) {

  if (A.get_shape()[2] != mx.size()) {
    std::runtime_error("A.col != x.size");
  }
  if (A.get_shape()[0] != C.get_shape()[0] ||
      A.get_shape()[1] != C.get_shape()[1]) {
    std::runtime_error("A.row != C.row");
  }

  T *x = mx.begin();
  T *y = C.begin();
  int M = A.get_shape()[0] * A.get_shape()[1];
  int N = A.get_shape()[2];

  for (int i = 0; i < C.get_nnz(); i++)
    y[i] = 0;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      y[i] += A.begin()[N * i + j] * x[j];
    }
  }
}

template <typename T, typename TENS_A, typename VEC, typename TENS_B>
bool test_send_tensvec_core(const size_t M, const size_t N, const size_t L,
                            double tol, monolish::tensor::tensor_Dense<T> &AA,
                            monolish::vector<T> &xx,
                            monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                            VEC &x, TENS_B &C) {
  ans_tensvec(AA, xx, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, x, C);
  monolish::blas::tensvec(A, x, C);
  monolish::util::recv(A, x, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type() + "+" + x.type() + "=" + C.type(),
                      resultC.begin(), ansC.begin(), resultC.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_send_tensvec(const size_t M, const size_t N, const size_t L,
                       double tol) {

  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);
  monolish::tensor::tensor_COO<T> seedC = get_random_structure_tensor<T>(M, N);

  MAT A(seedA); // M*N tensor
  MAT C(seedC); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  monolish::vector<T> vec(L, 0.0, 1.0, test_random_engine());

  return test_send_tensvec_core(M, N, L, tol, AA, vec, CC, A, vec, C);
}

template <typename MAT, typename T>
bool test_send_tensvec_crs(const size_t M, const size_t N, const size_t L,
                           double tol) {

  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);
  monolish::tensor::tensor_COO<T> seedC = get_random_structure_tensor<T>(M, N);

  MAT A(seedA);                               // M*N tensor
  monolish::tensor::tensor_Dense<T> C(seedC); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  monolish::vector<T> vec(L, 0.0, 1.0, test_random_engine());

  return test_send_tensvec_core(M, N, L, tol, AA, vec, CC, A, vec, C);
}

template <typename T, typename TENS_A, typename VEC, std::size_t K = 0,
          typename... Tp>
inline typename std::enable_if<K == sizeof...(Tp), bool>::type
test_send_tensvec_view_core3(const size_t M, const size_t N, const size_t L,
                             double tol, monolish::tensor::tensor_Dense<T> &AA,
                             monolish::vector<T> &xx,
                             monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                             VEC &x, std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, typename TENS_A, typename VEC, std::size_t K = 0,
          typename... Tp>
    inline typename std::enable_if <
    K<sizeof...(Tp), bool>::type test_send_tensvec_view_core3(
        const size_t M, const size_t N, const size_t L, double tol,
        monolish::tensor::tensor_Dense<T> &AA, monolish::vector<T> &xx,
        monolish::tensor::tensor_Dense<T> &CC, TENS_A &A, VEC &x,
        std::tuple<Tp...> &Cs) {
  A = AA;
  x = xx;
  if (!test_send_tensvec_core(M, N, L, tol, AA, xx, CC, A, x,
                              std::get<K>(Cs))) {
    return false;
  }
  return test_send_tensvec_view_core3<T, TENS_A, VEC, K + 1, Tp...>(
      M, N, L, tol, AA, xx, CC, A, x, Cs);
}

template <typename T, typename TENS_A, std::size_t J = 0, typename TENSES_C,
          typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_send_tensvec_view_core2(const size_t M, const size_t N, const size_t L,
                             double tol, monolish::tensor::tensor_Dense<T> &AA,
                             monolish::vector<T> &xx,
                             monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                             std::tuple<Tq...> &xs, TENSES_C &Cs) {
  return true;
}

template <typename T, typename TENS_A, std::size_t J = 0, typename TENSES_C,
          typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type test_send_tensvec_view_core2(
        const size_t M, const size_t N, const size_t L, double tol,
        monolish::tensor::tensor_Dense<T> &AA, monolish::vector<T> &xx,
        monolish::tensor::tensor_Dense<T> &CC, TENS_A &A, std::tuple<Tq...> &xs,
        TENSES_C &Cs) {
  if (!test_send_tensvec_view_core3(M, N, L, tol, AA, xx, CC, A,
                                    std::get<J>(xs), Cs)) {
    return false;
  }
  return test_send_tensvec_view_core2<T, TENS_A, J + 1, TENSES_C, Tq...>(
      M, N, L, tol, AA, xx, CC, A, xs, Cs);
}

template <typename T, std::size_t I = 0, typename VECS, typename TENSES_C,
          typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_send_tensvec_view_core1(const size_t M, const size_t N, const size_t L,
                             double tol, monolish::tensor::tensor_Dense<T> &AA,
                             monolish::vector<T> &xx,
                             monolish::tensor::tensor_Dense<T> &CC,
                             std::tuple<Tp...> &As, VECS &xs, TENSES_C &Cs) {
  return true;
}

template <typename T, std::size_t I = 0, typename VECS, typename TENSES_C,
          typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type test_send_tensvec_view_core1(
        const size_t M, const size_t N, const size_t L, double tol,
        monolish::tensor::tensor_Dense<T> &AA, monolish::vector<T> &xx,
        monolish::tensor::tensor_Dense<T> &CC, std::tuple<Tp...> &As, VECS &xs,
        TENSES_C &Cs) {
  if (!test_send_tensvec_view_core2(M, N, L, tol, AA, xx, CC, std::get<I>(As),
                                    xs, Cs)) {
    return false;
  }
  return test_send_tensvec_view_core1<T, I + 1, VECS, TENSES_C, Tp...>(
      M, N, L, tol, AA, xx, CC, As, xs, Cs);
}

template <typename T>
bool test_send_tensvec_view(const size_t M, const size_t N, const size_t L,
                            double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);
  monolish::tensor::tensor_COO<T> seedC = get_random_structure_tensor<T>(M, N);
  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedC);
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
  T1_2 x2(x2_, L / 2, {M, N, L});
  T1_3_ x3_(2 * M * N * L, 1, 0.0, 1.0);
  T1_3 x3(x3_, L / 2, {M, N, L});
  T1_4_ x4_({2 * M * N * L, 1, 1}, 0.0, 1.0);
  T1_4 x4(x4_, L / 2, {M, N, L});

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

  T1_1 z1({M, N}, 0.0, 1.0);
  T1_2_ z2_(2 * M * N, 0.0, 1.0);
  T1_2 z2(z2_, N / 2, {M, N});
  T1_3_ z3_(2 * M * N, 1, 0.0, 1.0);
  T1_3 z3(z3_, N / 2, {M, N});
  T1_4_ z4_({2 * M * N, 1, 1}, 0.0, 1.0);
  T1_4 z4(z4_, N / 2, {M, N});

  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_send_tensvec_view_core1(M, N, L, tol, AA, xx, CC, As, Bs, Cs);
}

template <typename T, typename TENS_A, typename VEC, typename TENS_B>
bool test_tensvec_core(const size_t M, const size_t N, const size_t L,
                       double tol, monolish::tensor::tensor_Dense<T> &AA,
                       monolish::vector<T> &xx,
                       monolish::tensor::tensor_Dense<T> &CC, TENS_A &A, VEC &x,
                       TENS_B &C) {
  ans_tensvec(AA, xx, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::blas::tensvec(A, x, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type() + "+" + x.type() + "=" + C.type(),
                      resultC.begin(), ansC.begin(), resultC.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_tensvec(const size_t M, const size_t N, const size_t L, double tol) {

  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);
  monolish::tensor::tensor_COO<T> seedC = get_random_structure_tensor<T>(M, N);

  MAT A(seedA); // M*N tensor
  MAT C(seedC); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  monolish::vector<T> vec(L, 0.0, 1.0, test_random_engine());

  return test_tensvec_core(M, N, L, tol, AA, vec, CC, A, vec, C);
}

template <typename MAT, typename T>
bool test_tensvec_crs(const size_t M, const size_t N, const size_t L,
                      double tol) {

  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);
  monolish::tensor::tensor_COO<T> seedC = get_random_structure_tensor<T>(M, N);

  MAT A(seedA);                               // M*N tensor
  monolish::tensor::tensor_Dense<T> C(seedC); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  monolish::vector<T> vec(L, 0.0, 1.0, test_random_engine());

  return test_tensvec_core(M, N, L, tol, AA, vec, CC, A, vec, C);
}

template <typename T, typename TENS_A, typename VEC, std::size_t K = 0,
          typename... Tp>
inline typename std::enable_if<K == sizeof...(Tp), bool>::type
test_tensvec_view_core3(const size_t M, const size_t N, const size_t L,
                        double tol, monolish::tensor::tensor_Dense<T> &AA,
                        monolish::vector<T> &xx,
                        monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                        VEC &x, std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, typename TENS_A, typename VEC, std::size_t K = 0,
          typename... Tp>
    inline typename std::enable_if <
    K<sizeof...(Tp), bool>::type
    test_tensvec_view_core3(const size_t M, const size_t N, const size_t L,
                            double tol, monolish::tensor::tensor_Dense<T> &AA,
                            monolish::vector<T> &xx,
                            monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                            VEC &x, std::tuple<Tp...> &Cs) {
  A = AA;
  x = xx;
  if (!test_tensvec_core(M, N, L, tol, AA, xx, CC, A, x, std::get<K>(Cs))) {
    return false;
  }
  return test_tensvec_view_core3<T, TENS_A, VEC, K + 1, Tp...>(
      M, N, L, tol, AA, xx, CC, A, x, Cs);
}

template <typename T, typename TENS_A, std::size_t J = 0, typename TENSES_C,
          typename... Tq>
inline typename std::enable_if<J == sizeof...(Tq), bool>::type
test_tensvec_view_core2(const size_t M, const size_t N, const size_t L,
                        double tol, monolish::tensor::tensor_Dense<T> &AA,
                        monolish::vector<T> &xx,
                        monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                        std::tuple<Tq...> &xs, TENSES_C &Cs) {
  return true;
}

template <typename T, typename TENS_A, std::size_t J = 0, typename TENSES_C,
          typename... Tq>
    inline typename std::enable_if <
    J<sizeof...(Tq), bool>::type
    test_tensvec_view_core2(const size_t M, const size_t N, const size_t L,
                            double tol, monolish::tensor::tensor_Dense<T> &AA,
                            monolish::vector<T> &xx,
                            monolish::tensor::tensor_Dense<T> &CC, TENS_A &A,
                            std::tuple<Tq...> &xs, TENSES_C &Cs) {
  if (!test_tensvec_view_core3(M, N, L, tol, AA, xx, CC, A, std::get<J>(xs),
                               Cs)) {
    return false;
  }
  return test_tensvec_view_core2<T, TENS_A, J + 1, TENSES_C, Tq...>(
      M, N, L, tol, AA, xx, CC, A, xs, Cs);
}

template <typename T, std::size_t I = 0, typename VECS, typename TENSES_C,
          typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_tensvec_view_core1(const size_t M, const size_t N, const size_t L,
                        double tol, monolish::tensor::tensor_Dense<T> &AA,
                        monolish::vector<T> &xx,
                        monolish::tensor::tensor_Dense<T> &CC,
                        std::tuple<Tp...> &As, VECS &xs, TENSES_C &Cs) {
  return true;
}

template <typename T, std::size_t I = 0, typename VECS, typename TENSES_C,
          typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type
    test_tensvec_view_core1(const size_t M, const size_t N, const size_t L,
                            double tol, monolish::tensor::tensor_Dense<T> &AA,
                            monolish::vector<T> &xx,
                            monolish::tensor::tensor_Dense<T> &CC,
                            std::tuple<Tp...> &As, VECS &xs, TENSES_C &Cs) {
  if (!test_tensvec_view_core2(M, N, L, tol, AA, xx, CC, std::get<I>(As), xs,
                               Cs)) {
    return false;
  }
  return test_tensvec_view_core1<T, I + 1, VECS, TENSES_C, Tp...>(
      M, N, L, tol, AA, xx, CC, As, xs, Cs);
}

template <typename T>
bool test_tensvec_view(const size_t M, const size_t N, const size_t L,
                       double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);
  monolish::tensor::tensor_COO<T> seedC = get_random_structure_tensor<T>(M, N);
  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedC);
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
  T1_2 x2(x2_, L / 2, {M, N, L});
  T1_3_ x3_(2 * M * N * L, 1, 0.0, 1.0);
  T1_3 x3(x3_, L / 2, {M, N, L});
  T1_4_ x4_({2 * M * N * L, 1, 1}, 0.0, 1.0);
  T1_4 x4(x4_, L / 2, {M, N, L});

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

  T1_1 z1({M, N}, 0.0, 1.0);
  T1_2_ z2_(2 * M * N, 0.0, 1.0);
  T1_2 z2(z2_, N / 2, {M, N});
  T1_3_ z3_(2 * M * N, 1, 0.0, 1.0);
  T1_3 z3(z3_, N / 2, {M, N});
  T1_4_ z4_({2 * M * N, 1, 1}, 0.0, 1.0);
  T1_4 z4(z4_, N / 2, {M, N});

  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_tensvec_view_core1(M, N, L, tol, AA, xx, CC, As, Bs, Cs);
}
