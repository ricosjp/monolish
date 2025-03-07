#include "../../test_utils.hpp"

template <typename T>
void ans_tensadd(const monolish::tensor::tensor_Dense<T> &A,
                 const monolish::tensor::tensor_Dense<T> &B,
                 monolish::tensor::tensor_Dense<T> &C) {

  if (!monolish::util::is_same_size(A, B)) {
    std::runtime_error("A.shape != B.shape");
  }

  for (int i = 0; i < A.get_nnz(); i++) {
    C.begin()[i] = A.begin()[i] + B.begin()[i];
  }
}

template <typename T, typename MAT_A, typename MAT_B, typename MAT_C>
bool test_send_tensadd_core(const size_t M, const size_t N, const size_t L,
                            double tol, monolish::tensor::tensor_Dense<T> &AA,
                            monolish::tensor::tensor_Dense<T> &BB,
                            monolish::tensor::tensor_Dense<T> &CC, MAT_A &A,
                            MAT_B &B, MAT_C &C) {
  ans_tensadd(AA, BB, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::tensadd(A, B, C);
  monolish::util::recv(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_send_tensadd(const size_t M, const size_t N, const size_t L,
                       double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);

  MAT_A A(seedA);
  MAT_B B(seedA);
  MAT_C C(seedA);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> BB(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedA);

  return test_send_tensadd_core(M, N, L, tol, AA, BB, CC, A, B, C);
}

template <typename T, typename MAT_A, typename MAT_B, std::size_t K = 0,
          typename... Tp>
inline typename std::enable_if<K == sizeof...(Tp), bool>::type
test_send_tensadd_view_core3(const size_t M, const size_t N, const size_t L,
                             double tol, monolish::tensor::tensor_Dense<T> &AA,
                             monolish::tensor::tensor_Dense<T> &BB,
                             monolish::tensor::tensor_Dense<T> &CC, MAT_A &A,
                             MAT_B &B, std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, typename MAT_A, typename MAT_B, std::size_t K = 0,
          typename... Tp>
    inline typename std::enable_if <
    K<sizeof...(Tp), bool>::type
    test_send_tensadd_view_core3(const size_t M, const size_t N, const size_t L,
                                 double tol,
                                 monolish::tensor::tensor_Dense<T> &AA,
                                 monolish::tensor::tensor_Dense<T> &BB,
                                 monolish::tensor::tensor_Dense<T> &CC,
                                 MAT_A &A, MAT_B &B, std::tuple<Tp...> &Cs) {
  A = AA;
  B = BB;
  if (!test_send_tensadd_core(M, N, L, tol, AA, BB, CC, A, B,
                              std::get<K>(Cs))) {
    return false;
  }
  return test_send_tensadd_view_core3<T, MAT_A, MAT_B, K + 1, Tp...>(
      M, N, L, tol, AA, BB, CC, A, B, Cs);
}

template <typename T, typename MAT_A, std::size_t J = 0, typename... Tp>
inline typename std::enable_if<J == sizeof...(Tp), bool>::type
test_send_tensadd_view_core2(const size_t M, const size_t N, const size_t L,
                             double tol, monolish::tensor::tensor_Dense<T> &AA,
                             monolish::tensor::tensor_Dense<T> &BB,
                             monolish::tensor::tensor_Dense<T> &CC, MAT_A &A,
                             std::tuple<Tp...> &Bs, std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, typename MAT_A, std::size_t J = 0, typename... Tp>
    inline typename std::enable_if <
    J<sizeof...(Tp), bool>::type test_send_tensadd_view_core2(
        const size_t M, const size_t N, const size_t L, double tol,
        monolish::tensor::tensor_Dense<T> &AA,
        monolish::tensor::tensor_Dense<T> &BB,
        monolish::tensor::tensor_Dense<T> &CC, MAT_A &A, std::tuple<Tp...> &Bs,
        std::tuple<Tp...> &Cs) {
  if (!test_send_tensadd_view_core3(M, N, L, tol, AA, BB, CC, A,
                                    std::get<J>(Bs), Cs)) {
    return false;
  }
  return test_send_tensadd_view_core2<T, MAT_A, J + 1, Tp...>(
      M, N, L, tol, AA, BB, CC, A, Bs, Cs);
}

template <typename T, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_send_tensadd_view_core1(const size_t M, const size_t N, const size_t L,
                             double tol, monolish::tensor::tensor_Dense<T> &AA,
                             monolish::tensor::tensor_Dense<T> &BB,
                             monolish::tensor::tensor_Dense<T> &CC,
                             std::tuple<Tp...> &As, std::tuple<Tp...> &Bs,
                             std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type test_send_tensadd_view_core1(
        const size_t M, const size_t N, const size_t L, double tol,
        monolish::tensor::tensor_Dense<T> &AA,
        monolish::tensor::tensor_Dense<T> &BB,
        monolish::tensor::tensor_Dense<T> &CC, std::tuple<Tp...> &As,
        std::tuple<Tp...> &Bs, std::tuple<Tp...> &Cs) {
  if (!test_send_tensadd_view_core2(M, N, L, tol, AA, BB, CC, std::get<I>(As),
                                    Bs, Cs)) {
    return false;
  }
  return test_send_tensadd_view_core1<T, I + 1, Tp...>(M, N, L, tol, AA, BB, CC,
                                                       As, Bs, Cs);
}

template <typename T>
bool test_send_tensadd_view(const size_t M, const size_t N, const size_t L,
                            double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> BB(seedA);
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

  T1 z1({M, N, L});
  T2_ z2_(2 * M * N * L, 0.0, 1.0);
  T2 z2(z2_, L / 2, {M, N, L});
  T3_ z3_(2 * M * N * L, 1, 0.0, 1.0);
  T3 z3(z3_, L / 2, {M, N, L});
  T4_ z4_({2 * M * N * L, 1, 1}, 0.0, 1.0);
  T4 z4(z4_, L / 2, {M, N, L});

  auto As = std::make_tuple(x1, x2, x3, x4);
  auto Bs = std::make_tuple(y1, y2, y3, y4);
  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_send_tensadd_view_core1(M, N, L, tol, AA, BB, CC, As, Bs, Cs);
}

template <typename T, typename MAT_A, typename MAT_B, typename MAT_C>
bool test_tensadd_core(const size_t M, const size_t N, const size_t L,
                       double tol, monolish::tensor::tensor_Dense<T> &AA,
                       monolish::tensor::tensor_Dense<T> &BB,
                       monolish::tensor::tensor_Dense<T> &CC, MAT_A &A,
                       MAT_B &B, MAT_C &C) {
  ans_tensadd(AA, BB, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::blas::tensadd(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_tensadd(const size_t M, const size_t N, const size_t L, double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);

  MAT_A A(seedA);
  MAT_B B(seedA);
  MAT_C C(seedA);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> BB(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedA);

  return test_tensadd_core(M, N, L, tol, AA, BB, CC, A, B, C);
}

template <typename T, typename MAT_A, typename MAT_B, std::size_t K = 0,
          typename... Tp>
inline typename std::enable_if<K == sizeof...(Tp), bool>::type
test_tensadd_view_core3(const size_t M, const size_t N, const size_t L,
                        double tol, monolish::tensor::tensor_Dense<T> &AA,
                        monolish::tensor::tensor_Dense<T> &BB,
                        monolish::tensor::tensor_Dense<T> &CC, MAT_A &A,
                        MAT_B &B, std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, typename MAT_A, typename MAT_B, std::size_t K = 0,
          typename... Tp>
    inline typename std::enable_if <
    K<sizeof...(Tp), bool>::type
    test_tensadd_view_core3(const size_t M, const size_t N, const size_t L,
                            double tol, monolish::tensor::tensor_Dense<T> &AA,
                            monolish::tensor::tensor_Dense<T> &BB,
                            monolish::tensor::tensor_Dense<T> &CC, MAT_A &A,
                            MAT_B &B, std::tuple<Tp...> &Cs) {
  A = AA;
  B = BB;
  if (!test_tensadd_core(M, N, L, tol, AA, BB, CC, A, B, std::get<K>(Cs))) {
    return false;
  }
  return test_tensadd_view_core3<T, MAT_A, MAT_B, K + 1, Tp...>(
      M, N, L, tol, AA, BB, CC, A, B, Cs);
}

template <typename T, typename MAT_A, std::size_t J = 0, typename... Tp>
inline typename std::enable_if<J == sizeof...(Tp), bool>::type
test_tensadd_view_core2(const size_t M, const size_t N, const size_t L,
                        double tol, monolish::tensor::tensor_Dense<T> &AA,
                        monolish::tensor::tensor_Dense<T> &BB,
                        monolish::tensor::tensor_Dense<T> &CC, MAT_A &A,
                        std::tuple<Tp...> &Bs, std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, typename MAT_A, std::size_t J = 0, typename... Tp>
    inline typename std::enable_if <
    J<sizeof...(Tp), bool>::type
    test_tensadd_view_core2(const size_t M, const size_t N, const size_t L,
                            double tol, monolish::tensor::tensor_Dense<T> &AA,
                            monolish::tensor::tensor_Dense<T> &BB,
                            monolish::tensor::tensor_Dense<T> &CC, MAT_A &A,
                            std::tuple<Tp...> &Bs, std::tuple<Tp...> &Cs) {
  if (!test_tensadd_view_core3(M, N, L, tol, AA, BB, CC, A, std::get<J>(Bs),
                               Cs)) {
    return false;
  }
  return test_tensadd_view_core2<T, MAT_A, J + 1, Tp...>(M, N, L, tol, AA, BB,
                                                         CC, A, Bs, Cs);
}

template <typename T, std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), bool>::type
test_tensadd_view_core1(const size_t M, const size_t N, const size_t L,
                        double tol, monolish::tensor::tensor_Dense<T> &AA,
                        monolish::tensor::tensor_Dense<T> &BB,
                        monolish::tensor::tensor_Dense<T> &CC,
                        std::tuple<Tp...> &As, std::tuple<Tp...> &Bs,
                        std::tuple<Tp...> &Cs) {
  return true;
}

template <typename T, std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), bool>::type
    test_tensadd_view_core1(const size_t M, const size_t N, const size_t L,
                            double tol, monolish::tensor::tensor_Dense<T> &AA,
                            monolish::tensor::tensor_Dense<T> &BB,
                            monolish::tensor::tensor_Dense<T> &CC,
                            std::tuple<Tp...> &As, std::tuple<Tp...> &Bs,
                            std::tuple<Tp...> &Cs) {
  if (!test_tensadd_view_core2(M, N, L, tol, AA, BB, CC, std::get<I>(As), Bs,
                               Cs)) {
    return false;
  }
  return test_tensadd_view_core1<T, I + 1, Tp...>(M, N, L, tol, AA, BB, CC, As,
                                                  Bs, Cs);
}

template <typename T>
bool test_tensadd_view(const size_t M, const size_t N, const size_t L,
                       double tol) {
  monolish::tensor::tensor_COO<T> seedA =
      get_random_structure_tensor<T>(M, N, L);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> BB(seedA);
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

  T1 z1({M, N, L});
  T2_ z2_(2 * M * N * L, 0.0, 1.0);
  T2 z2(z2_, L / 2, {M, N, L});
  T3_ z3_(2 * M * N * L, 1, 0.0, 1.0);
  T3 z3(z3_, L / 2, {M, N, L});
  T4_ z4_({2 * M * N * L, 1, 1}, 0.0, 1.0);
  T4 z4(z4_, L / 2, {M, N, L});

  auto As = std::make_tuple(x1, x2, x3, x4);
  auto Bs = std::make_tuple(y1, y2, y3, y4);
  auto Cs = std::make_tuple(z1, z2, z3, z4);

  return test_tensadd_view_core1(M, N, L, tol, AA, BB, CC, As, Bs, Cs);
}
