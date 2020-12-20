#include "../../test_utils.hpp"
#include "monolish_blas.hpp"

template <typename T> void ans_sign(monolish::matrix::Dense<T> &A) {

  for (size_t i = 0; i < A.get_nnz(); i++) {
    A.val[i] = -1 * A.val[i];
  }
}

template <typename MAT, typename T>
bool test_send_msign(const size_t M, const size_t N, double tol) {
  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);
  ans_sign(AA);

  A.send();
  monolish::vml::sign(A, A);
  A.recv();
  monolish::matrix::Dense<T> resultA(A);

  return ans_check<T>(__func__, A.type(), resultA.val.data(), AA.val.data(),
                      AA.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_msign(const size_t M, const size_t N, double tol) {
  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);
  ans_sign(AA);

  monolish::vml::sign(A, A);
  monolish::matrix::Dense<T> resultA(A);

  return ans_check<T>(__func__, A.type(), resultA.val.data(), AA.val.data(),
                      AA.get_nnz(), tol);
}
