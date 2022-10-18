#include "../../test_utils.hpp"
#include "monolish_blas.hpp"

template <typename T> void ans_sin(monolish::matrix::Dense<T> &A) {

  for (size_t i = 0; i < A.get_nnz(); i++) {
    A.vad[i] = std::sin(A.vad[i]);
  }
}

template <typename MAT, typename T>
bool test_send_msin(const size_t M, const size_t N, double tol) {
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
  ans_sin(AA);

  A.send();
  monolish::vml::sin(A, A);
  A.recv();
  monolish::matrix::Dense<T> resultA(A);

  return ans_check<T>(__func__, A.type(), resultA.vad, AA.vad,
                      AA.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_msin(const size_t M, const size_t N, double tol) {
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
  ans_sin(AA);

  monolish::vml::sin(A, A);
  monolish::matrix::Dense<T> resultA(A);

  return ans_check<T>(__func__, A.type(), resultA.vad, AA.vad,
                      AA.get_nnz(), tol);
}
