#include "../test_utils.hpp"

template <typename MAT, typename T>
bool test_send_transpose(const size_t M, const size_t N, double tol) {

  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }
  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N matrix

  if (A.type() != "COO")
    monolish::util::send(A);

  A.transpose();

  if (A.type() != "COO")
    A.recv();

  if (A.get_row() != N || A.get_col() != M) {
    std::cout << "transpose error, transA.row = " << A.get_row()
              << ", transA.col = " << A.get_col() << std::endl;
    return false;
  }

  if (A.type() != "COO")
    monolish::util::send(A);
  A.transpose();
  if (A.type() != "COO")
    A.recv();

  monolish::matrix::COO<T> ansA(A);
  if (ans_check<T>(__func__, A.type(), seedA.val.data(), ansA.val.data(),
                   ansA.get_nnz(), tol) == false) {
    return false;
  };

  return true;
}

template <typename MAT, typename T>
bool test_transpose(const size_t M, const size_t N, double tol) {

  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }
  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N matrix

  monolish::matrix::Dense<T> ansA(A);

  A.transpose();
  monolish::matrix::Dense<T> ansAT(A);

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      if (ansA.at(i, j) != ansAT.at(j, i)) {
        std::cout << "A(" << i << "," << j << ")=" << ansA.at(i, j) << ", A^T("
                  << j << "," << i << ")=" << ansAT.at(j, i) << std::endl;
        std::cout << "Error!!" << std::endl;
        std::cout << __func__ << "(" << A.type() << ")"
                  << ": fail" << std::endl;
        return false;
      }
    }
  }

  std::cout << __func__ << "(" << A.type() << ")"
            << ": pass" << std::endl;
  return true;
}

template <typename MAT, typename T>
bool test_transpose_elements(const size_t M, const size_t N, double tol) {
  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }
  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N matrix

  MAT B; // N*M matrix
  B.transpose(A);

  monolish::matrix::Dense<T> ansA(A);
  monolish::matrix::Dense<T> ansAt(B);

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      if (ansA.at(i, j) != ansAt.at(j, i)) {
        std::cout << __func__ << "(" << A.type() << ")"
                  << ": fail" << std::endl;
        return false;
      }
    }
  }
  std::cout << "test_transpose_elements"
            << "(" << get_type<T>() << ")" << std::flush;
  std::cout << ": pass" << std::endl;
  return true;
}

int main(int argc, char **argv) {

  if (argc != 3) {
    std::cout << "$1: row, $2: col" << std::endl;
    return 1;
  }

  const size_t M = atoi(argv[1]);
  const size_t N = atoi(argv[2]);

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  std::cout << "M=" << M << ", N=" << N << std::endl;

  if (test_send_transpose<monolish::matrix::Dense<double>, double>(
          M, N, 1.0e-6) == false) {
    return 1;
  }

  if (test_send_transpose<monolish::matrix::Dense<float>, float>(
          M, N, 1.0e-6) == false) {
    return 1;
  }

  if (test_send_transpose<monolish::matrix::COO<double>, double>(
          M, N, 1.0e-6) == false) {
    return 1;
  }

  if (test_send_transpose<monolish::matrix::COO<float>, float>(M, N, 1.0e-6) ==
      false) {
    return 1;
  }

  if (test_send_transpose<monolish::matrix::CRS<double>, double>(
          M, N, 1.0e-6) == false) {
    return 1;
  }

  if (test_send_transpose<monolish::matrix::CRS<float>, float>(M, N, 1.0e-6) ==
      false) {
    return 1;
  }

  if (test_transpose<monolish::matrix::Dense<double>, double>(M, N, 1.0e-6) ==
      false) {
    return 1;
  }

  if (test_transpose<monolish::matrix::Dense<float>, float>(M, N, 1.0e-6) ==
      false) {
    return 1;
  }

  if (test_transpose<monolish::matrix::COO<double>, double>(M, N, 1.0e-6) ==
      false) {
    return 1;
  }

  if (test_transpose<monolish::matrix::COO<float>, float>(M, N, 1.0e-6) ==
      false) {
    return 1;
  }

  if (test_transpose<monolish::matrix::CRS<double>, double>(M, N, 1.0e-6) ==
      false) {
    return 1;
  }

  if (test_transpose<monolish::matrix::CRS<float>, float>(M, N, 1.0e-6) ==
      false) {
    return 1;
  }

  if (test_transpose_elements<monolish::matrix::Dense<double>, double>(
          M, N, 1.0e-6) == false) {
    return 1;
  }

  if (test_transpose_elements<monolish::matrix::Dense<float>, float>(
          M, N, 1.0e-6) == false) {
    return 1;
  }

  if (test_transpose_elements<monolish::matrix::COO<double>, double>(
          M, N, 1.0e-6) == false) {
    return 1;
  }

  if (test_transpose_elements<monolish::matrix::COO<float>, float>(
          M, N, 1.0e-6) == false) {
    return 1;
  }

  if (test_transpose_elements<monolish::matrix::CRS<double>, double>(
          M, N, 1.0e-6) == false) {
    return 1;
  }

  if (test_transpose_elements<monolish::matrix::CRS<float>, float>(
          M, N, 1.0e-6) == false) {
    return 1;
  }

  return 0;
}
