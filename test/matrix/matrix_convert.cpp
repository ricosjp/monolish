#include "../test_utils.hpp"

template <typename MAT, typename T>
bool test_convert(const size_t M, const size_t N) {
  // ans COO (source)
  monolish::matrix::COO<T> ans_coo = get_random_structure_matrix<T>(M, N);

  // convert COO -> MAT
  MAT mat(ans_coo);

  // convert MAT -> result COO (dest.)
  monolish::matrix::COO<T> result_coo(mat);

  // check source == dest.
  if (result_coo.get_row() != ans_coo.get_row() ||
      result_coo.get_col() != ans_coo.get_col() ||
      result_coo.get_nnz() != ans_coo.get_nnz()) {

    std::cout << "error, row, col, nnz are different(COO2" << mat.type() << ")"
              << std::endl;
    std::cout << result_coo.get_row() << " != " << ans_coo.get_row()
              << std::endl;
    std::cout << result_coo.get_col() << " != " << ans_coo.get_col()
              << std::endl;
    std::cout << result_coo.get_nnz() << " != " << ans_coo.get_nnz()
              << std::endl;

    return false;
  }
  for (size_t i = 0; i < ans_coo.get_nnz(); i++) {

    if (result_coo.data()[i] != ans_coo.data()[i] ||
        result_coo.row_index[i] != ans_coo.row_index[i] ||
        result_coo.col_index[i] != ans_coo.col_index[i]) {

      std::cout << i << "\t" << result_coo.row_index[i] << ","
                << result_coo.col_index[i] << "," << result_coo.data()[i]
                << std::flush;
      std::cout << ", (ans: " << ans_coo.row_index[i] << ","
                << ans_coo.col_index[i] << "," << ans_coo.data()[i] << ")"
                << std::endl;
      return false;
    }
  }

  std::cout << __func__ << "(" << get_type<T>() << "," << mat.type() << ")"
            << std::flush;
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

  std::cout << "M=" << M << ", N=" << N << std::endl;

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  if (test_convert<monolish::matrix::Dense<double>, double>(M, N) == false) {
    return 1;
  }
  if (test_convert<monolish::matrix::CRS<double>, double>(M, N) == false) {
    return 1;
  }

  if (test_convert<monolish::matrix::Dense<float>, float>(M, N) == false) {
    return 1;
  }
  if (test_convert<monolish::matrix::CRS<float>, float>(M, N) == false) {
    return 1;
  }

  return 0;
}
