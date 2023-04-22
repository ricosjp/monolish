#include "../test_utils.hpp"

template <typename MAT, typename T>
bool test_convert(const size_t M, const size_t N, const size_t L) {
  size_t nnzrow = 27;
  if ((nnzrow < L)) {
    nnzrow = 27;
  } else {
    nnzrow = L - 1;
  }
  // ans tensor_COO (source)
  monolish::tensor::tensor_COO<T> ans_coo =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);

  // convert tensor_COO -> MAT
  MAT mat(ans_coo);

  // convert MAT -> result tensor_COO (dest.)
  monolish::tensor::tensor_COO<T> result_coo(mat);

  // check source == dest.
  if (result_coo.get_shape() != ans_coo.get_shape() ||
      result_coo.get_nnz() != ans_coo.get_nnz()) {

    std::cout << "error, shape, nnz are different(tensor_COO2" << mat.type()
              << ")" << std::endl;
    std::cout << result_coo.get_shape().size()
              << " != " << ans_coo.get_shape().size() << std::endl;
    for (auto i = 0; i < std::min({result_coo.get_shape().size(),
                                   ans_coo.get_shape().size()});
         i++) {
      std::cout << result_coo.get_shape()[i] << " != " << ans_coo.get_shape()[i]
                << std::endl;
    }
    std::cout << result_coo.get_nnz() << " != " << ans_coo.get_nnz()
              << std::endl;

    return false;
  }
  for (size_t i = 0; i < ans_coo.get_nnz(); i++) {

    if (result_coo.data()[i] != ans_coo.data()[i] ||
        result_coo.index[i] != ans_coo.index[i]) {

      std::cout << i << "\t";
      for (size_t j = 0; j < result_coo.index[i].size(); j++) {
        std::cout << result_coo.index[i][j] << ",";
      }
      std::cout << result_coo.data()[i] << std::flush;
      std::cout << ", (ans: ";
      for (size_t j = 0; j < ans_coo.index[i].size(); j++) {
        std::cout << ans_coo.index[i][j] << ",";
      }
      std::cout << ans_coo.data()[i] << ")" << std::endl;
      return false;
    }
  }

  std::cout << __func__ << "(" << get_type<T>() << "," << mat.type() << ")"
            << std::flush;
  std::cout << ": pass" << std::endl;

  return true;
}

int main(int argc, char **argv) {

  if (argc != 4) {
    std::cout << "$1: shape1, $2: shape2, $3 : shape3" << std::endl;
    return 1;
  }

  const size_t M = atoi(argv[1]);
  const size_t N = atoi(argv[2]);
  const size_t L = atoi(argv[3]);

  std::cout << "M=" << M << ", N=" << N << ", L=" << L << std::endl;

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  if (test_convert<monolish::tensor::tensor_Dense<double>, double>(M, N, L) ==
      false) {
    return 1;
  }

  if (test_convert<monolish::tensor::tensor_Dense<float>, float>(M, N, L) ==
      false) {
    return 1;
  }

  return 0;
}
