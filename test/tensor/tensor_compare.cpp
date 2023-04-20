#include "../test_utils.hpp"

template <typename T> bool test() {

  // same as test/test.mtx
  const int N = 3;
  const int NNZ = 8;

  // create C-pointer tensor_COO Matrix (same as test.mtx, but pointer is 0 origin!!)
  T *val_array = (T *)malloc(sizeof(T) * NNZ);
  int *col_array = (int *)malloc(sizeof(int) * NNZ);
  int *row_array = (int *)malloc(sizeof(int) * NNZ);
  std::vector<std::vector<size_t>> index_array(NNZ, std::vector<size_t>(2));

  // create tensor_COO type arrays
  val_array[0] = 1;
  index_array[0][0] = 0;
  index_array[0][1] = 0;
  val_array[1] = 2;
  index_array[1][0] = 0;
  index_array[1][1] = 1;
  val_array[2] = 3;
  index_array[2][0] = 0;
  index_array[2][1] = 2;
  val_array[3] = 4;
  index_array[3][0] = 1;
  index_array[3][1] = 0;
  val_array[4] = 5;
  index_array[4][0] = 1;
  index_array[4][1] = 2;
  val_array[5] = 6;
  index_array[5][0] = 2;
  index_array[5][1] = 0;
  val_array[6] = 7;
  index_array[6][0] = 2;
  index_array[6][1] = 1;
  val_array[7] = 8;
  index_array[7][0] = 2;
  index_array[7][1] = 2;

  monolish::tensor::tensor_COO<T> tensor_COO_A({N, N}, index_array, val_array);

  // create B
  val_array[7] = 9;
  monolish::tensor::tensor_COO<T> tensor_COO_B({N, N}, index_array, val_array);

  val_array[7] = 10;
  monolish::tensor::tensor_COO<T> tensor_COO_C({N, N}, index_array, val_array);

  if (tensor_COO_A == tensor_COO_B && tensor_COO_A == tensor_COO_C) {
    return false;
  }

  if (!(monolish::util::is_same_size(tensor_COO_A, tensor_COO_B, tensor_COO_C))) {
    return false;
  }

  if (!(monolish::util::is_same_structure(tensor_COO_A, tensor_COO_B, tensor_COO_C, tensor_COO_A, tensor_COO_B,
                                          tensor_COO_C))) {
    return false;
  }

  monolish::tensor::tensor_Dense<T> tensor_Dense_A(tensor_COO_A);
  monolish::tensor::tensor_Dense<T> tensor_Dense_B(tensor_COO_B);
  monolish::tensor::tensor_Dense<T> tensor_Dense_C(tensor_COO_C);

  if (tensor_Dense_A == tensor_Dense_B && tensor_Dense_A == tensor_Dense_C) {
    return false;
  }

  if (!(monolish::util::is_same_size(tensor_Dense_A, tensor_Dense_B, tensor_Dense_C))) {
    return false;
  }

  if (!(monolish::util::is_same_structure(tensor_Dense_A, tensor_Dense_B, tensor_Dense_C, tensor_Dense_A,
                                          tensor_Dense_B, tensor_Dense_C))) {
    return false;
  }

  std::cout << "Pass in " << get_type<T>() << " precision" << std::endl;
  return true;
}

int main(int argc, char **argv) {

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  if (!test<double>()) {
    return 1;
  }
  if (!test<float>()) {
    return 1;
  }

  return 0;
}
