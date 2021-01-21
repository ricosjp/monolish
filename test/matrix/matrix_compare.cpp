#include "../test_utils.hpp"

template <typename T> bool test() {

  // same as test/test.mtx
  const int N = 3;
  const int NNZ = 8;

  // create C-pointer COO Matrix (same as test.mtx, but pointer is 0 origin!!)
  T *val_array = (T *)malloc(sizeof(T) * NNZ);
  int *col_array = (int *)malloc(sizeof(int) * NNZ);
  int *row_array = (int *)malloc(sizeof(int) * NNZ);

  // create COO type arrays
  val_array[0] = 1;
  row_array[0] = 0;
  col_array[0] = 0;
  val_array[1] = 2;
  row_array[1] = 0;
  col_array[1] = 1;
  val_array[2] = 3;
  row_array[2] = 0;
  col_array[2] = 2;
  val_array[3] = 4;
  row_array[3] = 1;
  col_array[3] = 0;
  val_array[4] = 5;
  row_array[4] = 1;
  col_array[4] = 2;
  val_array[5] = 6;
  row_array[5] = 2;
  col_array[5] = 0;
  val_array[6] = 7;
  row_array[6] = 2;
  col_array[6] = 1;
  val_array[7] = 8;
  row_array[7] = 2;
  col_array[7] = 2;

  monolish::matrix::COO<T> COO_A(N, N, NNZ, row_array, col_array, val_array);

  // create B
  val_array[7] = 9;
  monolish::matrix::COO<T> COO_B(N, N, NNZ, row_array, col_array, val_array);

  val_array[7] = 10;
  monolish::matrix::COO<T> COO_C(N, N, NNZ, row_array, col_array, val_array);

  if (COO_A == COO_B && COO_A == COO_C) {
    return false;
  }

  if (!(monolish::util::is_same_size(COO_A, COO_B, COO_C))) {
    return false;
  }

  if (!(monolish::util::is_same_structure(COO_A, COO_B, COO_C, COO_A, COO_B,
                                          COO_C))) {
    return false;
  }

  monolish::matrix::CRS<T> CRS_A(COO_A);
  monolish::matrix::CRS<T> CRS_B(COO_B);
  monolish::matrix::CRS<T> CRS_C(COO_C);

  if (CRS_A == CRS_B && CRS_A == CRS_C) {
    return false;
  }

  if (!(monolish::util::is_same_size(CRS_A, CRS_B, CRS_C))) {
    return false;
  }

  if (!(monolish::util::is_same_structure(CRS_A, CRS_B, CRS_C, CRS_A, CRS_B,
                                          CRS_C))) {
    return false;
  }

  monolish::matrix::Dense<T> Dense_A(COO_A);
  monolish::matrix::Dense<T> Dense_B(COO_B);
  monolish::matrix::Dense<T> Dense_C(COO_C);

  if (Dense_A == Dense_B && Dense_A == Dense_C) {
    return false;
  }

  if (!(monolish::util::is_same_size(Dense_A, Dense_B, Dense_C))) {
    return false;
  }

  if (!(monolish::util::is_same_structure(Dense_A, Dense_B, Dense_C, Dense_A, Dense_B,
                                          Dense_C))) {
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
