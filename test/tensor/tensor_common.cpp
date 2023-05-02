#include "../test_utils.hpp"

template <typename T> bool test() {
  // same as test/test.mtx
  const int N = 3;
  const int NNZ = 8;

  monolish::tensor::tensor_Dense<T> list_dense(
      {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}); // create from initializer_list

  // create C-pointer COO Matrix (same as test.mtx, but pointer is 0 origin!!)
  T *val_array = (T *)malloc(sizeof(T) * NNZ);
  std::vector<std::vector<size_t>> index_array(NNZ, std::vector<size_t>(2));

  // create COO type arrays
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

  // test.mtx
  //	| 1 | 2 | 3 |
  //	| 4 | 0 | 5 |
  //	| 6 | 7 | 8 |

  // convert C-pointer -> monolish::COO
  monolish::tensor::tensor_COO<T> addr_COO({N, N}, index_array, val_array);

  // test print_all()
  // See https://stackoverflow.com/a/4191318 for testing cout output
  {
    std::ostringstream oss;
    std::streambuf *p_cout_streambuf = std::cout.rdbuf();
    std::cout.rdbuf(oss.rdbuf());
    addr_COO.print_all();
    std::cout.rdbuf(p_cout_streambuf); // restore
    std::stringstream ss;              // To set Float(T) output
    ss << std::scientific;
    ss << std::setprecision(std::numeric_limits<T>::max_digits10);
    ss << "1 1 " << 1.0 << std::endl;
    ss << "1 2 " << 2.0 << std::endl;
    ss << "1 3 " << 3.0 << std::endl;
    ss << "2 1 " << 4.0 << std::endl;
    ss << "2 3 " << 5.0 << std::endl;
    ss << "3 1 " << 6.0 << std::endl;
    ss << "3 2 " << 7.0 << std::endl;
    ss << "3 3 " << 8.0 << std::endl;
    if (oss.str() != ss.str()) {
      std::cout << "print addr_COO matrix mismatch" << std::endl;
      return false;
    }
  }

  // TODO
  /*
  // test transpose(), transpose(COO& B)
  {
    monolish::matrix::COO<T> transposed_COO1 = addr_COO;
    monolish::matrix::COO<T> transposed_COO2;
    transposed_COO1.transpose();
    transposed_COO2.transpose(addr_COO);
    std::ostringstream oss1;
    std::streambuf *p_cout_streambuf = std::cout.rdbuf();
    std::cout.rdbuf(oss1.rdbuf());
    transposed_COO1.print_all();
    std::ostringstream oss2;
    std::cout.rdbuf(oss2.rdbuf());
    transposed_COO2.print_all();
    std::cout.rdbuf(p_cout_streambuf); // restore
    if (oss1.str() != oss2.str()) {
      std::cout << "two transpose() function mismatch" << std::endl;
      return false;
    }
    if (addr_COO.at(0, 1) != transposed_COO1.at(1, 0)) {
      std::cout << "A(0,1) != A^T(1,0)" << std::endl;
      return false;
    }
  }
  */

  // TODO
  /*
  // test get_data_size()
  if (addr_COO.get_data_size() - 24.0e-9 * sizeof(T)) {
    std::cout << "get_data_size() failed" << std::endl;
    return false;
  }
  */

  // test type()
  if (addr_COO.type() != "tensor_COO") {
    std::cout << "type() is not tensor_COO" << std::endl;
    return false;
  }

  // test diag()
  auto shape = addr_COO.get_shape();
  size_t s = *std::min_element(shape.begin(), shape.end());
  monolish::vector<T> dv(s);
  addr_COO.diag(dv);
  if (dv[0] != 1.0 || dv[1] != 0.0 || dv[2] != 8.0) {
    std::cout << "diag() failed" << std::endl;
    return false;
  }

  // test changing matrix dimension
  //{set,get}_{row,col,nnz}()
  auto expanded_COO = addr_COO;
  expanded_COO.set_shape({4, 4});
  if (expanded_COO.get_shape().size() != 2 ||
      expanded_COO.get_shape()[0] != 4 || expanded_COO.get_shape()[1] != 4) {
    std::cout << "shape size mismatch" << std::endl;
    return false;
  }
  monolish::tensor::tensor_Dense<T> expanded_Dense(expanded_COO);
  expanded_COO.insert({3, 3}, 1.0);
  expanded_Dense.insert({3, 3}, 1.0);
  if (expanded_COO.get_nnz() != 9) {
    std::cout << "nnz size mismatch" << std::endl;
    return false;
  }
  {
    monolish::tensor::tensor_Dense<T> expanded_Dense_after_insertion(
        expanded_COO);
    if (expanded_Dense_after_insertion != expanded_Dense) {
      std::cout << "Dense matrix mismatch" << std::endl;
      return false;
    }
  }

  // expanded.mtx
  //	| 1 | 2 | 3 | 0 |
  //	| 4 | 0 | 5 | 0 |
  //	| 6 | 7 | 8 | 0 |
  //      | 0 | 0 | 0 | 1 |
  {
    std::ostringstream oss;
    std::streambuf *p_cout_streambuf = std::cout.rdbuf();
    std::cout.rdbuf(oss.rdbuf());
    expanded_COO.print_all();
    std::cout.rdbuf(p_cout_streambuf); // restore
    std::string res("%%MatrixMarket matrix coordinate real general\n");
    std::stringstream ss; // To set Float(T) output
    ss << std::scientific;
    ss << std::setprecision(std::numeric_limits<T>::max_digits10);
    ss << "1 1 " << 1.0 << std::endl;
    ss << "1 2 " << 2.0 << std::endl;
    ss << "1 3 " << 3.0 << std::endl;
    ss << "2 1 " << 4.0 << std::endl;
    ss << "2 3 " << 5.0 << std::endl;
    ss << "3 1 " << 6.0 << std::endl;
    ss << "3 2 " << 7.0 << std::endl;
    ss << "3 3 " << 8.0 << std::endl;
    ss << "4 4 " << 1.0 << std::endl;
    if (oss.str() != ss.str()) {
      std::cout << "print expanded matrix mismatch" << std::endl;
      return false;
    }
  }

  expanded_COO.insert({0, 0}, 3.0);
  if (expanded_COO.get_nnz() != 10) {
    std::cout << "nnz size mismatch after inserting duplicate element"
              << std::endl;
    return false;
  }
  expanded_COO.sort(true);
  if (expanded_COO.at({0, 0}) != 3.0) {
    std::cout << "sort and replace logic failed" << std::endl;
    return false;
  }
  if (expanded_COO.get_nnz() != 9) {
    std::cout << "nnz size mismatch after sort and merge" << std::endl;
    return false;
  }

  // test at(i, j)
  // non zero element
  if (addr_COO.at({0, 0}) != 1.0) {
    std::cout << "A(0, 0) != 1.0" << std::endl;
    return false;
  }
  // zero element
  if (addr_COO.at({1, 1}) != 0.0) {
    std::cout << "A(1, 1) != 0.0" << std::endl;
    return false;
  }

  return true;
}

template <typename T> bool default_constructor_test() {
  monolish::tensor::tensor_COO<T> A({2, 2});
  {
    std::ostringstream oss;
    std::streambuf *p_cout_streambuf = std::cout.rdbuf();
    std::cout.rdbuf(oss.rdbuf());
    A.print_all();
    std::cout.rdbuf(p_cout_streambuf); // restore
    std::string res("%%MatrixMarket matrix coordinate real general\n");
    std::stringstream ss; // To set Float(T) output
    ss << std::scientific;
    ss << std::setprecision(std::numeric_limits<T>::max_digits10);
    if (oss.str() != ss.str()) {
      std::cout << "empty A" << std::endl;
      return false;
    }
  }
  A.insert({1, 1}, 2.0);
  A.insert({0, 0}, 1.0);
  A.sort(false);
  {
    std::ostringstream oss;
    std::streambuf *p_cout_streambuf = std::cout.rdbuf();
    std::cout.rdbuf(oss.rdbuf());
    A.print_all();
    std::cout.rdbuf(p_cout_streambuf); // restore
    std::string res("%%MatrixMarket matrix coordinate real general\n");
    std::stringstream ss; // To set Float(T) output
    ss << std::scientific;
    ss << std::setprecision(std::numeric_limits<T>::max_digits10);
    ss << "1 1 " << 1.0 << std::endl;
    ss << "2 2 " << 2.0 << std::endl;
    if (oss.str() != ss.str()) {
      std::cout << "2 elements and sorted A" << std::endl;
      return false;
    }
  }

  return true;
}

template <typename T> bool fixed_size_test() {
  monolish::tensor::tensor_COO<T> A({2, 2});
  {
    std::ostringstream oss;
    std::streambuf *p_cout_streambuf = std::cout.rdbuf();
    std::cout.rdbuf(oss.rdbuf());
    A.print_all();
    std::cout.rdbuf(p_cout_streambuf); // restore
    std::string res("%%MatrixMarket matrix coordinate real general\n");
    std::stringstream ss; // To set Float(T) output
    ss << std::scientific;
    ss << std::setprecision(std::numeric_limits<T>::max_digits10);
    if (oss.str() != ss.str()) {
      std::cout << "empty A" << std::endl;
      return false;
    }
  }
  A.insert({1, 1}, 2.0);
  A.insert({0, 0}, 1.0);
  A.sort(false);
  {
    std::ostringstream oss;
    std::streambuf *p_cout_streambuf = std::cout.rdbuf();
    std::cout.rdbuf(oss.rdbuf());
    A.print_all();
    std::cout.rdbuf(p_cout_streambuf); // restore
    std::string res("%%MatrixMarket matrix coordinate real general\n");
    std::stringstream ss; // To set Float(T) output
    ss << std::scientific;
    ss << std::setprecision(std::numeric_limits<T>::max_digits10);
    ss << "1 1 " << 1.0 << std::endl;
    ss << "2 2 " << 2.0 << std::endl;
    if (oss.str() != ss.str()) {
      std::cout << "2 elements and sorted A" << std::endl;
      return false;
    }
  }

  // initialization constructor
  monolish::tensor::tensor_COO<T> B(A, 0.0);
  if (monolish::util::is_same_structure(A, B) != true && A == B) {
    std::cout << "init constructor error" << std::endl;
    return false;
  }

  std::cout << "Pass in " << get_type<T>() << " precision" << std::endl;

  return true;
}

template <typename T> bool reshape_test() {
  monolish::tensor::tensor_Dense<T> tensor_dense(
      {2, 6}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

  T val_orig = tensor_dense.at(1, 2);
  if(val_orig != 9){
    std::cout << "original value failed" << std::endl;
    return false;
  }

  tensor_dense.reshape(3, 4);
  T val_new = tensor_dense.at(2, 0);
  if(val_new != val_orig){
    std::cout << "reshaped value failed" << std::endl;
    return false;
  }

  return true;
}

int main(int argc, char **argv) {

  print_build_info();

  if (!test<double>()) {
    return 1;
  }
  if (!test<float>()) {
    return 1;
  }

  if (!default_constructor_test<double>()) {
    return 2;
  }
  if (!default_constructor_test<float>()) {
    return 2;
  }

  if (!fixed_size_test<double>()) {
    return 3;
  }
  if (!fixed_size_test<float>()) {
    return 3;
  }

  if (!reshape_test<double>()) {
    return 4;
  }
  if (!reshape_test<float>()) {
    return 4;
  }

  return 0;
}
