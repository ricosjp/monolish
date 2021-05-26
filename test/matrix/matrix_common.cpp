#include "../test_utils.hpp"

template <typename T> bool test() {
  // same as test/test.mtx
  const int N = 3;
  const int NNZ = 8;

  monolish::matrix::Dense<T> list_dense(
      3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9}); // create from initializer_list

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

  // test.mtx
  //	| 1 | 2 | 3 |
  //	| 4 | 0 | 5 |
  //	| 6 | 7 | 8 |

  // convert C-pointer -> monolish::COO
  monolish::matrix::COO<T> addr_COO(N, N, NNZ, row_array, col_array, val_array);

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

  // test transpose(), transpose(COO& B)
  {
    monolish::matrix::COO<T> transposed_COO1 = addr_COO;
    monolish::matrix::COO<T> transposed_COO2;
    transposed_COO1.transpose();
    addr_COO.transpose(transposed_COO2);
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

  // test get_data_size()
  if (addr_COO.get_data_size() - 24.0e-9 * sizeof(T)) {
    std::cout << "get_data_size() failed" << std::endl;
    return false;
  }

  // test type()
  if (addr_COO.type() != "COO") {
    std::cout << "type() is not COO" << std::endl;
    return false;
  }

  // test row(int i)
  monolish::vector<T> row1(addr_COO.get_row());
  addr_COO.row(1, row1);
  if (row1[0] != 4.0 || row1[1] != 0.0 || row1[2] != 5.0) {
    std::cout << "row(int) failed" << std::endl;
    return false;
  }

  // test col(int j)
  monolish::vector<T> col1(addr_COO.get_col());
  addr_COO.col(1, col1);
  if (col1[0] != 2.0 || col1[1] != 0.0 || col1[2] != 7.0) {
    std::cout << "col(int) failed" << std::endl;
    return false;
  }

  // test diag()
  size_t s = addr_COO.get_row() > addr_COO.get_col() ? addr_COO.get_col()
                                                     : addr_COO.get_row();
  monolish::vector<T> dv(s);
  addr_COO.diag(dv);
  if (dv[0] != 1.0 || dv[1] != 0.0 || dv[2] != 8.0) {
    std::cout << "diag() failed" << std::endl;
    return false;
  }

  // test changing matrix dimension
  //{set,get}_{row,col,nnz}()
  auto expanded_COO = addr_COO;
  expanded_COO.set_row(4);
  if (expanded_COO.get_row() != 4) {
    std::cout << "row size mismatch" << std::endl;
    return false;
  }
  expanded_COO.set_col(4);
  if (expanded_COO.get_col() != 4) {
    std::cout << "col size mismatch" << std::endl;
    return false;
  }
  monolish::matrix::Dense<T> expanded_Dense(expanded_COO);
  expanded_COO.insert(3, 3, 1.0);
  expanded_Dense[3][3] = 1.0;
  if (expanded_COO.get_nnz() != 9) {
    std::cout << "nnz size mismatch" << std::endl;
    return false;
  }
  {
    monolish::matrix::Dense<T> expanded_Dense_after_insertion(expanded_COO);
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

  expanded_COO.insert(0, 0, 3.0);
  if (expanded_COO.get_nnz() != 10) {
    std::cout << "nnz size mismatch after inserting duplicate element"
              << std::endl;
    return false;
  }
  expanded_COO.sort(true);
  if (expanded_COO.at(0, 0) != 3.0) {
    std::cout << "sort and replace logic failed" << std::endl;
    return false;
  }
  if (expanded_COO.get_nnz() != 9) {
    std::cout << "nnz size mismatch after sort and merge" << std::endl;
    return false;
  }

  // test at(i, j)
  // non zero element
  if (addr_COO.at(0, 0) != 1.0) {
    std::cout << "A(0, 0) != 1.0" << std::endl;
    return false;
  }
  // zero element
  if (addr_COO.at(1, 1) != 0.0) {
    std::cout << "A(1, 1) != 0.0" << std::endl;
    return false;
  }

  // convert monolish::COO -> monolish::CRS
  monolish::matrix::CRS<T> addr_CRS(addr_COO);

  //////////////////////////////////////////////////////

  // from file (MM format is 1 origin)
  monolish::matrix::COO<T> file_COO("../test.mtx");
  monolish::matrix::CRS<T> file_CRS(file_COO);

  // 	//check
  // 	if(file_CRS.get_row() != addr_CRS.get_row()) {return false;}
  // 	if(file_CRS.get_nnz() != addr_CRS.get_nnz()) {return false;}

  //////////////////////////////////////////////////////
  // create vector x = {10, 10, 10, ... 10}
  monolish::vector<T> x(N, 10);

  // create vector y
  monolish::vector<T> filey(N);
  monolish::vector<T> addry(N);

  monolish::util::send(x, filey, addry, file_CRS, addr_CRS);

  monolish::blas::matvec(file_CRS, x, filey);
  monolish::blas::matvec(addr_CRS, x, addry);

  monolish::util::recv(addry, filey);

  // ans check
  if (addry[0] != 60) {
    addry.print_all();
    return false;
  }
  if (addry[1] != 90) {
    addry.print_all();
    return false;
  }
  if (addry[2] != 210) {
    addry.print_all();
    return false;
  }

  if (filey[0] != 10) {
    filey.print_all();
    return false;
  }
  if (filey[1] != 0) {
    filey.print_all();
    return false;
  }
  if (filey[2] != 10) {
    filey.print_all();
    return false;
  }

  return true;
}

template <typename T> bool default_constructor_test() {
  monolish::matrix::COO<T> A;
  A.set_row(2);
  A.set_col(2);
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
  A.insert(1, 1, 2.0);
  A.insert(0, 0, 1.0);
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
  monolish::matrix::COO<T> A(2, 2);
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
  A.insert(1, 1, 2.0);
  A.insert(0, 0, 1.0);
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

  std::cout << "Pass in " << get_type<T>() << " precision" << std::endl;
  return true;
}

int main(int argc, char **argv) {

  // logger option
  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

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

  return 0;
}
