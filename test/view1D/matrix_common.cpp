#include "../test_utils.hpp"
#include "monolish_blas.hpp"

template <typename T> bool test(const size_t size) {

  //(x) monolish::vector = std::vector  = 123, 123, ..., 123
  monolish::vector<T> x(size);

  for(size_t i=0; i<size; i++){
    x[i] = i;
  }

  x.print_all();

  std::cout << "--- view[2-5] ---" << std::endl;

  monolish::view1D<monolish::vector<T>> v(x, 2, 5);
  v.print_all();

  std::cout << "--- view[2] = 12345 ---" << std::endl;
  v[2] = 12345;

  x.print_all();

  monolish::matrix::Dense<T> A(3,3, 1.0);

  std::cout << "--- get view[2:5] ---" << std::endl;
  monolish::view1D<monolish::matrix::Dense<T>> mv(A, 2, 5);
  mv[2] = 12345;
  mv.print_all();

  std::cout << "--- view[2] = 12345 ---" << std::endl;
  A.print_all();


  std::cout << "Pass in " << __func__ << "(" << get_type<T>() << ")"
            << " precision" << std::endl;
  return true;
}

int main(int argc, char **argv) {

  if (argc != 2) {
    std::cout << "error!, $1:vector size" << std::endl;
    return 1;
  }

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  size_t size = atoi(argv[1]);

  if (size <= 1) {
    return 1;
  }

  // exec and error check
  if (test<double>(size) == false) {
    std::cout << "error in double" << std::endl;
    return 1;
  }

//   if (test<float>(size) == false) {
//     std::cout << "error in float" << std::endl;
//     return 1;
//   }
  return 0;
}
