#include "math/v_tanh.hpp"

int main(int argc, char **argv) {

  if (argc != 2) {
    std::cout << "error!, $1:vector size" << std::endl;
    return 1;
  }

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  size_t size = atoi(argv[1]);
  std::cout << "size: " << size << std::endl;

  // scalar-vetor-add//
  if (test_vtanh<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_vtanh<float>(size, 1.0e-4) == false) {
    return 1;
  }

  if (test_send_vtanh<double>(size, 1.0e-8) == false) {
    return 1;
  }
  if (test_send_vtanh<float>(size, 1.0e-4) == false) {
    return 1;
  }
}
