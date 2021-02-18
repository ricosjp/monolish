#include <monolish_blas.hpp>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "$1 Dimension should be specified" << std::endl;
  }

  int DIM = std::stoi(argv[1]);

  monolish::matrix::COO<double> COO_A = monolish::util::toeplitz_plus_hankel_matrix<double>(DIM, 1.0, -1.0 / 3.0, -1.0 / 6.0);
  std::string fileA("A.mtx");
  COO_A.print_all(fileA);
  monolish::matrix::COO<double> COO_B = monolish::util::toeplitz_plus_hankel_matrix<double>(DIM, 11.0 / 20.0, 13.0 / 60.0, 1.0 / 120.0);
  std::string fileB("B.mtx");
  COO_B.print_all(fileB);
  return 0;
}
