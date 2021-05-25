#include <iostream>
#include <monolish_blas.hpp>
int main() {

  // Output log if you need
  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  size_t N = 100;

  // x = {1,1,...,1}, length N
  monolish::vector<double> x(N, 1.0);

  // Random vector length N with random values in the range 1.0 to 2.0
  monolish::vector<double> y(N, 1.0, 2.0);

  // send data to GPU
  monolish::util::send(x, y);

  // compute innerproduct
  double ans = monolish::blas::dot(x, y);

  std::cout << ans << std::endl;

  return 0;
}
