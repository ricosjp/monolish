#include<iostream>
#include<monolish_blas.hpp>
int main(){
  size_t N = 100;

  // x = {1,1,...,1}, length N
  monolish::vector<double> x(N, 1.0); 

  // Random vector length N with values in the range 1.0 to 2.0
  monolish::vector<double> y(N, 1.0, 2.0); 

  // compute innerproduct
  double ans = monolish::blas::dot(x, y); 

  std::cout << ans << std::endl;

  return 0;
}
