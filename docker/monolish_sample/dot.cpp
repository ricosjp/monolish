#include<iostream>
#include<istream>
#include<random>
#include<cmath>
#include<monolish_blas.hpp>

int main(int argc, char** argv){

	int size = 1000;

	// create random vector
	std::random_device random;
  	monolish::vector<double> x(size, 0.0, 1.0);
  	monolish::vector<double> y(size, 0.0, 1.0);
	
	// inner product
	double result = monolish::blas::dot(x, y);

	// output result
	std::cout << result << std::endl;

	return 0;
}
