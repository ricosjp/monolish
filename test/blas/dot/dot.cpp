#include<iostream>
#include"monolish_blas.hpp"

int main(int argc, char** argv){

	if(argc!=2){
		std::cout << "error $1 is vector size" << std::endl;
		return 1;
	}

	int size = atoi(argv[1]);
	monolish::vector<double> x(size, 1.0);
	monolish::vector<double> y(size, 2.0);

	double ans = monolish::blas::dot(x, y);

	std::cout << ans << std::endl;

	return 0;


}
