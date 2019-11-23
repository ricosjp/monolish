#include<iostream>
#include"monolish_blas.hpp"

int main(int argc, char** argv){

	if(argc!=2){
		std::cout << "error $1 is matrix filename" << std::endl;
		return 1;
	}

	char* file = argv[1];
	monolish::matrix::COO<double> COO(file);
	monolish::matrix::CRS<double> A(COO);

	monolish::vector<double> x(A.get_row(), 0.0);
	monolish::vector<double> y(A.get_row(), 1.0);

	monolish::blas::spmv(A, x, y);

	y.print_all();

	return 0;


}
