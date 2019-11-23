#include<iostream>
#include"../include/monolish_equation.hpp"

int main(int argc, char** argv){
	if(argc!=2){
		std::cout << "error $1 is matrix filename" << std::endl;
		return 1;
	}

	monolish::equation::LU LU_solver;

	char* file = argv[1];
	monolish::matrix::COO<double> COO(file);
	monolish::matrix::CRS<double> A(COO);

	monolish::vector<double> x(A.get_row(), 0.0);
	monolish::vector<double> b(A.get_row(), 1.0);

	LU_solver.solve(A, x, b);

	x.print_all();

	return 0;


}
