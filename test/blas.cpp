#include<iostream>
#include"../include/monolish_equation.hpp"

int main(){

	monolish::equation::cg cg;
	monolish::vector<double> x(1000000, 0.0);
	monolish::vector<double> b(1000000, 0.0);

	monolish::COO_matrix<double> COO("./test.mtx");

	//COO.output();

	monolish::CRS_matrix<double> A(COO);

//	A.output();


	return 0;


}

