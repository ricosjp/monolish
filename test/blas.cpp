#include<iostream>
#include"../include/monolish_equation.hpp"

int main(){

	monolish::equation::cg cg;
	monolish::vector<double> x(1000000, 0.0);
	monolish::vector<double> b(1000000, 0.0);

	monolish::matrix::COO<double> COO("./test.mtx");

	//COO.output();

	monolish::matrix::CRS<double> A(COO);

//	A.output();


	return 0;


}

