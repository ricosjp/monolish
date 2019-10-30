#include<iostream>
#include"../include/monolish_equation.hpp"
#include"../include/common/monolish_common.hpp"

int main(){

	monolish::equation::cg cg;
	monolish::vector<double> x(10, 0.0);
	monolish::vector<double> b(10, 0.0);

	cg.solve(x, b);


	return 0;


}
