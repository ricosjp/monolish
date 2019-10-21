#include<iostream>
#include"../include/monolish_equation.hpp"

int main(){

	monolish::equation::cg solver;

	std::cout << solver.test_func2() << std::endl;

	return 0;


}
