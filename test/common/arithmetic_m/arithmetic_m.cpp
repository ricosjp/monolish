#include<iostream>
#include<istream>
#include<chrono>
#include"../../test_utils.hpp"

int main(int argc, char** argv){

	if(argc!=2){
		std::cout << "error $1:matrix_name" << std::endl;
		return 1;
	}
	char* filename = argv[1];

	//monolish::util::set_log_level(3);
	//monolish::util::set_log_filename("./monolish_test_log.txt");
	
	monolish::matrix::COO<double> tmp_COO(filename);
	monolish::matrix::CRS<double> tmp_CRS(tmp_COO);

	monolish::matrix::COO<double> COO = tmp_COO;
	monolish::matrix::CRS<double> CRS = tmp_CRS;

	//create random vector x rand(0.1~1.0)
   	monolish::vector<double> x(CRS.size(), 0.1, 1.0);
   	monolish::vector<double> ansy(CRS.size(), 0.0);

	monolish::blas::spmv(tmp_CRS, x, ansy);

	monolish::vector<double> tsty = CRS * x;


	if(ansy!=tsty){
		std::cout << "error" << std::endl;
		tsty.print_all();
		ansy.print_all();
		return 1;
	}

	return 0;
}
