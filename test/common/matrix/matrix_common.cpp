#include<iostream>
#include<istream>
#include<chrono>
#include"../../test_utils.hpp"

int main(int argc, char** argv){

	//same as test/test.mtx
	const int N = 3;
	const int NNZ = 8;

	// logger option
	//monolish::util::set_log_level(3);
	//monolish::util::set_log_filename("./monolish_test_log.txt");

	// create C-pointer COO Matrix (same as test.mtx, but pointer is 0 origin!!)
	double* val_array = (double*)malloc(sizeof(double) * NNZ);
	int*    col_array = (int*)malloc(sizeof(int) * NNZ);
	int*    row_array = (int*)malloc(sizeof(int) * NNZ);

	// oh...crazy...
	val_array[0] = 1; row_array[0] = 0; col_array[0] = 0;
	val_array[1] = 2; row_array[1] = 0; col_array[1] = 1;
	val_array[2] = 3; row_array[2] = 0; col_array[2] = 2;
	val_array[3] = 4; row_array[3] = 1; col_array[3] = 0;
	val_array[4] = 5; row_array[4] = 1; col_array[4] = 2;
	val_array[5] = 6; row_array[5] = 2; col_array[5] = 0;
	val_array[6] = 7; row_array[6] = 2; col_array[6] = 1;
	val_array[7] = 8; row_array[7] = 2; col_array[7] = 0;

	// test.mtx
	//	| 1 | 2 | 3 |
	//	| 4 | 0 | 5 |
	//	| 6 | 7 | 8 |
	
	//convert C-pointer -> monolish::COO
	monolish::matrix::COO<double> addr_COO(N, NNZ, row_array, col_array, val_array);

	//convert monolish::COO -> monolish::CRS
	monolish::matrix::CRS<double> addr_CRS(addr_COO);


//////////////////////////////////////////////////////

	//from file (MM format is 1 origin)
	monolish::matrix::COO<double> file_COO("../../test.mtx");
	monolish::matrix::CRS<double> file_CRS(file_COO);

	//check
	if(file_CRS.get_row() != addr_CRS.get_row()) {return 1;}
	if(file_CRS.get_nnz() != addr_CRS.get_nnz()) {return 1;}

//////////////////////////////////////////////////////
	//create vector x = {10, 10, 10, ... 10}
	monolish::vector<double> x(N, 10);

	//create vector y
	monolish::vector<double> filey(N);
	monolish::vector<double> addry(N);

	monolish::blas::spmv(file_CRS, x, filey);
	monolish::blas::spmv(addr_CRS, x, addry);

	//ans check
	if(addry[0] != 60) {addry.print_all(); return 1;}
	if(addry[1] != 90) {addry.print_all(); return 1;}
	if(addry[2] != 210) {addry.print_all(); return 1;}

	if(filey[0] != 60) {filey.print_all(); return 1;}
	if(filey[1] != 90) {filey.print_all(); return 1;}
	if(filey[2] != 210) {filey.print_all(); return 1;}

	return 0;
}
