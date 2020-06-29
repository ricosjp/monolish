#include<iostream>
#include<istream>
#include<chrono>
#include<stdexcept>
#include"../../test_utils.hpp"

template<typename T>
bool test(){
	//same as test/test.mtx
	const int N = 3;
	const int NNZ = 8;

	// create C-pointer COO Matrix (same as test.mtx, but pointer is 0 origin!!)
	T* val_array = (T*)malloc(sizeof(T) * NNZ);
	int*    col_array = (int*)malloc(sizeof(int) * NNZ);
	int*    row_array = (int*)malloc(sizeof(int) * NNZ);

	// create COO type arrays
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
	monolish::matrix::COO<T> addr_COO(N, NNZ, row_array, col_array, val_array);

	//test at(i, j)
	//non zero element
	if (addr_COO.at(0, 0) != 1) { return false; }
	//zero element
	if (addr_COO.at(1, 1) != 0) { return false; }
	//out of range element
	try {
		addr_COO.at(3, 2);
		throw std::logic_error("at() should throw out_of_range()");
	} catch (std::out_of_range& exception) {}

	//convert monolish::COO -> monolish::CRS
	monolish::matrix::CRS<T> addr_CRS(addr_COO);

	//test operator[](i, j)
	//non zero element
	if (addr_COO.at(0, 1) != 2) { return false; }
	//zero element
	if (addr_COO.at(1, 1) != 0) { return false; }

//////////////////////////////////////////////////////

	//from file (MM format is 1 origin)
	monolish::matrix::COO<T> file_COO("../../test.mtx");
	monolish::matrix::CRS<T> file_CRS(file_COO);

// 	//check
// 	if(file_CRS.get_row() != addr_CRS.get_row()) {return false;}
// 	if(file_CRS.get_nnz() != addr_CRS.get_nnz()) {return false;}

//////////////////////////////////////////////////////
	//create vector x = {10, 10, 10, ... 10}
	monolish::vector<T> x(N, 10);

	//create vector y
	monolish::vector<T> filey(N);
	monolish::vector<T> addry(N);

	monolish::util::send(x, filey, addry, file_CRS, addr_CRS);

	monolish::blas::spmv(file_CRS, x, filey);
	monolish::blas::spmv(addr_CRS, x, addry);

	monolish::util::recv(addry, filey);

	//ans check
	if(addry[0] != 60) {addry.print_all();  return false;}
	if(addry[1] != 90) {addry.print_all();  return false;}
	if(addry[2] != 210) {addry.print_all();  return false;}

	if(filey[0] != 30) {filey.print_all();  return false;}
	if(filey[1] != 20) {filey.print_all();  return false;}
	if(filey[2] != 30) {filey.print_all();  return false;}

	return 0;
}

int main(int argc, char** argv){

	// logger option
	//monolish::util::set_log_level(3);
	//monolish::util::set_log_filename("./monolish_test_log.txt");
	
	if( test<double>() ) {return 1;}
	if( test<float>() ) {return 1;}
	return 0;
}
