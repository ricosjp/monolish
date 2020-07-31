#include<iostream>
#include<istream>
#include<sstream>
#include<iomanip>
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
	val_array[7] = 8; row_array[7] = 2; col_array[7] = 2;

	// test.mtx
	//	| 1 | 2 | 3 |
	//	| 4 | 0 | 5 |
	//	| 6 | 7 | 8 |
	
	//convert C-pointer -> monolish::COO
	monolish::matrix::COO<T> addr_COO(N, N, NNZ, row_array, col_array, val_array);

        //test print_all()
        //See https://stackoverflow.com/a/4191318 for testing cout output
        {
        std::ostringstream oss;
        std::streambuf* p_cout_streambuf = std::cout.rdbuf();
        std::cout.rdbuf(oss.rdbuf());
        addr_COO.print_all();
        std::cout.rdbuf(p_cout_streambuf); // restore
        std::stringstream ss; // To set Float(T) output
        ss << std::scientific;
        ss << std::setprecision(std::numeric_limits<T>::max_digits10);
        ss << "%%MatrixMarket matrix coordinate real general" << std::endl;
        ss << "3 3 8" << std::endl;
        ss << "1 1 " << 1.0 << std::endl;
        ss << "1 2 " << 2.0 << std::endl;
        ss << "1 3 " << 3.0 << std::endl;
        ss << "2 1 " << 4.0 << std::endl;
        ss << "2 3 " << 5.0 << std::endl;
        ss << "3 1 " << 6.0 << std::endl;
        ss << "3 2 " << 7.0 << std::endl;
        ss << "3 3 " << 8.0 << std::endl;
        if (oss.str() != ss.str()) { std::cout << "print addr_COO matrix mismatch" << std::endl; return false; }
        }

        //test changing matrix dimension
        //{set,get}_{row,col,nnz}()
        auto expanded_COO = addr_COO;
        expanded_COO.set_row(4);
        if (expanded_COO.get_row() != 4) { std::cout << "row size mismatch" << std::endl; return false; }
        expanded_COO.set_col(4);
        if (expanded_COO.get_col() != 4) { std::cout << "col size mismatch" << std::endl; return false; }
        //expanded_COO.insert(4, 4, 1.0);
        //expanded_COO.set_nnz(9);
        //if (expanded_COO.get_nnz() != 9) { std::cout << "nnz size mismatch" << std::endl; return false; }
        // expanded.mtx
	//	| 1 | 2 | 3 | 0 |
	//	| 4 | 0 | 5 | 0 |
	//	| 6 | 7 | 8 | 0 |
        //      | 0 | 0 | 0 | 1 |
        {
        std::ostringstream oss;
        std::streambuf* p_cout_streambuf = std::cout.rdbuf();
        std::cout.rdbuf(oss.rdbuf());
        expanded_COO.print_all();
        std::cout.rdbuf(p_cout_streambuf); // restore
        std::string res("%%MatrixMarket matrix coordinate real general\n");
        std::stringstream ss; // To set Float(T) output
        ss << std::scientific;
        ss << std::setprecision(std::numeric_limits<T>::max_digits10);
        ss << "%%MatrixMarket matrix coordinate real general" << std::endl;
        //ss << "4 4 9" << std::endl;
        ss << "4 4 8" << std::endl;
        ss << "1 1 " << 1.0 << std::endl;
        ss << "1 2 " << 2.0 << std::endl;
        ss << "1 3 " << 3.0 << std::endl;
        ss << "2 1 " << 4.0 << std::endl;
        ss << "2 3 " << 5.0 << std::endl;
        ss << "3 1 " << 6.0 << std::endl;
        ss << "3 2 " << 7.0 << std::endl;
        ss << "3 3 " << 8.0 << std::endl;
        // ss << "4 4 " << 1.0 << std::endl;
        if (oss.str() != ss.str()) { std::cout << "print expanded matrix mismatch" << std::endl; return false; }
        }

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

	monolish::blas::matvec(file_CRS, x, filey);
	monolish::blas::matvec(addr_CRS, x, addry);

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
