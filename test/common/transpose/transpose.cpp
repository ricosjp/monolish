#include<iostream>
#include<istream>
#include<sstream>
#include<iomanip>
#include<chrono>
#include<stdexcept>
#include"../../test_utils.hpp"

template<typename T, typename M>
bool test(){
	//same as test/test.mtx
	const int N = 3;
	const int NNZ = 8;

	// create C-pointer COO Matrix (same as test.mtx, but pointer is 0 origin!!)
	T* val_array = (T*)malloc(sizeof(T) * NNZ);
	int*    col_array = (int*)malloc(sizeof(int) * NNZ);
	int*    row_array = (int*)malloc(sizeof(int) * NNZ);

	// create COO type arrays
	//	| 1 | 2 | 3 |
	//	| 4 | 0 | 5 |
	//	| 6 | 7 | 8 |
	val_array[0] = 1; row_array[0] = 0; col_array[0] = 0;
	val_array[1] = 2; row_array[1] = 0; col_array[1] = 1;
	val_array[2] = 3; row_array[2] = 0; col_array[2] = 2;
	val_array[3] = 4; row_array[3] = 1; col_array[3] = 0;
	val_array[4] = 5; row_array[4] = 1; col_array[4] = 2;
	val_array[5] = 6; row_array[5] = 2; col_array[5] = 0;
	val_array[6] = 7; row_array[6] = 2; col_array[6] = 1;
	val_array[7] = 8; row_array[7] = 2; col_array[7] = 2;

	//convert C-pointer -> monolish::COO
	monolish::matrix::COO<T> ans_coo(N, N, NNZ, row_array, col_array, val_array);

    //convert
    M ret(ans_coo);

    ret.transpose();
    ret.transpose();

    bool check = true;
    for(size_t i=0; i<ans_coo.get_nnz(); i++){
        if( ret.get_row() != ans_coo.get_row() ||
            ret.get_col() != ans_coo.get_col() ||
            ret.get_nnz() != ans_coo.get_nnz()   ){

            std::cout << "error, row, col, nnz are different" << std::endl;
            std::cout << ret.get_row() << " != " << ans_coo.get_row() << std::endl;
            std::cout << ret.get_col() << " != " << ans_coo.get_col() << std::endl;
            std::cout << ret.get_nnz() << " != " << ans_coo.get_nnz() << std::endl;
            check = false;
            break;
        }

        if( ret.val[i]       != ans_coo.val[i] ||
            ret.row_index[i] != ans_coo.row_index[i] ||
            ret.col_index[i] != ans_coo.col_index[i]){

            std::cout << i << "\t" <<  ret.row_index[i] << "," << ret.col_index[i] << "," << ret.val[i] <<std::flush;
            std::cout << ", (ans: " <<  ans_coo.row_index[i] << "," << ans_coo.col_index[i] << "," << ans_coo.val[i] << ")" << std::endl;
            check = false;
        }
    }
    
    return check;
}

int main(int argc, char** argv){

	// logger option
	//monolish::util::set_log_level(3);
	//monolish::util::set_log_filename("./monolish_test_log.txt");
	
    // COO
	if( !test<double, monolish::matrix::COO<double>>() ) {return 1;}
	if( !test<float, monolish::matrix::COO<float>>() ) {return 1;}

    // Dense
// 	if( !test<double, monolish::matrix::Dense<double>>() ) {return 1;}
// 	if( !test<float, monolish::matrix::Dense<float>>() ) {return 1;}

    return 0;
}
