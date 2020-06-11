#include<iostream>
#include"../../test_utils.hpp"
#include"monolish_blas.hpp"

template <typename T>
void get_ans(monolish::matrix::CRS<T> &A, monolish::vector<T> &mx, monolish::vector<T> &my){

	if(mx.size() != my.size()){
		std::runtime_error("x.size != y.size");
	}

	T* x = mx.data();
	T* y = my.data();

	for(int i = 0; i < my.size(); i++)
		y[i] = 0;

	for(int i = 0; i < mx.size(); i++){
		for(int j = A.row_ptr[i]; j < A.row_ptr[i+1]; j++){
			y[i] += A.val[j] * x[A.col_ind[j]];
		}
	}
}

template <typename T>
bool test(const char* file, double tol, int iter, int check_ans){

	monolish::matrix::COO<T> COO(file);
    COO.drop(0.0);
	monolish::matrix::CRS<T> A(COO);

	monolish::vector<T> x(A.get_row(), 0.0, 1.0);
	monolish::vector<T> y(A.get_row(), 0.0, 1.0);

	monolish::vector<T> ansy(A.get_row());
	ansy = y;

	if(check_ans == 1){
		get_ans(A, x, ansy);
		monolish::util::send(A, x, y);
		monolish::blas::spmv(A, x, y);
		y.recv();
		if(ans_check<T>(y.data(), ansy.data(), y.size(), tol) == false){
			return false;
		};
	}
	monolish::util::send(A, x, y);

	auto start = std::chrono::system_clock::now();

	for(int i = 0; i < iter; i++){
		monolish::blas::spmv(A, x, y);
	}

	auto end = std::chrono::system_clock::now();
	double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()/1.0e+9;

	A.device_free();
	x.device_free();
	y.device_free();

	std::cout << "total time: " << sec << std::endl;

	return true;
}

int main(int argc, char** argv){

	if(argc!=4){
		std::cout << "error $1:matrix filename, $2: iter, $3:error check (1/0)" << std::endl;
		return 1;
	}

	char* file = argv[1];
	int iter = atoi(argv[2]);
	int check_ans = atoi(argv[3]);

	monolish::util::set_log_level(3);
	// monolish::util::set_log_filename("./monolish_test_log.txt");

	if( test<double>(file, 1.0e-8, iter, check_ans) == false){ return 1; }
	if( test<float>(file, 1.0e-8, iter, check_ans) == false){ return 1; }

	return 0;
}
