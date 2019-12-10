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
bool test(monolish::matrix::CRS<T> A, monolish::vector<T> x, monolish::vector<T> y, double tol, int iter, int check_ans){

	monolish::vector<double> ansy(A.get_row());
	ansy = y.copy();


	monolish::blas::spmv(A, x, y);

	if(check_ans == 1){
		get_ans(A, x, ansy);
		if(ans_check<T>(y.data(), ansy.data(), y.size(), tol) == false){
			return false;
		};
	}

	auto start = std::chrono::system_clock::now();

	for(int i = 0; i < iter; i++){
		std::cout << "iter:" << i << "\t" <<std::flush;
		monolish::blas::spmv(A, x, y);
	}

	auto end = std::chrono::system_clock::now();
	double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()/1.0e+9;

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

	monolish::matrix::COO<double> COO(file);
	monolish::matrix::CRS<double> A(COO);

	std::random_device random;
	monolish::vector<double> x(A.get_row(), 1);
	monolish::vector<double> y(A.get_row(), 1);

	bool result;
	if( test<double>(A, x, y, 1.0e-8, iter, check_ans) == false){ return 1; }

	y.print_all();

	return 0;
}
