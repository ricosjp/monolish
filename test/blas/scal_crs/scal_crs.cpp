#include<iostream>
#include"../../test_utils.hpp"
#include"monolish_blas.hpp"

template <typename T>
void get_ans(monolish::matrix::CRS<T> &A, const double alpha){

	for(int i = 0; i < A.get_nnz(); i++)
		A.val[i] = alpha * A.val[i];
}

template <typename T>
bool test(monolish::matrix::CRS<T> A, double alpha, double tol, int iter, int check_ans){

	monolish::matrix::CRS<double> ansA = A;

	monolish::blas::mscal(alpha, A);

	if(check_ans == 1){
		get_ans(ansA, alpha);
		if(ans_check<T>(A.val.data(), ansA.val.data(), A.get_nnz(), tol) == false){
			return false;
		};
	}

	auto start = std::chrono::system_clock::now();

	for(int i = 0; i < iter; i++){
		monolish::blas::mscal(alpha, A);
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

	//monolish::util::set_log_level(3);
	//monolish::util::set_log_filename("./monolish_test_log.txt");

	monolish::matrix::COO<double> COO(file);
	monolish::matrix::CRS<double> A(COO);

	double alpha = 123.0;

	bool result;
	if( test<double>(A, alpha, 1.0e-8, iter, check_ans) == false){ return 1; }

	return 0;
}
