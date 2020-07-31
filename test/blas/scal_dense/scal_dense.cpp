#include<iostream>
#include"../../test_utils.hpp"
#include"monolish_blas.hpp"

template <typename T>
void get_ans(const double alpha, monolish::matrix::Dense<T> &A){

	for(int i = 0; i < A.get_nnz(); i++)
		A.val[i] = alpha * A.val[i];
}

template <typename T>
bool test(const size_t M, const size_t N, const double tol, int iter, int check_ans){

	T alpha = 123.0;
	monolish::matrix::Dense<T> A(M, N, 0.0, 1.0); // M*N matrix
	monolish::matrix::Dense<T> ansA = A;

	if(check_ans == 1){
		get_ans(alpha, ansA);

	    A.send();
		monolish::blas::mscal(alpha, A);
		A.recv();

		if(ans_check<T>(A.val.data(), ansA.val.data(), A.get_nnz(), tol) == false){
			return false;
		};
	}
    A.send();

	auto start = std::chrono::system_clock::now();

	for(int i = 0; i < iter; i++){
		monolish::blas::mscal(alpha, A);
	}

	auto end = std::chrono::system_clock::now();
	double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()/1.0e+9;

	A.device_free();

	std::cout << "average time: " << sec/iter << std::endl;

	return true;
}

int main(int argc, char** argv){

	if(argc!=5){
		std::cout << "error $1: row, $2: col, $3: iter, $4: error check (1/0)" << std::endl;
		return 1;
	}

	const size_t M = atoi(argv[1]);
    const size_t N = atoi(argv[2]);
	int iter = atoi(argv[3]);
	int check_ans = atoi(argv[4]);

	//monolish::util::set_log_level(3);
	//monolish::util::set_log_filename("./monolish_test_log.txt");

	if( test<double>(M, N, 1.0e-8, iter, check_ans) == false){ return 1; }
	if( test<float>(M, N, 1.0e-8, iter, check_ans) == false){ return 1; }

	return 0;
}
