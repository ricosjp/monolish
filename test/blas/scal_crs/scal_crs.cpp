#include<iostream>
#include"../../test_utils.hpp"
#include"monolish_blas.hpp"

template <typename T>
void get_ans(const double alpha, monolish::matrix::CRS<T> &A){

	for(int i = 0; i < A.get_nnz(); i++)
		A.val[i] = alpha * A.val[i];
}

template <typename T>
bool test(char* file, double tol, int iter, int check_ans){

	T alpha = 123.0;
	monolish::matrix::COO<T> COO(file);
	monolish::matrix::CRS<T> A(COO);

	if(check_ans == 1){
		monolish::matrix::CRS<T> ansA = A;
		monolish::blas::mscal(alpha, A);
		get_ans(alpha, ansA);
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

	if(argc!=4){
		std::cout << "error $1:matrix filename, $2: iter, $3:error check (1/0)" << std::endl;
		return 1;
	}

	char* file = argv[1];
	int iter = atoi(argv[2]);
	int check_ans = atoi(argv[3]);

	//monolish::util::set_log_level(3);
	//monolish::util::set_log_filename("./monolish_test_log.txt");

	if( test<double>(file, 1.0e-8, iter, check_ans) == false){ return 1; }
	if( test<float>(file, 1.0e-8, iter, check_ans) == false){ return 1; }

	return 0;
}
