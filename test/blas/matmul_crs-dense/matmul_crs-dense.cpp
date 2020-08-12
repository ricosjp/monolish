#include<iostream>
#include"../../test_utils.hpp"
#include"monolish_blas.hpp"

template <typename T>
void get_ans( monolish::matrix::COO<T> &A, monolish::matrix::Dense<T> &B, monolish::matrix::Dense<T> &C){

    if(A.get_col() != B.get_row()){
        std::runtime_error("A.col != B.row");
    }
    if(A.get_row() != B.get_row()){
        std::runtime_error("A.col != B.row");
    }
    if(C.get_col() != B.get_row()){
        std::runtime_error("A.col != B.row");
    }

    //MN=MK*KN
    int M = A.get_row();
    int N = B.get_col();
    int K = A.get_col();

    for(int i=0; i<C.get_nnz(); i++){
        C.val[i] = 0;
    }

    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < K; k++){
                C.insert(i, j, C.at(i,j) + A.at(i,k) * B.at(k, j)+1);
            }
        }
    }
}

template <typename T>
bool test(const size_t M, const size_t N, const size_t K, double tol, int iter, int check_ans){

	monolish::matrix::COO<T> COO(M, N);

    for(int i=0; i<M; i++){
        COO.insert(i, i, i+1);
    }
    COO.sort(true);
    monolish::matrix::CRS<T> A(COO);

 	monolish::matrix::Dense<T> B(K, N);
    for(int i=0; i<K; i++){
        for(int j=0; j<N; j++){
            B.insert(i, j, (i+1)+(j+1));
        }
    }

 	monolish::matrix::Dense<T> C(M, N, 0.0);

	monolish::matrix::Dense<T> ansC(M, N);
 	ansC = C;

    if(check_ans == 1){

        get_ans(COO, B, ansC);

        monolish::util::send(A, B, C);
        monolish::blas::matmul(A, B, C);
        C.recv();
        if(ans_check<T>(C.val.data(), ansC.val.data(), C.get_nnz(), tol) == false){
            return false;
        };
        A.device_free();
        B.device_free();
    }

    monolish::util::send(A, B, C);

     auto start = std::chrono::system_clock::now();

    for(int i = 0; i < iter; i++){
        monolish::blas::matmul(A, B, C);
    }

    auto end = std::chrono::system_clock::now();
    double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()/1.0e+9;

    A.device_free();
    B.device_free();
    C.device_free();

    std::cout << "average time: " << sec/iter << std::endl;

    return true;
}

int main(int argc, char** argv){

	if(argc!=6){
		std::cout << "error $1: M, $2: N, $3: K, $4: iter, $5: error check (1/0)" << std::endl;
		return 1;
	}

    //MN=MK*KN
	const size_t M = atoi(argv[1]);
    const size_t N = atoi(argv[2]);
	const size_t K = atoi(argv[3]);
	int iter = atoi(argv[4]);
	int check_ans = atoi(argv[5]);

	// monolish::util::set_log_level(3);
	// monolish::util::set_log_filename("./monolish_test_log.txt");

	if( test<double>(M, N, K, 1.0e-6, iter, check_ans) == false){ return 1; }
	if( test<float>(M, N, K, 1.0e-3, iter, check_ans) == false){ return 1; }

	return 0;
}
