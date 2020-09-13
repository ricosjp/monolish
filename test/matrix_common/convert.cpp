#include"../test_utils.hpp"
#include"monolish_blas.hpp"
#define FUNC "convert"

template <typename MAT, typename T>
bool test(const size_t M, const size_t N, int iter, int check_ans){
    size_t nnzrow = 81;
    if( (nnzrow < N) ){
        nnzrow=81;
    }
    else{
        nnzrow = N - 1;
    }
    // ans COO (source)
    monolish::matrix::COO<T> ans_coo = monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

    //convert COO -> MAT
    MAT mat(ans_coo);

    // convert MAT -> result COO (dest.)
    monolish::matrix::COO<T> result_coo(mat);

    //check source == dest.
    bool check = true;
    if(check_ans == 1){
        for(size_t i=0; i<ans_coo.get_nnz(); i++){
            if( result_coo.get_row() != ans_coo.get_row() ||
                    result_coo.get_col() != ans_coo.get_col() ||
                    result_coo.get_nnz() != ans_coo.get_nnz()   ){

                std::cout << "error, row, col, nnz are different(COO2" << mat.get_format_name() << ")" << std::endl;
                std::cout << result_coo.get_row() << " != " << ans_coo.get_row() << std::endl;
                std::cout << result_coo.get_col() << " != " << ans_coo.get_col() << std::endl;
                std::cout << result_coo.get_nnz() << " != " << ans_coo.get_nnz() << std::endl;

                std::cout << "==ans==" << std::endl;
                ans_coo.print_all();
                std::cout << "==result==" << std::endl;
                result_coo.print_all();

                check = false;
                break;
            }

            if( result_coo.val[i]       != ans_coo.val[i] ||
                    result_coo.row_index[i] != ans_coo.row_index[i] ||
                    result_coo.col_index[i] != ans_coo.col_index[i]){

                std::cout << i << "\t" <<  result_coo.row_index[i] << "," << result_coo.col_index[i] << "," << result_coo.val[i] <<std::flush;
                std::cout << ", (ans: " <<  ans_coo.row_index[i] << "," << ans_coo.col_index[i] << "," << ans_coo.val[i] << ")" << std::endl;
                check = false;
            }
        }
    }

	auto start = std::chrono::system_clock::now();

    for(int i = 0; i < iter; i++){
        MAT A(ans_coo);
    }

	auto end = std::chrono::system_clock::now();
	double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()/1.0e+9;


    double time = sec / iter;
    std::cout << "func\tprec\tM\tN\ttime[sec]" << std::endl;
    std::cout << FUNC << "(COO2" << mat.get_format_name() << ")\t" << std::flush;
    std::cout << get_type<T>() << "\t" << std::flush;
    std::cout << M << "\t" << std::flush;
    std::cout << N << "\t" << std::flush;
    std::cout << time << "\t" << std::endl;

    
    return check;
}

int main(int argc, char** argv){

	if(argc!=7){
		std::cout << "error $1: precision (double or float) $2: format, $3: row, $4: col, $5: iter., 6: error check (1/0)" << std::endl;
		return 1;
	}

	const size_t M = atoi(argv[3]);
    const size_t N = atoi(argv[4]);
    int iter = atoi(argv[5]);
	int check_ans = atoi(argv[6]);

	//monolish::util::set_log_level(3);
	//monolish::util::set_log_filename("./monolish_test_log.txt");

    if(strcmp(argv[1],"double")==0){
        if( (strcmp(argv[2],"Dense") == 0) ){
            if( test<monolish::matrix::Dense<double>,double>(M, N, iter, check_ans) == false){ return 1; }
        }
        if( (strcmp(argv[2],"CRS") == 0) ){
            if( test<monolish::matrix::CRS<double>,double>(M, N, iter, check_ans) == false){ return 1; }
        }
    }

    if(strcmp(argv[1],"float")==0){
        if( (strcmp(argv[2],"Dense") == 0) ){
            if( test<monolish::matrix::Dense<float>,float>(M, N, iter, check_ans) == false){ return 1; }
        }
        if( (strcmp(argv[2],"CRS") == 0) ){
            if( test<monolish::matrix::CRS<float>,float>(M, N, iter, check_ans) == false){ return 1; }
        }
    }

    return 0;
}
