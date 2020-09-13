#include"../test_utils.hpp"
#include"monolish_blas.hpp"

template<typename T>
bool test(const char* file){
	monolish::matrix::COO<T> tmp_COO(file);
	monolish::matrix::CRS<T> tmp_CRS(tmp_COO);

	monolish::matrix::COO<T> COO = tmp_COO;
	monolish::matrix::CRS<T> CRS = tmp_CRS;

	//create random vector x rand(0.1~1.0)
   	monolish::vector<T> x(CRS.get_row(), 0.1, 1.0);
   	monolish::vector<T> ansy(CRS.get_row(), 0.0);

	monolish::util::send(tmp_CRS, CRS, x, ansy);

	monolish::blas::matvec(tmp_CRS, x, ansy);

	monolish::vector<T> tsty = CRS * x;

	monolish::util::recv(ansy, tsty);

	if(ansy!=tsty){
		std::cout << "error" << std::endl;
		tsty.print_all();
		ansy.print_all();
		return 1;
	}

	return 0;
}

int main(int argc, char** argv){

	if(argc!=2){
		std::cout << "error $1:matrix_name" << std::endl;
		return 1;
	}
	char* file = argv[1];

	//monolish::util::set_log_level(3);
	//monolish::util::set_log_filename("./monolish_test_log.txt");
	
	if(test<double>(file)){return 1;}
	if(test<float>(file)){return 1;}

	return 0;
}
