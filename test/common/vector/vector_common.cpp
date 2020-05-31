#include<iostream>
#include<istream>
#include<chrono>
#include<vector>
#include"../../test_utils.hpp"

template<typename T>
bool test(const size_t size){

	//(x) monolish::vector = std::vector  = 123, 123, ..., 123
	std::vector<T> std_vec_x(size, 123.0);
  	monolish::vector<T> x(std_vec_x);


	//(y) monolish::vector = T* = 0,1,2, ..., N-1
	T* dp = (T*)malloc(sizeof(T) * size);
	for(size_t i=0; i<size; i++){
		dp[i] = i;
	}
	monolish::vector<T> y(dp, dp+size);


	//(z) monolish::vector.operator[] = 0,1,2, ..., N-1
	monolish::vector<T> z(size);
	for(size_t i=0; i<z.size(); i++){
		z[i] = i;
	}


	//monolish::vector random(1.0~2.0) vector
  	monolish::vector<T> randvec(size, 1.0, 2.0);
 
 	//equal operator (z = rand(1~2)) on cpu
 	z = randvec;

 	//vec element add
 	z[1] = z[1] + 111; //z[1] = rand(1~2) + 124 + 111 = 235+rand(1~2)
 
 
 	// size check
 	if(x.size() != size || y.size() != size || z.size() != size){ return false; }
 
 
 	//gpu send and vec add
	monolish::util::send(x,y,z);
 	z += x + y; //rand(1~2) + 123+0, rand(1~2) + 123 + 1 ....
 
 	//copy (cpu and gpu)
  	monolish::vector<T> tmp = z;

	//recv vector z and tmp from gpu
	z.recv();
	tmp.recv();

 	//compare (on cpu)
 	if (tmp != z){
 		std::cout << "error, copy fail " << std::endl;
 		return false;	
 	}

 	// range-based for statement (on cpu)
 	for(auto& val : z){
 		val += 100; // z[1] = 336~337
 	}

	if( !(336 < z[1]  && z[1] < 337) ){
		std::cout << "error, z[1] = " << z[1] << std::endl;
		z.print_all();
		//z.print_all("./z.txt");
		return false;
	}

	return 0;
}

int main(int argc, char** argv){

	if(argc!=2){
		std::cout << "error $1:vector size (size>1)" << std::endl;
		return 1;
	}

	int size = atoi(argv[1]);
	if(size<=1){return 1;}

	// monolish::util::set_log_level(3);
	// monolish::util::set_log_filename("./monolish_test_log.txt");
	
	if (test<double>(size)) {return 1;}
	if (test<float>(size)) {return 1;}

	return 0;
}
