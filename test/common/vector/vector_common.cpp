#include<iostream>
#include<istream>
#include<chrono>
#include"../../test_utils.hpp"

int main(int argc, char** argv){

	if(argc!=2){
		std::cout << "error $1:vector size (size>1)" << std::endl;
		return 1;
	}

	int size = atoi(argv[1]);
	if(size<=1){return 1;}

	//monolish::util::set_log_level(3);
	//monolish::util::set_log_filename("./monolish_test_log.txt");

	//(x) monolish::vector = std::vector  = 123, 123, ..., 123
	std::vector<double> std_vec_x(size, 123.0);
  	monolish::vector<double> x(std_vec_x);


	//(y) monolish::vector = double* = 0,1,2, ..., N-1
	double* dp = (double*)malloc(sizeof(double) * size);
	for(size_t i=0; i<size; i++){
		dp[i] = i;
	}
	monolish::vector<double> y(dp, dp+size);


	//(z) monolish::vector.operator[] = 0,1,2, ..., N-1
	monolish::vector<double> z(size);
	for(size_t i=0; i<z.size(); i++){
		z[i] = i;
	}


	//monolish::vector random(1.0~2.0) vector
  	monolish::vector<double> randvec(size, 1.0, 2.0);
 
 
 	//equal operator (z = rand(1~2)) on cpu
 	z = randvec;

 	//vec element add
 	z[1] = z[1] + 111; //z[1] = rand(1~2) + 124 + 111 = 235+rand(1~2)
 
 
 	// size check
 	if(x.size() != size || y.size() != size || z.size() != size){ return 1; }
 
 
 	//gpu send and vec add
	monolish::util::send(x,y,z);
 	z += x + y; //rand(1~2) + 123+0, rand(1~2) + 123 + 1 ....
 
 	//copy (cpu and gpu)
  	monolish::vector<double> tmp = z;

	//recv vector z and tmp from gpu
	z.recv();
	tmp.recv();

 	//compare (on cpu)
 	if (tmp != z){
 		std::cout << "error, copy fail " << std::endl;
 		return 1;	
 	}

	z.recv();

 	// range-based for statement (on cpu)
 	for(auto& val : z){
 		val += 100; // z[1] = 336~337
 	}

	if( !(336 < z[1]  && z[1] < 337) ){
		std::cout << "error, z[1] = " << z[1] << std::endl;
		z.print_all();
		//z.print_all("./z.txt");
		return 1;
	}

	return 0;
}
