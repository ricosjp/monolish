#!/bin/bash
sh ./gen_vector_blas.sh > ./monolish_vector_blas.hpp
sh ./gen_matvec_blas.sh > ./monolish_matvec_blas.hpp
sh ./gen_matrix_blas.sh > ./monolish_matrix_blas.hpp
sh ./gen_tensor_blas.sh > ./monolish_tensor_blas.hpp
sh ./gen_tensvec_blas.sh > ./monolish_tensvec_blas.hpp
sh ./gen_tensmat_blas.sh > ./monolish_tensmat_blas.hpp
sh ./gen_mattens_blas.sh > ./monolish_mattens_blas.hpp
