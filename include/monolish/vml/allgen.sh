#!/bin/bash
bash ./gen_vector_vml.sh > ./monolish_vector_vml.hpp
bash ./gen_dense_vml.sh > ./monolish_dense_vml.hpp
bash ./gen_crs_vml.sh > ./monolish_crs_vml.hpp
bash ./gen_linearoperator_vml.sh > ./monolish_linearoperator_vml.hpp
bash ./gen_tensor_dense_vml.sh > ./monolish_tensor_dense_vml.hpp
bash ./gen_tensor_crs_vml.sh > ./monolish_tensor_crs_vml.hpp
