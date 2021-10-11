/**
 * Copyright 2021 RICOS Co. Ltd.
 *
 * This file is a part of ricosjp/monolish,
 * and distributed under Apache-2.0 License
 * https://github.com/ricosjp/monolish
 */

#include "cuda_runtime.h"
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " [device number]" << std::endl;
    return 1;
  }
  int device_number = std::stoi(argv[1]);

  cudaError_t cudaStatus;

  int count;
  cudaStatus = cudaGetDeviceCount(&count);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "CUDA API cudaGetDeviceCount failed" << cudaStatus << std::endl;
    return cudaStatus;
  }

  if (device_number >= count) {
    std::cerr << "Input device_number is larger than the number of GPU ("
              << device_number << " >= " << count << ")" << std::endl;
    return 1;
  }

  cudaDeviceProp prop;
  cudaStatus = cudaGetDeviceProperties(&prop, device_number);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "CUDA API cudaGetDeviceProperties failed" << std::endl;
    return cudaStatus;
  }

  std::cout << prop.major << prop.minor << std::endl;
  return 0;
}
