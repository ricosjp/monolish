# GPU device acceleration {#gpu_dev}

## Introduction
The following four classes have the `computable` attribute:
- monolish::vector
- monolish::view1D
- monolish::matrix::CRS
- monolish::matrix::Dense

These classes support computing on the GPU and have five functions for GPU programming.
- @ref monolish::vector.send() "send()"
- @ref monolish::vector.recv() "recv()"
- @ref monolish::vector.device_free() "device_free()"
- @ref monolish::vector.get_device_mem_stat() "get_device_mem_stat()"

When `libmonolish_cpu.so` is linked, send() and recv() do nothing, the CPU and GPU code can be shared.

When `libmonolish_gpu.so` is linked, these functions enable data communication with the GPU.

Each class is mapped to GPU memory by using the send() function.
The class to which the data is transferred to the GPU behaves differently, and it becomes impossible to perform operations on the elements of vectors and matrices.

Whether the data has been transferred to the GPU can be obtained by the get\_device\_mem\_stat() function.

The data mapped to the GPU is released from the GPU by recv() or device\_free().

Most of the functions are executed on the CPU or GPU according to get\_device\_mem\_stat() .
A copy constructor is special, it is a function that copies an instance of a class. So both CPU and GPU data will be copied.

For developers, there is a nonfree\_recv() function that receives data from the GPU without freeing the GPU memory.
However, in the current version, there is no way to explicitly change the status of GPU memory, so it is not useful for most users.

GPU programs using monolish are implemented with the following flow in mind.
1. First, the CPU generates data, and then
2. Transfer data from CPU to GPU
3. Calculate on GPU,
4. Finally, receive data from GPU to CPU

It is important to be aware that send and recv are not performed many times in order to reduce transfers.

## Compute innerproduct on GPU
A simple inner product program for GPU is shown below:

\code{.cpp}
#include<iostream>
#include<monolish_blas.hpp>
int main(){

  // Output log if you need
  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_log.txt");

  size_t N = 100;

  // x = {1,1,...,1}, length N
  monolish::vector<double> x(N, 1.0); 

  // Random vector length N with values in the range 1.0 to 2.0
  monolish::vector<double> y(N, 1.0, 2.0); 

  //send data to GPU
  monolish::util::send(x, y); 

  // compute innerproduct
  double ans = monolish::blas::dot(x, y); 

  std::cout << ans << std::endl;

  return 0;
}
\endcode

This sample code can be found in `/sample/blas/innerproduct/`.

This program can be compiled by the following command.
```
g++ -O3 -fopenmp innerproduct.cpp -o innerproduct_cpu.out -lmonolish_gpu
```

The following command runs this.
``` 
./innerproduct_gpu.out
```

A description of this program is given below:
- Each class has `send()` and `recv()` functions.
- monolish::util::send() is a convenient util function that can take variable length arguments.
- The scalar values are automatically synchronized between the CPU and GPU.
- The BLAS and VML functions in monolish automatically call the GPU functions when they receive data that has already been sent.
- When libmonolish\_cpu.so is linked, send() and recv() do nothing, the CPU and GPU code can be shared.
- In this program, `x` and `y` do not need to receive to the CPU, so the memory management was left to the automatic release by the destructor.

For a more advanced example, sample programs that implement CG methods using monolish::BLAS and monolish::VML can be found in `/sample/blas/cg_impl/`.

## Solve Ax=b on GPU
The following is a sample program that solves a linear equations; Ax=b using the conjugate gradient method with jacobi preconditioner on GPU.

This program requires only two lines of changes from the CPU program.

\code{.cpp}
#include<iostream>
#include<monolish_equation.hpp>

int main(){

  // Output log if you need
  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_log.txt");

  monolish::matrix::COO<double> A_COO("sample.mtx"); // Input from file

  // Edit the matrix as needed //
  // Execute A_COO.sort() after editing the matrix //

  monolish::matrix::CRS<double> A(A_COO); // Create CRS format and convert from COO format

  // Length A.row()
  // Random vector length A.row() with values in the range 1.0 to 2.0
  monolish::vector<double> x(A.get_row(), 1.0, 2.0); 
  monolish::vector<double> b(A.get_row(), 1.0, 2.0); 

  monolish::util::send(A, x, b);

  // Create CG class
  monolish::equation::CG<monolish::matrix::CRS<double>, double> solver; 

  // create jacobi preconditioner
  monolish::equation::Jacobi<monolish::matrix::CRS<double>, double> precond; 

  // Set preconditioner to CG solver
  solver.set_create_precond(precond); 
  solver.set_apply_precond(precond);

  // Set solver options
  solver.set_tol(1.0e-12);
  solver.set_maxiter(A.get_row()); 

  // if you need residual history
  // solver.set_print_rhistory(true);
  // solver.set_rhistory_filename("./a.txt");

  //Solve Ax=b by CG with jacobi
  monolish::util::solver_check(solver.solve(A, x, b)); 

  // Recv x from GPU
  monolish::util::recv(x);

  // Show answer
  x.print_all();
  
  return 0;
}
\endcode

This sample code can be found in `/sample/equation/cg/`.

- After creating the vectors and matrix A, send the data to the GPU using the monolish::util::send() function.
- For x that requires output, explicitly receive data to the CPU using the @ref monolish::vector.recv() "recv()" function. At this time, the memory of x on the GPU is released.
- The CPU/GPU memory of A, A_COO, and b is released by the destructor at the end of the function.

A sample program for a templated linear equation solver can be found at `sample/equation/templated_solver`.

## Environment variable
- LIBOMPTARGET\_DEBUG= [1 or 0] : Output debug information on OpenMP Offloading at runtime

- CUDA\_VISIBLE\_DEVICES= [Device num.] : Specify GPU device number

