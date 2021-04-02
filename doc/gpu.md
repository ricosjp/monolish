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
- @ref monolish::vector.device_free() "device\_free()"
- @ref monolish::vector.get_device_mem_stat() "get_device_mem_stat()"

When libmonolish\_cpu.so is linked, send() and recv() do nothing, the CPU and GPU code can be shared.

When libmonolish\_gpu.so is linked, these functions enable data communication with the GPU.

Each class is mapped to GPU memory by using the send() function.
The class to which the data is transferred to the GPU behaves differently, and it becomes impossible to perform operations on the elements of vectors and matrices.

Whether the data has been transferred to the GPU can be obtained by the get\_device\_mem\_stat() function.

The data mapped to the GPU is released from the GPU by recv() or device\_free().

For developers, there is a nonfree\_recv() function that receives data from the GPU without freeing the GPU memory.
However, in the current version, there is no way to explicitly change the status of GPU memory, so it is not useful for most users.

GPU progrms using monolish are implemented with the following flow in mind.
1. First, the CPU generates data, and then
2. Transfer data from CPU to GPU
3. Calculate on GPU,
4. Finally, receive data from GPU to CPU

It is important to be aware that send and recv are not performed many times in order to reduce transfers.

## Compute innerproduct on GPU
First, a simple inner product program for GPU is shown below:

\code{.cpp}
#include<iostream>
#include<monolish_blas.hpp>
void main(){
  size_t N = 100;
  monolish::vector<double> x(N, 1.0); // x = {1,1,...,1}, length N
  monolish::vector<double> y(N, 1.0, 2.0); // Random vector length N with values in the range 1.0 to 2.0

  monolish::util::send(x, y); //send data to GPU

  double ans = monolish::blas::dot(x, y); // compute innerproduct

  std::cout << ans << std::endl;
}
\endcode

- Each class has `send()` and `recv()` functions.
- monolish::util::send() is a convenient util function that can take variable length arguments.
- The scalar values are automatically synchronized between the CPU and GPU.
- The BLAS and VML functions in monolish automatically call the GPU functions when they receive data that has already been sent.
- When libmonolish\_cpu.so is linked, send() and recv() do nothing, the CPU and GPU code can be shared.
- In this program, `x` and `y` do not need to receive to the CPU, so the memory management was left to the automatic release by the destructor.

## Solve Ax=b on GPU
The following is a sample program that solves a linear equations; Ax=b using the conjugate gradient method with jacobi preconditioner on GPU.

Surprisingly, this program requires only two lines of changes from the CPU program.

\code{.cpp}
#include<iostream>
#include<monolish_equation.hpp>

void main(){
  monolish::matrix::COO<double> A_COO("test_matrix.mtx") // Input from file
  // Edit the matrix as needed //
  // Execute A_COO.sort() after editing the matrix //
  monolish::matrix::CRS<double> A(A_COO) // Create CRS format and convert from COO format

  monolish::vector<double> x(A.get_row(), 1.0, 2.0); length A.row
  monolish::vector<double> b(A.get_row(), 1.0, 2.0); // Random vector length N with values in the range 1.0 to 2.0

  monolish::util::send(A, x, b);

  monolish::equation::CG<monolish::matrix::CRS, double> solver; // Create CG class

  monolish::equation::Jacobi precond; // create jacobi preconditioner
  solver.set_create_precond(precond); // set preconditioner creation to CG solver
  solver.set_apply_precond(precond); // set preconditioner application function to CG solver

  solver.set_tol(1.0e-12);
  solver.set_maxiter(A.get_row()); 

  monolish::util::solver_check(solver.solve(A, x, b)); //solver Ax=b by CG with jacobi

  monolish::util::recv(x);

  x.print_all();
}
\endcode
- After creating the vectors and matrix A, send the data to the GPU using the monolish::util::send() function.
- For x that requires output, explicitly receive data to the CPU using the @ref monolish::vector.recv() "recv()" function. At this time, the memory of x on the GPU is released.
- The memory of A, A_COO, and b is released by the destructor at the end of the function.

## Environment variable
- LIBOMPTARGET\_DEBUG= [1 or 0]
- CUDA\_VISIBLE\_DEVICES= [Device num.]
