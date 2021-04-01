# monolish data types {#data_type}

## Vector

hogehoge...

monolish::vector

## Matrix

The implementation of the matrix storage format in monolish has several features.

The matrix format has the following two attributes;
1. `editable`
2. `computable`

A matrix storage format with `editable` attribute can reference and add matrix elements.

A matrix storage format with `computable` attribute can be used for computation and has functions such as transfer to GPU (see here for a list of computation functions).


In monolish, Dense matrix is considered as one of the sparse matrix formats.
Currently, there are four matrix storage formats available: Dense, COO, CRS, and LinearOperator.
The attributes of each matrix storage format are shown below;
- monolish::matrix::Dense: `editable and computable`
- monolish::matrix::COO: `editable`
- monolish::matrix::CRS: `computable`
- monolish::matrix::LinearOperator: `computable` (special format)


Matrices stored in COO format can be easily referenced, but the computation cannot be parallelized. COO does not have any attributes to compute.

Matrices stored in CRS format can be parallelized efficiently. CRS does not have the `editable` attribute.

Matrices stored in the Dense format can be edited and computed efficiently; Dense has both attributes.

The LinearOperator format is a special format that can be created from all matrix storage formats, but it does not have an `editable` attribute. 
LinearOperator does not have all utility functions and don't support GPU.

Matirx storage formats belong to one of three attribute groups: `editable` / `computable` / `editable and computable`.
when a new matrix storage format is implemented, all the matrices with the same attribute group will have the same functionality.
For example, We will implement ELL, JAD and other storage formats, it will be implemented as a class with `computable` attributes like CRS.
(This is a beautiful implementation policy. But probably the policy will be broken....).

![](./img/matrix_convert.png)

# Matrix Util functions list {#MatUtil_list_md}

| func                          | COO (editable)              | Dense (editable/computable) | CRS (computable) |
|-------------------------------|-----------------------------|------------------------------|---------------|
| at                            | CPU                         | CPU                          | don't impl.   |
| insert                        | CPU                         | CPU                          | don't impl.   |
| row()/col()/diag()            | CPU                         | CPU/GPU                      | CPU/GPU       |
| type()                        | CPU                         | CPU                          | CPU           |
| get_data_size()               | CPU                         | CPU                          | CPU           |
| get_row()   / get_col()       | CPU                         | CPU                          | CPU           |
| print_all()                   | CPU                         | CPU                          | CPU           |
| print_all(std::string file)   | CPU                         | CPU                          | CPU           |
| send                          | don't support GPU           | CPU/GPU                      | CPU/GPU       |
| recv                          | don't support GPU           | CPU/GPU                      | CPU/GPU       |
| nonfree_recv                  | don't support GPU           | CPU/GPU                      | CPU/GPU       |
| device_free                   | don't support GPU           | CPU/GPU                      | CPU/GPU       |
