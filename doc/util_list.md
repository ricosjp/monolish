# Matrix storage format conversion
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
