# Matrix Util list {#MatUtil_list_md}

| func                          | COO(Util可能) | Dense(Util可能) | CRS(演算のみ) | ELL(演算のみ) |
|-------------------------------|---------------|-----------------|---------------|---------------|
| at                            | CPU           | CPU             | 作らない      | 作らない      |
| insert                        | CPU           | CPU             | 作らない      | 作らない      |
| row()   / col() / diag()      | CPU           | CPU/GPU         | CPU/GPU       | none          |
| type()                        | CPU           | CPU             | CPU           | none          |
| get_data_size()               | CPU           | CPU             | CPU           | none          |
| get_row()   / get_col()       | CPU           | CPU             | CPU           | none          |
| print_all()                   | CPU           | CPU             | CPU           | none          |
| print_all(std::string   file) | CPU           | CPU             | none          | none          |
| create   from array           | CPU           | CPU             | CPU           | none          |
| create   from std::vector     | CPU           | CPU             | CPU           | none          |
| create   from MM format file  | CPU           | CPU             | 作らない      | 作らない      |
| create   from COO             | CPU           | CPU             | CPU           | none          |
| create   from Dense           | CPU           | CPU             | 作らない      | 作らない      |
| sort(COOだけ)                 | CPU           | N/A             | N/A           | N/A           |
| send                          | N/A           | CPU/GPU         | CPU/GPU       | none          |
| recv                          | N/A           | CPU/GPU         | CPU/GPU       | none          |
| nonfree_recv                  | N/A           | CPU/GPU         | CPU/GPU       | none          |
| device_free                   | N/A           | CPU/GPU         | CPU/GPU       | none          |
| slice                         | none          | none            | 作らない      | 作らない      |
| insert_row()   / insert_col() | none          | none            | 作らない      | 作らない      |
| norm                          | none          | none            | 作らない      | 作らない      |
| drop_zero()                   | none          | none            | 作らない      | 作らない      |
| is_symm()                     | none          | none            | 作らない      | 作らない      |
| is_diag_dominant              | none          | none            | 作らない      | 作らない      |
| is_positive_definite()        | none          | none            | 作らない      | 作らない      |
| is_positive_definite()        | none          | none            | 作らない      | 作らない      |
