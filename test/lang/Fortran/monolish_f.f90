program f_call_c
    Use Iso_C_Binding

    !dense 2x2 matrix in COO format
    Integer :: n=2, nnz=4, ierr
    Real(C_double) :: x(2)
    Real(C_double) :: y(2)
    Real(C_double) :: val(4)
    Integer(C_int) :: col(4)
    Integer(C_int) :: row(4)

    x(1) = 1.0
    x(2) = 2.0

    ! dammy
    y(1) = 111.0
    y(2) = 111.0

    val(1) = 1.0
    val(2) = 2.0
    val(3) = 3.0
    val(4) = 4.0

    row(1) = 1
    row(2) = 1
    row(3) = 2
    row(4) = 2

    col(1) = 1
    col(2) = 2
    col(3) = 1
    col(4) = 2

    ! SpMV is...
    ! | 1 | 2 | * | 1 | = | 5  |
    ! | 3 | 4 |   | 2 |   | 11 |

!     print *, "Fortran calling C, passing"
!     print *, "i=",i,"x=",x
    ierr = monolish_spmv(n, nnz, row, col, val, x, y)

    print *, "y0=",y(1)
    print *, "y1=",y(2)
end program f_call_c
