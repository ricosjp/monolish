program f_call_c
    integer :: i=1, ierr
    double precision :: x=3.14159
    print *, "Fortran calling C, passing"
    print *, "i=",i,"x=",x
    ierr = cfun(i,x)
end program f_call_c
