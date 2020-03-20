#include <stdio.h>
#include <iostream>

extern "C"{
int cfun_(int *ip, double *xp)
{
	int i = *ip;
	double x = *xp;

	printf("This is in C function...\n");
	printf("i = %d, x = %g\n", i, x);
	return 0;
}
}
