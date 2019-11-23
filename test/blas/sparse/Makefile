all:
	make -B -e FUNC=spmv_crs -e ARCH=cpu -f Makefile.cpu

cpu:
	make -B -e FUNC=spmv_crs -e ARCH=cpu -f Makefile.cpu

gpu:
	make -B -e FUNC=spmv_crs -e ARCH=gpu -f Makefile.gpu
	
clean:
	- rm *.out
