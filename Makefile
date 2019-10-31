all:cpu

cpu:
	make -f Makefile.cpu

gpu:
	make -f Makefile.gpu

tst:
	cd test; make

clean:
	- make -f Makefile.cpu clean 
	- make -f Makefile.gpu clean 
