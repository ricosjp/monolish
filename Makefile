all:cpu

cpu:
	make -j -f Makefile.cpu

gpu:
	make -j -f Makefile.gpu

tst:
	cd test; make

clean:
	- make -f Makefile.cpu clean 
	- make -f Makefile.gpu clean 
