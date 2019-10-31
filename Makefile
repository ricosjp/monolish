all:cpu

cpu:
	make -j -f Makefile.cpu

gpu:
	make -j -f Makefile.gpu

libs:
	make -j -f Makefile.cpu libs

tst:
	cd test; make

clean:
	- make -f Makefile.cpu clean 
	- make -f Makefile.gpu clean 
