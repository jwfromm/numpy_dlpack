all: numpy_dlpack.c
	gcc -shared -o libmain.so -fPIC numpy_dlpack.c
