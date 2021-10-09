all: main.c
	gcc -shared -o libmain.so		\
		-g3 -O0 -fPIC		\
		-I /root/miniconda/envs/python38/include/python3.8	\
		-L /root/miniconda/envs/python38/lib/		\
		main.c
	

