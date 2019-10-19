

run_clean:
	g++ host/opencl.cpp -o ./executables/exe.out -std=c++11 -Wall -framework OpenCL
	./executables/exe.out

stringify:
	python2 ./build_tools/stringify.py

run_clean_db:
	g++ host/opencl.cpp -o ./executables/exe.out -std=c++11 -Wall -framework OpenCL
	gdb ./executables/exe.out



run: 
	$(ECHO)  g++ host/main.cpp -o ./executables/exe.out -std=c++11 -Wall -framework OpenCL
	$(ECHO) rm ./IO/output.csv 
	$(ECHO)  ./executables/exe.out >> ./IO/output.csv
	$(ECHO)  python2 ./visualization/data_post_processing.py

new_run:
	g++ new_host/opencl.cpp -o ./executables/new_exe.out -Wall -framework OpenCL
	./executables/new_exe.out 



rundb:
	g++ -g opencl.cpp -o ./executables/exe.out -Wall -framework OpenCL
	gdb ./exe.out



