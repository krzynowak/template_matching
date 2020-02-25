#Simple makefile for nvcc compilation via cygwin on Windows 10

result: main.obj data_io.obj matcher.obj
	nvcc --gpu-architecture=compute_61 main.obj data_io.obj matcher.obj -o template_match


main.obj:
	nvcc --gpu-architecture=compute_61 main.cu -c main.obj
data_io.obj:
	nvcc --gpu-architecture=compute_61 data_io.cu -c data_io.obj
matcher.obj:
	nvcc --gpu-architecture=compute_61 matcher.cu -c matcher.obj

clean:
	rm -f *.obj *.exe *.exp *.lib