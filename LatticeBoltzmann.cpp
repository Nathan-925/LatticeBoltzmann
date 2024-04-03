#include <iostream>
#include <cmath>
#include <chrono>
#include <CL/opencl.hpp>

#include "cl/lattice.cpp"

using namespace cl;
using namespace std::chrono;

float* getLatticeVectors(float c){
	
	return new float[]{
			0,0,
			c,0,
			0,c,
			-c,0,
			0,-c,
			c,c,
			-c,c,
			-c,-c,
			c,-c};
}

float* getMatrixM(float c){
	float* e = getLatticeVectors(c);
	float* m = new float[9*9];
	for(int i = 0; i < 9; i++){
		m[0*9+i] = 1;
		m[1*9+i] = e[2*i];
		m[2*9+i] = e[2*i+1];
		m[3*9+i] = 3*(pow(e[2*i], 2) + pow(e[2*i+1], 2)) - 2*pow(c, 2);
		m[4*9+i] = pow(e[2*i], 2) - pow(e[2*i+1], 2);
		m[5*9+i] = e[2*i]*e[2*i+1];
		m[6*9+i] = (3*(pow(e[2*i], 2) + pow(e[2*i+1], 2)) - 4*pow(c, 2))*e[2*i];
		m[7*9+i] = (3*(pow(e[2*i], 2) + pow(e[2*i+1], 2)) - 4*pow(c, 2))*e[2*i+1];
		m[8*9+i] = 0.5*(9*pow(pow(e[2*i], 2) + pow(e[2*i+1], 2), 2) - 15*pow(c, 2)*(pow(e[2*i], 2) + pow(e[2*i+1], 2)) + 2*pow(c, 4));
	}
	delete e;
	return m;
}

int main(){
	float* m = getMatrixM(1.5);
	for(int i = 0; i < 9; i++){
		for(int j = 0; j < 9; j++){
			printf("%.2f\t", m[i*9+j]);
		}
		std::cout << std::endl;
	}
	
	std::vector<Platform> platforms;
	Platform::get(&platforms);
	Platform platform = platforms[0];
	std::cout << "Using Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
	
	std::vector<Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	Device device = devices[0];
	std::cout << "Using Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
	std::cout << "Max Work Size: " << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0] << std::endl;
	
	Context context({device});
	Program::Sources sources;
	sources.push_back({lattice, sizeof(lattice)-1});
	
	Program program(context, sources);
	if(program.build({device}) != CL_SUCCESS){
		std::cout << "Error Building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		exit(1);
	}
	
	int n = 100;
	Buffer bufferA(context, CL_MEM_READ_WRITE, sizeof(int)*n);
	Buffer bufferB(context, CL_MEM_READ_WRITE, sizeof(int)*n);
	Buffer bufferC(context, CL_MEM_READ_WRITE, sizeof(int)*n);
	
	int a[n];
	int b[n];
	for(int i = 0; i < n; i++){
		a[i] = i;
		b[i] = i;
	}
	
	CommandQueue queue(context, device);
	queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(int)*n, a);
	queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, sizeof(int)*n, b);
	
	KernelFunctor<Buffer&, Buffer&, Buffer&> add(program, "add");
	
	auto start = high_resolution_clock::now();
	add(EnqueueArgs(queue, NDRange(n)), bufferA, bufferB, bufferC);
	auto stop = high_resolution_clock::now();
	
	int c[n];
	queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(int)*n, c);
	
	std::cout << "A\tB\tC\n";
	for(int i = 0; i < n; i++){
		std::cout << a[i] << "\t" << b[i] << "\t" << c[i] << std::endl;
	}
	
	std::cout << "Time: " << duration_cast<nanoseconds>(stop - start).count() << std::endl;
}