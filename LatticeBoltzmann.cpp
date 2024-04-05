#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <CL/opencl.hpp>

#include "cl/lattice.cpp"

using namespace cl;
using namespace std::chrono;

enum State{
	FLUID, SOLID
};

int main(){
	
	std::vector<Platform> platforms;
	Platform::get(&platforms);
	Platform platform = platforms[0];
	std::cout << "Using Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
	
	std::vector<Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	Device device = devices[0];
	std::cout << "Using Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
	std::cout << "Max Work Group Size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
	int groupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	std::cout << "Using Work Group Size: " << groupSize << std::endl;
	
	Context context({device});
	Program::Sources sources;
	sources.push_back({lattice, sizeof(lattice)-1});
	
	Program program(context, sources);
	if(program.build({device}) != CL_SUCCESS){
		std::cout << "Error Building:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		exit(1);
	}
	
	CommandQueue queue(context, device);
	
	int nx = groupSize*3;
	int ny = nx;
	int numSteps = 10000;
	cl_float2 g{0, 0};
	cl_float4 s{1, 1, 1, 1};
	
	printf("Grid Size: %dx%d\n", nx, ny);
	std::cout << numSteps << " Steps" << std::endl;
	
	int bufferSize = nx*ny*sizeof(float);
	Buffer f0[9];
	Buffer f1[9];
	for(int i = 0; i < 9; i++){
		f0[i] = Buffer(context, CL_MEM_READ_WRITE, bufferSize);
		f1[i] = Buffer(context, CL_MEM_READ_WRITE, bufferSize);
	}
	Buffer states(context, CL_MEM_READ_WRITE, nx*ny*sizeof(unsigned int));
	
	Kernel lbCollProp(program, "LBCollProp");
	lbCollProp.setArg(0, nx);
	lbCollProp.setArg(1, ny);
	lbCollProp.setArg(2, g);
	lbCollProp.setArg(3, s);
	lbCollProp.setArg(4, states);
	
	Kernel lbExchange(program, "LBExchange");
	lbExchange.setArg(0, nx);
	lbExchange.setArg(1, ny);
	lbExchange.setArg(2, groupSize);
	
	Event lbCollPropEvent;
	Event lbExchangePre[] = {lbCollPropEvent};
	
	Buffer* fIn = f0;
	Buffer* fOut = f1;
	
	auto start = high_resolution_clock::now();
	
	for(int i = 0; i < numSteps; i++){
		
		for(int j = 0; j < 9; j++){
			lbCollProp.setArg(i+11, fIn[i]);
			lbCollProp.setArg(i+20, fOut[i]);
		}
		
		lbExchange.setArg(3, fOut[1]);
		lbExchange.setArg(4, fOut[3]);
		lbExchange.setArg(5, fOut[5]);
		lbExchange.setArg(6, fOut[6]);
		lbExchange.setArg(7, fOut[7]);
		lbExchange.setArg(8, fOut[8]);
		
		queue.enqueueNDRangeKernel(lbCollProp, NullRange, NDRange(nx, ny), NDRange(groupSize), NULL, &lbCollPropEvent);
		
		queue.enqueueNDRangeKernel(lbExchange, NullRange, NDRange(ny), NDRange(groupSize), lbExchangePre);
		
		queue.enqueueBarrierWithWaitList();
		
	}
	
	auto stop = high_resolution_clock::now();
	
	std::cout << "Time: " << duration_cast<nanoseconds>(stop - start).count() << std::endl;
}