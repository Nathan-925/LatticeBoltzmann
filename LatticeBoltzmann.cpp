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
	
	std::vector<Platform> platforms;
	Platform::get(&platforms);
	Platform platform = platforms[0];
	std::cout << "Using Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
	
	std::vector<Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	Device device = devices[0];
	std::cout << "Using Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
	std::cout << "Max Work Group Size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
	int numThreads = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	std::cout << "Using Work Group Size: " << numThreads << std::endl;
	
	Context context({device});
	Program::Sources sources;
	sources.push_back({lattice, sizeof(lattice)-1});
	
	Program program(context, sources);
	if(program.build({device}) != CL_SUCCESS){
		std::cout << "Error Building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		exit(1);
	}
	
	CommandQueue queue(context, device);
	
	auto start = high_resolution_clock::now();
	
	auto stop = high_resolution_clock::now();
	
	std::cout << "Time: " << duration_cast<nanoseconds>(stop - start).count() << std::endl;
}