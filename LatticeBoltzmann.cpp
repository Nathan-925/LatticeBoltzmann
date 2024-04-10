#include <iostream>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <chrono>
#include <CL/opencl.hpp>
#include <gif.h>

#include "cl/lattice.cpp"

using namespace cl;
using namespace std::chrono;

enum State{
	FLUID, SOLID
};

void errCheck(cl_int err, std::string s){
	if(err != CL_SUCCESS){
		printf("Error %d: %s\n", err, s.c_str());
		exit(1);
	}
}

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
	int groupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()/2;
	std::cout << "Using Work Group Size: " << groupSize << std::endl;
	std::cout << "Local Memory Size: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE >() << std::endl;
	
	Context context({device});
	Program::Sources sources;
	sources.push_back({lattice, sizeof(lattice)-1});
	
	Program program(context, sources);
	std::string clArgs = "-DGROUP_SIZE="+std::to_string(groupSize);
	if(program.build({device}, clArgs.c_str()) != CL_SUCCESS){
		std::cout << "Error Building:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		exit(1);
	}
	
	CommandQueue queue(context, device);
	cl_int result;
	
	int nx = groupSize*3;
	int ny = nx;
	int numSteps = 1000;
	cl_float2 g{0, -1};
	cl_float4 s{1, 1, 1, 1};
	
	printf("Grid Size: %dx%d\n", nx, ny);
	std::cout << numSteps << " Steps" << std::endl;
	
	int bufferSize = nx*ny*sizeof(cl_float);
	Buffer f0[9];
	Buffer f1[9];
	
	int xMin = nx/3;
	int xMax = nx-nx/3;
	int yMin = ny/2-ny/8;
	int yMax = ny/2+ny/8;
	
	for(int i = 0; i < 9; i++){
		f0[i] = Buffer(context, CL_MEM_READ_WRITE, bufferSize);
		float* fMap = (float*)queue.enqueueMapBuffer(f0[i], true, CL_MAP_WRITE , 0, bufferSize);
		for(int j = 0; j < nx*ny; j++){
			int jx = j%nx;
			int jy = j/nx;
			if(i == 1 && jx > xMin && jx < xMax && jy > yMin && jy < yMax){
				fMap[j] = 1;
			}
			else{
				fMap[j] = 0;
			}
		}
		queue.enqueueUnmapMemObject(f0[i], fMap);
		
		f1[i] = Buffer(context, CL_MEM_READ_WRITE, bufferSize);
	}
	
	Buffer states(context, CL_MEM_READ_WRITE, nx*ny*sizeof(cl_uint));
	cl_uint* sMap = (cl_uint*)queue.enqueueMapBuffer(states, true, CL_MAP_WRITE , 0, nx*ny*sizeof(cl_uint));
	for(int i = 0; i < nx*ny; i++){
		sMap[i] = FLUID;
	}
	queue.enqueueUnmapMemObject(states, sMap);
	
	Buffer image(context, CL_MEM_READ_WRITE, nx*ny*sizeof(cl_uint));
	
	Kernel lbCollProp(program, "LBCollProp");
	result = lbCollProp.setArg(0, (cl_int)nx);
	errCheck(result, "LBCollProp arg 0");
	result = lbCollProp.setArg(1, (cl_int)ny);
	errCheck(result, "LBCollProp arg 1");
	result = lbCollProp.setArg(2, g);
	errCheck(result, "LBCollProp arg 2");
	result = lbCollProp.setArg(3, s);
	errCheck(result, "LBCollProp arg 3");
	result = lbCollProp.setArg(4, states);
	errCheck(result, "LBCollProp arg 4");
	
	Kernel lbExchange(program, "LBExchange");
	lbExchange.setArg(0, (cl_int)nx);
	lbExchange.setArg(1, (cl_int)ny);
	lbExchange.setArg(2, (cl_int)(nx/groupSize));
	
	Kernel writeImage(program, "writeImage");
	writeImage.setArg(0, (cl_int)nx);
	writeImage.setArg(1, (cl_int)ny);
	writeImage.setArg(2, image);
	
	Event lbCollPropEvent;
	Event lbExchangeEvent;
	Event writeImageEvent;
	std::vector<Event> lbExchangePre{lbCollPropEvent};
	std::vector<Event> writeImagePre{lbExchangeEvent};
	std::vector<Event> mapImagePre{writeImageEvent};
	
	Buffer* fIn = f0;
	Buffer* fOut = f1;
	
	GifWriter gif;
	GifBegin(&gif, "out.gif", nx, ny, 10);
	
	auto start = high_resolution_clock::now();
	
	for(int i = 0; i < numSteps; i++){
		
		for(int j = 0; j < 9; j++){
			result = lbCollProp.setArg(j+5, fIn[j]);
			errCheck(result, "LBCollProp arg "+std::to_string(j+5));
			result = lbCollProp.setArg(j+14, fOut[j]);
			errCheck(result, "LBCollProp arg "+std::to_string(j+14));
			
			result = writeImage.setArg(j+3, fOut[j]);
			errCheck(result, "writeImage arg "+std::to_string(j+3));
		}
		
		lbExchange.setArg(3, fOut[1]);
		lbExchange.setArg(4, fOut[3]);
		lbExchange.setArg(5, fOut[5]);
		lbExchange.setArg(6, fOut[6]);
		lbExchange.setArg(7, fOut[7]);
		lbExchange.setArg(8, fOut[8]);
		
		result = queue.enqueueNDRangeKernel(lbCollProp, NullRange, NDRange(nx, ny), NDRange(groupSize), NULL, &lbCollPropEvent);
		errCheck(result, "LBCollProp");
		lbExchangePre[0] = lbCollPropEvent;
		
		result = queue.enqueueNDRangeKernel(lbExchange, NullRange, NDRange(ny), NDRange(groupSize), &lbExchangePre, &lbExchangeEvent);
		errCheck(result, "LBExchange");
		writeImagePre[0] = lbExchangeEvent;
		
		result = queue.enqueueNDRangeKernel(writeImage, NullRange, NDRange(nx, ny), NDRange(groupSize), &writeImagePre, &writeImageEvent);
		errCheck(result, "writeImage");
		mapImagePre[0] = writeImageEvent;
		
		uint8_t* imgMap = (uint8_t*)queue.enqueueMapBuffer(image, true, CL_MAP_READ, 0, nx*ny*sizeof(cl_uint), &mapImagePre);
		GifWriteFrame(&gif, imgMap, nx, ny, 10);
		queue.enqueueUnmapMemObject(image, imgMap);
		
		result = writeImageEvent.wait();
		errCheck(result, "Wait");
		
		std::swap(fIn, fOut);
		
		std::cout << "\r[" << ceil((float)i*100/numSteps) << "%] " << i << std::flush;
		
	}
	std::cout << std::endl;
	
	auto stop = high_resolution_clock::now();
	
	std::cout << "Time (s): " << (duration_cast<milliseconds>(stop - start).count()/1000.0) << std::endl;
	
	GifEnd(&gif);
}