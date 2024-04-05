enum State{
	FLUID, SOLID
};

constant float m[81] = {
	1, 1, 1, 1, 1, 1, 1, 1, 1,
	0, 1, 0, -1, 0, 1, -1, -1, 1,
	0, 0, 1, 0, -1, 1, 1, -1, -1,
	-2, 1, 1, 1, 1, 4, 4, 4, 4,
	0, 1, -1, 1, -1, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 1, -1, 1, -1,
	0, -1, 0, 1, 0, 2, -2, -2, 2,
	0, 0, -1, 0, 1, 2, 2, -2, -2,
	1, -2, -2, -2, -2, 4, 4, 4, 4
};

constant float mInverse[81] = {
	4.0/9, 0, 0, -2.0/9, 0, 0, 0, 0, 1.0/9,
	1.0/9, 1.0/3, 0, 1.0/36, 1.0/4, 0, -1.0/6, 0, -1.0/18,
	1.0/9, 0, 1.0/3, 1.0/36, -1.0/4, 0, 0, -1.0/6, -1.0/18,
	1.0/9, -1.0/3, 0, 1.0/36, 1.0/4, 0, 1.0/6, 0, -1.0/18,
	1.0/9, 0, -1.0/3, 1.0/36, -1.0/4, 0, 0, 1.0/6, -1.0/18,
	1.0/36, 1.0/12, 1.0/12, 1.0/36, 0, 1.0/4, 1.0/12, 1.0/12, 1.0/36,
	1.0/36, -1.0/12, 1.0/12, 1.0/36, 0, -1.0/4, -1.0/12, 1.0/12, 1.0/36,
	1.0/36, -1.0/12, -1.0/12, 1.0/36, 0, 1.0/4, -1.0/12, -1.0/12, 1.0/36,
	1.0/36, 1.0/12, -1.0/12, 1.0/36, 0, -1.0/4, 1.0/12, -1.0/12, 1.0/36,
};


void kernel LBCollProp(const int nx, const int ny, const float2 g, const float4 s,
					   global unsigned int* states, local float* feOut, local float* fwOut, local float* fneOut, local float* fnwOut, local float* fswOut, local float* fseOut,
					   global float* fr0, global float* fe0, global float* fn0, global float* fw0, global float* fs0, global float* fne0, global float* fnw0, global float* fsw0, global float* fse0,
					   global float* fr1, global float* fe1, global float* fn1, global float* fw1, global float* fs1, global float* fne1, global float* fnw1, global float* fsw1, global float* fse1){
	const int gx = get_global_id(0);
	const int gy = get_global_id(1);
	const int lx = get_local_id(0);
	const int localSize = get_local_size(0);
	
	int k = nx*gy + gx;
	
	float fIn[9];
	fIn[0] = fr0[k];
	fIn[1] = fe0[k];
	fIn[2] = fn0[k];
	fIn[3] = fw0[k];
	fIn[4] = fs0[k];
	fIn[5] = fne0[k];
	fIn[6] = fnw0[k];
	fIn[7] = fsw0[k];
	fIn[8] = fse0[k];
	
	unsigned int state = states[k];
	if(state == FLUID){
		float mom[9] = {0};
		for(int i = 0; i < 9; i++){
			for(int j = 0; j < 9; j++){
				mom[i] += m[i*9+j]*fIn[j];
			}
		}
		
		float kVec[9];
		kVec[0] = 0;
		kVec[1] = g.x;
		kVec[2] = g.y;
		kVec[3] = -s.x*(mom[3] - 3*(pow(mom[1], 2) + pow(mom[1], 2)));
		kVec[4] = -s.y*(mom[4] - (pow(mom[1], 2) - pow(mom[1], 2)));
		kVec[5] = -s.y*(mom[5] - mom[1]*mom[2]);
		kVec[6] = -s.z*mom[6];
		kVec[7] = -s.z*mom[7];
		kVec[8] = -s.w*mom[8];
		
		for(int i = 0; i < 9; i++){
			for(int j = 0; j < 9; j++){
				fIn[i] += mInverse[i*9+j]*kVec[j];
			}
		}
	}
	else if(state == SOLID){
		
	}
	
	if(lx == 0){
		feOut[lx+1] = fIn[1];
		fneOut[lx+1] = fIn[5];
		fseOut[lx+1] = fIn[8];
		
		fwOut[localSize-1] = fIn[3];
		fnwOut[localSize-1] = fIn[6];
		fswOut[localSize-1] = fIn[7];
	}
	else if(lx == localSize-1){
		feOut[0] = fIn[1];
		fneOut[0] = fIn[5];
		fseOut[0] = fIn[8];
		
		fwOut[lx-1] = fIn[3];
		fnwOut[lx-1] = fIn[6];
		fswOut[lx-1] = fIn[7];
	}
	else{
		feOut[lx+1] = fIn[1];
		fneOut[lx+1] = fIn[5];
		fseOut[lx+1] = fIn[8];
		
		fwOut[lx-1] = fIn[3];
		fnwOut[lx-1] = fIn[6];
		fswOut[lx-1] = fIn[7];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(gx > 0 && gy > 0 && gx < nx-1 && gy < ny-1){
		fr1[k] = fIn[0];
		fe1[k] = feOut[lx];
		fw1[k] = fwOut[lx];
		
		k = nx*(gy+1) + gx;
		fn1[k] = fIn[2];
		fne1[k] = fneOut[lx];
		fnw1[k] = fnwOut[lx];
		
		k = nx*(gy-1) + gx;
		fs1[k] = fIn[4];
		fse1[k] = fseOut[lx];
		fsw1[k] = fswOut[lx];
	}
}

void kernel LBExchange(const int nx, const int ny, const int groupWidth, global float* fe1, global float* fw1, global float* fne1, global float* fnw1, global float* fsw1, global float* fse1){
	const int gx = get_global_id(0);
	const int gy = get_global_id(1);
	const int lx = get_local_id(0);
	
	int xStart, xTarget;
	int start, target;
	int i;
	for(i = 0; i < groupWidth; i++){
		xStart = i*groupWidth;
		xTarget = xStart + groupWidth;
		start = nx*gy + xStart;
		target = nx*gy + xTarget;
		
		fw1[target] = fw1[start];
		fnw1[target] = fnw1[start];
		fsw1[target] = fsw1[start];
	}
	
	for(i = groupWidth-1; i >= 0; i--){
		xTarget = i*groupWidth;
		xStart = xTarget + groupWidth;
		target = nx*gy + xTarget;
		start = nx*gy + xStart;
		
		fe1[target] = fe1[start];
		fne1[target] = fne1[start];
		fse1[target] = fse1[start];
	}
}