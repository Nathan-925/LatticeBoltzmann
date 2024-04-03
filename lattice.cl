enum State{
	FLUID, SOLID
};

void kernel LBCollProp(const int nx, const int ny, global unsigned int* states,
					   global float* fr0, global float* fe0, global float* fn0, global float* fw0, global float* fs0, global float* fne0, global float* fnw0, global float* fsw0, global float* fse0,
					   global float* fr1, global float* fe1, global float* fn1, global float* fw1, global float* fs1, global float* fne1, global float* fnw1, global float* fsw1, global float* fse1){
	const int gx = get_global_id(0);
	const int gy = get_global_id(1);
	const int lx = get_local_id(0);
	const int localSize = get_local_size(0);
	
	int k = nx*gy + gx;
	
	local float feOut[localSize];
	local float fwOut[localSize];
	local float fneOut[localSize];
	local float fnwOut[localSize];
	local float fseOut[localSize];
	local float fswOut[localSize];
	
	float frIn = fr0[k];
	float feIn = fe0[k];
	float fnIn = fn0[k];
	float fwIn = fw0[k];
	float fsIn = fs0[k];
	float fneIn = fne0[k];
	float fnwIn = fnw0[k];
	float fswIn = fsw0[k];
	float fseIn = fse0[k];
	
	if(states[k] == FLUID){
		
	}
	else if(states[k] == SOLID){
		
	}
	
	if(lx == 0){
		feOut[lx+1] = feIn;
		fneOut[lx+1] = fneIn;
		fseOut[lx+1] = fseIn;
		
		fwOut[localSize-1] = fwIn;
		fnwOut[localSize-1] = fnwIn;
		fswOut[localSize-1] = fswIn;
	}
	else if(lx == localSize-1){
		feOut[0] = feIn;
		fneOut[0] = fneIn;
		fseOut[0] = fseIn;
		
		fwOut[lx-1] = fwIn;
		fnwOut[lx-1] = fnwIn;
		fswOut[lx-1] = fswIn;
	}
	else{
		feOut[lx+1] = feIn;
		fneOut[lx+1] = fneIn;
		fseOut[lx+1] = fseIn;
		
		fwOut[lx-1] = fwIn;
		fnwOut[lx-1] = fnwIn;
		fswOut[lx-1] = fswIn;
	}
	
	barrier();
	
	if(gx > 0 && gy > 0 && gx < nx-1 && gy < ny-1){
		fr1[k] = frIn;
		fe1[k] = feOut[lx];
		fw1[k] = fwOut[lx];
		
		k = nx*(gy+1) + gx;
		fn1[k] = fnIn;
		fne1[k] = fneOut[lx];
		fnw1[k] = fnwOut[lx];
		
		k = nx*(gy-1) + gx;
		fs1[k] = fsIn;
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