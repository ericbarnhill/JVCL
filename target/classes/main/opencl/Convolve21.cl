
kernel void Convolve21(global float* f, global float* g, global float* r, const int ri, const int rj, const int gi, const int hgi, const int hgie) 
{
	
    int i = get_global_id(0);
	int j = get_global_id(1);
	
	float sum = 0;
	int ai;
	for (int p =  0; p < gi; p++) {
		ai = i + p - hgie;
		if (ai >= 0 && ai < ri) {
			sum += f[ai+j*ri]*g[gi-1-p];
		}
	}
	r[i+j*ri] = sum;

	barrier(CLK_LOCAL_MEM_FENCE);
		
}


