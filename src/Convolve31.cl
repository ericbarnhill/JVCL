
kernel void Convolve31(global float* f, global float* g, global float* r, const int ri, const int rj, const int gi, const int hgi, const int hgie) 
{
	
    int i = get_global_id(0);
	int j = get_global_id(1);
	int k = get_global_id(2);

	float sum = 0;
	int ai;
	for (int p =  0; p < gi; p++) {
		ai = i + p - hgie;
		if (ai >= 0 && ai < ri) {
			sum += f[ai+j*ri+k*ri*rj]*g[gi-1-p];
		}
	}
	r[i+j*ri+k*ri*rj] = sum;

	barrier(CLK_LOCAL_MEM_FENCE);
		
}


