
float2 mult(float2 a, float2 b){ 
    float2 res; 
    res.x = a.x * b.x - a.y * b.y;
    res.y = a.x * b.y + a.y * b.x;
    return res; 
}

kernel void Convolve31Complex(global float2* f, global float2* g, global float2* r, const int ri, const int rj, const int gi, const int hgi, const int hgie) 
{
	
    int i = get_global_id(0);
	int j = get_global_id(1);
	int k = get_global_id(2);
	float2 sum = 0;
	int ai;
	for (int p =  0; p < gi; p++) {
		ai = i + p - hgie;
		if (ai >= 0 && ai < ri) {
			sum += mult(f[ai+j*ri+k*ri*rj],g[gi-1-p]);
		}
	}
	r[i+j*ri+k*ri*rj] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);
		
}

