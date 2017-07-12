float2 mult(float2 a, float2 b){ 
    float2 res; 
    res.x = a.x * b.x - a.y * b.y;
    res.y = a.x * b.y + a.y * b.x;
    return res; 
}

kernel void Convolve2dComplex(global float2* f, global float2* g, global float2* r, const int ri, const int rj, const int gi, const int gj, const int hgi, const int hgj, const int hgie, const int hgje) {
	
    int i = get_global_id(0);
    int j = get_global_id(1);
	int ai, aj, fInd, gInd;
	float2 sum = 0;
	for (int p = 0; p < gi; p++) {
		for (int q = 0; q < gj; q++) {
			ai = i + (p - hgie);
			aj = j + (q - hgje);
			if (ai >= 0 && ai < ri) {
				if (aj >= 0 && aj < rj) {
					fInd = aj*ri + ai;
					gInd = (gj-1-q)*gi + (gi-1-p);
					sum += mult(f[fInd], g[gInd]);
				}
			}
		}
	}
	r[j*ri+i] = sum;

	barrier(CLK_LOCAL_MEM_FENCE);
}
