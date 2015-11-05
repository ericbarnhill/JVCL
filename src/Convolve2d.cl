
kernel void Convolve2d(global float* f, global float* g, global float* r, const int ri, const int rj, const int gi, const int gj, const int hgi, const int hgj, const int hgie, const int hgje) {
	
    int i = get_global_id(0);
    int j = get_global_id(1);

	float sum = 0;
	int ai, aj, fInd, gInd;
	for (int p = 0; p < gi; p++) {
		for (int q = 0; q < gj; q++) {
			ai = i + (p - hgi);
			aj = j + (q - hgj);
			if (ai >= 0 && ai < ri) {
				if (aj >= 0 && aj < rj) {
                	fInd = aj*ri + ai;
					gInd = (gj-1-q)*gi + (gi-1-p);
					sum += f[fInd]*g[gInd];
				}
			}
		}
	}
	r[j*ri + i] = sum;

	barrier(CLK_LOCAL_MEM_FENCE);
}
