

kernel void Convolve3d(global float* f, global float* g, global float* r, const int ri, const int rj, const int rk, const int gi, const int gj, const int gk, const int hgi, const int hgj, const int hgk, const int hgie, const int hgje, const int hgke)
{
	
    int i = get_global_id(0);
    int j = get_global_id(1);
	int k = get_global_id(2);
	float sum = 0;
	int ai, aj, ak, fInd, gInd;
	for (int p = 0; p < gi; p++) {
		for (int q = 0; q < gj; q++) {
			for (int s = 0; s < gk; s++) {
				ai = i + (p - hgie);
				aj = j + (q - hgje);
				ak = k + (s - hgke);
				if (ai >= 0 && ai < ri) {
					if (aj >= 0 && aj < rj) {
						if (ak >= 0 && ak < rk) {
							fInd = ak*ri*rj + aj*ri + ai;
							gInd = (gi-1-p)*gi*gj +(gj-1-q)*gi + (gk-1-s);
							sum += f[fInd]*g[gInd];
								
						}
					}
				}
			}
		}
	}
	r[k*ri*rj + j*ri +i] = sum; 
	
	barrier(CLK_LOCAL_MEM_FENCE);
}


