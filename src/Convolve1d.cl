#define ZERO_BOUNDARY 0
#define MIRROR_BOUNDARY 1
#define PERIODIC_BOUNDARY 2

kernel void Convolve1d(global float *vector,global float *mask,global float *result,const int vectorLength,
const int maskLength, const int halfLength, const int boundaryConditions) 
{
	
    int n = get_global_id(0);
	int adjN;
	float sum;
	
	if ( n >= halfLength && n < vectorLength-halfLength) {
        sum = (float)0;
        for(int p = 0; p < maskLength; p++) {
                adjN = n - halfLength + p;
                result[n] += vector[adjN] * mask[p];
        }
    } else if (boundaryConditions > 0) {
		sum = (float)0;
        for(int p = 0; p < maskLength; p++) {
            adjN = n - halfLength + p;
			if (adjN < 0) {
				if (boundaryConditions == MIRROR_BOUNDARY) {
					adjN = abs(adjN);
					vector[n] += vector[adjN]*mask[p];
				} else if (boundaryConditions == PERIODIC_BOUNDARY) {
					adjN = vectorLength + adjN;
					result[n] += vector[adjN]*mask[p];
				} 
			} // if outside boundary
			if (adjN >= vectorLength) {
				if (boundaryConditions == MIRROR_BOUNDARY) {
					adjN = 2*vectorLength - adjN - 1;
					result[n] += vector[adjN]*mask[p];
				} else if (boundaryConditions == PERIODIC_BOUNDARY) {
					adjN = adjN - vectorLength;
					result[n] += vector[adjN]*mask[p];
				} 
			} // if outside boundary
        }
	}


	barrier(CLK_LOCAL_MEM_FENCE);
		
}


