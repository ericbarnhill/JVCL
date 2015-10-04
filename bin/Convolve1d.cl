kernel void Convolve1d(global float *input,global float *mask,global float *output,int vectorLength,
int kernelLength, int halfLength) 
{
	
    int x = get_global_id(0);

	
	if ( x >= halfLength && x < vectorLength-halfLength) {
        float sum = (float)0;
        for(int p = 0; p < kernelLength; p++) {
                int i = x - halfWidth + p;
                sum += input[i] * mask[p];
            }
        }
        output[y*vectorLength+x] = sum;
    } else {
	output[y*vectorLength+x] = (float)0;
	}
}
