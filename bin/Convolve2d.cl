// A simple image convolution kernel

kernel void Convolve2d(global float *input,global float *mask,global float *output,int imageWidth, int imageHeight,
int kernelWidth, int kernelHeight, int halfWidth, int halfHeight) 
{
	
    int x = get_global_id(0);
    int y = get_global_id(1);
	
	if ( x >= halfWidth && x < imageWidth-halfWidth && y >= halfHeight && y < imageHeight - halfHeight) {
        float sum = (float)0;
        for(int p = 0; p < kernelWidth; p++) {
            for(int q = 0; q < kernelHeight; q++) {
                int mi = q*kernelWidth + p;
                int ix = x - halfWidth + p;
                int iy = y - halfHeight + q;
                int i = iy*imageWidth + ix;
                sum += input[i] * mask[mi];
            }
        }
        output[y*imageWidth+x] = sum;
    } else {
		output[y*imageWidth+x] = (float)0;
	}
}


