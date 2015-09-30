// A simple image convolution kernel

kernel void Convolve3d(global float *input,global float *mask,global float *output,int imageWidth, int imageHeight,int imageDepth,
int kernelWidth, int kernelHeight, int kernelDepth, int halfWidth, int halfHeight, int halfDepth) 
{
	
    int x = get_global_id(0);
    int y = get_global_id(1);
	int z = get_global_id(2);
	int imageArea = imageWidth*imageHeight;
	int kernelArea = kernelWidth*kernelHeight;
	
	if ( x >= halfWidth && x < imageWidth-halfWidth && y >= halfHeight && y < imageHeight - halfHeight &&
			z >= halfDepth && z < imageDepth-halfDepth) {
        float sum = (float)0;
        for (int p = 0; p < kernelWidth; p++) {
            for (int q = 0; q < kernelHeight; q++) {
				for (int r = 0; r < kernelDepth; r++) {
		            int mi = r*kernelArea + q*kernelWidth + p;
		            int ix = x - halfWidth + p;
		            int iy = y - halfHeight + q;
					int iz = z - halfDepth + r;
		            int i = iz*imageArea + iy*imageWidth + ix;
		            sum += input[i] * mask[mi];
				}
            }
        }
        output[z*imageArea + y*imageWidth + x] = sum;
    } else {
		output[z*imageArea + y*imageWidth + x] = (float)0;
	}
	
}


