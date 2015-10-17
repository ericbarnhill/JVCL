// A simple image convolution kernel
#define ZERO_BOUNDARY 0
#define MIRROR_BOUNDARY 1
#define PERIODIC_BOUNDARY 2


kernel void Convolve2d(global float *image,global float *mask,global float *result,const int imageWidth, const int imageHeight,
const int kernelWidth, const int kernelHeight, const int halfWidth, const int halfHeight, const int boundaryConditions) {
	int adjX, adjY, maskIndex, imageIndex;
	float sum;
    int x = get_global_id(0);
    int y = get_global_id(1);
	
	if ( x >= halfWidth && x < imageWidth-halfWidth && y >= halfHeight && y < imageHeight - halfHeight) {
        sum = (float)0;
        for(int p = 0; p < kernelWidth; p++) {
            for(int q = 0; q < kernelHeight; q++) {
                adjX = x - halfWidth + p;
                adjY = y - halfHeight + q;
				maskIndex = q*kernelWidth + p;
                imageIndex = adjY*imageWidth + adjX;
                sum += image[imageIndex] * mask[maskIndex];
            }
        }
        result[y*imageWidth+x] = sum;
    } else {
		sum = (float)0;
		for (int p = 0; p < kernelWidth; p++) {
			for (int q = 0; q < kernelHeight; q++) {
				adjX = x + (p - halfWidth);
				adjY = y + (q - halfHeight);
				if (adjX < 0) {
					if (boundaryConditions == MIRROR_BOUNDARY) {
						adjX = abs(adjX);
					} else if (boundaryConditions == PERIODIC_BOUNDARY) {
						adjX = imageWidth + adjX;
					}
				} else if (adjX >= imageWidth) {
					if (boundaryConditions == MIRROR_BOUNDARY) {
						adjX = 2*imageWidth - adjX - 1;
					} else if (boundaryConditions == PERIODIC_BOUNDARY) {
						adjX = adjX - imageWidth;
					}
				}
				if (adjY < 0) {
					if (boundaryConditions == MIRROR_BOUNDARY) {
						adjY = abs(adjY);
					} else if (boundaryConditions == PERIODIC_BOUNDARY) {
						adjY = imageWidth + adjY;
					}
				} else if (adjY >= imageHeight) {
					if (boundaryConditions == MIRROR_BOUNDARY) {
						adjY = 2*imageHeight - adjY - 1;
					} else if (boundaryConditions == PERIODIC_BOUNDARY) {
						adjY = adjY - imageHeight;
					}
				}
				int maskIndex = q*kernelWidth + p;
                int imageIndex = adjY*imageWidth + adjX;
                sum += image[imageIndex] * mask[maskIndex];
			} // for q
		} // for p
	} // if boundary


		barrier(CLK_LOCAL_MEM_FENCE);
}
