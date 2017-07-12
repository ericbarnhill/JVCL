// A simple image convolution kernel

kernel void ConvolveTest(global uchar4 *input,global float *mask,global uchar4 *output,int imageWidth, int imageHeight,
int kernelWidth, int kernelHeight, int kernelOriginX, int kernelOriginY) 
{
	
    int gx = get_global_id(0);

    if (gx >= kernelOriginX &&
        gx < imageWidth - (kernelWidth-kernelOriginX-1) 
    {
        float4 sum = (float4)0;
        for(int mx=0; mx<kernelWidth; mx++)
        {
                int mi = mx;
                int ix = gx - kernelOriginX + mx;
                int i = ix;
                sum += convert_float4(input[i]) * mask[mi];

        }
        uchar4 result = convert_uchar4_sat(sum);
        output[gx] = result;
    }
    else
    {
        if (gx >= 0 && gx < imageWidth)
        {
            output[gx] = (uchar4)0;
        }
    }
	
}


