// kernel test

kernel void Test(global float *input,global float *output) 
{
	
    int gx = get_global_id(0);
	output[gx] = (float)input[gx];
	//output[gx] = 3;
}

