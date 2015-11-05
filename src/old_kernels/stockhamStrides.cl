
__kernel void stockhamStrides(  __global float * real, __global float * imag, int is, int N, int powN, int blockSize) 
{
	const int block = get_group_id(0);
	const int x = get_local_id(0);
	int index = block*blockSize+x;
	//const int  tid = bx * blockSize + tx; 
	const float TWOPI = 2*3.14159265359;

	int l = N/2; 
	int m = 1;
	int j = x;
	// int j = tid;
	int k = 0;

	float theta;

	float c0_real, c0_imag, c1_real, c1_imag;
	float real_diff, imag_diff;

	for(int i =0 ; i < powN; i++) {

	  k = x - j*m;    
	  theta = - TWOPI * j / (2.0 * l);
	  
	  c0_real = real[index + k + j*m];
	  c0_imag = imag[index + k + j*m];

	  c1_real = real[index + k + j*m + l*m];
	  c1_imag = imag[index + k + j*m + l*m];
	  
	  real[index + k + 2*j*m] = c0_real + c1_real;
	  imag[index + k + 2*j*m] = c0_imag + c1_imag;
	  
	  real_diff = c0_real - c1_real;
	  imag_diff = c0_imag - c1_imag;
	  
	  real[index + k + 2*j*m + m] = cos(theta)*(real_diff) - sin(theta)*(imag_diff);
	  imag[index + k + 2*j*m + m] = cos(theta)*(imag_diff) + sin(theta)*(real_diff);  

	  j = j/2;
	  l = l/2;
	  m = m*2;

	}

	barrier(CLK_LOCAL_MEM_FENCE);
}
