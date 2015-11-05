
__kernel void stockhamStride(  __global float * real, __global float * imag, const int is, const int N, const int powN, const int blockSize) 
{
	int block = get_group_id(0);
	int x = get_local_id(0);
	int index = block*blockSize+x;
	const float TWOPI = 2*3.14159265359;

	int j = x;
	int l = N/2; 
	int m = 1;
	int k;

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

	  j >> 1;
	  l >> 1;
	  m << 1;

	}

	barrier(CLK_LOCAL_MEM_FENCE);
}
