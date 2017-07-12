
__kernel void stockham2(  __global float * real, __global float * imag, const int is, const int blockSize,
	const int jDivisor, const int l, const int m) 
{
	int block = get_group_id(0);
	int x = get_local_id(0);
	int index = block*blockSize+x;
	const float TWOPI = 2*3.14159265359;

	int j = x / jDivisor;

	float theta;

	float c0_real, c0_imag, c1_real, c1_imag;
	float real_diff, imag_diff;



	  int k = x - j*m;    
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
}
