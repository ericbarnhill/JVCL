
__kernel void stockhamStride(  __global float * real, __global float * imag, int is, int N, int powN, int blockSize) {

  const int bx = get_group_id(0);
  const int tx = get_local_id(0);
  inst ind = bx*N;
  const int tid = ind + tx; 
  const float TWOPI = 2*3.14159265359;
  //real[tid] = bx;
  //imag[tid] = tx;
  int l = N/2; 
  int m = 1;
  int j = tid;
  int k = 0;

  float theta;

  float c0_real, c0_imag, c1_real, c1_imag;
  float real_diff, imag_diff;

  for(int i =0 ; i < powN; i++) {

      k = tid - j*m;    
      theta = - TWOPI * j / (2.0 * l);
      if (k + j*m < N) {
		  c0_real = real[ind + k + j*m];
		  c0_imag = imag[ind + k + j*m];
	 } else break;
	if (k + j*m + l*m < N) {
      c1_real = real[ind + k + j*m + l*m];
      c1_imag = imag[ind + k + j*m + l*m];
	} else break;
    if (k + 2*j*m < N) {
      real[ind + k + 2*j*m] = c0_real + c1_real;
      imag[ind + k + 2*j*m] = c0_imag + c1_imag;
    } else break;
      real_diff = c0_real - c1_real;
      imag_diff = c0_imag - c1_imag;
    if (k + 2*j*m + m < N) {
      real[ind + k + 2*j*m + m] = cos(theta)*(real_diff) - sin(theta)*(imag_diff);
      imag[ind + k + 2*j*m + m] = cos(theta)*(imag_diff) + sin(theta)*(real_diff);  
	} else break;
      j = j/2;
      l = l/2;
      m = m*2;

      barrier(CLK_LOCAL_MEM_FENCE);
   }

	ind = ind + N;
  for(int i =0 ; i < powN; i++) {

      k = tid - j*m;    
      theta = - TWOPI * j / (2.0 * l);
      if (k + j*m < N) {
		  c0_real = real[ind + k + j*m];
		  c0_imag = imag[ind + k + j*m];
	 } else break;
	if (k + j*m + l*m < N) {
      c1_real = real[ind + k + j*m + l*m];
      c1_imag = imag[ind + k + j*m + l*m];
	} else break;
    if (k + 2*j*m < N) {
      real[ind + k + 2*j*m] = c0_real + c1_real;
      imag[ind + k + 2*j*m] = c0_imag + c1_imag;
    } else break;
      real_diff = c0_real - c1_real;
      imag_diff = c0_imag - c1_imag;
    if (k + 2*j*m + m < N) {
      real[ind + k + 2*j*m + m] = cos(theta)*(real_diff) - sin(theta)*(imag_diff);
      imag[ind + k + 2*j*m + m] = cos(theta)*(imag_diff) + sin(theta)*(real_diff);  
	} else break;
      j = j/2;
      l = l/2;
      m = m*2;

      barrier(CLK_LOCAL_MEM_FENCE);
   }


}
