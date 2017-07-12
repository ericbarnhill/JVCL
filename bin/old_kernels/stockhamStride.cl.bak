
__kernel void stockhamStride(  __global float * real, __global float * imag, int is, int N, int powN, int blockSize) {

  const int bx = get_group_id(0);
  const int tx = get_local_id(0);
  const int blk = (tx / N)*N;
  const int td = tx % N;
  const int tid = blk + td; 
  const float TWOPI = 2*3.14159265359;
  int l = N/2; 
  int m = 1;
 // int j = tid;
  int j = td;
  int k = 0;
  //real[tx] = td; imag[tx] = td; return;
  float theta;

  float c0_real, c0_imag, c1_real, c1_imag;
  float real_diff, imag_diff;

  for(int i =0 ; i < powN; i++) {

      k = td - j*m;    
      //k = tid - j*m;    
      theta = - TWOPI * j / (2.0 * l);
      if (k + j*m < N) {
		  c0_real = real[blk + k + j*m];
		  c0_imag = imag[blk + k + j*m];
	 } else break;
	if (k + j*m + l*m < N) {
      c1_real = real[blk + k + j*m + l*m];
      c1_imag = imag[blk + k + j*m + l*m];
	} else break;
    if (k + 2*j*m < N) {
      real[blk + k + 2*j*m] = c0_real + c1_real;
      imag[blk + k + 2*j*m] = c0_imag + c1_imag;
    } else break;
      real_diff = c0_real - c1_real;
      imag_diff = c0_imag - c1_imag;
    if (k + 2*j*m + m < N) {
      real[blk + k + 2*j*m + m] = cos(theta)*(real_diff) - sin(theta)*(imag_diff);
      imag[blk + k + 2*j*m + m] = cos(theta)*(imag_diff) + sin(theta)*(real_diff);  
	} else break;
      j = j/2;
      l = l/2;
      m = m*2;

      barrier(CLK_LOCAL_MEM_FENCE);
   }


}
