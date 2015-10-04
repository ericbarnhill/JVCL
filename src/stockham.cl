#define BLOCK_SIZE 64

__kernel void stockham(  __global float * r_real, __global float * r_imag,
//          int is, unsigned n, unsigned powN)
          int is, int n, int powN)
{

  const size_t bx = get_group_id(0);
  const size_t tx = get_local_id(0);
  const unsigned  tid = bx * BLOCK_SIZE + tx; 
  const float TWOPI = 2*3.14159265359;
/*
  r_real[tid] = bx;
  r_imag[tid] = tx;
 return;
*/
  int l = n/2; 
  int m = 1;
  int j = tid;
  int k=0;

  float theta;

  float c0_real, c0_imag, c1_real, c1_imag;
  float real_diff, imag_diff;

  for(int i =0 ; i < powN; i++)
    {
      k = tid - j*m;
      
      //do main computation here.
      
      theta = is * TWOPI * j / (2.0 * l);
      
      c0_real = r_real[k + j*m];
      c0_imag = r_imag[k + j*m];

      c1_real = r_real[k + j*m + l*m];
      c1_imag = r_imag[k + j*m + l*m];

      //y[k + 2*j*m] = c0 + c1
      
      r_real[k + 2*j*m] = c0_real + c1_real;
      r_imag[k + 2*j*m] = c0_imag + c1_imag;
      
      real_diff = c0_real - c1_real;
      imag_diff = c0_imag - c1_imag;
      
      r_real[k + 2*j*m + m] = cos(theta)*(real_diff) - sin(theta)*(imag_diff);
      r_imag[k + 2*j*m + m] = cos(theta)*(imag_diff) + sin(theta)*(real_diff);  

      //change values of m,l,j
      j = j/2;
      l = l/2;
      m = m*2;

      barrier(CLK_LOCAL_MEM_FENCE);
    }


}
