/* 
   Stockham's implementation of radix-2 fft.
 */

#define TWOPI 6.28318530718

__kernel void stockhamRadix2(__global float2* src, /*input array*/ 
                         __global float2* dst, /*output array*/
                         const int p,          /*block size*/
                         const int tt) {        /*number of threads*/
		const int t = tt / p;
     	const int gid = get_global_id(0);
		const int k = gid & (p - 1);

		src += gid;
		dst += (gid << 1) - k; 

		const float2 in1 = src[0];
		const float2 in2 = src[t];
		dst[0] = in1 + in2;
		const float2 diff = in1 - in2;
		const float theta = -TWOPI * k / p;
		float cs;
		float sn = sincos(theta, &cs);
		const float2 temp = (float2) (diff.x * cs - diff.y * sn,
		                              diff.y * cs + diff.x * sn);
		dst[p] = temp;
		//dst[0] = in1 + temp;
		//dst[p] = in1 - temp;
		//dst[p] = (float2)(cos(theta)*diff.x - sin(theta)*diff.y,
		//		cos(theta)*diff.y - sin(theta)*diff.x);

}
