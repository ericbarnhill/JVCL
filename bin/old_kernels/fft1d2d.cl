#define PI 3.14159265358979323846

kernel void fft1d2d(global float* real, global float* imag, global float* evenR, global float* evenI, 
	global float* oddR, global float* oddI, int N){
	
	int x = get_global_id(0);
	int y = get_global_id(1);

	if (k < N/2) {
			float phaseAngle;
			float angleR, angleI, productR, productI;
			phaseAngle = - 2.0f * (float)PI * (float)k / (float)N;	
			angleR = cos(phaseAngle);
			angleI = sin(phaseAngle);
			productR = angleR*oddR[k] - angleI*oddI[k];
			productI = angleR*oddI[k] + angleI*oddR[k];
			real[k] = evenR[k] + productR;
			imag[k] = evenI[k] + productI;
			real[k+N/2] = evenR[k] - productR;
			imag[k+N/2] = evenI[k] - productI;
	}
}

