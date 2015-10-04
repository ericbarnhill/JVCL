package jvcl;

public class CPUFFT {
	
	public CPUFFT() {}
	 	 static int tally;
	 
	// Cooley–Tukey FFT (in-place, divide-and-conquer)
	// Higher memory requirements and redundancy although more intuitive
	void fft_simple(float[] real, float[] imag)
	{
	    int N = real.length;
	    if (N <= 1) return;
	 
	    // divide
	    float[] evenR = new float[N/2];
	    float[] evenI = new float[N/2];
	    float[] oddR = new float[N/2];
	    float[] oddI = new float[N/2];
	    for (int n = 0; n < N/2; n++) {
	    	evenR[n] = real[n*2];
	    	evenI[n] = imag[n*2];
	    	oddR[n] = real[n*2+1];
	    	oddI[n] = imag[n*2+1];
	    }
	 
	    // conquer
	    	fft_simple(evenR, evenI);
	    	fft_simple(oddR, oddI);
	 
	    float[] angle = new float[2];
	    float[] product = new float[2];
		    for (int k = 0; k < N/2; k++)  {
			    	float phaseAngle = - 2.0f * (float)Math.PI * (float)k / (float)N;	
			    	angle[0] = (float)Math.cos(phaseAngle);
					angle[1] = (float)Math.sin(phaseAngle);
					product = complexMult(angle, new float[] {oddR[k], oddI[k]});
					real[k] = evenR[k] + product[0];
					imag[k] = evenI[k] + product[1];
					real[k+N/2] = evenR[k] - product[0];
					imag[k+N/2] = evenI[k] - product[1];
			}
	}
	
	static private float[] complexMult(float[] a, float[]b) {
		float[] c = new float[2];
		c[0] = a[0]*b[0] - a[1]*b[1];
		c[1] = a[0]*b[1] + a[1]*b[0];
		return c;
	}
	
	public static void main(String[] args) {

		float[] real = new float[64];
		float[] imag = new float[64];
		
		for (int n = 0; n < 64; n++) {
			real[n] = 10*(float)Math.cos(2*Math.PI*(n+1)/64);
			imag[n] = 10*(float)Math.sin(2*Math.PI*(n+1)/64);
			//System.out.format("%.2f + i%.2f ", real[n], imag[n]);
			//if ((n+1) % 8 == 0) System.out.format("%n");
		}
	 
	    // forward fft
	    new CPUFFT().fft_simple(real, imag);
	    System.out.println("REAL");
	    for (int n = 0; n < 64; n++) {
			System.out.format("%.2f ",real[n]);
			if ((n+1) % 16 == 0) System.out.println();
		}
	    System.out.println("IMAG");
	    for (int n = 0; n < 16; n++) {
			System.out.format("%.2f ",imag[n]);
		}
	    System.out.println();
	    
	}
	
	

}
