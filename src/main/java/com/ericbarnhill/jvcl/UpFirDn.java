package com.ericbarnhill.jvcl;

import org.apache.commons.numbers.complex.Complex;

import com.ericbarnhill.arrayMath.ArrayMath;

public class UpFirDn {

	static int upValue, downValue;

	static Complex[] upFirDn(Complex[] f, Complex[] g, int upFactor, int downFactor) {
		if (upFactor > 1) {
			f = interpolateZeros(f, upFactor);
		}
		// f = Unrolled.convolve_10(f, g);
		f = ConvolverFactory.getConvolver().convolve(f,g);
		if (downFactor > 1) {
			f = decimate(f, downFactor);
		}
		return f;
	}

	static Complex[][] upFirDn(Complex[][] f, Complex[] g, int upFactor, int downFactor, int dim) {
		if (dim < 0 || dim > 1) {
			throw new RuntimeException("Switch to invalid dimension");
		}
		if (dim == 1) f = ArrayMath.shiftDim(f);
		int height = f[0].length;
		for (int i = 0; i <height-1; i++) {
			f[i] = upFirDn(f[i], g, upFactor, downFactor);
		}
		if (dim ==1) f = ArrayMath.shiftDim(f);
		return f;
	}

	static Complex[][] upFirDn(Complex[][] f, Complex[] g, int upFactor, int downFactor) {
		return upFirDn(f, g, upFactor, downFactor, 1);
	}

	public static double[] upFirDn(double[] f, double[] g, int upFactor, int downFactor) {

		if (upFactor > 1) {
			f = interpolateZeros(f, upFactor);
		}
		//f = Unrolled.convolve_10(f, g);
		f = ConvolverFactory.getConvolver().convolve(f, g);
		if (downFactor > 1) {
			f = decimate( f, downFactor);
		}
		return f;
	}

	public static double[][] upFirDn(double[][] f, double[] g, int upFactor, int downFactor, int dim) {
		if (dim < 0 || dim > 1) {
			throw new RuntimeException("Switch to invalid dimension");
		}
		if (dim == 0) f = ArrayMath.shiftDim(f);
		int fi = f.length;
		if (upFactor > 1) {
			for (int i = 0; i < fi; i++) {
				f[i] = interpolateZeros(f[i], upFactor);
			}
		}
		for (int i = 0; i < fi; i++) {
			//f[i] = Unrolled.convolve_10(f[i],g);
			f[i] = ConvolverFactory.getConvolver(f, g).convolve(f[i],g);

		}
		if (downFactor > 1) {
			for (int i = 0; i < fi; i++) {
				f[i] = decimate(f[i], downFactor);
			}
		}
		if (dim == 0) f = ArrayMath.shiftDim(f);
		return f;
	}

	public static double[][] upFirDn(double[][] f, double[] g, int upFactor, int downFactor) {
		return upFirDn(f, g, upFactor, downFactor, 1);
	}

	public static double[][][] upFirDn(double[][][] f, double[] g, int upFactor, int downFactor, int dim) {
		if (dim < 0 || dim > 1) {
			throw new RuntimeException("Switch to invalid dimension");
		}
		if (dim == 0) f = ArrayMath.shiftDim(f , 1);
		if (dim == 1) f = ArrayMath.shiftDim(f , 2);
		int fi = f.length;
		int fj = f[0].length;
		if (upFactor > 1) {
			for (int i = 0; i < fi; i++) {
				for (int j = 0; j < fj; j++) {
					//f[i][j] = Unrolled.convolve_10(f[i][j],g);
					f[i][j] = ConvolverFactory.getConvolver(f,g).convolve(f[i][j],g);
				}
			}
		}
		if (dim == 0) f = ArrayMath.shiftDim(f , 1);
		if (dim == 1) f = ArrayMath.shiftDim(f , 2);
		return f;
	}

	public static double[][][] upFirDn(double[][][] f, double[] g, int upFactor, int downFactor) {
		return upFirDn(f, g, upFactor, downFactor, 0);
	}

	public static double[] interpolateZeros(double[] vec, int factor) {
		int length = vec.length;
		double[] interpVec = new double[length*factor-(factor-1)];
		for (int n = 0; n < length; n++) {
			interpVec[n*factor] = vec[n];
		}
		return interpVec;
	}

	public static Complex[] interpolateZeros(Complex[] vec, int factor) {
		int length = vec.length;
		Complex[] interpVec = new Complex[length*factor - (factor-1)];
		for (int n = 0; n < length; n++) {
			interpVec[n*factor] = vec[n];
		}
		return interpVec;
	}

	public static float[] interpolateZeros(float[] vec, int factor) {
		int length = vec.length;
		float[] interpVec = new float[length*factor - (factor-1)];
		for (int n = 0; n < length; n++) {
			interpVec[n*factor] = vec[n];
		}
		return interpVec;
	}

	public static double[] decimate(double[] vec, int factor) {
		int length = vec.length / factor + vec.length % factor;
		double[] deciVec = new double[length]; // off by one removed EB jan 2016
		for (int n = 0; n < length; n++) {
			deciVec[n] = vec[n*factor];
		}
		return deciVec;
	}

	public static Complex[] decimate(Complex[] vec, int factor) {
		int length = vec.length / factor + vec.length % factor;
		Complex[] deciVec = new Complex[length];
		for (int n = 0; n < length; n++) {
			deciVec[n] = vec[n*factor];
		}
		return deciVec;
	}

	public static float[] decimate(float[] vec, int factor) {
		int length = vec.length;
		float[] deciVec = new float[length];
		for (int n = 0; n < length; n++) {
			deciVec[n] = vec[n*factor];
		}
		return deciVec;
	}

}
