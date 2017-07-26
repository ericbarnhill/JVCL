package com.ericbarnhill.jvcl;

import org.apache.commons.numbers.complex.Complex;

import com.ericbarnhill.arrayMath.ArrayMath;

public class UpFirDn {

	static int upValue, downValue;

	static Complex[] upFirDn(Complex[] f, Complex[] g, int upFactor, int downFactor) {
		if (upFactor > 1) {
			f = JVCLUtils.interpolateZeros(f, upFactor);
		}
		// f = Unrolled.convolve_10(f, g);
		f = FDCPU.convolve(f,g);
		if (downFactor > 1) {
			f = JVCLUtils.decimate(f, downFactor);
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
			f = JVCLUtils.interpolateZeros(f, upFactor);
		}
		//f = Unrolled.convolve_10(f, g);
		f = FDCPU.convolve(f, g);
		if (downFactor > 1) {
			f = JVCLUtils.decimate( f, downFactor);
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
				f[i] = JVCLUtils.interpolateZeros(f[i], upFactor);
			}
		}
		for (int i = 0; i < fi; i++) {
			//f[i] = Unrolled.convolve_10(f[i],g);
			f[i] = FDCPU.convolve(f[i],g);

		}
		if (downFactor > 1) {
			for (int i = 0; i < fi; i++) {
				f[i] = JVCLUtils.decimate(f[i], downFactor);
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
					f[i][j] = FDCPU.convolve(f[i][j],g);
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

}
