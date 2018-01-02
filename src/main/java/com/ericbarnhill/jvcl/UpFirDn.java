package com.ericbarnhill.jvcl;

import org.apache.commons.numbers.complex.Complex;

import com.ericbarnhill.arrayMath.ArrayMath;

 public class UpFirDn {

	 int upValue, downValue;
     ConvolverFactory convolverFactory;
     ConvolverFactory.ConvolutionType convolutionType;

     public UpFirDn() {
         convolverFactory = new ConvolverFactory();
         this.convolutionType = ConvolverFactory.ConvolutionType.FDCPU;
     }

     public UpFirDn(ConvolverFactory.ConvolutionType convolutionType) {
         convolverFactory = new ConvolverFactory();
         this.convolutionType = convolutionType;
     }

	 Complex[] upFirDn(Complex[] f, Complex[] g, int upFactor, int downFactor) {
        ConvolverComplex c = (ConvolverComplex)convolverFactory.getConvolver(ConvolverFactory.DataType.COMPLEX, convolutionType);
		if (upFactor > 1) {
			f = interpolateZeros(f, upFactor);
		}
		// f = Unrolled.convolve_10(f, g);
		f = c.convolve(f,g);
		if (downFactor > 1) {
			f = decimate(f, downFactor);
		}
		return f;
	}

	 Complex[][] upFirDn(Complex[][] f, Complex[] g, int upFactor, int downFactor, int dim) {
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

	 Complex[][] upFirDn(Complex[][] f, Complex[] g, int upFactor, int downFactor) {
		return upFirDn(f, g, upFactor, downFactor, 1);
	}

	 public double[] upFirDn(double[] f, double[] g, int upFactor, int downFactor) {
        ConvolverDouble c = (ConvolverDouble)convolverFactory.getConvolver(ConvolverFactory.DataType.DOUBLE, convolutionType);
		if (upFactor > 1) {
			f = interpolateZeros(f, upFactor);
		}
		//f = Unrolled.convolve_10(f, g);
		f = c.convolve(f,g);
		if (downFactor > 1) {
			f = decimate( f, downFactor);
		}
		return f;
	}

	 public double[][] upFirDn(double[][] f, double[] g, int upFactor, int downFactor, int dim) {
		if (dim < 0 || dim > 1) {
			throw new RuntimeException("Switch to invalid dimension");
		}
		if (dim == 1) f = ArrayMath.shiftDim(f);
		int height = f.length;
		for (int i = 0; i < height; i++) {
			f[i] = upFirDn(f[i], g, upFactor, downFactor);
		}
		if (dim ==1) f = ArrayMath.shiftDim(f);
		return f;
	}

	 public double[][] upFirDn(double[][] f, double[] g, int upFactor, int downFactor) {
		return upFirDn(f, g, upFactor, downFactor, 1);
	}

	 public double[][][] upFirDn(double[][][] f, double[] g, int upFactor, int downFactor, int dim) {
        ConvolverDouble c = (ConvolverDouble)convolverFactory.getConvolver(ConvolverFactory.DataType.DOUBLE, convolutionType);
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
                    f[i][j] = upFirDn(f[i][j], g, upFactor, downFactor); 
				}
			}
		}
		if (dim == 0) f = ArrayMath.shiftDim(f , 1);
		if (dim == 1) f = ArrayMath.shiftDim(f , 2);
		return f;
	}

	 public double[][][] upFirDn(double[][][] f, double[] g, int upFactor, int downFactor) {
		return upFirDn(f, g, upFactor, downFactor, 0);
	}

	 public double[] interpolateZeros(double[] vec, int factor) {
		int length = vec.length;
		double[] interpVec = new double[length*factor-(factor-1)];
		for (int n = 0; n < length; n++) {
			interpVec[n*factor] = vec[n];
		}
		return interpVec;
	}

	 public Complex[] interpolateZeros(Complex[] vec, int factor) {
		int length = vec.length;
		Complex[] interpVec = new Complex[length*factor - (factor-1)];
		for (int n = 0; n < length; n++) {
			interpVec[n*factor] = vec[n];
		}
		return interpVec;
	}

	 public float[] interpolateZeros(float[] vec, int factor) {
		int length = vec.length;
		float[] interpVec = new float[length*factor - (factor-1)];
		for (int n = 0; n < length; n++) {
			interpVec[n*factor] = vec[n];
		}
		return interpVec;
	}

	 public double[] decimate(double[] vec, int factor) {
		int length = vec.length / factor + vec.length % factor;
		double[] deciVec = new double[length]; // off by one removed EB jan 2016
		for (int n = 0; n < length; n++) {
			deciVec[n] = vec[n*factor];
		}
		return deciVec;
	}

	 public Complex[] decimate(Complex[] vec, int factor) {
		int length = vec.length / factor + vec.length % factor;
		Complex[] deciVec = new Complex[length];
		for (int n = 0; n < length; n++) {
			deciVec[n] = vec[n*factor];
		}
		return deciVec;
	}

	 public float[] decimate(float[] vec, int factor) {
		int length = vec.length;
		float[] deciVec = new float[length];
		for (int n = 0; n < length; n++) {
			deciVec[n] = vec[n*factor];
		}
		return deciVec;
	}

}
