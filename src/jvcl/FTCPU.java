package jvcl;

import org.apache.commons.math4.complex.Complex;
import org.apache.commons.math4.complex.ComplexUtils;

import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_2D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_3D;

public class FTCPU {
		
	public FTCPU() {
	}
	
	public Complex[] convolve(Complex[] vector, double[] kernel) {
		int vectorLength = vector.length;
		int kernelLength = kernel.length;
		int kernelLengthInterleaved = kernelLength*2;
		double[] v = JVCLUtils.zeroPad(ComplexUtils.complex2Interleaved(vector), kernelLengthInterleaved);
		double[] k = JVCLUtils.zeroPad(
						JVCLUtils.real2Interleaved(
								JVCLUtils.deepCopyToPadded(kernel, vectorLength)
								)
						, kernelLengthInterleaved);
		int adjLength = v.length;
		DoubleFFT_1D fft = new DoubleFFT_1D(adjLength);
		fft.complexForward(v);
		fft.complexForward(k);
		for (int n = 0; n < adjLength; n++) {
			v[n] *= k[n];
		}
		fft.complexInverse(v, true);
		return ComplexUtils.interleaved2Complex(JVCLUtils.stripPadding(v, kernelLengthInterleaved));
	}
	
	public double[] convolve(double[] vector, double[] kernel) {
		return ComplexUtils.complex2Real(convolve(ComplexUtils.real2Complex(vector), kernel));
	}
	
	public Complex[][] convolve(Complex[][] image, double[][] kernel) {
		
		int imageWidth = image.length;
		int imageHeight = image[0].length;
		int kernelWidth = kernel.length;
		int kernelWidthInterleaved = kernelWidth*2;
		int kernelHeight = kernel[0].length;
		int kernelHeightInterleaved = kernelHeight*2;
		double[][] v = JVCLUtils.zeroPad(ComplexUtils.complex2Interleaved(image), kernelWidthInterleaved, kernelHeightInterleaved);
		double[][] k = JVCLUtils.zeroPad(
						JVCLUtils.real2Interleaved(
								JVCLUtils.deepCopyToPadded(kernel, imageWidth, imageHeight)
								)
						, kernelWidth, kernelHeight);
		int adjWidth = v.length;
		int adjHeight = v[0].length;
		DoubleFFT_2D fft = new DoubleFFT_2D(adjWidth, adjHeight);
		fft.complexForward(v);
		fft.complexForward(k);
		for (int x = 0; x < adjWidth; x++) {
			for (int y = 0; y < adjHeight; y++) {
				v[x][y] *= k[x][y];
			}
		}
		fft.complexInverse(v, true);
		return ComplexUtils.interleaved2Complex(JVCLUtils.stripPadding(v, kernelWidthInterleaved, kernelHeightInterleaved));
		
	}
	
	public double[][] convolve(double[][] vector, double[][] kernel) {
		return ComplexUtils.complex2Real(convolve(ComplexUtils.real2Complex(vector), kernel));
	}
	
	public Complex[][][] convolve(Complex[][][] image, double[][][] kernel) {
		
		int imageWidth = image.length;
		int imageHeight = image[0].length;
		int imageDepth = image[0][0].length;
		int kernelWidth = kernel.length;
		int kernelWidthInterleaved = kernelWidth*2;
		int kernelHeight = kernel[0].length;
		int kernelHeightInterleaved = kernelHeight*2;
		int kernelDepth = kernel[0][0].length;
		int kernelDepthInterleaved = kernelDepth*2;
		double[][][] v = JVCLUtils.zeroPad(ComplexUtils.complex2Interleaved(image), 
				kernelWidthInterleaved, kernelHeightInterleaved, kernelDepthInterleaved);
		double[][][] k = JVCLUtils.zeroPad(
						JVCLUtils.real2Interleaved(
								JVCLUtils.deepCopyToPadded(kernel, imageWidth, imageHeight, imageDepth)
								)
						, kernelWidth, kernelHeight, kernelDepth);
		int adjWidth = v.length;
		int adjHeight = v[0].length;
		int adjDepth = v[0][0].length;
		DoubleFFT_3D fft = new DoubleFFT_3D(adjWidth, adjHeight, adjDepth);
		fft.complexForward(v);
		fft.complexForward(k);
		for (int x = 0; x < adjWidth; x++) {
			for (int y = 0; y < adjHeight; y++) {
				for (int z = 0; z < adjDepth; z++) {
					v[x][y][z] *= k[x][y][z];
				}
			}
		}
		fft.complexInverse(v, true);
		return ComplexUtils.interleaved2Complex(JVCLUtils.stripPadding(v, 
				kernelWidthInterleaved, kernelHeightInterleaved, kernelDepthInterleaved));
		
	}

	public double[][][] convolve(double[][][] vector, double[][][] kernel) {
		return ComplexUtils.complex2Real(convolve(ComplexUtils.real2Complex(vector), kernel));
	}
}


