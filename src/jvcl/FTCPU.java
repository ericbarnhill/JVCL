package jvcl;

import java.util.Arrays;

import org.apache.commons.math4.complex.Complex;
import org.apache.commons.math4.complex.ComplexUtils;

import arrayMath.ArrayMath;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_2D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_3D;


public class FTCPU {
		
	public FTCPU() {
	}
	
	public Complex[] convolve(Complex[] vector, Complex[] kernel) {
		int vectorLength = vector.length;
		int kernelLength = kernel.length;
		double[] v = ComplexUtils.complex2Interleaved(
						JVCLUtils.zeroPadBoundaries(vector, kernelLength)
					);
		double[] k = JVCLUtils.deepCopyToPadded(
						ComplexUtils.complex2Interleaved(
							JVCLUtils.zeroPadBoundaries(kernel, kernelLength)
						),
					v.length);
		DoubleFFT_1D fft = new DoubleFFT_1D(v.length/2);
		fft.complexForward(v);
		fft.complexForward(k);
		ArrayMath.multiply(v, k);
		fft.complexInverse(v, true);
		return 	JVCLUtils.stripEndPadding(
					JVCLUtils.stripBorderPadding(
						ComplexUtils.interleaved2Complex(
								v), 
						kernelLength),
					vectorLength);
	}
	
	public double[] convolve(double[] image, double[] kernel) {
		return ComplexUtils.complex2Real(
				convolve(ComplexUtils.real2Complex(image), ComplexUtils.real2Complex(kernel)
						)
				);
	}
	
	public Complex[][] convolve(Complex[][] image, Complex[][] kernel) {
		int imageWidth = image.length;
		int imageHeight = image[0].length;
		int kernelWidth = kernel.length;
		int kernelHeight = kernel[0].length;
		int imageWidthInterleaved = imageWidth;
		int imageHeightInterleaved = 2*imageHeight;
		int kernelWidthInterleaved = kernelWidth;
		int kernelHeightInterleaved = 2*kernelHeight;
		int paddedWidth = imageWidthInterleaved+2*kernelWidthInterleaved;
		int paddedHeight = imageHeightInterleaved+2*kernelHeightInterleaved;
		double[][] v = JVCLUtils.zeroPadBoundaries(
							ComplexUtils.complex2Interleaved(image),
					kernelWidthInterleaved, kernelHeightInterleaved);
		double[][] k = JVCLUtils.deepCopyToPadded(
						JVCLUtils.zeroPadBoundaries(
								ComplexUtils.complex2Interleaved(kernel),
						kernelWidthInterleaved, kernelHeightInterleaved),
						paddedWidth, paddedHeight);
		DoubleFFT_2D fft = new DoubleFFT_2D(paddedWidth, paddedHeight/2);
		fft.complexForward(v);
		fft.complexForward(k);
		ArrayMath.multiply(v,k);
		fft.complexInverse(v, true);
		return ComplexUtils.interleaved2Complex(
				JVCLUtils.stripBorderPadding(
					v, kernelWidthInterleaved, kernelHeightInterleaved)
				);
	}
	
	public Complex[][][] convolve(Complex[][][] volume, Complex[][][] kernel) {
		
		int volumeWidth = volume.length;
		int volumeHeight = volume[0].length;
		int volumeDepth = volume[0][0].length;
		int kernelWidth = kernel.length;
		int kernelHeight = kernel[0].length;
		int kernelDepth = kernel[0][0].length;
		int volumeWidthInterleaved = 2*volumeWidth;
		int volumeHeightInterleaved = 2*volumeHeight;
		int volumeDepthInterleaved = 2*volumeDepth;
		int kernelWidthInterleaved = 2*kernelWidth;
		int kernelHeightInterleaved = 2*kernelHeight;
		int kernelDepthInterleaved = 2*kernelDepth;
		int paddedWidth = JVCLUtils.nextPwr2(volumeWidthInterleaved+2*kernelWidthInterleaved);
		int paddedHeight = JVCLUtils.nextPwr2(volumeHeightInterleaved+2*kernelHeightInterleaved);
		int paddedDepth = JVCLUtils.nextPwr2(volumeDepthInterleaved+2*kernelDepthInterleaved);
		double[][][] v = JVCLUtils.zeroPadBoundaries(
							ComplexUtils.complex2Interleaved(volume), 
					kernelWidthInterleaved, kernelHeightInterleaved, kernelDepthInterleaved
				);
		double[][][] k = JVCLUtils.deepCopyToPadded(
					JVCLUtils.zeroPadBoundaries(
						ComplexUtils.complex2Interleaved(volume), 
						kernelWidthInterleaved, kernelHeightInterleaved, kernelDepthInterleaved), 
					paddedWidth, paddedHeight, paddedDepth);
		DoubleFFT_3D fft = new DoubleFFT_3D(paddedWidth, paddedHeight, paddedDepth);
		fft.complexForward(v);
		fft.complexForward(k);
		ArrayMath.multiply(v, k);
		fft.complexInverse(v, true);
		return ComplexUtils.interleaved2Complex(
					JVCLUtils.stripBorderPadding(v,
					kernelWidthInterleaved, kernelHeightInterleaved, kernelDepthInterleaved)
			);
		
	}

	public double[] fft(double[] vec, boolean forward) {
		DoubleFFT_1D fft = new DoubleFFT_1D(vec.length/2);
		if (forward) {
			fft.complexForward(vec);
		} else {
			fft.complexInverse(vec, true);
		}
		return vec;
	}
	
	public static void main(String[] args) {
		double[] test = JVCLUtils.fillWithGradient(1024);
		FTCPU ftcpu = new FTCPU();
		JVCLUtils.display(
				JVCLUtils.double2Float(ftcpu.convolve(test, new double[] {0, 1, 0})), "test", 8
		);
	}
	
}


