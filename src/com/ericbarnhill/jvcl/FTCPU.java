/*
 * (c) Eric Barnhill 2016 All Rights Reserved.
 *
 * This file is part of the Java Volumetric Convolution Library (JVCL). JVCL is free software:
 * you can redistribute it and/or modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * JVCL is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details. You should have received a copy of
 * the GNU General Public License along with JVCL.  If not, see http://www.gnu.org/licenses/ .
 *
 * This code uses software from the Apache Software Foundation.
 * The Apache Software License can be found at: http://www.apache.org/licenses/LICENSE-2.0.txt .
 *
 * This code uses software from the JogAmp project.
 * Jogamp information and software license can be found at: https://jogamp.org/ .
 *
 * This code uses methods from the JTransforms package by Piotr Wendykier.
 * JTransforms information and software license can be found at: https://github.com/wendykierp/JTransforms .
 *
 */
package com.ericbarnhill.jvcl;

import org.apache.commons.numbers.complex.Complex;
import org.apache.commons.numbers.complex.ComplexUtils;

import com.ericbarnhill.arrayMath.ArrayMath;

import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_2D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_3D;

/**
 * This class performs Fourier-domain convolutions on the CPU.
 * Arrays must be rectangular i.e. non-ragged. Array and kernel must have the same dimension.
 *
 * @author ericbarnhill
 * @since 0.1
 */
public class FTCPU {

	/**
	 * Convolve 1D {@code Complex[]} array with 1D {@code Complex[]} g
	 * @param f {@code Complex[]} array
	 * @param g {@code Complex[]} g
	 * @return {@code Complex[]}
	 */
	public static Complex[] convolve(Complex[] f, Complex[] g) {
		final int fi = f.length;
		final int gi = g.length;
		final int pad = gi*2;
		final double[] v = ComplexUtils.complex2Interleaved(
						JVCLUtils.zeroPadBoundaries(f, pad)
					);
		final double[] k = JVCLUtils.deepCopyToPadded(
						ComplexUtils.complex2Interleaved(
							JVCLUtils.zeroPadBoundaries(g, pad)
						),
					v.length);
		final DoubleFFT_1D fft = new DoubleFFT_1D(v.length/2);
		fft.complexForward(v);
		fft.complexForward(k);
		ArrayMath.multiply(v, k);
		fft.complexInverse(v, true);
		return 	JVCLUtils.stripBorderPadding(
						ComplexUtils.interleaved2Complex(
								v),
						pad, pad);
	}

	/**
	 * Convolve 1D {@code double[]} array with 1D {@code double[]} g
	 * @param f {@code double[]} array
	 * @param g {@code double[]} g
	 * @return {@code double[]}
	 */
	public static double[] convolve(double[] f, double[] g) {
		return ComplexUtils.complex2Real(
				convolve(ComplexUtils.real2Complex(f), ComplexUtils.real2Complex(g)
						)
				);
	}

	/**
	 * Convolve 2D {@code Complex[][]} array with 2D {@code Complex[][]} g
	 * @param f {@code Complex[][]} array
	 * @param g {@code Complex[][]} g
	 * @return {@code Complex[][]}
	 */
	public static Complex[][] convolve(Complex[][] f, Complex[][] g) {
		final int fi = f.length;
		final int fj = f[0].length;
		final int gi = g.length;
		final int gj = g[0].length;
		final int fiInterleaved = fi;
		final int fjInterleaved = 2*fj;
		final int giInterleaved = gi;
		final int gjInterleaved = 2*gj;
		final int fiPadded = fiInterleaved+2*giInterleaved;
		final int fjPadded = fjInterleaved+2*gjInterleaved;
		final double[][] v = JVCLUtils.zeroPadBoundaries(
							ComplexUtils.complex2Interleaved(f),
					giInterleaved, gjInterleaved);
		final double[][] k = JVCLUtils.deepCopyToPadded(
						JVCLUtils.zeroPadBoundaries(
								ComplexUtils.complex2Interleaved(g),
						giInterleaved, gjInterleaved),
						fiPadded, fjPadded);
		final DoubleFFT_2D fft = new DoubleFFT_2D(fiPadded, fjPadded/2);
		fft.complexForward(v);
		fft.complexForward(k);
		ArrayMath.multiply(v,k);
		fft.complexInverse(v, true);
		return ComplexUtils.interleaved2Complex(
				JVCLUtils.stripBorderPadding(
					v, 2*giInterleaved, 2*gjInterleaved)
				);
	}

	/**
	 * Convolve 3D {@code Complex[][][]} array with 2D {@code Complex[][][]} g
	 * @param f {@code Complex[][][]} array
	 * @param g {@code Complex[][][]} g
	 * @return {@code Complex[][][]}
	 */
	public static Complex[][][] convolve(Complex[][][] f, Complex[][][] g) {

		final int fi = f.length;
		final int fj = f[0].length;
		final int fk = f[0][0].length;
		final int gi = g.length;
		final int gj = g[0].length;
		final int gk = g[0][0].length;
		final int fiInterleaved = 2*fi;
		final int fjInterleaved = 2*fj;
		final int fkInterleaved = 2*fk;
		final int giInterleaved = 2*gi;
		final int gjInterleaved = 2*gj;
		final int gkInterleaved = 2*gk;
		final int fiPadded = JVCLUtils.nextPwr2(fiInterleaved+2*giInterleaved);
		final int fjPadded = JVCLUtils.nextPwr2(fjInterleaved+2*gjInterleaved);
		final int paddedDepth = JVCLUtils.nextPwr2(fkInterleaved+2*gkInterleaved);
		final double[][][] v = JVCLUtils.zeroPadBoundaries(
							ComplexUtils.complex2Interleaved(f),
					giInterleaved, gjInterleaved, gkInterleaved
				);
		final double[][][] k = JVCLUtils.deepCopyToPadded(
					JVCLUtils.zeroPadBoundaries(
						ComplexUtils.complex2Interleaved(f),
						giInterleaved, gjInterleaved, gkInterleaved),
					fiPadded, fjPadded, paddedDepth);
		final DoubleFFT_3D fft = new DoubleFFT_3D(fiPadded, fjPadded, paddedDepth);
		fft.complexForward(v);
		fft.complexForward(k);
		ArrayMath.multiply(v, k);
		fft.complexInverse(v, true);
		return ComplexUtils.interleaved2Complex(
					JVCLUtils.stripBorderPadding(v,
					giInterleaved, gjInterleaved, gkInterleaved)
			);

	}


}


