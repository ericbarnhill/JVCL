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

import edu.emory.mathcs.jtransforms.fft.FloatFFT_1D;
import edu.emory.mathcs.jtransforms.fft.FloatFFT_2D;
import edu.emory.mathcs.jtransforms.fft.FloatFFT_3D;

/**
 * This class performs Fourier-domain convolutions on the CPU.
 * Arrays must be rectangular i.e. non-ragged. Array and kernel must have the same number of dimensions.
 *
 * @author ericbarnhill
 * @since 0.1
 */
public class ConvolverFloatFTCPU extends ConvolverFloat {

    ConvolverComplexFTCPU ccf;
    ConvolverFloatFTCPU() {
        ccf = new ConvolverComplexFTCPU();
    }
	/**
	 * Convolve 1D {@code float[]} array with 1D {@code float[]} g
	 * @param f {@code float[]} array
	 * @param g {@code float[]} g
	 * @return {@code float[]}
	 */
	public float[] convolve(float[] f, float[] g) {
		return ComplexUtils.complex2RealFloat(
				ccf.convolve(ComplexUtils.real2Complex(f), ComplexUtils.real2Complex(g)
						)
				);
	}

	/**
	 * Convolve 2D {@code float[][]} array with 2D {@code float[][]} g
	 * @param f {@code float[][]} array
	 * @param g {@code float[][]} g
	 * @return {@code float[][]}
	 */
	public float[][] convolve(float[][] f, float[][] g) {
		return ComplexUtils.complex2RealFloat(
				ccf.convolve(ComplexUtils.real2Complex(f), ComplexUtils.real2Complex(g)
						)
				);
	}

	/**
	 * Convolve 1D {@code float[][][]} array with 1D {@code float[][][]} g
	 * @param f {@code float[][][]} array
	 * @param g {@code float[][][]} g
	 * @return {@code float[][][]}
	 */
	public float[][][] convolve(float[][][] f, float[][][] g) {
		return ComplexUtils.complex2RealFloat(
				ccf.convolve(ComplexUtils.real2Complex(f), ComplexUtils.real2Complex(g)
						)
				);
	}


    public Float[] convolve(Float[] f, Float[] g) {
        return ArrayMath.box(convolve(ArrayMath.unbox(f), ArrayMath.unbox(g)));
    }

    public Float[][] convolve(Float[][] f, Float[] g) {
        return ArrayMath.box(convolve(ArrayMath.unbox(f), ArrayMath.convertTo2d(ArrayMath.unbox(g))));
    }

    public Float[][] convolve(Float[][] f, Float[][] g) {
        return ArrayMath.box(convolve(ArrayMath.unbox(f), ArrayMath.unbox(g)));
    }

    public Float[][][] convolve(Float[][][] f, Float[] g) {
        return ArrayMath.box(convolve(ArrayMath.unbox(f), ArrayMath.convertTo3d(ArrayMath.convertTo2d(ArrayMath.unbox(g)))));
    }

    public Float[][][] convolve(Float[][][] f, Float[][] g) {
        return ArrayMath.box(convolve(ArrayMath.unbox(f), ArrayMath.convertTo3d(ArrayMath.unbox(g))));
    }

    public Float[][][] convolve(Float[][][] f, Float[][][] g) {
        return ArrayMath.box(convolve(ArrayMath.unbox(f), ArrayMath.unbox(g)));
    }


}


