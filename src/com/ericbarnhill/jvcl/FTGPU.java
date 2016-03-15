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

import java.util.Random;

import org.apache.commons.math4.complex.Complex;
import org.apache.commons.math4.complex.ComplexUtils;

import com.ericbarnhill.arrayMath.ArrayMath;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

/**
 * This class performs Fourier-domain convolutions on the GPU. Note that the Stockham implementation used
 * is quite simple and it does not outperform the CPU method except in specialised cases. Users are welcome
 * to fork the Stockham implementation and add improvements.
 *
 * Arrays must be rectangular i.e. non-ragged. Array and kernel must have the same dimension.
 *
 * @author ericbarnhill
 * @since 0.1
 * @see StockhamFFT
 */
class FTGPU {

    CLDevice device;
    CLCommandQueue queue;
    CLProgram program;
    CLContext context;
    CLKernel Kernel;
    StockhamFFT s;
    boolean debug;

	FTGPU() {
		String path = "openCL/";
		context = CLContext.create();
		device = context.getMaxFlopsDevice();
		queue = device.createCommandQueue();
	    String source = JVCLUtils.readFile(path+"stockhamStride.cl");
		program = context.createProgram(source).build();
		Kernel = program.createCLKernel("stockhamStride");
	}

	Complex[] convolve(Complex[] f, Complex[] g) {

		final int fi = f.length;
		final int gi = g.length;
		final int padFront = gi;
		final int padBackF = JVCLUtils.nextPwr2(fi+2*gi) - fi - gi; // rounds out to next power of 2
		final int padBackG = JVCLUtils.nextPwr2(fi+2*gi) - gi - gi; // rounds out to next power of 2
		final int totalLengthF = padFront + fi + padBackF;
		final int totalLengthG = padFront + gi + padBackG;
		System.out.format("%d %d %d %d %d %d %d %n", fi, gi, padFront, padBackF, padBackG, totalLengthF, totalLengthG);
		Complex[] fPad = JVCLUtils.zeroPadBoundaries(f, padFront, padBackF);
		Complex[] gPad = JVCLUtils.zeroPadBoundaries(g, padFront, padBackG);
		s = new StockhamFFT(context, device, program, Kernel, queue);
		int N = totalLengthF;
		JVCLUtils.display(fPad, "prefft", 16);
		fPad = s.fft(fPad, true, N);
		gPad = s.fft(gPad, true, N);
		JVCLUtils.display(fPad, "premult", 16);
		ArrayMath.multiply(fPad, gPad);
		JVCLUtils.display(fPad, "postmult", 16);
		fPad = s.fft(fPad, false, N);
		JVCLUtils.display(fPad, "postifft", 16);
		s.close();
		fPad = JVCLUtils.stripBorderPadding(fPad, padFront, padBackF);
		//fPad = JVCLUtils.stripBorderPadding(fPad, fi/2, fi/2);
		JVCLUtils.display(fPad, "post strip", 16);
		return fPad;
	}

	Complex[][] convolve(Complex[][] f, Complex[][] g) {

		final int fi = f.length;
		final int fj = f[0].length;
		final int gi = g.length;
		final int gj = g[0].length;
		final int padFrontI = gi;
		final int padBackFI = JVCLUtils.nextPwr2(fi+2*gi) - fi - gi; // rounds out to next power of 2
		final int padBackGI = JVCLUtils.nextPwr2(fi+2*gi) - gi - gi; // rounds out to next power of 2
		final int padFrontJ = gj;
		final int padBackFJ = JVCLUtils.nextPwr2(fj+2*gj) - fj - gj; // rounds out to next power of 2
		final int padBackGJ = JVCLUtils.nextPwr2(fj+2*gj) - gj - gj; // rounds out to next power of 2
		final int totalLengthFI = padFrontI + fi + padBackFI;
		final int totalLengthGI = padFrontI + gi + padBackGI;
		final int totalLengthFJ = padFrontJ + fj + padBackFJ;
		final int totalLengthGJ = padFrontJ + gj + padBackGJ;
		Complex[][] fPad = JVCLUtils.zeroPadBoundaries(f, padFrontI, padBackFI, padFrontJ, padBackFJ);
		Complex[][] gPad = JVCLUtils.zeroPadBoundaries(g, padFrontI, padBackGI, padFrontJ, padBackGJ);
		s = new StockhamFFT(context, device, program, Kernel, queue);
		int NI = totalLengthFI;
		int NJ = totalLengthFJ;
		fPad = s.fft(fPad, true, NI);
		gPad = s.fft(gPad, true, NI);
		fPad = ArrayMath.shiftDim(fPad);
		gPad = ArrayMath.shiftDim(gPad);
	    fPad = s.fft(fPad, true, NJ);
	    gPad = s.fft(gPad, true, NJ);
		ArrayMath.multiply(fPad, gPad);
		fPad = s.fft(fPad, false, NJ);
		fPad = ArrayMath.shiftDim(fPad);
        fPad = s.fft(fPad, false, NI);
		s.close();
		return JVCLUtils.stripBorderPadding(fPad, padFrontI, padBackFI, padFrontJ, padBackFJ);
		//return JVCLUtils.stripBorderPadding(fPad, fi/2, fi/2, fj/2, fj/2);
	}

	/**
	 * Should be called as destructor method.
	 */
	void close() {
		context.release();
	}

	static void main(String[] args) {
		Random random = new Random();
		int arraySize = 128;
		int gSize = 7;
		long start, end, duration;
		FTGPU ftgpu = new FTGPU();
		System.out.format("Array size %d Kernel Size %d \n", arraySize, gSize);
		// 1D
		double[] f = ArrayMath.secondOrder(arraySize);
		double[] lap1D = new double[] {1, -2, 1};
		start = System.currentTimeMillis();
		Complex[] result = ftgpu.convolve(ComplexUtils.real2Complex(f), ComplexUtils.real2Complex(lap1D));
		end = System.currentTimeMillis();
		duration = end-start;
		System.out.format("1D FT on GPU: %.1f sec %n", duration / 1000.0 );
		JVCLUtils.display(ComplexUtils.complex2Real(result), "result", 16);
		System.out.println("Done");
		ftgpu.close();
		/*
		// 2D
		ftgpu = new FTGPU();
	    double[][] f2 = ArrayMath.secondOrder(arraySize, arraySize);
	    double[][] lap2D = JVCLUtils.laplacian2();
	    start = System.currentTimeMillis();
	    Complex[][] result2 = ftgpu.convolve(ComplexUtils.real2Complex(f2), ComplexUtils.real2Complex(lap2D));
	    end = System.currentTimeMillis();
	    duration = end-start;
	    System.out.format("1D FT on GPU: %.1f sec %n", duration / 1000.0 );
	    JVCLUtils.display(ComplexUtils.complex2Real(result2), "result", 32);
	    System.out.println("Done");
	    ftgpu.close();
	    */
	}

}
