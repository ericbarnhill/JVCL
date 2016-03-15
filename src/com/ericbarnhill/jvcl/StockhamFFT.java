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

import java.nio.FloatBuffer;
import java.util.ArrayList;

import org.apache.commons.math4.complex.Complex;
import org.apache.commons.math4.complex.ComplexUtils;

import com.ericbarnhill.arrayMath.ArrayMath;
import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.demos.fft.CLFFTPlan;

/**
 * Stockham FFT implementation in JogAmp. Note that this implementation is quite simple.
 * Users are encouraged to fork it and add optimisations such as prime radixes, spin
 * vectors, etc.
 *
 * @author ericbarnhill
 * @since 0.1
 * @see FTGPU
 */
class StockhamFFT {

	CLFFTPlan fft;
    JVCLUtils ds;
    CLDevice device;
    CLProgram program;
    CLContext context;
    long totalTime;
    CLKernel Kernel;
    CLCommandQueue queue;
    int local, global;
    CLBuffer<FloatBuffer> CLBufferReal;
	CLBuffer<FloatBuffer> CLBufferImag;
	FloatBuffer realBuffer;
	FloatBuffer imagBuffer;

	StockhamFFT() {
		context = CLContext.create();
		device = context.getMaxFlopsDevice();
		queue = device.createCommandQueue();
	    String sourceStride = JVCLUtils.readFile("src/stockhamStride.cl");
		program = context.createProgram(sourceStride).build();
		Kernel = program.createCLKernel("stockhamStride");
	}

	StockhamFFT(CLContext context, CLDevice device, CLProgram program, CLKernel Kernel, CLCommandQueue queue) {
		this.context = context;
		this.device = device;
		this.program = program;
		this.Kernel = Kernel;
		this.queue = queue;
		Kernel = program.createCLKernel("stockhamStride");
	}

	void fft(float[] real, float[] imag, boolean isForward, int N) {
	    final int len = real.length;
	    final int blocks = len / N;
	    final int pwr2 = (int)(Math.log(N)/Math.log(2));
		if (!isForward) {
			imag = ArrayMath.multiply(imag,  -1);
		}
		CLBufferReal = context.createFloatBuffer(len);
		CLBufferImag = context.createFloatBuffer(len);
		realBuffer = CLBufferReal.getBuffer();
		imagBuffer = CLBufferImag.getBuffer();
		realBuffer.clear();
		realBuffer.put(real).rewind();
		imagBuffer.clear();
		imagBuffer.put(imag).rewind();
		Kernel.setArg(0, CLBufferReal)
			.setArg(1, CLBufferImag)
			.setArg(2, -1)
			.setArg(3, N)
			.setArg(4,  pwr2)
			.setArg(5, len);
		queue.putWriteBuffer(CLBufferReal, true);
		queue.putWriteBuffer(CLBufferImag, true);
		queue.put2DRangeKernel(Kernel, 0, 0, N, blocks, 0, 0);
		queue.putReadBuffer(CLBufferReal, true);
		queue.putReadBuffer(CLBufferImag, true);
		realBuffer.get(real);
		imagBuffer.get(imag);
		CLBufferReal.release();
		CLBufferImag.release();
		final double scaling = 1 / (float)N;
		if (!isForward) {
			for (int n = 0; n < len; n++) {
				real[n] *= scaling;
				imag[n] *= -scaling;
			}
		}
	}

	void fft(ArrayList<float[]> split, boolean isForward, int N) {
		fft(split.get(0), split.get(1), isForward, N);
	}

	Complex[] fft(Complex[] c, boolean isForward, int N) {
		float[] re = ComplexUtils.complex2RealFloat(c);
		float[] im = ComplexUtils.complex2ImaginaryFloat(c);
		fft(re, im, isForward, N);
		Complex[] cFFT = ComplexUtils.split2Complex(re, im);
		return cFFT;
	}

   Complex[][] fft(Complex[][] c, boolean isForward, int N) {
       final int ci = c.length;
       final int cj = c[0].length;
       Complex[][] cFFT = ComplexUtils.initialize(new Complex[ci][cj]);
       for (int i = 0; i < ci; i++) {
            cFFT[i] = fft(c[i], isForward, N);
       }
       return cFFT;
    }

   Complex[][][] fft(Complex[][][] c, boolean isForward, int N) {
       final int ci = c.length;
       final int cj = c[0].length;
       final int ck = c[0][0].length;
       Complex[][][] cFFT = ComplexUtils.initialize(new Complex[ci][cj][ck]);
       for (int i = 0; i < ci; i++) {
           cFFT[i] = fft(cFFT[i], isForward, N);
       }
       return cFFT;
    }


	/**
	 * should be called as destructor method
	 */
	void close() {
		Kernel.release();
	}

}
