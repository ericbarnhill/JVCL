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

import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.WRITE_ONLY;
import static java.lang.Math.min;

import java.nio.FloatBuffer;

import org.apache.commons.numbers.complex.Complex;
import org.apache.commons.numbers.complex.ComplexUtils;

import com.ericbarnhill.arrayMath.ArrayMath;
import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

/**
 * This class performs Finite-Differences convolutions on the GPU. OpenCL is required.
 * Because of the need to create GPU objects, the FDGPU methods are not static and must be instantiated.
 *
 * @author ericbarnhill
 * @since 0.1
 */

public class ConvolverComplexFDGPU extends ConvolverComplex {

    CLDevice device;
    CLCommandQueue queue;
    CLProgram program1d, program2d, program3d, program1dComplex, program2dComplex, program3dComplex,
    	program21, program21Complex, program31, program31Complex;
    CLContext context;
    int localWorkSize;

	public ConvolverComplexFDGPU() {
		//try {
			context = CLContext.create();
			device = context.getMaxFlopsDevice();
	        queue = device.createCommandQueue();
	        String path="openCL/";
	        String source1d = readFile(path+"Convolve1d.cl");
	        String source2d = readFile(path+"Convolve2d.cl");
	        String source3d = readFile(path+"Convolve3d.cl");
	        String source1dComplex = readFile(path+"Convolve1dComplex.cl");
	        String source2dComplex = readFile(path+"Convolve2dComplex.cl");
	        String source3dComplex = readFile(path+"Convolve3dComplex.cl");
	        String source21 = readFile(path+"Convolve21.cl");
	        String source21Complex = readFile(path+"Convolve21Complex.cl");
	        String source31 = readFile(path+"Convolve31.cl");
	        String source31Complex = readFile(path+"Convolve31Complex.cl");
	        program1d = context.createProgram(source1d).build();
	        program2d = context.createProgram(source2d).build();
	        program3d = context.createProgram(source3d).build();
	        program21 = context.createProgram(source21).build();
	        program31 = context.createProgram(source31).build();
	        program1dComplex = context.createProgram(source1dComplex).build();
	        program2dComplex = context.createProgram(source2dComplex).build();
	        program3dComplex = context.createProgram(source3dComplex).build();
	        program21Complex = context.createProgram(source21Complex).build();
	        program31Complex = context.createProgram(source31Complex).build();
			localWorkSize = min(device.getMaxWorkGroupSize(), 32);  // Local work size dimensions
		//} catch (Exception e) {
		//	throw new NoGPUException();
		//}
	}

	/**
	 * Convolve 1D {@code Complex[]} array with 1D {@code Complex[]} kernel
	 * @param f {@code Complex[]} array
	 * @param g {@code Complex[]} kernel
	 * @return {@code Complex[]}
	 */
	public Complex[] convolve(Complex[] f, Complex[] g) {
		final int fi = f.length;
		final int gi = g.length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		Complex[] fPad = ArrayMath.zeroPadBoundaries(f, hgi, hgie);
		Complex[] r = ArrayMath.zeroPadBoundaries(new Complex[fi], hgi, hgie);
		final int ri = r.length;
    	CLBuffer<FloatBuffer> clF = context.createFloatBuffer(ri*2, READ_ONLY);
        CLBuffer<FloatBuffer> clG = context.createFloatBuffer(gi*2, READ_ONLY);
        CLBuffer<FloatBuffer> clR = context.createFloatBuffer(ri*2, WRITE_ONLY);
        clF.getBuffer().put(ComplexUtils.complex2InterleavedFloat(fPad)).rewind();
        clG.getBuffer().put(ComplexUtils.complex2InterleavedFloat(g)).rewind();
        CLKernel Kernel = program1dComplex.createCLKernel("Convolve1dComplex");
        Kernel.putArg(clF)
        	.putArg(clG)
        	.putArg(clR)
        	.putArg(ri)
        	.putArg(gi)
        	.putArg(hgi)
        	.putArg(hgie);
        queue.putWriteBuffer(clF, false)
        	.putWriteBuffer(clG, false)
        	.put1DRangeKernel(Kernel, 0, roundUp(ri, localWorkSize),0)
        	.putReadBuffer(clR, true);
		float[] result = new float[ri*2];
		clR.getBuffer().get(result);
		clF.release();
		clG.release();
		clR.release();
        return ComplexUtils.interleaved2Complex(result);

	}
    
	/**
	 * Convolve 2D {@code Complex[][]} array with 1D {@code Complex[]} kernel.
	 * @param f {@code Complex[][]} array
	 * @param g {@code Complex[]} kernel
	 * @param dim orientation of kernel (0 or 1)
	 * @return {@code Complex[][]}
	 */
	public Complex[][] convolve(Complex[][] f, Complex[] g, int dim) {
		if (dim == 1) {
			f = ArrayMath.shiftDim(f);
		}
		final int fi = f.length;
		final int fj = f[0].length;
		final int gi = g.length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		Complex[][] fPad = ArrayMath.zeroPadBoundaries(f, hgi, hgie, 0, 0);
		Complex[][] r = ArrayMath.zeroPadBoundaries(new Complex[fi][fj], hgi, hgie, 0, 0);
		final int ri = r.length;
    	CLBuffer<FloatBuffer> clF = context.createFloatBuffer(ri*fj*2, READ_ONLY);
        CLBuffer<FloatBuffer> clG = context.createFloatBuffer(gi*2, READ_ONLY);
        CLBuffer<FloatBuffer> clR = context.createFloatBuffer(ri*fj*2, WRITE_ONLY);
        clF.getBuffer().put(ComplexUtils.complex2InterleavedFloat(ArrayMath.vectorize(fPad))).rewind();
        clG.getBuffer().put(ComplexUtils.complex2InterleavedFloat(g)).rewind();
        CLKernel Kernel = program21Complex.createCLKernel("Convolve21Complex");
        Kernel.putArg(clF)
        	.putArg(clG)
        	.putArg(clR)
        	.putArg(ri)
        	.putArg(fj)
        	.putArg(gi)
        	.putArg(hgi)
        	.putArg(hgie);
        queue.putWriteBuffer(clF, false)
        	.putWriteBuffer(clG, false)
        	.put2DRangeKernel(Kernel, 0, 0, ri, fj, 0,0)
        	.putReadBuffer(clR, true);
		float[] resultVec = new float[ri*fj*2];
		clR.getBuffer().get(resultVec);
		clF.release();
		clG.release();
		clR.release();
		Complex[][] result = ArrayMath.devectorize(ComplexUtils.interleaved2Complex(resultVec), ri);
		if (dim == 1) result = ArrayMath.shiftDim(result);
        return result;
	}

	/**
	 * Convolve 2D {@code Complex[][]} array with 1D {@code Complex[]} kernel.
	 * @param f {@code Complex[][]} array
	 * @param g {@code Complex[]} kernel
	 * @return {@code Complex[][]}
	 */
	public Complex[][] convolve(Complex[][] f, Complex[] g) {
		return convolve(f, g, 0);
	}

	/**
	 * Convolve 3D {@code Complex[][][]} array with 1D {@code Complex[]} kernel.
	 * @param f {@code Complex[][][]} array
	 * @param g {@code Complex[]} kernel
	 * @param dim orientation of kernel (0,1, or 2)
	 * @return {@code Complex[][][]}
	 */
	public Complex[][][] convolve(Complex[][][] f, Complex[] g, int dim) {
		if (dim == 1) f = ArrayMath.shiftDim(f, 1);
		if (dim == 2) f = ArrayMath.shiftDim(f, 2);
		final int fi = f.length;
		final int fj = f[0].length;
		final int fk = f[0][0].length;
		final int gi = g.length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		Complex[][][] fPad = ArrayMath.zeroPadBoundaries(f, hgi, hgie, 0, 0, 0, 0);
		Complex[][][] r = ArrayMath.zeroPadBoundaries(new Complex[fi][fj][fk], hgi, hgie, 0, 0, 0, 0);
		final int ri = r.length;
    	CLBuffer<FloatBuffer> clF = context.createFloatBuffer(ri*fj*fk*2, READ_ONLY);
        CLBuffer<FloatBuffer> clG = context.createFloatBuffer(gi*2, READ_ONLY);
        CLBuffer<FloatBuffer> clR = context.createFloatBuffer(ri*fj*fk*2, WRITE_ONLY);
        clF.getBuffer().put(ComplexUtils.complex2InterleavedFloat(ArrayMath.vectorize(fPad))).rewind();
        clG.getBuffer().put(ComplexUtils.complex2InterleavedFloat(g)).rewind();
        CLKernel Kernel = program31Complex.createCLKernel("Convolve31Complex");
        Kernel.putArg(clF)
        	.putArg(clG)
        	.putArg(clR)
        	.putArg(ri)
        	.putArg(fj)
        	.putArg(gi)
        	.putArg(hgi)
        	.putArg(hgie);
        queue.putWriteBuffer(clF, false)
        	.putWriteBuffer(clG, false)
        	.put3DRangeKernel(Kernel, 0, 0, 0, ri, fj, fk, 0,0,0)
        	.putReadBuffer(clR, true);
		float[] resultVec = new float[ri*fj*fk*2];
		clR.getBuffer().get(resultVec);
		clF.release();
		clG.release();
		clR.release();
        Complex[][][] result = ArrayMath.devectorize(ComplexUtils.interleaved2Complex(resultVec), ri, fj);
        if (dim == 1) result = ArrayMath.shiftDim(result, 2);
		if (dim == 2) result = ArrayMath.shiftDim(result, 1);
		return result;
	}

	/**
	 * Convolve 2D {@code Complex[][]} array with 2D {@code Complex[][]} kernel.
	 * @param f {@code Complex[][]} array
	 * @param g {@code Complex[][]} kernel
	 * @return {@code Complex[][]}
	 */
	public Complex[][] convolve(Complex[][] f, Complex[][] g) {
		final int fi = f.length;
		final int fj = f[0].length;
		final int gi = g.length;
		final int gj = g[0].length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgj = (int)( (gj - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
		Complex[][] fPad = ArrayMath.zeroPadBoundaries(f, hgi, hgie, hgj, hgje);
		Complex[][] r = ArrayMath.zeroPadBoundaries(ComplexUtils.initialize(new Complex[fi][fj]),
				hgi, hgie, hgj, hgje);
		final int ri = r.length;
		final int rj = r[0].length;
    	CLBuffer<FloatBuffer> clF = context.createFloatBuffer(ri*rj*2, READ_ONLY);
        CLBuffer<FloatBuffer> clG = context.createFloatBuffer(gi*gj*2, READ_ONLY);
        CLBuffer<FloatBuffer> clR = context.createFloatBuffer(ri*rj*2, WRITE_ONLY);
        clF.getBuffer().put(ArrayMath.vectorize(ComplexUtils.complex2InterleavedFloat(fPad, 0)))
        	.rewind();
        clG.getBuffer().put(ArrayMath.vectorize(ComplexUtils.complex2InterleavedFloat(g, 0)))
    		.rewind();
        CLKernel Kernel = program2dComplex.createCLKernel("Convolve2dComplex");
        Kernel.putArg(clF)
        	.putArg(clG)
        	.putArg(clR)
        	.putArg(ri)
        	.putArg(rj)
        	.putArg(gi)
        	.putArg(gj)
        	.putArg(hgi)
        	.putArg(hgj)
        	.putArg(hgie)
        	.putArg(hgje);
        queue.putWriteBuffer(clF, false)
        	.putWriteBuffer(clG, false)
        	.put2DRangeKernel(Kernel, 0, 0, ri, rj, 0,0)
        	.putReadBuffer(clR, true);
		float[] result = new float[ri*rj*2];
		clR.getBuffer().get(result);
		clF.release();
        clG.release();
        clR.release();
        return ArrayMath.devectorize(ComplexUtils.interleaved2Complex(result), ri);
	}

	/**
	 * Convolve 3D {@code Complex[][][]} array with 3D {@code Complex[][][]} kernel.
	 * @param f {@code Complex[][][]} array
	 * @param g {@code Complex[][][]} kernel
	 * @return {@code Complex[][][]}
	 */
	public Complex[][][] convolve(Complex[][][] f, Complex[][][] g) {
		final int fi = f.length;
		final int fj = f[0].length;
		final int fk = f[0][0].length;
		final int gi = g.length;
		final int gj = g[0].length;
		final int gk = g[0][0].length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgj = (int)( (gj - 1) / 2.0);
		final int hgk = (int)( (gk - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
		final int hgke = (gk % 2 == 0) ? hgk + 1 : hgk;
		//Complex[][][] fPad = ArrayMath.zeroPadBoundaries(f, hgi, hgie, hgj, hgje, hgk, hgke);
		Complex[][][] fPad = ArrayMath.zeroPadBoundaries(f, 1);
		//Complex[][][] r = ArrayMath.zeroPadBoundaries(ComplexUtils.initialize(new Complex[fi][fj][fk]),
		//		hgi, hgie, hgj, hgje, hgk, hgke);
		Complex[][][] r = ArrayMath.zeroPadBoundaries(ComplexUtils.initialize(new Complex[fi][fj][fk]),1);
		final int ri = r.length;
		final int rj = r[0].length;
		final int rk = r[0][0].length;
    	CLBuffer<FloatBuffer> clF = context.createFloatBuffer(ri*rj*rk*2, READ_ONLY);
        CLBuffer<FloatBuffer> clG = context.createFloatBuffer(gi*gj*gk*2, READ_ONLY);
        CLBuffer<FloatBuffer> clR = context.createFloatBuffer(ri*rj*rk*2, WRITE_ONLY);
        clF.getBuffer().put(ArrayMath.vectorize(ComplexUtils.complex2InterleavedFloat(fPad, 0)))
        	.rewind();
        clG.getBuffer().put(ArrayMath.vectorize(ComplexUtils.complex2InterleavedFloat(g, 0)))
    		.rewind();
        CLKernel Kernel = program3dComplex.createCLKernel("Convolve3dComplex");
        Kernel.putArg(clF)
        	.putArg(clG)
        	.putArg(clR)
        	.putArg(ri)
        	.putArg(rj)
        	.putArg(rk)
        	.putArg(gi)
        	.putArg(gj)
        	.putArg(gk)
        	.putArg(hgi)
        	.putArg(hgj)
        	.putArg(hgk)
        	.putArg(hgie)
        	.putArg(hgje)
        	.putArg(hgke);
        queue.putWriteBuffer(clF, false)
        	.putWriteBuffer(clG, false)
        	.put3DRangeKernel(Kernel, 0, 0, 0, ri, rj, rk, 0,0,0)
        	.putReadBuffer(clR, true);
		float[] result = new float[ri*rj*rk*2];
		clR.getBuffer().get(result);
		clF.release();
        clG.release();
        clR.release();
        return ArrayMath.devectorize(ComplexUtils.interleaved2Complex(result), ri, rj);
	}

	/**
	 * Should be called as destructor method.
	 */
	public void close() {
        context.release();
	}

    public Complex[][][] convolve(Complex[][][] f, Complex[] g) {
        return convolve(f, ArrayMath.convertTo3d(ArrayMath.convertTo2d(g)));
    }

    public Complex[][][] convolve(Complex[][][] f, Complex[][] g) {
        return convolve(f, ArrayMath.convertTo3d(g));
    }
}
