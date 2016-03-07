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

public class FTGPU {
		
    CLDevice device;
    CLCommandQueue queue;
    CLProgram program;
    CLContext context;
    CLKernel Kernel;
    StockhamGPUStride s;
    boolean debug;
	
	public FTGPU() {
		String path = "/home/ericbarnhill/barnhill-eclipse-workspace/JVCL/";
		context = CLContext.create();
		device = context.getMaxFlopsDevice();
		queue = device.createCommandQueue();
	    String source = JVCLUtils.readFile(path+"src/stockhamStride.cl");
		program = context.createProgram(source).build();
		Kernel = program.createCLKernel("stockhamStride");
	}

	public Complex[] convolve(Complex[] vector, Complex[] kernel) {
		
		int vectorLength = vector.length;
		int kernelLength = kernel.length;
		int vectorLengthInterleaved = 2*vectorLength;
		int kernelLengthInterleaved = 2*kernelLength;
		int paddedLength = JVCLUtils.nextPwr2(vectorLengthInterleaved+2*kernelLengthInterleaved);
		float[] v = JVCLUtils.deepCopyToPadded(
						JVCLUtils.zeroPadBoundaries(
							JVCLUtils.double2Float(
								ComplexUtils.complex2Interleaved(vector) 
							),
						kernelLengthInterleaved),
					paddedLength);
		float[] k = JVCLUtils.deepCopyToPadded(
						JVCLUtils.zeroPadBoundaries(
							JVCLUtils.double2Float(
								ComplexUtils.complex2Interleaved(kernel) 
							),
						kernelLengthInterleaved),
					paddedLength);
		s = new StockhamGPUStride(context, device, program, Kernel, queue);
		int N = v.length / 2;
		s.fft(v, true, N);
		s.fft(k, true, N);
		ArrayMath.multiply(v, k);
		s.fft(v, false, N);
		//s.close();
		return ComplexUtils.interleaved2Complex(
					JVCLUtils.stripEndPadding(
						JVCLUtils.stripBorderPadding(
							JVCLUtils.float2Double(v), kernelLengthInterleaved), 
						vectorLengthInterleaved)
				);
					
	}

	public Complex[][] convolve(Complex[][] image, Complex[][] kernel) {
		debug = false;
		long t1 = 0; long t2 = 0; long t3 = 0; long t4 = 0; long t5 = 0;long t6 = 0;
		int imageWidth = image.length;
		int imageHeight = image[0].length;
		int kernelWidth = kernel.length;
		int kernelHeight = kernel[0].length;
		int imageWidthInterleaved = 2*imageWidth;
		int imageHeightInterleaved = 2*imageHeight;
		int kernelWidthInterleaved = 2*kernelWidth;
		int kernelHeightInterleaved = 2*kernelHeight;
		int paddedWidth = JVCLUtils.nextPwr2(imageWidthInterleaved+2*kernelWidthInterleaved);
		int paddedHeight = JVCLUtils.nextPwr2(imageHeightInterleaved+2*kernelHeightInterleaved);
		if (debug) t1 = System.currentTimeMillis();
		float[] v = ArrayMath.vectorise(
						JVCLUtils.deepCopyToPadded(
							JVCLUtils.zeroPadBoundaries(
								JVCLUtils.double2Float(
									ComplexUtils.complex2Interleaved(image)
								), 
							kernelWidthInterleaved, kernelHeightInterleaved),
						paddedWidth, paddedHeight)
					);
		float[] k = ArrayMath.vectorise(
						JVCLUtils.deepCopyToPadded(
							JVCLUtils.zeroPadBoundaries(
								JVCLUtils.double2Float(
									ComplexUtils.complex2Interleaved(kernel)
								), 
							kernelWidthInterleaved, kernelHeightInterleaved),
						paddedWidth, paddedHeight)
					);
		s = new StockhamGPUStride(context, device, program, Kernel, queue);
		int N = paddedWidth / 2;
		if (debug) t2 = System.currentTimeMillis();
		s.fft(v, true, N);
		s.fft(k, true, N);
		if (debug) t3 = System.currentTimeMillis();
		JVCLUtils.shiftVectorDim(v, paddedWidth);
		JVCLUtils.shiftVectorDim(k, paddedWidth);
		N = paddedHeight / 2;
		if (debug) t4 = System.currentTimeMillis();
		s.fft(v, true, N);
		s.fft(k, true, N);
		ArrayMath.multiply(v, k);
		s.fft(v, false, N);
		if (debug) t5 = System.currentTimeMillis();
		JVCLUtils.shiftVectorDim(v, paddedHeight);
		s.fft(v, false, N);
		if (debug) t6 = System.currentTimeMillis();
		if (debug) System.out.format("%d %d %d %d %d %n", t2-t1, t3-t2, t4-t3, t5-t4, t6-t5);
		return ComplexUtils.interleaved2Complex(
					JVCLUtils.stripEndPadding(
						JVCLUtils.stripBorderPadding(
							ArrayMath.devectorise(
								JVCLUtils.float2Double(v)
							, paddedWidth),
						kernelWidthInterleaved, kernelHeightInterleaved), 
					imageWidthInterleaved, imageHeightInterleaved)
				);
					
	}
	
	public Complex[][][] convolve(Complex[][][] image, Complex[][][] kernel) {
		
		int imageWidth = image.length;
		int imageHeight = image[0].length;
		int imageDepth = image[0][0].length;
		int kernelWidth = kernel.length;
		int kernelHeight = kernel[0].length;
		int kernelDepth = kernel[0][0].length;
		int imageWidthInterleaved = 2*imageWidth;
		int imageHeightInterleaved = 2*imageHeight;
		int imageDepthInterleaved = 2*imageDepth;
		int kernelWidthInterleaved = 2*kernelWidth;
		int kernelHeightInterleaved = 2*kernelHeight;
		int kernelDepthInterleaved = 2*kernelDepth;
		int paddedWidth = JVCLUtils.nextPwr2(imageWidthInterleaved+2*kernelWidthInterleaved);
		int paddedHeight = JVCLUtils.nextPwr2(imageHeightInterleaved+2*kernelHeightInterleaved);
		int paddedDepth = JVCLUtils.nextPwr2(imageDepthInterleaved+2*kernelDepthInterleaved);
		float[] v = ArrayMath.vectorise(
						JVCLUtils.deepCopyToPadded(
							JVCLUtils.zeroPadBoundaries(
								JVCLUtils.double2Float(
									ComplexUtils.complex2Interleaved(image) 
								),
							kernelWidthInterleaved, kernelHeightInterleaved, kernelDepthInterleaved),
						paddedWidth, paddedHeight, paddedDepth)
					);
		float[] k = ArrayMath.vectorise(
						JVCLUtils.deepCopyToPadded(
							JVCLUtils.zeroPadBoundaries(
								JVCLUtils.double2Float(
									ComplexUtils.complex2Interleaved(kernel) 
								),
							kernelWidthInterleaved, kernelHeightInterleaved, kernelDepthInterleaved),
						paddedWidth, paddedHeight, paddedDepth)
					);
		s = new StockhamGPUStride(context, device, program, Kernel, queue);
		int N = paddedWidth / 2;
		s.fft(v, true, N);
		s.fft(k, true, N);
		JVCLUtils.shiftVectorDim(v, paddedWidth);
		JVCLUtils.shiftVectorDim(k, paddedWidth);
		N = paddedHeight / 2;
		s.fft(v, true, N);
		s.fft(k, true, N);
		ArrayMath.multiply(v, k);
		s.fft(v, false, N);
		JVCLUtils.shiftVectorDim(v, paddedHeight);
		s.fft(v, true, N);
		s.close();
		return ComplexUtils.interleaved2Complex(
					JVCLUtils.stripEndPadding(
						JVCLUtils.stripBorderPadding(
							ArrayMath.devectorise(
								JVCLUtils.float2Double(v)
							, paddedWidth, paddedHeight),
						kernelWidthInterleaved, kernelHeightInterleaved, kernelDepthInterleaved), 
					imageWidthInterleaved, imageHeightInterleaved, imageDepthInterleaved)
				);
					
	}
	
	public void close() {
		context.release();
	}

	public static void main(String[] args) {
		Random random = new Random();
		int arraySize = 512;
		int kernelSize = 7;
		long start, end, duration;
		FTGPU ftgpu = new FTGPU();
		System.out.format("Array size %d Kernel Size %d \n", arraySize, kernelSize);
		double[][] array = new double[arraySize][arraySize];
		double[][] kernel = new double[kernelSize][kernelSize];
		for (int x = 0; x < arraySize; x++) {
			for (int y = 0; y < arraySize; y++) {
				if (x < kernelSize && y < kernelSize) {
					kernel[x][y] = random.nextDouble();
				}
				array[x][y] = random.nextDouble();
			}
		}
		start = System.currentTimeMillis();
		for (int t = 0; t < 10; t++) {
			ftgpu.convolve(ComplexUtils.real2Complex(array), ComplexUtils.real2Complex(kernel));
			System.out.println("---");
		}
		end = System.currentTimeMillis();
		duration = end-start;
		System.out.format("2D FT on GPU: %.1f sec %n", duration / 1000.0 );
		System.out.println("Done");
		ftgpu.close();
	}

}
