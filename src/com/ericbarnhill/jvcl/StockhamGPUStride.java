package com.ericbarnhill.jvcl;

import java.nio.FloatBuffer;

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

public class StockhamGPUStride {
	
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
    

    
	public StockhamGPUStride() {	
		context = CLContext.create();
		device = context.getMaxFlopsDevice();
		queue = device.createCommandQueue();
	    String sourceStride = JVCLUtils.readFile("src/stockhamStride.cl");
		program = context.createProgram(sourceStride).build();
		Kernel = program.createCLKernel("stockhamStride");
	}
	
	public StockhamGPUStride(CLContext context, CLDevice device, CLProgram program, CLKernel Kernel, CLCommandQueue queue) {
		this.context = context;
		this.device = device;
		this.program = program;
		this.Kernel = Kernel;
		this.queue = queue;
		Kernel = program.createCLKernel("stockhamStride");
	}
	
	void fft(float[] real, float[] imag, boolean isForward, int N) {
	    int len = real.length;
	    int blocks = len / N;
	    int pwr2 = (int)(Math.log(N)/Math.log(2));
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
	
	void fft(float[] interleaved, boolean isForward, int N) {
		Complex[] c = ComplexUtils.interleaved2Complex(interleaved);
		fft(ComplexUtils.complex2RealFloat(c), ComplexUtils.complex2ImaginaryFloat(c), isForward, N);
	}

	void fft(Complex[] c, boolean isForward, int N) {
		fft(ComplexUtils.complex2RealFloat(c), ComplexUtils.complex2ImaginaryFloat(c), isForward, N);
	}
	
	void fft(Complex[][] c, boolean isForward, int N) {
		fft(ComplexUtils.complex2RealFloat(ArrayMath.vectorise(c)),
				ComplexUtils.complex2ImaginaryFloat(ArrayMath.vectorise(c)), isForward, N);
	}
	
	public void close() {
		Kernel.release();
	}

	public static void main(String[] args) {
		CLContext context = CLContext.create();
		CLDevice device = context.getMaxFlopsDevice();
		CLCommandQueue queue = device.createCommandQueue();
		String source = JVCLUtils.readFile("src/stockhamStride.cl");
		CLProgram program = context.createProgram(source).build();
		CLKernel Kernel = program.createCLKernel("stockhamStride");
		StockhamGPUStride s = new StockhamGPUStride(context, device, program, Kernel, queue);
		
		float[][] f = JVCLUtils.double2Float(JVCLUtils.fillWithSecondOrder(512, 512));
		for (int n = 0; n < 10; n++) {
			long t1 = System.currentTimeMillis();
			s.fft(ArrayMath.vectorise(f), ArrayMath.vectorise(f), true, 512);
			long t2 = System.currentTimeMillis();
			System.out.format("%.3f %n", (t2-t1) / 1000.0);
		}
		s.close();
	}
	
	
	
}
