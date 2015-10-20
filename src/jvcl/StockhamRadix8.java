package jvcl;

import java.nio.FloatBuffer;
import java.util.Arrays;

import org.apache.commons.math4.complex.Complex;
import org.apache.commons.math4.complex.ComplexUtils;
import org.apache.commons.math4.util.FastMath;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.demos.fft.CLFFTPlan;

import arrayMath.ArrayMath;

public class StockhamRadix8 {
	
	CLFFTPlan fft;
    JVCLUtils ds;
    CLDevice device;
    CLProgram program;
    CLContext context;
    long totalTime;
    CLKernel Kernel;
    CLCommandQueue queue;
    int local, global;
    CLBuffer<FloatBuffer> CLBufferSrc;
	CLBuffer<FloatBuffer> CLBufferDest;  
	FloatBuffer src;
	FloatBuffer dest;
    

    
	public StockhamRadix8() {	
		context = CLContext.create();
		device = context.getMaxFlopsDevice();
		queue = device.createCommandQueue();
	    String sourceStride = JVCLUtils.readFile("src/StockhamRadix8.cl");
		program = context.createProgram(sourceStride).build();
		Kernel = program.createCLKernel("stockhamRadix8");
	}
	
	public StockhamRadix8(CLContext context, CLDevice device, CLProgram program, CLKernel Kernel, CLCommandQueue queue) {
		this.context = context;
		this.device = device;
		this.program = program;
		this.Kernel = Kernel;
		this.queue = queue;
		Kernel = program.createCLKernel("stockhamStride");
	}
	
	void fft(float[] srcArray, float[] destArray, boolean isForward, int N) {
	    int len = srcArray.length;
	    int blocks = len / N;
	    int threads = N / 8;
	    int logN = (int)(Math.log(N)/Math.log(2));
	    int sign;
		if (isForward) {
			sign = 1;
		} else {
			sign = -1;
		}
		int global = threads;
		int local = 16;
		int blockSize = 1;
		//for (int n = 0; n < 1; n++) { //test
		for (int n = 0; n < logN / 3; n++) {
			CLBufferSrc = context.createFloatBuffer(len);
			CLBufferDest = context.createFloatBuffer(len);
			src = CLBufferSrc.getBuffer();
			dest = CLBufferDest.getBuffer();
			src.clear();
			src.put(srcArray).rewind();
			dest.clear();
			dest.put(destArray).rewind();
			Kernel.setArg(0, CLBufferSrc)
				.setArg(1, CLBufferDest)	
				.setArg(2, blockSize)
				.setArg(3, threads);
			queue.putWriteBuffer(CLBufferSrc, true);
			queue.put2DRangeKernel(Kernel, 0, 0, N, blocks, N, 1);
			queue.putReadBuffer(CLBufferDest, true);
			//src.get(srcArray);
			dest.get(destArray);
			srcArray = JVCLUtils.deepCopy(destArray);
			blockSize *= 8;
		}
		CLBufferSrc.release();
		CLBufferDest.release();
		//testing release -
		if (!isForward) {
			for (int n = 0; n < N; n++) {
				destArray[n] = destArray[n] / (float)N;
			}
		}
	}
	
	public void close() {
		Kernel.release();
	}

	public static void main(String[] args) {
		/*
		CLContext context = CLContext.create();
		CLDevice device = context.getMaxFlopsDevice();
		CLCommandQueue queue = device.createCommandQueue();
		String source = JVCLUtils.readFile("src/stockhamRadix8.cl");
		CLProgram program = context.createProgram(source).build();
		CLKernel Kernel = program.createCLKernel("stockhamStride");
		StockhamRadix8 s = new StockhamRadix8(context, device, program, Kernel, queue);
		*/
		StockhamRadix8 s = new StockhamRadix8();

		int N = 512;
		float[] testR = new float[N];
		float[] testI = new float[N];
		for (int n = 0; n < N; n++) {
				testR[n] = 10*(float)FastMath.cos(2*FastMath.PI*(n+1)/(double)(N/2.0));
				testI[n] = 10*(float)FastMath.sin(2*FastMath.PI*(n+1)/(double)(N/2.0));
		}
		System.out.println(Arrays.toString(ArrayMath.round(testR)));		
		float[] dst = new float[N*2];
		s.fft(JVCLUtils.split2interleaved(testR, testI), dst, true, N);
		Complex[] result = ComplexUtils.interleaved2Complex(dst);
		//System.out.println(Arrays.toString(ArrayMath.round(ComplexUtils.complex2RealFloat(result))));
		System.out.println(Arrays.toString(ArrayMath.round(dst)));		
		s.close();
	}
	
	
	
}
