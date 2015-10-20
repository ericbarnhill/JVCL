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

public class StockhamRadix2 {
	
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
    

    
	public StockhamRadix2() {	
		context = CLContext.create();
		device = context.getMaxFlopsDevice();
		queue = device.createCommandQueue();
	    String sourceStride = JVCLUtils.readFile("src/StockhamRadix2.cl");
		program = context.createProgram(sourceStride).build();
		Kernel = program.createCLKernel("stockhamRadix2");
	}
	
	public StockhamRadix2(CLContext context, CLDevice device, CLProgram program, CLKernel Kernel, CLCommandQueue queue) {
		this.context = context;
		this.device = device;
		this.program = program;
		this.Kernel = Kernel;
		this.queue = queue;
		Kernel = program.createCLKernel("stockhamStride");
	}
	
	float[] fft(float[] srcArray, float[] destArray, boolean isForward, int N) {
	    int len = srcArray.length;
	    int threads = N / 2;
	    int logN = (int)(Math.log(N)/Math.log(2));
	    int sign;
	    float[] finalArray = null;
		if (isForward) {
			sign = 1;
		} else {
			sign = -1;
		}
		int blockSize = 1;
		JVCLUtils.display(srcArray, "premethod");
		for (int n = 0; n < logN; n++) {
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
			queue.put1DRangeKernel(Kernel, 0, threads, 16);
			queue.putReadBuffer(CLBufferDest, true);
			queue.finish();
			src.get(srcArray);
			dest.get(destArray);
			JVCLUtils.display(srcArray, "postmethod");
			float[] temp = JVCLUtils.deepCopy(destArray);
			destArray = JVCLUtils.deepCopy(srcArray);
			srcArray = JVCLUtils.deepCopy(temp);
			JVCLUtils.display(srcArray, "postswap");
			
			blockSize *= 2;
			CLBufferSrc.release();
			CLBufferDest.release();
		}
		
		//testing release -
		if (!isForward) {
			for (int n = 0; n < N; n++) {
				destArray[n] = destArray[n] / (float)N;
			}
		}

		return srcArray;
	}
	
	public void close() {
		Kernel.release();
	}

	public static void main(String[] args) {
		/*
		CLContext context = CLContext.create();
		CLDevice device = context.getMaxFlopsDevice();
		CLCommandQueue queue = device.createCommandQueue();
		String source = JVCLUtils.readFile("src/stockhamRadix2.cl");
		CLProgram program = context.createProgram(source).build();
		CLKernel Kernel = program.createCLKernel("stockhamStride");
		StockhamRadix2 s = new StockhamRadix2(context, device, program, Kernel, queue);
		*/
		StockhamRadix2 s = new StockhamRadix2();

		int N = 32;
		float[] testR = new float[N];
		float[] testI = new float[N];
		for (int n = 0; n < N; n++) {
				testR[n] = 10*(float)FastMath.cos(2*FastMath.PI*(n+1)/(double)(N/2.0));
				testI[n] = 10*(float)FastMath.sin(2*FastMath.PI*(n+1)/(double)(N/2.0));
		}
		System.out.println(Arrays.toString(ArrayMath.round(testR)));		
		float[] dst = new float[N*2];
		float[] resultf = s.fft(JVCLUtils.split2interleaved(testR, testI), dst, true, N);
		Complex[] result = ComplexUtils.interleaved2Complex(resultf);
		JVCLUtils.display(result, "final");
		//System.out.println(Arrays.toString(ArrayMath.round(dst)));		
		s.close();
	}
	
	
}
