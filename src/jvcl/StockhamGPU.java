package jvcl;

import java.nio.FloatBuffer;
import java.util.Arrays;

import org.apache.commons.math4.util.FastMath;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.demos.fft.CLFFTPlan;

import arrayMath.ArrayMath;

public class StockhamGPU {
	
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
	int N;
    

    
	public StockhamGPU() {	
	}
	
	public StockhamGPU(CLContext context, CLDevice device, CLProgram program, CLKernel Kernel, CLCommandQueue queue) {
		this.context = context;
		this.device = device;
		this.program = program;
		this.Kernel = Kernel;
		this.queue = queue;
		Kernel = program.createCLKernel("stockham");
	}
	
	void fft(float[] real, float[] imag, boolean isForward) {
	    int blockSize = real.length;
	    N = blockSize;
		int sign;
		if (isForward) {
			sign = 1;
		} else {
			sign = -1;
		}
		CLBufferReal = context.createFloatBuffer(N);
		CLBufferImag = context.createFloatBuffer(N);
		realBuffer = CLBufferReal.getBuffer();
		imagBuffer = CLBufferImag.getBuffer();
		realBuffer.clear();
		realBuffer.put(real).rewind();
		imagBuffer.clear();
		imagBuffer.put(imag).rewind();
		Kernel.setArg(0, CLBufferReal)
			.setArg(1, CLBufferImag)	
			.setArg(2, sign)
			.setArg(3, N)
			.setArg(4, (int)(Math.log(N)/Math.log(2)))
			.setArg(5, blockSize);
		queue.putWriteBuffer(CLBufferReal, true);
		queue.putWriteBuffer(CLBufferImag, true);
		queue.put1DRangeKernel(Kernel, 0, N*2, N);
		queue.putReadBuffer(CLBufferReal, true);
		queue.putReadBuffer(CLBufferImag, true);
		realBuffer.get(real);
		imagBuffer.get(imag);
		CLBufferReal.release();
		CLBufferImag.release();
		//testing release -
		if (!isForward) {
			for (int n = 0; n < N; n++) {
				real[n] = real[n] / (float)N;
				imag[n] = imag[n] / (float)N;
			}
		}
	}
	

	private static int roundUp(int groupSize, int globalSize) {
	    int r = globalSize % groupSize;
	    if (r == 0) {
	        return globalSize;
	    } else {
	        return globalSize + groupSize - r;
	    }
	}

	public void close() {
		Kernel.release();
	}

	public static void main(String[] args) {
		CLContext context = CLContext.create();
		CLDevice device = context.getMaxFlopsDevice();
		CLCommandQueue queue = device.createCommandQueue();
		String source = JVCLUtils.readFile("src/stockham.cl");
		CLProgram program = context.createProgram(source).build();
		CLKernel Kernel = program.createCLKernel("stockham");
		StockhamGPU s = new StockhamGPU(context, device, program, Kernel, queue);
		int N = 256;
		float[] testR = new float[N];
		float[] testI = new float[N];
		for (int n = 0; n < N; n++) {
				testR[n] = 10*(float)FastMath.cos(2*FastMath.PI*(n+1)/(double)(N/2.0));
				testI[n] = 10*(float)FastMath.sin(2*FastMath.PI*(n+1)/(double)(N/2.0));
		}
		System.out.println(Arrays.toString(ArrayMath.round(testR)));
		s.fft(testR, testI, true);
		System.out.println(Arrays.toString(ArrayMath.round(testR)));
		System.out.println(Arrays.toString(ArrayMath.round(testI)));
		s.close();
	}
	
	
	
}
