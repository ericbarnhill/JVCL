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
	
	void fft(float[] real, float[] imag, boolean isForward, int N, int iter) {
	    int len = real.length;
	    int blocks = len / N;
	    int pwr2 = (int)(Math.log(N)/Math.log(2));
		int sign;
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
			//.setArg(4, pwr2)
			.setArg(4,  pwr2)
			.setArg(5, len);
		queue.putWriteBuffer(CLBufferReal, true);
		queue.putWriteBuffer(CLBufferImag, true);
		queue.put2DRangeKernel(Kernel, 0, 0, N, blocks, N, 1);
		queue.putReadBuffer(CLBufferReal, true);
		queue.putReadBuffer(CLBufferImag, true);
		realBuffer.get(real);
		imagBuffer.get(imag);
		CLBufferReal.release();
		CLBufferImag.release();
		//testing release -
		if (!isForward) {
			for (int n = 0; n < len; n++) {
				real[n] = real[n] / (float)N;
				imag[n] = - imag[n] / (float)N;
			}
		}
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
		
		//System.out.println(Arrays.toString(ArrayMath.divide(ArrayMath.round(ArrayMath.multiply(testR, 100)), 100.0f)));
		//for (int i = 0; i < 6; i++) {
			int N = 256;
			float[] testR = new float[N];
			float[] testI = new float[N];
			for (int n = 0; n < N/4; n++) {
				testR[n] = 10*(float)FastMath.cos(2*FastMath.PI*(n+1)/(double)(N/4.0));
				testI[n] = 10*(float)FastMath.sin(2*FastMath.PI*(n+1)/(double)(N/4.0));
			}
			for (int n = N/4; n < N/2; n++) {
				testR[n] = 10*(float)FastMath.cos(2*FastMath.PI*((n-N/4)+1)/(double)(N/12.0));
				testI[n] = 10*(float)FastMath.sin(2*FastMath.PI*((n-N/4)+1)/(double)(N/12.0));
			}
			for (int n = N/2; n < 3*N/4; n++) {
				testR[n] = 10*(float)FastMath.cos(2*FastMath.PI*((n-N/2)+1)/(double)(N/32.0));
				testI[n] = 10*(float)FastMath.sin(2*FastMath.PI*((n-N/2)+1)/(double)(N/32.0));
			}
			for (int n = 3*N/4; n < N; n++) {
				testR[n] = 10*(float)FastMath.cos(2*FastMath.PI*((n-3*N/4)+1)/(double)(N/16.0));
				testI[n] = 10*(float)FastMath.sin(2*FastMath.PI*((n-3*N/4)+1)/(double)(N/16.0));
			}
			JVCLUtils.display(JVCLUtils.split2interleaved(testR, testI), "", 8);
			s.fft(testR, testI, true, N/4, 0);
			JVCLUtils.display(JVCLUtils.split2interleaved(testR, testI), "", 8);
			s.fft(testR, testI, false, N/4, 0);
			JVCLUtils.display(JVCLUtils.split2interleaved(testR, testI), "", 8);
		//}
		s.close();
	}
	
	
	
}
