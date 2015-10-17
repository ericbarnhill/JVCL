package jvcl;

import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.demos.fft.CLFFTPlan;

public class StockhamGPUStrideNoDS {
	
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
    
    
	public StockhamGPUStrideNoDS() {	
		context = CLContext.create();
		device = context.getMaxFlopsDevice();
		queue = device.createCommandQueue();
	    String sourceStride = JVCLUtils.readFile("src/stockhamStride.cl");
		program = context.createProgram(sourceStride).build();
		Kernel = program.createCLKernel("stockhamStride");
	}
	
	
	public StockhamGPUStrideNoDS(CLContext context, CLDevice device, CLCommandQueue queue, CLProgram program) {
		this.context = context;
		this.device = device;
		this.queue = queue;
		this.program = program;
		Kernel = program.createCLKernel("stockhamStride");
	}
	
	void fft(float[] real, float[] imag, boolean isForward, int xDim, int yDim, boolean xStyle) {
		boolean debug = false;
		//if (blockSize >= 2048) debug = true;
		if (debug) System.out.println("----------------------------------");
	    if (xStyle) {
	    	N = xDim;
	    } else {
	    	N = yDim;
	    }
		int sign;
		if (isForward) {
			sign = 1;
		} else {
			sign = -1;
		}
		int currRunCapacity;
		long start = 0;long fillBuffers = 0; long copyData = 0; long runKernel = 0; long readOut = 0;
		if (debug) start = System.currentTimeMillis();
    	currRunCapacity = xDim*yDim;
    	CLBufferReal = context.createFloatBuffer(currRunCapacity);
    	CLBufferImag = context.createFloatBuffer(currRunCapacity);
		realBuffer = CLBufferReal.getBuffer();
		imagBuffer = CLBufferImag.getBuffer();
		if (debug) {
			fillBuffers = System.currentTimeMillis();
			System.out.format("Create buffers %.5f %n", (fillBuffers - start)/1000.0);
		}
		int blockSize;
		if (xStyle) {
			for (int x = 0; x < xDim; x++) {
				for (int y = 0; y < yDim; y++) {
					realBuffer.put(real[x + xDim*y]);
					imagBuffer.put(imag[x + xDim*y]);
				}
			}
			blockSize = xDim;
		} else {
			for (int x = 0; x < xDim; x++) {
				for (int y = 0; y < yDim; y++) {
					realBuffer.put(real[x*yDim + y]);
					imagBuffer.put(imag[x*yDim + y]);
				}
			}
			blockSize = yDim;
		}
		realBuffer.rewind();
		imagBuffer.rewind();
		if (debug) {
			copyData = System.currentTimeMillis();
			System.out.format("Copy data %.5f %n", (copyData - fillBuffers)/1000.0);
		}
		//if (true) copyData = System.currentTimeMillis();
		Kernel.setArg(0, CLBufferReal)
			.setArg(1, CLBufferImag)	
			.setArg(2, sign)
			.setArg(3, blockSize)
			.setArg(4, (int)(Math.log(blockSize)/Math.log(2)))
			.setArg(5, blockSize);
		queue.putWriteBuffer(CLBufferReal, true);
		queue.putWriteBuffer(CLBufferImag, true);
		//queue.put1DRangeKernel(Kernel, 0, blocks*blockSize, blockSize);
		queue.put2DRangeKernel(Kernel, 0, 0, blockSize, blockSize, 16, 16);
		queue.putReadBuffer(CLBufferReal, true);
		queue.putReadBuffer(CLBufferImag, true);
		//queue.finish();
		//if (debug) {
		if (true) {
			//runKernel = System.currentTimeMillis();
			//System.out.format("Run kernel %.5f %n", (runKernel - copyData)/1000.0);
		}
		realBuffer.get(real);
		imagBuffer.get(imag);
		//System.arraycopy(tempReal, 0, real, 0, currRunCapacity);
		//System.arraycopy(tempImag, 0, imag, 0, currRunCapacity);
		CLBufferReal.release();
		CLBufferImag.release();
		if (!isForward) {
			for (int n = 0; n < N; n++) {
				real[n] = real[n] / (float)N;
				imag[n] = imag[n] / (float)N;
			} // for n
		} // if is forward
		if (debug) {
			readOut = System.currentTimeMillis();
			System.out.format("Read out %.5f %n", (readOut - runKernel)/1000.0);
		}
		if (debug) System.out.println("------------------------------------");
	}
	
	public void close() {
		Kernel.release();
	}

	public static void main(String[] args) {
		StockhamGPUStrideNoDS s = new StockhamGPUStrideNoDS();
		int[] dims = new int[] {64,128,256,512, 1024, 2048};
		for (int d = 0; d < dims.length; d++) {
			int dim = dims[d]*dims[d];
			float[] array = new float[dim];
			System.out.println("FFT dim size " + dims[d]);
			s.fft(array, array, true, dims[d], dims[d], true);
			System.out.println("done dim size "+dims[d]);
		}
	
	}
	
}
