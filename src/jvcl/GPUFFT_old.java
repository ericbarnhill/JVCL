package jvcl;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.io.*;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.demos.fft.*;
import com.jogamp.opencl.demos.fft.CLFFTPlan.InvalidContextException;
import com.jogamp.common.nio.PointerBuffer;

import edu.emory.mathcs.jtransforms.fft.DoubleFFT_2D;
import static java.lang.System.*;
import static com.jogamp.opencl.CLMemory.Mem.*;
import static java.lang.Math.*;


public class GPUFFT_old {
	
	float[] real;
	float[] imag;
	int N;
	CLFFTPlan fft;
    DimShifter ds;
    CLDevice device;
    CLCommandQueue queue;
    CLProgram program;
    CLContext context;
	List<CLBuffer<FloatBuffer>> buffersEvenR;
	List<CLBuffer<FloatBuffer>> buffersEvenI;
	List<CLBuffer<FloatBuffer>> buffersOddR;
	List<CLBuffer<FloatBuffer>> buffersOddI;
	
	public GPUFFT_old() {
		buffersEvenR = new ArrayList<CLBuffer<FloatBuffer>>();
		buffersEvenI = new ArrayList<CLBuffer<FloatBuffer>>();
		buffersOddR = new ArrayList<CLBuffer<FloatBuffer>>();
		buffersOddI = new ArrayList<CLBuffer<FloatBuffer>>();		
	}
	 	 
	 
	// Cooley–Tukey FFT (in-place, divide-and-conquer)
	// Higher memory requirements and redundancy although more intuitive
	void fft_simple(float[] real, float[] imag)
	{
		context = CLContext.create();
        double[] result;
        try{
            device = context.getMaxFlopsDevice();
            queue = device.createCommandQueue();
            Path currentRelativePath = Paths.get("");
            String s = currentRelativePath.toAbsolutePath().toString();
            String source = readFile("src/fft1d.cl");
            program = context.createProgram(source).build(); 
		    N = real.length;
		    int half = N>>1;
		    if (N <= 1) return;
		    
		    CLBuffer<FloatBuffer> bufferReal = context.createFloatBuffer(N);
		    CLBuffer<FloatBuffer> bufferImag = context.createFloatBuffer(N);
		    bufferReal.getBuffer().put(real).rewind();
			bufferImag.getBuffer().put(imag).rewind();	
		    divideAndBuffer(real, imag, 0);
		    int iterations = buffersEvenR.size();
		    CLKernel Kernel = program.createCLKernel("fft1d");
		    bufferReal.getBuffer().rewind();
	    	bufferImag.getBuffer().rewind();
	    	 Kernel.setArg(0, bufferReal)
				.setArg(1, bufferImag);
	    	 
		    //for (int n = 0; n < N/2; n++) {
			//	System.out.format("%.2f %.2f ", bufferReal.getBuffer().get(n), bufferImag.getBuffer().get(n));
			//}
		    float[] temp = new float[64];
		    int[] n = new int[] {0, 2, 3, 4, 5, 1};
		    for (int m = 1; m >= 1; m--) {
		    	int stride = (int)( (N) / Math.pow(2,m));
			    System.out.println("iteration "+m+" stride "+stride);
		    	CLBuffer<FloatBuffer> bufferEvenR = buffersEvenR.get(m);
		    	CLBuffer<FloatBuffer> bufferEvenI = buffersEvenI.get(m);
		    	CLBuffer<FloatBuffer> bufferOddR = buffersOddR.get(m);
		    	CLBuffer<FloatBuffer> bufferOddI = buffersOddI.get(m);
		    	
		    	bufferEvenR.getBuffer().rewind();
		    	bufferEvenI.getBuffer().rewind();
		    	bufferOddR.getBuffer().rewind();
		    	bufferOddI.getBuffer().rewind();
		    	Kernel.setArg(2, bufferEvenR)
		    		.setArg(3, bufferEvenI)
		    		.setArg(4, bufferOddR)
		    		.setArg(5, bufferOddI)
		    		.setArg(6, N)
		    		.setArg(7, stride);
		    	queue.putWriteBuffer(bufferReal, true);
				queue.putWriteBuffer(bufferImag, true);
		    	queue.putWriteBuffer(bufferEvenR, true);
		    	queue.putWriteBuffer(bufferEvenI, true);
		    	queue.putWriteBuffer(bufferOddR, true);
		    	queue.putWriteBuffer(bufferOddI, true);		    	
		    	queue.put1DRangeKernel(Kernel, 0, N, 0);
		    	queue.putReadBuffer(bufferReal, true);
			    queue.putReadBuffer(bufferImag, true);
		    	queue.putBarrier();
			    for (int z = 0; z < 64; z++) {
			    	temp[z] += bufferReal.getBuffer().get();
					System.out.format("%.2f ", temp[z]);
					if ((z+1) % 16 == 0) System.out.println();
			    }
			    bufferReal.getBuffer().rewind();
			    System.out.println();	    
			    
		    }
		    
		    bufferReal.getBuffer().get(real);
		    bufferImag.getBuffer().get(imag);
		    System.out.println("----------------");
		    for (int z = 0; z < 64; z++) {
				System.out.format("%.2f ", real[z]);
				if ((z+1) % 16 == 0) System.out.println();
		    }
		    System.out.println();
		    queue.finish();
	    } finally {
            // cleanup all resources associated with this context.
            context.release();
        }
        return;
    }
	 
	
	void divideAndBuffer(float[] real, float[] imag, int iteration) {
		int length = real.length;
		if (length == 1) return;
		int half = length>>1;
		float[] evenR = new float[half];
		float[] evenI = new float[half];
		float[] oddR = new float[half];
		float[] oddI = new float[half];
		for (int n = 0; n < half; n++) {
			evenR[n] = real[n*2];
			evenI[n] = imag[n*2];
			oddR[n] = real[n*2+1];
			oddI[n] = imag[n*2+1];
		}
		CLBuffer<FloatBuffer> bufferEvenR, bufferEvenI, bufferOddR, bufferOddI;
		if (buffersEvenR.size() > iteration) {
			bufferEvenR = buffersEvenR.remove(iteration);
			bufferEvenI = buffersEvenI.remove(iteration);
			bufferOddR = buffersOddR.remove(iteration);
			bufferOddI = buffersOddI.remove(iteration);
		} else {
			bufferEvenR = context.createFloatBuffer(N/2);
			bufferEvenI = context.createFloatBuffer(N/2);
			bufferOddR = context.createFloatBuffer(N/2);
			bufferOddI = context.createFloatBuffer(N/2);
		}
		int pos = bufferEvenR.getBuffer().position();
		//if (iteration == 5) System.out.format("capacity %d position %d arraylength %d %n", bufferEvenR.getBuffer().capacity(), pos, evenR.length);
		for (int n = 0; n < evenR.length; n++) {
			bufferEvenR.getBuffer().put(evenR[n]);
			bufferEvenI.getBuffer().put(evenI[n]);
			bufferOddR.getBuffer().put(oddR[n]);
			bufferOddI.getBuffer().put(oddI[n]);
		}
		
		buffersEvenR.add(iteration, bufferEvenR);
		buffersEvenI.add(iteration, bufferEvenI);
		buffersOddR.add(iteration, bufferOddR);
		buffersOddI.add(iteration, bufferOddI);
		/*
		if (iteration == 4) {
			for (int n = 0; n < half; n++) {
					System.out.format("%.2f %.2f ", evenR[n], oddR[n]);
			}
			System.out.println();
			for (int n = 0; n < length; n++) {
				System.out.format("%.2f ", real[n]);	
			}
			System.out.println();
			for (int n = 0; n < N/2; n++) {
				System.out.format("%.2f %.2f ", bufferEvenR.getBuffer().get(n), bufferOddR.getBuffer().get(n));
			}
			System.out.println();
			System.out.println("-----------------------------");
		}
		*/
		divideAndBuffer(evenR, evenI, iteration+1);
		divideAndBuffer(oddR, oddI, iteration+1);
		//System.out.format("Iteration %d %n", iteration);
		return;
	}
	
	
	public static void main(String[] args) {

		float[] real = new float[64];
		float[] imag = new float[64];
		
		for (int n = 0; n < 64; n++) {
			real[n] = 10*(float)Math.cos(2*Math.PI*(n+1)/64);
			imag[n] = 10*(float)Math.sin(2*Math.PI*(n+1)/64);
			//System.out.format("%.2f + i%.2f ", real[n], imag[n]);
			//if ((n+1) % 8 == 0) System.out.format("%n");
		}
	 
	    // forward fft
	    //new GPUFFT().fft_simple(real, imag);
	 
	    for (int n = 0; n < 64; n++) {
			System.out.format("%.2f ",real[n]);
			if ((n+1)%16 == 0) System.out.println();
		}
	    /*
	    System.out.println("----------");
	    for (int n = 0; n < 16; n++) {
			System.out.format("%.2f ",imag[n]);
			if ((n+1)%16 == 0) System.out.println();
		}
	    System.out.println();
	    */
	}
	

    private static int roundUp(int groupSize, int globalSize) {
        int r = globalSize % groupSize;
        if (r == 0) {
            return globalSize;
        } else {
            return globalSize + groupSize - r;
        }
    }
    
    private static String readFile(String fileName)
    {
        try
        {
            BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(fileName)));
            StringBuffer sb = new StringBuffer();
            String line = null;
            while (true)
            {
                line = br.readLine();
                if (line == null)
                {
                    break;
                }
                sb.append(line).append("\n");
            }
            br.close();
            return sb.toString();
        }
        catch (IOException e)
        {
            e.printStackTrace();
            System.exit(1);
            return null;
        }
    }

	
	
	

}
