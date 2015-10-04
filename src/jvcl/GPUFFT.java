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

public class GPUFFT {
	
	CLFFTPlan fft;
    DimShifter ds;
    CLDevice device;
    CLCommandQueue queue;
    CLProgram program;
    CLContext context;
    CLBuffer<FloatBuffer> bufferReal;
    CLBuffer<FloatBuffer> bufferImag;
    CLBuffer<FloatBuffer> bufferEvenR;
    CLBuffer<FloatBuffer> bufferEvenI;
    CLBuffer<FloatBuffer> bufferOddR;
    CLBuffer<FloatBuffer> bufferOddI;
    long totalTime;
    CLKernel Kernel;
    int local, global;
    
    
	public GPUFFT(int size, int global, int local) {
		try {
			context = CLContext.create();
			device = context.getMaxFlopsDevice();
	        queue = device.createCommandQueue();
	        Path currentRelativePath = Paths.get("");
	        String s = currentRelativePath.toAbsolutePath().toString();
	        String source = readFile("src/fft1d.cl");
		    program = context.createProgram(source).build();
		    Kernel = program.createCLKernel("fft1d");
		    bufferReal = context.createFloatBuffer(size, WRITE_ONLY);
	    	bufferImag = context.createFloatBuffer(size, WRITE_ONLY);
	    	bufferEvenR = context.createFloatBuffer(size, READ_ONLY);
	    	bufferEvenI = context.createFloatBuffer(size, READ_ONLY);
	    	bufferOddR = context.createFloatBuffer(size, READ_ONLY);
	    	bufferOddI = context.createFloatBuffer(size, READ_ONLY);
			this.global = global;
			this.local = local;
		} catch (Exception e) {}		
		
	}
	 
	// Cooley–Tukey FFT (in-place, divide-and-conquer)
	// Higher memory requirements and redundancy although more intuitive
	void fft_simple(float[] real, float[] imag) {
	    int N = real.length;
	    if (N <= 1) return;	
    try{
	    // divide
	    float[] evenR = new float[N/2];
	    float[] evenI = new float[N/2];
	    float[] oddR = new float[N/2];
	    float[] oddI = new float[N/2];
	    for (int n = 0; n < N/2; n++) {
	    	evenR[n] = real[n*2];
	    	evenI[n] = imag[n*2];
	    	oddR[n] = real[n*2+1];
	    	oddI[n] = imag[n*2+1];
	    }
	 
	    // conquer
    	fft_simple(evenR, evenI);
    	fft_simple(oddR, oddI);
    	
    	// combine
    	bufferReal.getBuffer().clear();
    	bufferReal.getBuffer().put(real);
    	bufferReal.getBuffer().rewind();
    	bufferImag.getBuffer().clear();
    	bufferImag.getBuffer().put(imag);
    	bufferImag.getBuffer().rewind();
    	bufferEvenR.getBuffer().clear();
    	bufferEvenR.getBuffer().put(evenR);
    	bufferEvenR.getBuffer().rewind();
    	bufferEvenI.getBuffer().clear();
    	bufferEvenI.getBuffer().put(evenI);
    	bufferEvenI.getBuffer().rewind();
    	bufferOddR.getBuffer().clear();
    	bufferOddR.getBuffer().put(oddR);
    	bufferOddR.getBuffer().rewind();
    	bufferOddI.getBuffer().clear();
    	bufferOddI.getBuffer().put(oddI);
    	bufferOddI.getBuffer().rewind();
    	Kernel.setArg(0, bufferReal)
			.setArg(1, bufferImag)	
	    	.setArg(2, bufferEvenR)
			.setArg(3, bufferEvenI)
			.setArg(4, bufferOddR)
			.setArg(5, bufferOddI)
			.setArg(6, N);
    	queue.putWriteBuffer(bufferReal, false);
		queue.putWriteBuffer(bufferImag, false);
    	queue.putWriteBuffer(bufferEvenR, false);
    	queue.putWriteBuffer(bufferEvenI, false);
    	queue.putWriteBuffer(bufferOddR, false);
    	queue.putWriteBuffer(bufferOddI, false);	
    	queue.put1DRangeKernel(Kernel, 0, global, local);
    	queue.putReadBuffer(bufferReal, false);
	    queue.putReadBuffer(bufferImag, false);
    	bufferReal.getBuffer().get(real);
    	bufferImag.getBuffer().get(imag);
	    //System.out.println(" Time: "+ (time5-time1));
	    } finally {
	    }
	}
	
	public static void main(String[] args) {

		float[] real = new float[4096];
		float[] imag = new float[4096];
		
		for (int n = 0; n < 4096; n++) {
			real[n] = (float)Math.cos(2*Math.PI*(n+1)/1024);
			imag[n] = (float)Math.sin(2*Math.PI*(n+1)/1024);
			//System.out.format("%.2f + i%.2f ", real[n], imag[n]);
			//if ((n+1) % 8 == 0) System.out.format("%n");
		}
	 
	    // forward fft
		GPUFFT g = null;
	    int[] locals = {256, 512, 1024, real.length};
	    int[] globals = {256, 1024, real.length, real.length*6};
	    for (int x = 0; x < 4; x++) {
	    	System.out.println("Global: "+locals[x]);
	    	for (int y = 0; y < 4; y++) {
		    	System.out.println("Local: "+locals[x]);
			    g = new GPUFFT(real.length, locals[x], locals[x]);
			    long time1 = System.currentTimeMillis();
			    for (int n = 0; n < 5; n++) {
			    	g.fft_simple(real, imag);
			    	 long time2 = System.currentTimeMillis();
			 	    System.out.println("Time "+(time2-time1)/1000.0);
			 	    time1 = time2;
			    }
	    	}
	    }
	    g.context.release();
	    
	    System.out.println("REAL");
	    for (int n = 0; n < 64; n++) {
			System.out.format("%.2f ",real[n]);
			if ((n+1) % 16 == 0) System.out.println();
		}
	    System.out.println("IMAG");
	    for (int n = 0; n < 16; n++) {
			System.out.format("%.2f ",imag[n]);
		}
	    System.out.println();
	    
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
