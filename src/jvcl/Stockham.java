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

public class Stockham {
	
	CLFFTPlan fft;
    DimShifter ds;
    CLDevice device;
    CLCommandQueue queue;
    CLProgram program;
    CLContext context;
    long totalTime;
    CLKernel Kernel;
    int local, global;
    
    
	public Stockham() {
		try {
			context = CLContext.create();
			device = context.getMaxFlopsDevice();
	        queue = device.createCommandQueue();
	        
		} catch (Exception e) {}		
		
	}
	 
	// Cooley–Tukey FFT (in-place, divide-and-conquer)
	// Higher memory requirements and redundancy although more intuitive
	void fft_simple(float[] real, float[] imag) {
	    int N = real.length;
    try{
        String source = readFile("src/stockham.cl");
	    program = context.createProgram(source).build();
	    Kernel = program.createCLKernel("stockham");
    	// combine
    	 CLBuffer<FloatBuffer> bufferReal;
    	 CLBuffer<FloatBuffer> bufferImag;   
    	bufferReal = context.createFloatBuffer(N);
    	bufferImag = context.createFloatBuffer(N);
    	bufferReal.getBuffer().clear();
    	bufferReal.getBuffer().put(real);
    	bufferReal.getBuffer().rewind();
    	bufferImag.getBuffer().clear();
    	bufferImag.getBuffer().put(imag);
    	bufferImag.getBuffer().rewind();
    	Kernel.setArg(0, bufferReal)
			.setArg(1, bufferImag)	
			.setArg(2, -1)
    		.setArg(3, N)
    		.setArg(4, (int)(Math.log(N)/Math.log(2)));
    	queue.putWriteBuffer(bufferReal, false);
		queue.putWriteBuffer(bufferImag, false);
    	queue.put1DRangeKernel(Kernel, 0, roundUp(N,64), 64);
    	queue.putReadBuffer(bufferReal, false);
	    queue.putReadBuffer(bufferImag, false);
    	bufferReal.getBuffer().get(real);
    	bufferImag.getBuffer().get(imag);
	    //System.out.println(" Time: "+ (time5-time1));
	    } finally {
	    }
	}
	
	public static void main(String[] args) {

		float[] real = new float[128];
		float[] imag = new float[128];
		
		for (int n = 0; n < 128; n++) {
			real[n] = (float)Math.cos(2*Math.PI*(n+1)/128)+2;
			imag[n] = (float)Math.sin(2*Math.PI*(n+1)/128)+2;
			//System.out.format("%.2f + i%.2f ", real[n], imag[n]);
			//if ((n+1) % 8 == 0) System.out.format("%n");
		}
	 
	    // forward fft
		Stockham s = new Stockham();
    	s.fft_simple(real, imag);
			    	
	    s.context.release();
	    
	    System.out.println("REAL");
	    for (int n = 0; n < 128; n++) {
			System.out.format("%.2f ",real[n]);
			if ((n+1) % 16 == 0) System.out.println();
		}
	    System.out.println("IMAG");
	    for (int n = 0; n < 128; n++) {
			System.out.format("%.2f ",imag[n]);
			if ((n+1) % 16 == 0) System.out.println();
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

	private static int roundUp(int groupSize, int globalSize) {
	    int r = globalSize % groupSize;
	    if (r == 0) {
	        return globalSize;
	    } else {
	        return globalSize + groupSize - r;
	    }
	}

}