package jvcl;

//TODO: 2d against 1d methods etc.

import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.WRITE_ONLY;
import static java.lang.Math.min;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

public class FDGPU {
	
    CLDevice device;
    CLCommandQueue queue;
    CLProgram program1d, program2d, program3d;
    CLContext context;
    int boundaryConditions;
	final int ZERO_BOUNDARY = 0;
	final int MIRROR_BOUNDARY = 1;
	final int PERIODIC_BOUNDARY = 2;
	
	public FDGPU(int boundaryConditions) {
		context = CLContext.create();
		this.boundaryConditions = boundaryConditions;
	}
	
	public FDGPU() {
		context = CLContext.create();
		device = context.getMaxFlopsDevice();
        queue = device.createCommandQueue();
        String source1d = JVCLUtils.readFile("src/Convolve1d.cl");
        String source2d = JVCLUtils.readFile("src/Convolve2d.cl");
        String source3d = JVCLUtils.readFile("src/Convolve3d.cl");
        program1d = context.createProgram(source1d).build(); 
        program2d = context.createProgram(source2d).build(); 
        program3d = context.createProgram(source3d).build(); 
		this.boundaryConditions = ZERO_BOUNDARY;
	}

	public double[] convolve(double[] vector, double[] kernel) {
        double[] result;
        int vectorLength = vector.length;
        float[] vectorFloat = new float[vectorLength];
        for (int n = 0; n < vectorLength; n++) vectorFloat[n] = (float)vector[n];
        int kernelLength = kernel.length;
        float[] kernelFloat = new float[kernelLength];
        for (int n = 0; n < kernelLength; n++) kernelFloat[n] = (float)kernel[n];
        int halfLength = kernelLength / 2;
    	CLBuffer<FloatBuffer> clVector = context.createFloatBuffer(vectorLength, READ_ONLY);
        CLBuffer<FloatBuffer> clKernel = context.createFloatBuffer(kernelLength, READ_ONLY);
        CLBuffer<FloatBuffer> clOutput = context.createFloatBuffer(vectorLength, WRITE_ONLY);
        clVector.getBuffer().put(vectorFloat).rewind();
        clKernel.getBuffer().put(kernelFloat).rewind();
        CLKernel Kernel = program1d.createCLKernel("Convolve1d");
        Kernel.putArg(clVector)
        	.putArg(clKernel)
        	.putArg(clOutput)
        	.putArg(vectorLength)
        	.putArg(kernelLength)
        	.putArg(halfLength)
        	.putArg(boundaryConditions);
        queue.putWriteBuffer(clVector, false)
        	.putWriteBuffer(clKernel, false)
        	.put1DRangeKernel(Kernel, 0, JVCLUtils.roundUp(vectorLength, 32),0)
        	.putReadBuffer(clOutput, true);
		result = new double[vectorLength];
		for (int x = 0; x < vectorLength; x++) {
				result[x] = clOutput.getBuffer().get(x);
		}
     
        return result;    		
	}
	
	public double[][] convolve(double[][] image, double[][] kernel) {
        double[][] result;
        try{
         
            int imageWidth = image.length;
            int imageHeight = image[0].length;
            int imageArea = imageWidth*imageHeight;
            float[] image1d = new float[imageArea];
            int index = 0;
            for (int x = 0; x < imageWidth; x++) {
            	for (int y = 0; y < imageHeight; y++) {
            		index = x + imageWidth*y;
            		image1d[index] = (float)image[x][y];
            	}
            }
            int kernelWidth = kernel.length;
            int kernelHeight = kernel[0].length;
            int kernelArea = kernelWidth*kernelHeight;
            int halfWidth = kernelWidth / 2;
            int halfHeight = kernelHeight / 2;
            float[] kernel1d = new float[kernelArea];
            for (int x = 0; x < kernelWidth; x++) {
            	for (int y = 0; y < kernelHeight; y++) {
            		index = x + kernelWidth*y;
            		kernel1d[index] = (float)kernel[x][y];
            	}
            }
        	CLBuffer<FloatBuffer> clImage = context.createFloatBuffer(imageArea, READ_ONLY);
            CLBuffer<FloatBuffer> clKernel = context.createFloatBuffer(kernelArea, READ_ONLY);
            CLBuffer<FloatBuffer> clOutput = context.createFloatBuffer(imageArea, WRITE_ONLY);
            clImage.getBuffer().put(image1d).rewind();
            clKernel.getBuffer().put(kernel1d).rewind();
            CLKernel Kernel = program2d.createCLKernel("Convolve2d");
            Kernel.putArg(clImage)
            	.putArg(clKernel)
            	.putArg(clOutput)
            	.putArg(imageWidth)
            	.putArg(imageHeight)
            	.putArg(kernelWidth)
            	.putArg(kernelHeight)
            	.putArg(halfWidth)
            	.putArg(halfHeight)
            	.putArg(boundaryConditions);
            queue.putWriteBuffer(clImage, false)
            	.putWriteBuffer(clKernel, false)
            	.put2DRangeKernel(Kernel, 0, 0, JVCLUtils.roundUp(imageWidth, 32),JVCLUtils.roundUp(imageHeight,32), 0,0)
            	.putReadBuffer(clOutput, true);
    		result = new double[imageWidth][imageHeight];
    		for (int x = 0; x < imageWidth; x++) {
    			for (int y = 0; y < imageHeight; y++) {
    				result[x][y] = clOutput.getBuffer().get(x + y*imageWidth);
    			}
    		}
    		 clImage.release();
             clKernel.release();
             clOutput.release();
        } finally {
           
        }
        return result;
	}

	public double[][][] convolve(double[][][] image, double[][][] kernel) {
        //out.println("created "+context);
        double[][][] result;
        try{
            
            int imageWidth = image.length;
            int imageHeight = image[0].length;
            int imageDepth = image[0][0].length;
            int imageArea = imageWidth*imageHeight;
            int imageVolume = imageArea*imageDepth;
            float[] image1d = new float[imageVolume];
            int index = 0;
            for (int x = 0; x < imageWidth; x++) {
            	for (int y = 0; y < imageHeight; y++) {
            		for (int z = 0; z < imageDepth; z++) {
	            		index = x + imageWidth*y + imageArea*z;
	            		image1d[index] = (float)image[x][y][z];
            		}
            	}
            }
            int kernelWidth = kernel.length;
            int kernelHeight = kernel[0].length;
            int kernelDepth = kernel[0][0].length;
            int kernelArea = kernelWidth*kernelHeight;
            int kernelVolume = kernelArea*kernelDepth;
            int halfWidth = kernelWidth / 2;
            int halfHeight = kernelHeight / 2;
            int halfDepth = kernelDepth / 2;
            float[] kernel1d = new float[kernelVolume];
            for (int x = 0; x < kernelWidth; x++) {
            	for (int y = 0; y < kernelHeight; y++) {
            		for (int z = 0; z < kernelDepth; z++) {
	            		index = x + kernelWidth*y + kernelArea*z;
	            		kernel1d[index] = (float)kernel[x][y][z];
            		}
            	}
            }
        	int localWorkSize = min(device.getMaxWorkGroupSize(), 32);  // Local work size dimensions
        	int globalX = JVCLUtils.roundUp(imageWidth, localWorkSize);
        	int globalY = JVCLUtils.roundUp(imageHeight, localWorkSize);
        	int globalZ = JVCLUtils.roundUp(imageDepth, localWorkSize);
        	CLBuffer<FloatBuffer> clImage = context.createFloatBuffer(imageVolume, READ_ONLY);
            CLBuffer<FloatBuffer> clKernel = context.createFloatBuffer(kernelVolume, READ_ONLY);
            CLBuffer<FloatBuffer> clOutput = context.createFloatBuffer(imageVolume, WRITE_ONLY);
            clImage.getBuffer().put(image1d).rewind();
            clKernel.getBuffer().put(kernel1d).rewind();
            CLKernel Kernel = program3d.createCLKernel("Convolve3d");
            Kernel.putArg(clImage)
            	.putArg(clKernel)
            	.putArg(clOutput)
            	.putArg(imageWidth)
            	.putArg(imageHeight)
            	.putArg(imageDepth)
            	.putArg(kernelWidth)
            	.putArg(kernelHeight)
            	.putArg(kernelDepth)
            	.putArg(halfWidth)
            	.putArg(halfHeight)
            	.putArg(halfDepth)
            	.putArg(boundaryConditions);
            queue.putWriteBuffer(clImage, false)
            	.putWriteBuffer(clKernel, false)
            	.put3DRangeKernel(Kernel, 0, 0, 0, globalX, globalY, globalZ, 0,0,0)
            	.putReadBuffer(clOutput, true);
    		result = new double[imageWidth][imageHeight][imageDepth];
    		for (int x = 0; x < imageWidth; x++) {
    			for (int y = 0; y < imageHeight; y++) {
    				for (int z = 0; z < imageDepth; z++) {
    					result[x][y][z] = clOutput.getBuffer().get(x + y*imageWidth + z*imageArea);
    				}
    			}
    		}
        }finally{
        }
        return result;
	}

	public void close() {
        context.release();
	}
}
