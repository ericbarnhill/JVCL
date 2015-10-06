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
	
    DimShift ds;
    CLDevice device;
    CLCommandQueue queue;
    CLProgram program;
    CLContext context;
	
	public FDGPU() {
		ds = new DimShift();
	}

	public double[] convolve(double[] vector, double[] kernel) {
		context = CLContext.create();
        double[] result;
        try{
            device = context.getMaxFlopsDevice();
            queue = device.createCommandQueue();
            String source = readFile("src/Convolve1d.cl");
            program = context.createProgram(source).build(); 
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
            CLKernel Kernel = program.createCLKernel("Convolve1d");
            Kernel.putArg(clVector)
            	.putArg(clKernel)
            	.putArg(clOutput)
            	.putArg(vectorLength)
            	.putArg(kernelLength)
            	.putArg(halfLength);
            queue.putWriteBuffer(clVector, false)
            	.putWriteBuffer(clKernel, false)
            	.put1DRangeKernel(Kernel, 0, roundUp(vectorLength, 32),0)
            	.putReadBuffer(clOutput, true);
    		result = new double[vectorLength];
    		for (int x = 0; x < vectorLength; x++) {
    				result[x] = clOutput.getBuffer().get(x);
    		}
        }finally{
            // cleanup all resources associated with this context.
            context.release();
        }
        return result;    		
	}
	
	public double[][] convolve(double[][] image, double[][] kernel) {
		context = CLContext.create();
        double[][] result;
        try{
            device = context.getMaxFlopsDevice();
            queue = device.createCommandQueue();
            String source = readFile("src/Convolve2d.cl");
            program = context.createProgram(source).build(); 
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
            CLKernel Kernel = program.createCLKernel("Convolve2d");
            Kernel.putArg(clImage)
            	.putArg(clKernel)
            	.putArg(clOutput)
            	.putArg(imageWidth)
            	.putArg(imageHeight)
            	.putArg(kernelWidth)
            	.putArg(kernelHeight)
            	.putArg(halfWidth)
            	.putArg(halfHeight);
            queue.putWriteBuffer(clImage, false)
            	.putWriteBuffer(clKernel, false)
            	.put2DRangeKernel(Kernel, 0, 0, roundUp(imageWidth, 32),roundUp(imageHeight,32), 0,0)
            	.putReadBuffer(clOutput, true);
    		result = new double[imageWidth][imageHeight];
    		for (int x = 0; x < imageWidth; x++) {
    			for (int y = 0; y < imageHeight; y++) {
    				result[x][y] = clOutput.getBuffer().get(x + y*imageWidth);
    			}
    		}
        } finally {
            // cleanup all resources associated with this context.
            context.release();
        }
        return result;
	}

	public double[][][] convolve(double[][][] image, double[][][] kernel) {
		context = CLContext.create();
        //out.println("created "+context);
        double[][][] result;
        try{
            device = context.getMaxFlopsDevice();
            queue = device.createCommandQueue();
            String source = readFile("src/Convolve3d.cl");
            program = context.createProgram(source).build(); 
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
        	int globalX = roundUp(imageWidth, localWorkSize);
        	int globalY = roundUp(imageHeight, localWorkSize);
        	int globalZ = roundUp(imageDepth, localWorkSize);
        	CLBuffer<FloatBuffer> clImage = context.createFloatBuffer(imageVolume, READ_ONLY);
            CLBuffer<FloatBuffer> clKernel = context.createFloatBuffer(kernelVolume, READ_ONLY);
            CLBuffer<FloatBuffer> clOutput = context.createFloatBuffer(imageVolume, WRITE_ONLY);
            clImage.getBuffer().put(image1d).rewind();
            clKernel.getBuffer().put(kernel1d).rewind();
            CLKernel Kernel = program.createCLKernel("Convolve3d");
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
            	.putArg(halfDepth);
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
            context.release();
        }
        return result;
	}

    private static int roundUp(int groupSize, int globalSize) {
        int r = globalSize % groupSize;
        if (r == 0) {
            return globalSize;
        } else {
            return globalSize + groupSize - r;
        }
    }
    
    private static String readFile(String fileName) {
        try  {
            BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(fileName)));
            StringBuffer sb = new StringBuffer();
            String line = null;
            while (true) {
                line = br.readLine();
                if (line == null) {
                    break;
                }
                sb.append(line).append("\n");
            }
            br.close();
            return sb.toString();
        }
        catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
            return null;
        }
    }

	
}
