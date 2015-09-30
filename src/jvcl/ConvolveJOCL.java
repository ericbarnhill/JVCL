package jvcl;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.*;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.demos.fft.*;
import com.jogamp.opencl.demos.fft.CLFFTPlan.InvalidContextException;

import static java.lang.System.*;
import static com.jogamp.opencl.CLMemory.Mem.*;
import static java.lang.Math.*;

public class ConvolveJOCL {
	
    CLFFTPlan fft;
	
	public ConvolveJOCL() {}

	public double[] convolveFDJOCL(double[] vector, double[] kernel) throws IOException {
		CLContext context = CLContext.create();
        double[] result;
        try{
            CLDevice device = context.getMaxFlopsDevice();
            CLCommandQueue queue = device.createCommandQueue();
            Path currentRelativePath = Paths.get("");
            String s = currentRelativePath.toAbsolutePath().toString();
            String source = readFile("src/dualTree/Convolve2d.cl");
            CLProgram program = context.createProgram(source).build(); 
            int vectorLength = vector.length;
            float[] vectorFloat = new float[vectorLength];
            for (int n = 0; n < vectorLength; n++) vectorFloat[n] = (float)vector[n];
            int kernelLength = kernel.length;
            float[] kernelFloat = new float[kernelLength];
            for (int n = 0; n < kernelLength; n++) kernelFloat[n] = (float)kernel[n];
            int halfLength = kernelLength / 2;
        	int localWorkSize = min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
        	CLBuffer<FloatBuffer> clVector = context.createFloatBuffer(vectorLength, READ_ONLY);
            CLBuffer<FloatBuffer> clKernel = context.createFloatBuffer(kernelLength, READ_ONLY);
            CLBuffer<FloatBuffer> clOutput = context.createFloatBuffer(vectorLength, WRITE_ONLY);
            clVector.getBuffer().put(vectorFloat).rewind();
            clKernel.getBuffer().put(kernelFloat).rewind();
            CLKernel Kernel = program.createCLKernel("Convolve2d");
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
	
	public double[][] convolveFDJOCL(double[][] image, double[][] kernel) throws IOException {
		CLContext context = CLContext.create();
        double[][] result;
        try{
            CLDevice device = context.getMaxFlopsDevice();
            CLCommandQueue queue = device.createCommandQueue();
            Path currentRelativePath = Paths.get("");
            String s = currentRelativePath.toAbsolutePath().toString();
            String source = readFile("src/dualTree/Convolve2d.cl");
            CLProgram program = context.createProgram(source).build(); 
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
            // TODO get work size finished
        	int localWorkSize = min(device.getMaxWorkGroupSize(), 256);  // Local work size dimensions
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

	public double[][][] convolveFDJOCL(double[][][] image, double[][][] kernel) throws IOException {
		CLContext context = CLContext.create();
        //out.println("created "+context);
        double[][][] result;
        try{
            CLDevice device = context.getMaxFlopsDevice();
            CLCommandQueue queue = device.createCommandQueue();
            Path currentRelativePath = Paths.get("");
            String s = currentRelativePath.toAbsolutePath().toString();
            //System.out.println("Current relative path is: " + s);
            String source = readFile("src/dualTree/Convolve3d.cl");
            CLProgram program = context.createProgram(source).build(); 
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
            // cleanup all resources associated with this context.
            context.release();
        }
        return result;
	}

	public double[] convolveFTJOCL(double[] vector, double[] kernel, boolean isComplex) {
		
		int vectorLength = vector.length;
		int kernelLength = kernel.length;
		CLContext context = CLContext.create();
		double[] result;
        try{
            CLDevice device = context.getMaxFlopsDevice();
            CLCommandQueue queue = device.createCommandQueue();
	
			float[] paddedVector;
			float[] paddedKernel;
			
			if (isComplex) {
				paddedVector = new float[vectorLength+2*kernelLength];
				paddedKernel = new float[vectorLength+2*kernelLength];
			} else {
				paddedVector = new float[2*(vectorLength+2*kernelLength)];
				paddedKernel = new float[2*(vectorLength+2*kernelLength)];
			}
			
			for (int x = 0; x < vectorLength; x++) {
				if (isComplex) {
					paddedVector[x+kernelLength] = (float)vector[x];
				} else {
					paddedVector[x*2+kernelLength*2] = (float)vector[x];
				}
			}
			
			for (int x = 0; x < kernelLength; x++) {
				if (isComplex) {
					paddedKernel[x+kernelLength] = (float)kernel[x];
				} else {
					paddedKernel[x*2+kernelLength*2] = (float)kernel[x];
				}
			}
			int paddedLength = paddedVector.length;
			try {
				fft = new CLFFTPlan(context, new int[]{paddedLength}, CLFFTPlan.CLFFTDataFormat.InterleavedComplexFormat);
			} catch (InvalidContextException e) {
				e.printStackTrace();
			}
			CLBuffer<FloatBuffer> clVectorIn = context.createFloatBuffer(paddedLength, READ_ONLY);
            CLBuffer<FloatBuffer> clVectorOut = context.createFloatBuffer(paddedLength, WRITE_ONLY);
            clVectorIn.getBuffer().put(paddedVector).rewind();
            CLBuffer<FloatBuffer> clKernelIn = context.createFloatBuffer(paddedLength, WRITE_ONLY);
            CLBuffer<FloatBuffer> clKernelOut = context.createFloatBuffer(paddedLength, WRITE_ONLY);
            clKernelIn.getBuffer().put(paddedVector).rewind();
	        fft.executeInterleaved(queue, 1, CLFFTPlan.CLFFTDirection.Forward, clVectorIn, clVectorOut, null, null);
	        fft.executeInterleaved(queue, 1, CLFFTPlan.CLFFTDirection.Forward, clKernelIn, clKernelOut, null, null);
			for (int n = 0; n < paddedLength; n++) {
				paddedVector[n] = clVectorOut.getBuffer().get(n)*clKernelOut.getBuffer().get(n);
			}
			clVectorIn.getBuffer().clear();
			clVectorIn.getBuffer().put(paddedVector).rewind();
			clVectorOut.getBuffer().clear();
	        fft.executeInterleaved(queue, 1, CLFFTPlan.CLFFTDirection.Forward, clVectorIn, clVectorOut, null, null);
			
			result = new double[vectorLength];
			for (int x = 0; x < vectorLength; x++) {
				if (isComplex) {
					vector[x] = paddedVector[x+kernelLength];
				} else {
					vector[x] = paddedVector[x*2+kernelLength*2];
				}
			}		
        } finally {
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
