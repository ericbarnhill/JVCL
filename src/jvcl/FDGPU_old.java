package jvcl;

import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.WRITE_ONLY;
import static java.lang.Math.min;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

import org.apache.commons.math4.complex.Complex;
import org.apache.commons.math4.complex.ComplexUtils;
import org.apache.commons.math4.util.IntegerSequence;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

public class FDGPU_old {
	
    CLDevice device;
    CLCommandQueue queue;
    CLProgram program1d, program2d, program3d, program1dComplex, program2dComplex, program3dComplex;
    CLContext context;
    int boundaryConditions;
	final int ZERO_BOUNDARY = 0;
	final int MIRROR_BOUNDARY = 1;
	final int PERIODIC_BOUNDARY = 2;
	/*
	public FDGPU_old(int boundaryConditions) {
		String path = "/home/ericbarnhill/barnhill-eclipse-workspace/JVCL/";
		context = CLContext.create();
		device = context.getMaxFlopsDevice();
        queue = device.createCommandQueue();
        String source1d = JVCLUtils.readFile(path+"src/Convolve1d.cl");
        String source2d = JVCLUtils.readFile(path+"src/Convolve2d.cl");
        String source3d = JVCLUtils.readFile(path+"src/Convolve3d.cl");
        String source1dComplex = JVCLUtils.readFile(path+"src/Convolve1dComplex.cl");
        String source2dComplex = JVCLUtils.readFile(path+"src/Convolve2dComplex.cl");
        String source3dComplex = JVCLUtils.readFile(path+"src/Convolve3dComplex.cl");
        program1d = context.createProgram(source1d).build(); 
        program2d = context.createProgram(source2d).build(); 
        program3d = context.createProgram(source3d).build(); 
        program1dComplex = context.createProgram(source1dComplex).build(); 
        program2dComplex = context.createProgram(source2dComplex).build(); 
        program3dComplex = context.createProgram(source3dComplex).build(); 
		this.boundaryConditions = boundaryConditions;
	}
	
	public FDGPU_old() {
		context = CLContext.create();
		device = context.getMaxFlopsDevice();
        queue = device.createCommandQueue();
        Path currentRelativePath = Paths.get("");
        String s = currentRelativePath.toAbsolutePath().toString();
        System.out.println("Current relative path is: " + s);
        String path="/home/ericbarnhill/barnhill-eclipse-workspace/JVCL/";
        String source1d = JVCLUtils.readFile(path+"src/Convolve1d.cl");
        String source2d = JVCLUtils.readFile(path+"src/Convolve2d.cl");
        String source3d = JVCLUtils.readFile(path+"src/Convolve3d.cl");
        String source1dComplex = JVCLUtils.readFile(path+"src/Convolve1dComplex.cl");
        String source2dComplex = JVCLUtils.readFile(path+"src/Convolve2dComplex.cl");
        String source3dComplex = JVCLUtils.readFile(path+"src/Convolve3dComplex.cl");
        program1d = context.createProgram(source1d).build(); 
        program2d = context.createProgram(source2d).build(); 
        program3d = context.createProgram(source3d).build(); 
        program1dComplex = context.createProgram(source1dComplex).build(); 
        program2dComplex = context.createProgram(source2dComplex).build(); 
        program3dComplex = context.createProgram(source3dComplex).build(); 
		this.boundaryConditions = ZERO_BOUNDARY;
	}

	public double[] convolve(double[] vector, double[] kernel) {
        int vectorLength = vector.length;
        int kernelLength = kernel.length;
        int halfLength = kernelLength / 2;
    	CLBuffer<FloatBuffer> clVector = context.createFloatBuffer(vectorLength, READ_ONLY);
        CLBuffer<FloatBuffer> clKernel = context.createFloatBuffer(kernelLength, READ_ONLY);
        CLBuffer<FloatBuffer> clOutput = context.createFloatBuffer(vectorLength, WRITE_ONLY);
        clVector.getBuffer().put(JVCLUtils.double2Float(vector)).rewind();
        clKernel.getBuffer().put(JVCLUtils.double2Float(kernel)).rewind();
        //clOutput.getBuffer().put(result).rewind();
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
		float[] result = new float[vectorLength];
		clOutput.getBuffer().get(result);     
        return JVCLUtils.float2Double(result);    		
	}
	

	public Complex[] convolve(Complex[] vector, Complex[] kernel) {
        int vectorLength = vector.length;
        int kernelLength = kernel.length;
        int halfLength = kernelLength / 2;
		float[] result = new float[vectorLength*2];
    	CLBuffer<FloatBuffer> clVector = context.createFloatBuffer(vectorLength*2, READ_ONLY); // *2 for interleaved
        CLBuffer<FloatBuffer> clKernel = context.createFloatBuffer(kernelLength*2, READ_ONLY);
        CLBuffer<FloatBuffer> clOutput = context.createFloatBuffer(vectorLength*2, WRITE_ONLY);
        clVector.getBuffer().put(ComplexUtils.complex2InterleavedFloat(vector)).rewind();
        clKernel.getBuffer().put(ComplexUtils.complex2InterleavedFloat(kernel)).rewind();
        CLKernel Kernel = program1dComplex.createCLKernel("Convolve1dComplex");
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
		clOutput.getBuffer().get(result);
        return ComplexUtils.interleaved2Complex(result);    		
	}
	
	
	public double[][] convolve(double[][] image, double[][] kernel) {
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
    		float[] result = new float[imageArea];
    		 clOutput.getBuffer().get(result);
    		 clImage.release();
             clKernel.release();
             clOutput.release();
        return JVCLUtils.float2Double(JVCLUtils.devectorise(result, imageWidth, 1));
	}
	

	public Complex[][] convolve(Complex[][] image, Complex[][] kernel) {
            int imageWidth = image.length;
            int imageHeight = image[0].length;
            int imageArea = imageWidth*imageHeight;
            int kernelWidth = kernel.length;
            int kernelHeight = kernel[0].length;
            int kernelArea = kernelWidth*kernelHeight;
            int halfWidth = kernelWidth / 2;
            int halfHeight = kernelHeight / 2;
    		float[] result = new float[imageArea*2];
        	CLBuffer<FloatBuffer> clImage = context.createFloatBuffer(imageArea*2, READ_ONLY);
            CLBuffer<FloatBuffer> clKernel = context.createFloatBuffer(kernelArea*2, READ_ONLY);
            CLBuffer<FloatBuffer> clOutput = context.createFloatBuffer(imageArea*2, WRITE_ONLY);
            float[][] imageTemp1 = ComplexUtils.complex2InterleavedFloat(image);
            float[] imageTemp2 = JVCLUtils.vectorise(ComplexUtils.complex2InterleavedFloat(image), 1);
            clImage.getBuffer().put(JVCLUtils.vectorise(ComplexUtils.complex2InterleavedFloat(image), 1)).rewind();
            clKernel.getBuffer().put(JVCLUtils.vectorise(ComplexUtils.complex2InterleavedFloat(kernel), 1)).rewind();
            clOutput.getBuffer().put(result).rewind();
            CLKernel Kernel = program2dComplex.createCLKernel("Convolve2dComplex");
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
            	.put2DRangeKernel(Kernel, 0, 0, JVCLUtils.roundUp(imageWidth*2, 32),JVCLUtils.roundUp(imageHeight, 32), 0, 0)
            	.putReadBuffer(clOutput, true);
   		 	clOutput.getBuffer().get(result);
   		 	//JVCLUtils.display(result, "result", 8);
            clOutput.release();
            return ComplexUtils.interleaved2Complex(JVCLUtils.devectorise(result, imageWidth, 1));
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

	public static void main(String[] args) {
		/*
		int length = 64;
		double[] vec = new double[length];
		double[] kern = new double[]{1, -2, 1};
		for ( int i : IntegerSequence.range(0,length-1)) {
			vec[i] = i*i;
		}
		FDGPU_old fdgpu = new FDGPU_old();
		double[] res1 = fdgpu.convolve(vec, kern);
		double[] res2 = ComplexUtils.complex2Real(fdgpu.convolve(ComplexUtils.real2Complex(vec), ComplexUtils.real2Complex(kern)));
		double[] res3 = new FDCPUNaive().convolve(vec, kern);
		System.out.println(Arrays.toString(res1));
		System.out.println(Arrays.toString(res2));
		System.out.println(Arrays.toString(res3));
		
		int length = 64;
		double[][] vec = new double[length][length];
		double[][] kern = JVCLUtils.devectorise(new double[]{0, 1, 0, 1, -4, 1, 0, 1, 0}, 3, 0);
		for ( int i : IntegerSequence.range(0,length-1)) {
			for ( int j : IntegerSequence.range(0,length-1)) {
				vec[i][j] = i*i + j*j;
			}
		}
		FDGPU_old fdgpu = new FDGPU_old();
		try {
			double[][] res1 = fdgpu.convolve(vec, kern);
			double[][] res2 = ComplexUtils.complex2Real(fdgpu.convolve(ComplexUtils.real2Complex(vec), ComplexUtils.real2Complex(kern)));
			double[][] res3 = new FDCPUNaive().convolve(vec, kern);
			System.out.println(Arrays.toString(res1[10]));
			System.out.println(Arrays.toString(res2[11]));
			System.out.println(Arrays.toString(res3[10]));
		} finally {
			fdgpu.close();
		}
	}
	*/
}
