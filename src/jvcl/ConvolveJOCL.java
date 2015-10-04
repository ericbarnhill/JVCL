package jvcl;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
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
import com.jogamp.common.nio.PointerBuffer;

import edu.emory.mathcs.jtransforms.fft.DoubleFFT_2D;
import static java.lang.System.*;
import static com.jogamp.opencl.CLMemory.Mem.*;
import static java.lang.Math.*;

public class ConvolveJOCL {
	
    CLFFTPlan fft;
    DimShifter ds;
    CLDevice device;
    CLCommandQueue queue;
    CLProgram program;
    CLContext context;
	
	public ConvolveJOCL() {
		ds = new DimShifter();
	}

	public double[] convolveFDJOCL(double[] vector, double[] kernel) throws IOException {
		context = CLContext.create();
        double[] result;
        try{
            device = context.getMaxFlopsDevice();
            queue = device.createCommandQueue();
            Path currentRelativePath = Paths.get("");
            String s = currentRelativePath.toAbsolutePath().toString();
            String source = readFile("src/Convolve1d.cl");
            program = context.createProgram(source).build(); 
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
	
	public double[][] convolveFDJOCL(double[][] image, double[][] kernel) throws IOException {
		context = CLContext.create();
        double[][] result;
        try{
            device = context.getMaxFlopsDevice();
            queue = device.createCommandQueue();
            Path currentRelativePath = Paths.get("");
            String s = currentRelativePath.toAbsolutePath().toString();
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
		context = CLContext.create();
        //out.println("created "+context);
        double[][][] result;
        try{
            device = context.getMaxFlopsDevice();
            queue = device.createCommandQueue();
            Path currentRelativePath = Paths.get("");
            String s = currentRelativePath.toAbsolutePath().toString();
            //System.out.println("Current relative path is: " + s);
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
            // cleanup all resources associated with this context.
            context.release();
        }
        return result;
	}

	public double[] convolveFTJOCL(double[] vector, double[] kernel, boolean isComplex) {
		int vectorLength = vector.length;
		int kernelLength = kernel.length;
		context = CLContext.create();
		double[] result;
        try{
            device = context.getMaxFlopsDevice();
            queue = device.createCommandQueue();
            String source = readFile("src/naivefft.cl");
            program = context.createProgram(source).build(); 
	
			float[] paddedVector;
			float[] paddedKernel;
			
			if (isComplex) {
				paddedVector = new float[nextPwr2(vectorLength+2*kernelLength)];
				paddedKernel = new float[nextPwr2(vectorLength+2*kernelLength)];
			} else {
				paddedVector = new float[nextPwr2(2*(vectorLength+2*kernelLength))];
				paddedKernel = new float[nextPwr2(2*(vectorLength+2*kernelLength))];
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
			int log2Length = (int)( Math.log(paddedLength) / Math.log(2) );

			/*
            CLBuffer<FloatBuffer> clSpinPrecompute = context.createFloatBuffer(paddedLength / 2, WRITE_ONLY);
            CLKernel sfac = program.createCLKernel("spinFact");	            	
            sfac.setArg(0, clSpinPrecompute)
        		.setArg(1, paddedLength);
            queue.put1DRangeKernel(sfac, 0, vectorLength, 0)
            	.putReadBuffer(clSpinPrecompute, false)
            	.putBarrier();
            sfac.release();
            */

			CLKernel radix8Img = program.createCLKernel("naivefft");
            CLBuffer<FloatBuffer> clVectorIn = context.createFloatBuffer(paddedLength, READ_ONLY);
            CLBuffer<FloatBuffer> clVectorOut = context.createFloatBuffer(paddedLength, WRITE_ONLY);
            clVectorIn.getBuffer().put(paddedVector).rewind();
            radix8Img.putArg(clVectorIn);
			radix8Img.putArg(clVectorOut);
			radix8Img.putArg(paddedLength);
			//radix8Img.putArg(32);
            queue.putWriteBuffer(clVectorIn, false);
            queue.put1DRangeKernel(radix8Img, 0, paddedLength, 0);
            queue.putReadBuffer(clVectorOut, false);
            queue.putBarrier();
            queue.finish();
            for (int n = 0; n < paddedVector.length; n++) {
            	paddedKernel[n] = clVectorOut.getBuffer().get(n);
            }

            queue = device.createCommandQueue();
            CLKernel radix8Ker = program.createCLKernel("naivefft");
            CLBuffer<FloatBuffer> clKernelIn = context.createFloatBuffer(paddedLength, READ_ONLY);
            CLBuffer<FloatBuffer> clKernelOut = context.createFloatBuffer(paddedLength, WRITE_ONLY);
            clKernelIn.getBuffer().put(paddedKernel).rewind();
            radix8Ker.putArg(clKernelIn);
			radix8Ker.putArg(clKernelOut);
			radix8Ker.putArg(paddedLength);
			//radix8Ker.putArg(32);
            queue.putWriteBuffer(clKernelIn, false);
            queue.put1DRangeKernel(radix8Ker, 0, paddedLength, 0);
            queue.putReadBuffer(clKernelOut, false);
            queue.putBarrier();
            queue.finish();
                        

    		// DEBUG 
    		System.out.println("POSTFFT");
            for (int n = 0; n < 12; n++) System.out.print(clVectorOut.getBuffer().get(n) + " "); 
            System.out.println();
            
            
            
            
            
            /*
            fft(clVectorIn, clVectorOut, clSpinPrecompute, log2Length, true, dims);
            CLBuffer<FloatBuffer> clKernelIn = context.createFloatBuffer(paddedLength, WRITE_ONLY);
            CLBuffer<FloatBuffer> clKernelOut = context.createFloatBuffer(paddedLength, WRITE_ONLY);
            clKernelIn.getBuffer().put(paddedKernel).rewind();
            fft(clKernelIn, clKernelOut, clSpinPrecompute, log2Length, true, dims);
			for (int n = 0; n < paddedLength; n++) {
				float vectorOut = clVectorOut.getBuffer().get(n);
				float kernelOut = clKernelOut.getBuffer().get(n);
				paddedVector[n] = vectorOut*kernelOut;
			}
			clVectorIn.getBuffer().clear();
			clVectorIn.getBuffer().put(paddedVector).rewind();
			clVectorOut.getBuffer().clear();
            fft(clKernelIn, clKernelOut, clSpinPrecompute, log2Length, false, dims);
			result = new double[vectorLength];
			for (int x = 0; x < vectorLength; x++) {
				if (isComplex) {
					result[x] = paddedVector[x+kernelLength];
				} else {
					result[x] = paddedVector[x*2+kernelLength*2];
				}
			}		
			*/
            result = null;
        } finally {
            context.release();
        }
		return result;
	}
	
	int nextPwr2(int length) {
		
		int pwr2Length = 1;
		do {
			pwr2Length *= 2;
		} while(pwr2Length < length);
		return pwr2Length;
	}
	
	public double[][] convolveFTJOCL(double[][] image, double[] kernel, boolean isComplex, int dim) {
		if (dim > 1) throw new RuntimeException("Invalid dim");
		if (dim == 0) image = ds.shiftDim(image);
		int height = image.length;
		for (int n = 0; n < height; n++) {
			image[n] = convolveFTJOCL(image[n], kernel, isComplex);
		}
		if (dim == 0) {
			return ds.shiftDim(image);
		} else {
			return image;
		}
	}
	
	public double[][] convolveFTJOCL(double[][] image, double[] kernel, boolean isComplex) {
		return convolveFTJOCL(image, kernel, isComplex, 0);
	}
	

	public double[][] convolveFTJOCL(double[][] image, double[][] kernel, boolean isComplex) {
		CLContext context = CLContext.create();
		int imageWidth = image.length;
		int imageHeight = image[0].length;
		int kernelWidth = kernel.length;
		int kernelHeight = kernel[0].length;
		double[][] result;
		try{
            CLDevice device = context.getMaxFlopsDevice();
            CLCommandQueue queue = device.createCommandQueue();
			double[][] paddedImage;
			double[][] paddedKernel;
			
			if (isComplex) {
				paddedImage = new double[imageWidth+2*kernelWidth][imageHeight+2*kernelHeight];
				paddedKernel = new double[imageWidth+2*kernelWidth][imageHeight+2*kernelHeight];
			} else {
				paddedImage = new double[2*(imageWidth+2*kernelWidth)][2*(imageHeight+2*kernelHeight)];
				paddedKernel = new double[2*(imageWidth+2*kernelWidth)][2*(imageHeight+2*kernelHeight)];
			}
			
			for (int x = 0; x < imageWidth; x++) {
				for (int y = 0; y < imageHeight; y++) {
					if (isComplex) {
						paddedImage[x+kernelWidth][y+kernelHeight] = image[x][y];
					} else {
						paddedImage[x*2+kernelWidth*2][y*2+kernelHeight*2] = image[x][y];
					}
				}
			}
			
			for (int x = 0; x < kernelWidth; x++) {
				for (int y = 0; y < kernelHeight; y++) {
					if (isComplex) {
						paddedKernel[x+kernelWidth][y+kernelHeight] = kernel[x][y];
					} else {
						paddedKernel[x*2+kernelWidth*2][y*2+kernelHeight*2] = kernel[x][y];
					}
				}
			}
			
			int paddedWidth = paddedImage.length;
			int paddedHeight = paddedImage[0].length;
			int paddedArea = paddedWidth*paddedHeight;
			float[] image1d = new float[paddedArea];
			float[] kernel1d = new float[paddedArea];
			for (int x = 0; x < paddedWidth; x++) {
				for (int y = 0; y < paddedHeight; y++) {
					image1d[x + y*paddedWidth] = (float)paddedImage[x][y];
					kernel1d[x + y*paddedWidth] = (float)paddedKernel[x][y];
				}
			}
			try {
				fft = new CLFFTPlan(context, new int[]{paddedWidth, paddedHeight}, CLFFTPlan.CLFFTDataFormat.InterleavedComplexFormat);
			} catch (InvalidContextException e) {
				e.printStackTrace();
			}

			CLBuffer<FloatBuffer> clVectorIn = context.createFloatBuffer(paddedArea, READ_ONLY);
            CLBuffer<FloatBuffer> clVectorOut = context.createFloatBuffer(paddedArea, WRITE_ONLY);
            clVectorIn.getBuffer().put(image1d).rewind();
            CLBuffer<FloatBuffer> clKernelIn = context.createFloatBuffer(paddedArea, WRITE_ONLY);
            CLBuffer<FloatBuffer> clKernelOut = context.createFloatBuffer(paddedArea, WRITE_ONLY);
            clKernelIn.getBuffer().put(kernel1d).rewind();
	        fft.executeInterleaved(queue, 1, CLFFTPlan.CLFFTDirection.Forward, clVectorIn, clVectorOut, null, null);
	        fft.executeInterleaved(queue, 1, CLFFTPlan.CLFFTDirection.Forward, clKernelIn, clKernelOut, null, null);
			for (int n = 0; n < paddedArea; n++) {
				image1d[n] = clVectorOut.getBuffer().get(n)*clKernelOut.getBuffer().get(n);
			}
			clVectorIn.getBuffer().clear();
			clVectorIn.getBuffer().put(image1d).rewind();
			clVectorOut.getBuffer().clear();
	        fft.executeInterleaved(queue, 1, CLFFTPlan.CLFFTDirection.Inverse, clVectorIn, clVectorOut, null, null);
	        result = new double[imageHeight][imageWidth];
			for (int x = 0; x < imageHeight; x++) {
				for (int y = 0; y < imageWidth; y++) {
					if (isComplex) {
						result[x][y] = image1d[x+kernelWidth + (y+kernelHeight) * paddedWidth];
					} else {
						result[x][y] = image1d[x*2+kernelWidth*2 + (y*2+kernelHeight*2) * paddedWidth];
					}
				}
			}		
		} finally {
            context.release();
        }
		return result;
		
	}
	
	public double[][][] convolveFTJOCL(double[][][] image, double[][][] kernel, boolean isComplex) {
		CLContext context = CLContext.create();
		int imageWidth = image.length;
		int imageHeight = image[0].length;
		int imageDepth = image[0][0].length;
		int kernelWidth = kernel.length;
		int kernelHeight = kernel[0].length;
		int kernelDepth = kernel[0][0].length;
		double[][][] result;
		try{
            CLDevice device = context.getMaxFlopsDevice();
            CLCommandQueue queue = device.createCommandQueue();
			double[][][] paddedImage;
			double[][][] paddedKernel;
			
			if (isComplex) {
				paddedImage = new double[imageWidth+2*kernelWidth][imageHeight+2*kernelHeight][imageDepth+2*kernelDepth];
				paddedKernel = new double[imageWidth+2*kernelWidth][imageHeight+2*kernelHeight][imageDepth+2*kernelDepth];
			} else {
				paddedImage = new double[2*(imageWidth+2*kernelWidth)][2*(imageHeight+2*kernelHeight)][2*(imageDepth+2*kernelDepth)];
				paddedKernel = new double[2*(imageWidth+2*kernelWidth)][2*(imageHeight+2*kernelHeight)][2*(imageDepth+2*kernelDepth)];
			}
			
			for (int x = 0; x < imageWidth; x++) {
				for (int y = 0; y < imageHeight; y++) {
					for (int z = 0; z < imageDepth; z++) {
						if (isComplex) {
							paddedImage[x+kernelWidth][y+kernelHeight][z+kernelDepth] = image[x][y][z];
						} else {
							paddedImage[x*2+kernelWidth*2][y*2+kernelHeight*2][z*2+kernelDepth*2] = image[x][y][z];
						}
					}
				}
			}
			
			for (int x = 0; x < kernelWidth; x++) {
				for (int y = 0; y < kernelHeight; y++) {
					for (int z = 0; z < kernelDepth; z++) {
						if (isComplex) {
							paddedKernel[x+kernelWidth][y+kernelHeight][z+kernelDepth] = kernel[x][y][z];
						} else {
							paddedKernel[x*2+kernelWidth*2][y*2+kernelHeight*2][z*2+kernelDepth*2] = kernel[x][y][z];
						}
					}
				}
			}
			
			int paddedWidth = paddedImage.length;
			int paddedHeight = paddedImage[0].length;
			int paddedDepth = paddedImage[0][0].length;
			int paddedArea = paddedWidth*paddedHeight;
			int paddedVolume = paddedArea*paddedDepth;
			float[] image1d = new float[paddedVolume];
			float[] kernel1d = new float[paddedVolume];
			for (int x = 0; x < paddedWidth; x++) {
				for (int y = 0; y < paddedHeight; y++) {
					for (int z = 0; z < imageDepth; z++) {
						image1d[x + y*paddedWidth + z*paddedArea] = (float)paddedImage[x][y][z];
						kernel1d[x + y*paddedWidth + z*paddedArea] = (float)paddedKernel[x][y][z];
					}
				}
			}
			try {
				fft = new CLFFTPlan(context, new int[]{paddedWidth, paddedHeight, paddedDepth}, CLFFTPlan.CLFFTDataFormat.InterleavedComplexFormat);
			} catch (InvalidContextException e) {
				e.printStackTrace();
			}

			CLBuffer<FloatBuffer> clVectorIn = context.createFloatBuffer(paddedVolume, READ_ONLY);
            CLBuffer<FloatBuffer> clVectorOut = context.createFloatBuffer(paddedVolume, WRITE_ONLY);
            clVectorIn.getBuffer().put(image1d).rewind();
            CLBuffer<FloatBuffer> clKernelIn = context.createFloatBuffer(paddedVolume, WRITE_ONLY);
            CLBuffer<FloatBuffer> clKernelOut = context.createFloatBuffer(paddedVolume, WRITE_ONLY);
            clKernelIn.getBuffer().put(kernel1d).rewind();
	        fft.executeInterleaved(queue, 1, CLFFTPlan.CLFFTDirection.Forward, clVectorIn, clVectorOut, null, null);
	        fft.executeInterleaved(queue, 1, CLFFTPlan.CLFFTDirection.Forward, clKernelIn, clKernelOut, null, null);
			for (int n = 0; n < paddedVolume; n++) {
				image1d[n] = clVectorOut.getBuffer().get(n)*clKernelOut.getBuffer().get(n);
			}
			clVectorIn.getBuffer().clear();
			clVectorIn.getBuffer().put(image1d).rewind();
			clVectorOut.getBuffer().clear();
	        fft.executeInterleaved(queue, 1, CLFFTPlan.CLFFTDirection.Inverse, clVectorIn, clVectorOut, null, null);
	        result = new double[imageHeight][imageWidth][imageDepth];
			for (int x = 0; x < imageHeight; x++) {
				for (int y = 0; y < imageWidth; y++) {
					for (int z = 0; z < imageDepth; z++) {
						if (isComplex) {
							result[x][y][z] = image1d[x+kernelWidth + (y+kernelHeight) * paddedWidth + 
							                          (z+kernelDepth) * paddedArea];
						} else {
							result[x][y][z] = image1d[x*2+kernelWidth*2 + (y*2+kernelHeight*2) * paddedWidth +
							                          (z*2+kernelDepth*2) * paddedArea];
						}
					}
				}
			}		
		} finally {
            context.release();
        }
		return result;
		
	}

	public double[][][] convolveFTJOCL(double[][][] volume, double[][] kernel, boolean isComplex, int dim) {
		if (dim == 0) volume = ds.shiftDim(volume, 2);
		if (dim == 1) volume = ds.shiftDim(volume, 1);
		if (dim > 2) throw new RuntimeException("Invalid dim");
		
		int volumeWidth = volume.length;
		
		for (int x = 0; x < volumeWidth; x++) {
				volume[x] = convolveFTJOCL(volume[x], kernel, isComplex);
		}
		return volume;
	}
	
	public double[][][] convolveFTJOCL(double[][][] volume, double[][] kernel, boolean isComplex) {
		return convolveFTJOCL(volume, kernel, isComplex, 0);
	}
	
		
	
	void fft(CLBuffer<FloatBuffer> in, CLBuffer<FloatBuffer> out, CLBuffer<FloatBuffer> spin,
			int log2Length, boolean forward, int[] dims)	{	
		// DEBUG 
		System.out.println("within FFT");
        for (int n = 0; n < 12; n++) System.out.print(in.getBuffer().get(n) + " ");
        System.out.println();
		
		int iter;	
		int flag = 0;	
		int width = (int)Math.pow(2,log2Length);	
		
	    CLKernel brev = program.createCLKernel("bitReverse");
	    CLKernel bfly = program.createCLKernel("butterfly");
	    CLKernel norm = program.createCLKernel("norm");
		
	    int nDims = dims.length;
	    CLBuffer<IntBuffer> gws = context.createIntBuffer(nDims);
	    CLBuffer<IntBuffer> lws = context.createIntBuffer(nDims);
	    CLBuffer<IntBuffer> offsets = context.createIntBuffer(nDims);
	    for (int x = 0; x < nDims; x++) {
	    	gws.getBuffer().put(dims[x]);
	    	lws.getBuffer().put(0);
	    	offsets.getBuffer().put(0);
	    }
		flag = (forward) ? 0x00000000 : 0x80000000;
	    //System.out.println("SPIN");
	    //for (int n = 0; n < 48; n++) {
		//	System.out.format("%.3f ", spin.getBuffer().get(n));
		//}
	    //System.out.println();
		
		brev.putArg(out)
			.putArg(in)
			.putArg(log2Length)
			.putArg(width);
		
		queue.putWriteBuffer(in, false)
			.putWriteBuffer(spin, false);
		
		/* Reverse bit ordering */	
		if (nDims == 1) {
			queue.put1DRangeKernel(brev, 0, dims[0], 0);
		} else if (nDims == 2) {
			queue.put2DRangeKernel(brev, 0, 0, dims[0], dims[1], 0, 0);
		} else if (nDims == 3) {
			queue.put3DRangeKernel(brev,  0,  0, 0,  dims[0], dims[1], dims[2], 0, 0, 0);
		}
		queue.putReadBuffer(out, false);
		queue.finish();
		
		// DEBUG 
		System.out.println("POST BIT REVERSAL");
        for (int n = 0; n < 12; n++) System.out.print(out.getBuffer().get(n) + " ");
        System.out.println(); 
        
        in.getBuffer().clear();
        for (int n = 0; n < out.getCLCapacity(); n++) in.getBuffer().put(out.getBuffer().get(n));
        out.getBuffer().clear();
		
		queue.putWriteBuffer(in, false)
			.putWriteBuffer(spin, false);
		
		/* Reverse bit ordering */	
		if (nDims == 1) {
			queue.put1DRangeKernel(brev, 0, dims[0], 0);
		} else if (nDims == 2) {
			queue.put2DRangeKernel(brev, 0, 0, dims[0], dims[1], 0, 0);
		} else if (nDims == 3) {
			queue.put3DRangeKernel(brev,  0,  0, 0,  dims[0], dims[1], dims[2], 0, 0, 0);
		}
		queue.putReadBuffer(out, false);
		queue.finish();
		
		// DEBUG 
		System.out.println("POST BIT REVERSAL");
        for (int n = 0; n < 12; n++) System.out.print(out.getBuffer().get(n) + " ");
        System.out.println(); 
        
        

		bfly.setArg(0, out)
			.setArg(1, spin)
			.setArg(2, log2Length)
			.setArg(3, width)
			.setArg(5, flag);
		
        CLBuffer<FloatBuffer> pre = context.createFloatBuffer(out.getBuffer().capacity());
        CLBuffer<FloatBuffer> post = context.createFloatBuffer(out.getBuffer().capacity());
		/* Perform Butterfly Operations*/	
		queue.putWriteBuffer(out, false);
		for (iter=1; iter <= log2Length; iter++){
			bfly.setArg(4, iter);
			if (nDims == 1) {
				queue.put1DRangeKernel(bfly, 0, dims[0], 0);
			} else if (nDims == 2) {
				queue.put2DRangeKernel(bfly, 0, 0, dims[0], dims[1], 0, 0);
			} else if (nDims == 3) {
				queue.put3DRangeKernel(bfly,  0,  0, 0,  dims[0], dims[1], dims[2], 0, 0, 0);
			}
		}
		queue.putReadBuffer(post, false);

		// DEBUG 
		System.out.println("POST FFT");
        for (int n = 0; n < 12; n++) System.out.print(post.getBuffer().get(n) + " ");
        System.out.println();
		
		norm.putArg(out)
			.putArg(width);
        
		if (!forward) {	
			if (nDims == 1) {
				queue.put1DRangeKernel(norm, 0, dims[0], 0);
			} else if (nDims == 2) {
				queue.put2DRangeKernel(norm, 0, 0, dims[0], dims[1], 0, 0);
			} else if (nDims == 3) {
				queue.put3DRangeKernel(norm,  0,  0, 0,  dims[0], dims[1], dims[2], 0, 0, 0);
			}
		}	
		
		brev.release();
		bfly.release();
		norm.release();
			
		return;	
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
		
	CLBuffer<FloatBuffer> arrayToBuffer(double[] array) {
		CLBuffer<FloatBuffer> buffer = context.createFloatBuffer(array.length);
		for (int n = 0; n < array.length; n++) {
			buffer.getBuffer().put((float)array[n]);
		}
		buffer.getBuffer().rewind();
		return buffer;
	}
	
	CLBuffer<IntBuffer> arrayToBuffer(int[] array) {
		CLBuffer<IntBuffer> buffer = context.createIntBuffer(array.length);
		for (int n = 0; n < array.length; n++) {
			buffer.getBuffer().put(array[n]);
		}
		buffer.getBuffer().rewind();
		return buffer;
	}
	
	CLBuffer<FloatBuffer> arrayToBuffer(double[][] array) {
		int width = array.length;
		int height = array[0].length;
		int area = width*height;
		CLBuffer<FloatBuffer> buffer = context.createFloatBuffer(area);
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				buffer.getBuffer().put((float)array[x][y]);
			}
		}
		buffer.getBuffer().rewind();
		return buffer;
	}
	
	CLBuffer<FloatBuffer> arrayToBuffer(double[][][] array) {
		int width = array.length;
		int height = array[0].length;
		int depth = array[0][0].length;
		int area = width*height;
		int volume = area*depth;
		CLBuffer<FloatBuffer> buffer = context.createFloatBuffer(volume);
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				for (int z = 0; z < depth; z++) {
					buffer.getBuffer().put((float)array[x][y][z]);
				}
			}
		}
		buffer.getBuffer().rewind();
		return buffer;
	}

}
