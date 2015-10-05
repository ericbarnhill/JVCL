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
		Stockham s = new Stockham();
		
		double[] paddedReal;
		double[] paddedImag;
		double[] paddedKernel;
		double[] paddedKernelImag; // just a placeholder
		int newSize;
		if (isComplex) {
			newSize = vectorLength / 2 + 2 * kernelLength;
		} else {
			newSize = vectorLength + 2 * kernelLength;
		}
		paddedReal = new double[newSize];
		paddedImag = new double[newSize];
		paddedKernel = new double[newSize];
		paddedKernelImag = new double[newSize];
		if (isComplex) {
			for (int x = 0; x < vectorLength/2; x++) {
					paddedReal[x+kernelLength] = vector[x*2];
					paddedImag[x+kernelLength] = vector[x*2+1];
			}
		} else {
				paddedReal[x*2+kernelLength*2] = vector[x];
			}
		}
		
		for (int x = 0; x < kernelHeight; x++) {
			if (isComplex) {
				paddedKernel[x+kernelHeight] = kernel[x];
			} else {
				paddedKernel[x*2+kernelHeight*2] = kernel[x];
			}
		}
		
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
	
	public double[][][] convolveFTJOCL(double[][][] real, double[][][] imag, double kernel) {
		Stockham s = new Stockham();
		
		
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
