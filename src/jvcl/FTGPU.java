package jvcl;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Random;

import org.apache.commons.math4.complex.Complex;
import org.apache.commons.math4.complex.ComplexUtils;

import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;

public class FTGPU {
	
    CLDevice device;
    CLCommandQueue queue;
    CLProgram program, programStride, program2, programMult;
    CLContext context;
    CLKernel Kernel, KernelMult;
    StockhamGPUStride s;
    boolean debug;
	
	public FTGPU() {
		context = CLContext.create();
		device = context.getMaxFlopsDevice();
		queue = device.createCommandQueue();
	    String source = JVCLUtils.readFile("src/stockham.cl");
	    String source2 = JVCLUtils.readFile("src/stockham2.cl");
	    String sourceStride = JVCLUtils.readFile("src/stockhamStride.cl");
	    String sourceMult = JVCLUtils.readFile("src/vecMult.cl");
		program = context.createProgram(source).build();
		programStride = context.createProgram(sourceStride).build();
		program2 = context.createProgram(sourceStride).build();
		//programMult = context.createProgram(sourceMult).build();

	}

	public Complex[] convolve(Complex[] vector, double[] kernel, boolean isComplex) {
		
		int vectorLength = vector.length;
		int kernelLength = kernel.length;
		int kernelLengthInterleaved = kernelLength*2;
		float[] v = JVCLUtils.double2Float(
						JVCLUtils.zeroPad(ComplexUtils.complex2Interleaved(vector), kernelLengthInterleaved)
					);
		float[] k = JVCLUtils.double2Float(
						JVCLUtils.zeroPad(
								JVCLUtils.real2Interleaved(
										JVCLUtils.deepCopyToPadded(kernel, 
												JVCLUtils.nextPwr2(vectorLength))
								)
						,kernelLengthInterleaved)
					);
		s = new StockhamGPUStride(context, device, queue, program);
		int adjLength = v.length / 2;
		s.fft1d(v, true, adjLength);
		s.fft1d(k, true, adjLength);
		for (int n = 0; n < adjLength; n++) {
			v[n] *= k[n];
		}
		s.fft1d(v, false, adjLength);
		s.close();
		return ComplexUtils.interleaved2Complex(JVCLUtils.stripPadding(JVCLUtils.float2Double(v), kernelLengthInterleaved));		
	}

	public double[][] convolve(double[][] image, double[][] kernel, boolean isComplex) {
		
		long start = 0;
		long end = 0;

		int imageWidth = image.length;
		int imageHeight = image[0].length;
		
		int kernelWidth = kernel.length;
		int kernelHeight = kernel[0].length;
				
		int newWidth, newHeight;
		newWidth = JVCLUtils.nextPwr2(imageWidth+2*kernelWidth);
		newHeight = JVCLUtils.nextPwr2(imageHeight+2*kernelHeight);
		s = new StockhamGPU(context, device, queue, program, newWidth);	

		float[][] paddedReal = new float[newWidth][newHeight];
		float[][] paddedImag = new float[newWidth][newHeight];
		float[][] paddedKernel = new float[newWidth][newHeight];
		float[][] paddedKernelImag = new float[newWidth][newHeight];
		
		if (isComplex) {
			for (int x = 0; x < imageWidth / 2; x++) {
				for (int y = 0; y < imageHeight; y++) {
					paddedReal[x+kernelWidth][y+kernelHeight] = (float)image[x*2][y];
					paddedImag[x+kernelWidth][y+kernelHeight] = (float)image[x*2+1][y];
				} 
			}
		} else {
			for (int x = 0; x < imageWidth; x++) {
				for (int y = 0; y < imageHeight; y++) {
					paddedReal[x+kernelWidth][y+kernelHeight] = (float)image[x][y];
				} 
			}
		}
		
		for (int x = 0; x < kernelWidth; x++) {
			for (int y = 0; y < kernelHeight; y++) {
					paddedKernel[x+kernelWidth][y+kernelHeight] = (float)kernel[x][y];
			}
		}
		// begin FFT
		if (debug) {
			System.out.println("FFTs forward");
			start = System.currentTimeMillis();
		}
		for (int x = 0; x < newWidth; x++) {
			s.fft(paddedReal[x], paddedImag[x], true);
			s.fft(paddedKernel[x], paddedKernelImag[x], true);
		}
		if (debug) {
			end = System.currentTimeMillis();
			System.out.format("Duration %.3f %n", (end-start)/1000.0);
			System.out.println("Dim shifting");
			start = System.currentTimeMillis();
		}
		paddedReal =  JVCLUtils.shiftDim(paddedReal);
		paddedImag = JVCLUtils.shiftDim(paddedImag);
		paddedKernel = JVCLUtils.shiftDim(paddedKernel);
		paddedKernelImag = JVCLUtils.shiftDim(paddedKernelImag);
		if (debug) {
			end = System.currentTimeMillis();
			System.out.format("Duration %.3f %n", (end-start)/1000.0);
			System.out.println("Second FFT");
			start = System.currentTimeMillis();
		}
		for (int y = 0; y < newHeight; y++) {
			s.fft(paddedReal[y], paddedImag[y], true);
			s.fft(paddedKernel[y], paddedKernelImag[y], true);
		}
		if (debug) {
			end = System.currentTimeMillis();
			System.out.format("Duration %.3f %n", (end-start)/1000.0);
			System.out.println("Multiplying arrays");
			start = System.currentTimeMillis();
		}
		for (int x = 0; x < newWidth; x++) {
			for (int y = 0; y < newHeight; y++) {
					paddedReal[x][y] *= paddedKernel[x][y];
					paddedImag[x][y] *= paddedKernelImag[x][y];
			}
		}
		if (debug) {
			end = System.currentTimeMillis();
			System.out.format("Duration %.3f %n", (end-start)/1000.0);
			System.out.println("FFT inverse");
			start = System.currentTimeMillis();
		}
		for (int y = 0; y < newHeight; y++) {
			s.fft(paddedReal[y], paddedImag[y], false);
		}
		if (debug) {
			end = System.currentTimeMillis();
			System.out.format("Duration %.3f %n", (end-start)/1000.0);
			System.out.println("Dim Shifting");
			start = System.currentTimeMillis();
		}
		paddedReal =  JVCLUtils.shiftDim(paddedReal);
		paddedImag = JVCLUtils.shiftDim(paddedImag);
		if (debug) {
			end = System.currentTimeMillis();
			System.out.format("Duration %.3f %n", (end-start)/1000.0);
			System.out.println("Second FFT");
			start = System.currentTimeMillis();
		}
		for (int x = 0; x < newWidth; x++) {
			s.fft(paddedReal[x], paddedImag[x], false);
		}
		if (debug) {
			end = System.currentTimeMillis();
			System.out.format("Duration %.3f %n", (end-start)/1000.0);
		}
		
		double[][] result = new double[imageWidth][imageHeight];
		if (isComplex) {
			for (int x = 0; x < imageWidth / 2; x++) {
				for (int y = 0; y < imageHeight; y++) {
					result[x*2][y] = paddedReal[x+kernelWidth][y+kernelHeight];
					result[x*2+1][y] = paddedImag[x+kernelWidth][y+kernelHeight];
				} 
			}
		} else {
			for (int x = 0; x < imageWidth; x++) {
				for (int y = 0; y < imageHeight; y++) {
					result[x][y] = paddedReal[x+kernelWidth][y+kernelHeight];
				} 
			}
		}
		s.close();
		return result;
		
	}
	
	public double[][] convolveStride(double[][] image, double[][] kernel, boolean isComplex) {
				
		long start = 0;
		long end = 0;

		int imageWidth = image.length;
		int imageHeight = image[0].length;
		
		//if (imageWidth == 1024) debug = true;
		
		int kernelWidth = kernel.length;
		int kernelHeight = kernel[0].length;
				
		int newWidth, newHeight;
		newWidth = JVCLUtils.nextPwr2(imageWidth+2*kernelWidth);
		newHeight = JVCLUtils.nextPwr2(imageHeight+2*kernelHeight);
		int newArea = newWidth*newHeight;

		//str = new StockhamGPUStride();
		s2 = new StockhamGPUStrideNoDS(context, device, queue, programStride);	
		//str = new StockhamGPUStride(context, device, queue, programStride);	

		float[][] paddedReal = new float[newWidth][newHeight];
		float[][] paddedImag = new float[newWidth][newHeight];
		float[][] paddedKernel = new float[newWidth][newHeight];
		float[][] paddedKernelImag = new float[newWidth][newHeight];
		
		if (isComplex) {
			for (int x = 0; x < imageWidth / 2; x++) {
				for (int y = 0; y < imageHeight; y++) {
					paddedReal[x+kernelWidth][y+kernelHeight] = (float)image[x*2][y];
					paddedImag[x+kernelWidth][y+kernelHeight] = (float)image[x*2+1][y];
				} 
			}
		} else {
			for (int x = 0; x < imageWidth; x++) {
				for (int y = 0; y < imageHeight; y++) {
					paddedReal[x+kernelWidth][y+kernelHeight] = (float)image[x][y];
				} 
			}
		}
		
		for (int x = 0; x < kernelWidth; x++) {
			for (int y = 0; y < kernelHeight; y++) {
					paddedKernel[x+kernelWidth][y+kernelHeight] = (float)kernel[x][y];
			}
		}
		
		float[] paddedRVec = JVCLUtils.vectorise(paddedReal);
		float[] paddedIVec = JVCLUtils.vectorise(paddedImag);
		float[] paddedRKern = JVCLUtils.vectorise(paddedReal);
		float[] paddedIKern = JVCLUtils.vectorise(paddedImag);
		s2.fft(paddedRVec, paddedIVec, true, newWidth, newHeight, true);
		s2.fft(paddedRKern, paddedIKern, true, newWidth, newHeight, true);
		s2.fft(paddedRVec, paddedIVec, true, newWidth, newHeight, false);
		s2.fft(paddedRKern, paddedIKern, true, newWidth, newHeight, false);
		
		for (int n = 0; n < newArea; n++) {
					paddedRVec[n] *= paddedRKern[n];
					paddedIVec[n] *= paddedIKern[n];
		}
		
		if (debug) {
			end = System.currentTimeMillis();
			System.out.format("Duration %.3f %n", (end-start)/1000.0);
			System.out.println("FFT inverse");
			start = System.currentTimeMillis();
		}
		
		s2.fft(paddedRVec, paddedIVec, false, newWidth, newHeight, false);
		s2.fft(paddedRVec, paddedIVec, false, newWidth, newHeight, false);

		paddedReal = JVCLUtils.devectorise(paddedRVec, newWidth);
		paddedImag = JVCLUtils.devectorise(paddedIVec, newWidth);
		
		if (isComplex) {
			for (int x = 0; x < imageWidth / 2; x++) {
				for (int y = 0; y < imageHeight; y++) {
					image[x*2][y] = paddedReal[x+kernelWidth][y+kernelHeight];
					image[x*2+1][y] = paddedImag[x+kernelWidth][y+kernelHeight];
				} 
			}
		} else {
			for (int x = 0; x < imageWidth; x++) {
				for (int y = 0; y < imageHeight; y++) {
					image[x][y] = paddedReal[x+kernelWidth][y+kernelHeight];
				} 
			}
		}
		//str.close();
		if (debug == true) System.exit(0);
		return image;
		
	}
		
	public double[][][] convolve(double[][][] volume, double[][][] kernel, boolean isComplex) {
		
		int volumeWidth = volume.length;
		int volumeHeight = volume[0].length;
		int volumeDepth = volume[0][0].length;
		
		int kernelWidth = kernel.length;
		int kernelHeight = kernel[0].length;
		int kernelDepth = kernel[0][0].length;
				
		int newWidth, newHeight, newDepth;
		if (isComplex) {
			newWidth = JVCLUtils.nextPwr2(volumeWidth + 2*kernelWidth);
			newHeight = JVCLUtils.nextPwr2(volumeHeight + 2*kernelHeight);		
			newDepth = JVCLUtils.nextPwr2(volumeDepth + 2*kernelDepth);
		} else {
			newWidth = JVCLUtils.nextPwr2(2*(volumeWidth+2*kernelWidth));
			newHeight = JVCLUtils.nextPwr2(2*(volumeHeight+2*kernelHeight));	
			newDepth = JVCLUtils.nextPwr2(2*(volumeDepth+2*kernelDepth));
		}
		s = new StockhamGPU(context, device, queue, program, newWidth);	

		float[][][] paddedReal = new float[newWidth][newHeight][newDepth];
		float[][][] paddedImag = new float[newWidth][newHeight][newDepth];
		float[][][] paddedKernel = new float[newWidth][newHeight][newDepth];
		float[][][] paddedKernelImag = new float[newWidth][newHeight][newDepth];
		
		if (isComplex) {
			for (int x = 0; x < volumeWidth / 2; x++) {
				for (int y = 0; y < volumeHeight; y++) {
					for (int z = 0; z < volumeDepth; z++) {
						paddedReal[x+kernelWidth][y+kernelHeight][z+kernelDepth] = (float)volume[x*2][y][z];
						paddedImag[x+kernelWidth][y+kernelHeight][z+kernelDepth] = (float)volume[x*2+1][y][z];
					}
				} 
			}
		} else {
			for (int x = 0; x < volumeWidth; x++) {
				for (int y = 0; y < volumeHeight; y++) {
					for (int z = 0; z < volumeDepth; z++) {
						paddedReal[x+kernelWidth][y+kernelHeight][z+kernelDepth] = (float)volume[x][y][z];
					}
				} 
			}
		}
		
		for (int x = 0; x < kernelWidth; x++) {
			for (int y = 0; y < kernelHeight; y++) {
				for (int z = 0; z < volumeDepth; z++) {
					paddedKernel[x+kernelWidth][y+kernelHeight][z+kernelDepth] = (float)kernel[x][y][z];
				}
			}
		}
		
		for (int x = 0; x < newWidth; x++) {
			for (int y = 0; y < newHeight; y++) {
				s.fft(paddedReal[x][y], paddedImag[x][y], true);
				s.fft(paddedKernel[x][y], paddedKernelImag[x][y], true);
			}
		}

		paddedReal =  JVCLUtils.shiftDim(paddedReal);
		paddedImag = JVCLUtils.shiftDim(paddedImag);
		paddedKernel = JVCLUtils.shiftDim(paddedKernel);
		paddedKernelImag = JVCLUtils.shiftDim(paddedKernelImag);

		for (int y = 0; y < newHeight; y++) {
			for (int z = 0; z < newDepth; z++) {
				s.fft(paddedReal[y][z], paddedImag[y][z], true);
				s.fft(paddedKernel[y][z], paddedKernelImag[y][z], true);
			}
		}

		paddedReal =  JVCLUtils.shiftDim(paddedReal);
		paddedImag = JVCLUtils.shiftDim(paddedImag);
		paddedKernel = JVCLUtils.shiftDim(paddedKernel);
		paddedKernelImag = JVCLUtils.shiftDim(paddedKernelImag);

		for (int z = 0; z < newDepth; z++) {
			for (int x = 0; x < newWidth; x++) {
				s.fft(paddedReal[z][x], paddedImag[z][x], true);
				s.fft(paddedKernel[z][x], paddedKernelImag[z][x], true);
			}
		}
		
		for (int x = 0; x < newWidth; x++) {
			for (int y = 0; y < newHeight; y++) {
				for (int z = 0; z < newDepth; z++) {
					paddedReal[x][y][z] *= paddedKernel[x][y][z];
					paddedImag[x][y][z] *= paddedKernelImag[x][y][z];
				}
			}
		}

		for (int z = 0; z < newDepth; z++) {
			for (int x = 0; x < newWidth; x++) {
				s.fft(paddedReal[z][x], paddedImag[z][x], false);
			}
		}

		paddedReal =  JVCLUtils.shiftDim(paddedReal, 2);
		paddedImag = JVCLUtils.shiftDim(paddedImag, 2);

		for (int y = 0; y < newHeight; y++) {
			for (int z = 0; z < newDepth; z++) {
				s.fft(paddedReal[y][z], paddedImag[y][z], false);
			}
		}

		paddedReal =  JVCLUtils.shiftDim(paddedReal, 2);
		paddedImag = JVCLUtils.shiftDim(paddedImag, 2);

		for (int x = 0; x < newWidth; x++) {
			for (int y = 0; y < newHeight; y++) {
				s.fft(paddedReal[x][y], paddedImag[x][y], false);
			}
		}

		
		double[][][] result = new double[volumeWidth][volumeHeight][volumeHeight];
		if (isComplex) {
			for (int x = 0; x < volumeWidth / 2; x++) {
				for (int y = 0; y < volumeHeight; y++) {
					for (int z = 0; z < newDepth; z++) {
						result[x*2][y][z] = paddedReal[x+kernelWidth][y+kernelHeight][z+kernelDepth];
						result[x*2+1][y][z] = paddedImag[x+kernelWidth][y+kernelHeight][z+kernelDepth];
					}
				} 
			}
		} else {
			for (int x = 0; x < volumeWidth; x++) {
				for (int y = 0; y < volumeHeight; y++) {
					for (int z = 0; z < newDepth; z++) {
						result[x][y][z] = paddedReal[x+kernelWidth][y+kernelHeight][z+kernelDepth];
					}
				} 
			}
		}
		return result;
		
	}
	
	public double[][][] convolve(double[][][] volume, double[][] kernel, boolean isComplex) {
		return convolve(volume, kernel, isComplex, 0);
	}
		public void setDebug(boolean debugSet) {
		debug = debugSet;
	}
	
	public void close() {
		context.release();
	}

	public static void main(String[] args) {
		Random random = new Random();
		int arraySize = 128;
		int kernelSize = 7;
		long start, end, duration;
		FTGPU ftgpu = new FTGPU();
		System.out.format("Array size %d Kernel Size %d \n", arraySize, kernelSize);
		double[][] array = new double[arraySize][arraySize];
		double[][] kernel = new double[kernelSize][kernelSize];
		for (int x = 0; x < arraySize; x++) {
			for (int y = 0; y < arraySize; y++) {
				if (x < kernelSize && y < kernelSize) {
					kernel[x][y] = random.nextDouble();
				}
				array[x][y] = random.nextDouble();
			}
		}
		start = System.currentTimeMillis();
		for (int t = 0; t < 3; t++) {
			ftgpu.convolve(array, kernel, false);
			System.out.println("---");
		}
		end = System.currentTimeMillis();
		duration = end-start;
		System.out.format("2D FT on GPU: %.1f sec %n", duration / 1000.0 );
		System.out.println("Done");
		ftgpu.close();
	}

}
