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

public class FTGPU {
	
    DimShift ds;
    CLDevice device;
    CLCommandQueue queue;
    CLProgram program;
    CLContext context;
	
	public FTGPU() {
		ds = new DimShift();
	}

	public double[] convolve(double[] vector, double[] kernel, boolean isComplex) {
		
		int vectorLength = vector.length;
		int kernelLength = kernel.length;
		StockhamGPU s = new StockhamGPU();
		
		float[] paddedReal;
		float[] paddedImag;
		float[] paddedKernel;
		float[] paddedKernelImag; // only has values post-fft
		int newLength;
		if (isComplex) {
			newLength = nextPwr2(vectorLength / 2 + 2 * kernelLength);
		} else {
			newLength = nextPwr2(vectorLength + 2 * kernelLength);
		}
		paddedReal = new float[newLength];
		paddedImag = new float[newLength];
		paddedKernel = new float[newLength];
		paddedKernelImag = new float[newLength];

		if (isComplex) {
			for (int x = 0; x < vectorLength/2; x++) {
					paddedReal[x+kernelLength] = (float)vector[x*2];
					paddedImag[x+kernelLength] = (float)vector[x*2+1];
			}
		} else {
			for (int x = 0; x < vectorLength; x++) {
				paddedReal[x+kernelLength] = (float)vector[x];
			}
		}
		
		for (int x = 0; x < kernelLength; x++) {
			paddedKernel[x+kernelLength] = (float)kernel[x];
		}

		s.fft(paddedReal, paddedImag, true);
		s.fft(paddedKernel, paddedKernelImag, true);
		
		for (int x = 0; x < newLength; x++) {
			paddedReal[x] *= paddedKernel[x];
			paddedImag[x] *= paddedKernelImag[x];
		}
		
		s.fft(paddedReal, paddedImag, false);
		// RE-INTERLEAVE
		double[] result = new double[vectorLength];
		if (isComplex) {
			for (int x = 0; x < vectorLength / 2; x++) {
				result[x*2] = paddedReal[x+kernelLength];
				result[x*2+1] = paddedImag[x+kernelLength];
			}
		} else {
			for (int x = 0; x < vectorLength; x++) {
				result[x] = paddedReal[x+kernelLength];
			}
		}		
		return result;
		
	}
	
	public double[][] convolve(double[][] image, double[] kernel, boolean isComplex, int dim) {
		if (dim > 1) throw new RuntimeException("Invalid dim");
		if (dim == 0) image = ds.shiftDim(image);
		int height = image.length;
		for (int n = 0; n < height; n++) {
			image[n] = convolve(image[n], kernel, isComplex);
		}
		if (dim == 0) {
			return ds.shiftDim(image);
		} else {
			return image;
		}
	}
	
	public double[][] convolve(double[][] image, double[] kernel, boolean isComplex) {
		return convolve(image, kernel, isComplex, 0);
	}
	
	
	public double[][][] convolve(double[][][] volume, double[][] kernel, boolean isComplex, int dim) {
		if (dim == 0) volume = ds.shiftDim(volume, 2);
		if (dim == 1) volume = ds.shiftDim(volume, 1);
		if (dim > 2) throw new RuntimeException("Invalid dim");
		
		int volumeWidth = volume.length;
		
		for (int x = 0; x < volumeWidth; x++) {
				volume[x] = convolve(volume[x], kernel, isComplex);
		}
		return volume;
	}

	public double[][] convolve(double[][] image, double[][] kernel, boolean isComplex) {
		
		int imageWidth = image.length;
		int imageHeight = image[0].length;
		
		int kernelWidth = kernel.length;
		int kernelHeight = kernel[0].length;
				
		StockhamGPU s = new StockhamGPU();
		int newWidth, newHeight;
		if (isComplex) {
			newWidth = imageWidth+2*kernelWidth;
			newHeight = imageHeight+2*kernelHeight;			
		} else {
			newWidth = 2*(imageWidth+2*kernelWidth);
			newHeight = 2*(imageHeight+2*kernelHeight);	
		}

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
		
		for (int x = 0; x < newWidth; x++) {
			s.fft(paddedReal[x], paddedImag[x], true);
			s.fft(paddedKernel[x], paddedKernelImag[x], true);
		}
		paddedReal =  ds.shiftDim(paddedReal);
		paddedImag = ds.shiftDim(paddedImag);
		paddedKernel = ds.shiftDim(paddedKernel);
		paddedKernelImag = ds.shiftDim(paddedKernelImag);
		for (int y = 0; y < newHeight; y++) {
			s.fft(paddedReal[y], paddedImag[y], true);
			s.fft(paddedKernel[y], paddedKernelImag[y], true);
		}
		
		for (int x = 0; x < newWidth; x++) {
			for (int y = 0; y < newHeight; y++) {
					paddedReal[x][y] *= paddedKernel[x][y];
					paddedImag[x][y] *= paddedKernelImag[x][y];
			}
		}

		for (int y = 0; y < newHeight; y++) {
			s.fft(paddedReal[y], paddedImag[y], false);
		}
		paddedReal =  ds.shiftDim(paddedReal);
		paddedImag = ds.shiftDim(paddedImag);
		for (int x = 0; x < newWidth; x++) {
			s.fft(paddedReal[x], paddedImag[x], false);
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
		return result;
		
	}
		
	public double[][][] convolve(double[][][] volume, double[][][] kernel, boolean isComplex) {
		
		int volumeWidth = volume.length;
		int volumeHeight = volume[0].length;
		int volumeDepth = volume[0][0].length;
		
		int kernelWidth = kernel.length;
		int kernelHeight = kernel[0].length;
		int kernelDepth = kernel[0][0].length;
				
		StockhamGPU s = new StockhamGPU();
		int newWidth, newHeight, newDepth;
		if (isComplex) {
			newWidth = volumeWidth + 2*kernelWidth;
			newHeight = volumeHeight + 2*kernelHeight;		
			newDepth = volumeDepth + 2*kernelDepth;
		} else {
			newWidth = 2*(volumeWidth+2*kernelWidth);
			newHeight = 2*(volumeHeight+2*kernelHeight);	
			newDepth = 2*(volumeDepth+2*kernelDepth);
		}

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

		paddedReal =  ds.shiftDim(paddedReal);
		paddedImag = ds.shiftDim(paddedImag);
		paddedKernel = ds.shiftDim(paddedKernel);
		paddedKernelImag = ds.shiftDim(paddedKernelImag);

		for (int y = 0; y < newHeight; y++) {
			for (int z = 0; z < newDepth; z++) {
				s.fft(paddedReal[y][z], paddedImag[y][z], true);
				s.fft(paddedKernel[y][z], paddedKernelImag[y][z], true);
			}
		}

		paddedReal =  ds.shiftDim(paddedReal);
		paddedImag = ds.shiftDim(paddedImag);
		paddedKernel = ds.shiftDim(paddedKernel);
		paddedKernelImag = ds.shiftDim(paddedKernelImag);

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

		paddedReal =  ds.shiftDim(paddedReal, 2);
		paddedImag = ds.shiftDim(paddedImag, 2);

		for (int y = 0; y < newHeight; y++) {
			for (int z = 0; z < newDepth; z++) {
				s.fft(paddedReal[y][z], paddedImag[y][z], false);
			}
		}

		paddedReal =  ds.shiftDim(paddedReal, 2);
		paddedImag = ds.shiftDim(paddedImag, 2);

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
	
	int nextPwr2(int length) {
		
		int pwr2Length = 1;
		do {
			pwr2Length *= 2;
		} while(pwr2Length < length);
		return pwr2Length;
	}


		
}
