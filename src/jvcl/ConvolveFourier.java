package jvcl;

import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_2D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_3D;

public class ConvolveFourier {
	
	public ConvolveFourier() {}
	
	public double[] convolveFT(double[] vector, double[] kernel, boolean isComplex) {
		
		int vectorHeight = vector.length;
		int kernelHeight = kernel.length;
				
		DoubleFFT_1D fft = new DoubleFFT_1D(vectorHeight+2*kernelHeight);
		double[] paddedVector;
		double[] paddedKernel;
		
		if (isComplex) {
			paddedVector = new double[vectorHeight+2*kernelHeight];
			paddedKernel = new double[vectorHeight+2*kernelHeight];
		} else {
			paddedVector = new double[2*(vectorHeight+2*kernelHeight)];
			paddedKernel = new double[2*(vectorHeight+2*kernelHeight)];
		}
		
		for (int x = 0; x < vectorHeight; x++) {
			if (isComplex) {
				paddedVector[x+kernelHeight] = vector[x];
			} else {
				paddedVector[x*2+kernelHeight*2] = vector[x];
			}
		}
		
		for (int x = 0; x < kernelHeight; x++) {
			if (isComplex) {
				paddedKernel[x+kernelHeight] = kernel[x];
			} else {
				paddedKernel[x*2+kernelHeight*2] = kernel[x];
			}
		}
		
		fft.complexForward(paddedVector);
		fft.complexForward(paddedKernel);
		
		int paddedHeight = paddedVector.length;
		
		for (int x = 0; x < paddedHeight; x++) {
			paddedVector[x] *= paddedKernel[x];
		}
		
		fft.complexInverse(paddedVector, true);
		
		double[] result = new double[vectorHeight];
		for (int x = 0; x < vectorHeight; x++) {
			if (isComplex) {
				vector[x] = paddedVector[x+kernelHeight];
			} else {
				vector[x] = paddedVector[x*2+kernelHeight*2];
			}
		}		
		return result;
		
	}
	
	public double[][] convolveFT(double[][] image, double[][] kernel, boolean isComplex) {
		
		int imageHeight = image.length;
		int imageWidth = image[0].length;
		
		int kernelHeight = kernel.length;
		int kernelWidth = kernel[0].length;
				
		DoubleFFT_2D fft = new DoubleFFT_2D(imageHeight+2*kernelHeight, imageWidth+2*kernelWidth);
		double[][] paddedImage;
		double[][] paddedKernel;
		
		if (isComplex) {
			paddedImage = new double[imageHeight+2*kernelHeight][imageWidth+2*kernelWidth];
			paddedKernel = new double[imageHeight+2*kernelHeight][imageWidth+2*kernelWidth];
		} else {
			paddedImage = new double[2*(imageHeight+2*kernelHeight)][2*(imageWidth+2*kernelWidth)];
			paddedKernel = new double[2*(imageHeight+2*kernelHeight)][2*(imageWidth+2*kernelWidth)];
		}
		
		for (int x = 0; x < imageHeight; x++) {
			for (int y = 0; y < imageWidth; y++) {
				if (isComplex) {
					paddedImage[x+kernelHeight][y+kernelWidth] = image[x][y];
				} else {
					paddedImage[x*2+kernelHeight*2][y*2+kernelWidth*2] = image[x][y];
				}
			}
		}
		
		for (int x = 0; x < kernelHeight; x++) {
			for (int y = 0; y < kernelWidth; y++) {
				if (isComplex) {
					paddedKernel[x+kernelHeight][y+kernelWidth] = kernel[x][y];
				} else {
					paddedKernel[x*2+kernelHeight*2][y*2+kernelWidth*2] = kernel[x][y];
				}
			}
		}
		
		fft.complexForward(paddedImage);
		fft.complexForward(paddedKernel);
		
		int paddedHeight = paddedImage.length;
		int paddedWidth = paddedImage[0].length;
		
		for (int x = 0; x < paddedHeight; x++) {
			for (int y = 0; y < paddedWidth; y++) {
					paddedImage[x][y] *= paddedKernel[x][y];
			}
		}
		
		fft.complexInverse(paddedImage, true);
		
		double[][] result = new double[imageHeight][imageWidth];
		for (int x = 0; x < imageHeight; x++) {
			for (int y = 0; y < imageWidth; y++) {
				if (isComplex) {
					image[x][y] = paddedImage[x+kernelHeight][y+kernelWidth];
				} else {
					image[x][y] = paddedImage[x*2+kernelHeight*2][y*2+kernelWidth*2];
				}
			}
		}		
		return result;
		
	}
		
	public double[][][] convolveFT(double[][][] volume, double[][][] kernel, boolean isComplex) {
		
		int volumeHeight = volume.length;
		int volumeWidth = volume[0].length;
		int volumeDepth = volume[0][0].length;
		
		int kernelHeight = kernel.length;
		int kernelWidth = kernel[0].length;
		int kernelDepth = kernel[0][0].length;
				
		DoubleFFT_3D fft = new DoubleFFT_3D(volumeHeight+2*kernelHeight, volumeWidth+2*kernelWidth, volumeDepth+2*kernelDepth);
		double[][][] paddedVolume;
		double[][][] paddedKernel;
		
		if (isComplex) {
			paddedVolume = new double[volumeHeight+2*kernelHeight][volumeWidth+2*kernelWidth][volumeDepth+2*kernelDepth];
			paddedKernel = new double[volumeHeight+2*kernelHeight][volumeWidth+2*kernelWidth][volumeDepth+2*kernelDepth];
		} else {
			paddedVolume = new double[2*(volumeHeight+2*kernelHeight)][2*(volumeWidth+2*kernelWidth)][2*(volumeDepth+2*kernelDepth)];
			paddedKernel = new double[2*(volumeHeight+2*kernelHeight)][2*(volumeWidth+2*kernelWidth)][2*(volumeDepth+2*kernelDepth)];
		}
		
		for (int x = 0; x < volumeHeight; x++) {
			for (int y = 0; y < volumeWidth; y++) {
				for (int z = 0; z < volumeDepth; z++) {
					if (isComplex) {
						paddedVolume[x+kernelHeight][y+kernelWidth][z+kernelDepth] = volume[x][y][z];
					} else 
						paddedVolume[x*2+kernelHeight*2][y*2+kernelWidth*2][z*2+kernelDepth*2] = volume[x][y][z];
				}
			}
		}
		
		for (int x = 0; x < kernelHeight; x++) {
			for (int y = 0; y < kernelWidth; y++) {
				for (int z = 0; z < kernelDepth; z++) {
					if (isComplex) {
						paddedKernel[x+kernelHeight][y+kernelWidth][z+kernelDepth] = kernel[x][y][z];
					} else {
						paddedKernel[x*2+kernelHeight*2][y*2+kernelWidth*2][z*2+kernelDepth*2] = kernel[x][y][z];
					}
				}
			}
		}
		
		fft.complexForward(paddedVolume);
		fft.complexForward(paddedKernel);
		
		int paddedHeight = paddedVolume.length;
		int paddedWidth = paddedVolume[0].length;
		int paddedDepth = paddedVolume[0][0].length;
		
		for (int x = 0; x < paddedHeight; x++) {
			for (int y = 0; y < paddedWidth; y++) {
				for (int z = 0; z < paddedDepth; z++) {
					paddedVolume[x][y][z] *= paddedKernel[x][y][z];
				}
			}
		}
		
		fft.complexInverse(paddedVolume, true);
		
		double[][][] result = new double[volumeHeight][volumeWidth][volumeDepth];
		for (int x = 0; x < volumeHeight; x++) {
			for (int y = 0; y < volumeWidth; y++) {
				for (int z = 0; z < volumeDepth; z++) {
					if (isComplex) {
						volume[x][y][z] = paddedVolume[x+kernelHeight][y+kernelWidth][z+kernelDepth];
					} else 
						volume[x][y][z] = paddedVolume[x*2+kernelHeight*2][y*2+kernelWidth*2][z*2+kernelDepth*2];
				}
			}
		}		
		return result;
		
	}

}
