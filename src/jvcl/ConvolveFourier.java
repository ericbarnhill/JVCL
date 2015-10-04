package jvcl;

import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_2D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_3D;

public class ConvolveFourier {
	
	DimShifter ds;
	
	public ConvolveFourier() {
		ds = new DimShifter();
	}
	
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
				result[x] = paddedVector[x+kernelHeight];
			} else {
				result[x] = paddedVector[x*2+kernelHeight*2];
			}
		}		
		return result;
		
	}
	
	public double[][] convolveFT(double[][] image, double[] kernel, boolean isComplex, int dim) {
		if (dim > 1) throw new RuntimeException("Invalid dim");
		if (dim == 0) image = ds.shiftDim(image);
		int height = image.length;
		for (int n = 0; n < height; n++) {
			image[n] = convolveFT(image[n], kernel, isComplex);
		}
		if (dim == 0) {
			return ds.shiftDim(image);
		} else {
			return image;
		}
	}
	
	public double[][] convolveFT(double[][] image, double[] kernel, boolean isComplex) {
		return convolveFT(image, kernel, isComplex, 0);
	}
	
	public double[][][] convolveFT(double[][][] volume, double[] kernel, boolean isComplex, int dim) {
		if (dim == 0) volume = ds.shiftDim(volume, 2);
		if (dim == 1) volume = ds.shiftDim(volume, 1);
		if (dim > 2) throw new RuntimeException("Invalid dim");
		
		int volumeWidth = volume.length;
		int volumeHeight = volume[0].length;
		
		for (int x = 0; x < volumeWidth; x++) {
			for (int y = 0; y < volumeHeight; y++) {
				volume[x][y] = convolveFT(volume[x][y], kernel, isComplex);
			}
		}
		return volume;
		
	}
	
	public double[][][] convolveFT(double[][][] volume, double[] kernel, boolean isComplex) {
		return convolveFT(volume, kernel, isComplex, 0);
	}
		
	public double[][] convolveFT(double[][] image, double[][] kernel, boolean isComplex) {
		
		int imageWidth = image.length;
		int imageHeight = image[0].length;
		
		int kernelWidth = kernel.length;
		int kernelHeight = kernel[0].length;
				
		DoubleFFT_2D fft = new DoubleFFT_2D(imageWidth+2*kernelWidth, imageHeight+2*kernelHeight);
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
		
		fft.complexForward(paddedImage);
		fft.complexForward(paddedKernel);
		
		int paddedWidth = paddedImage.length;
		int paddedHeight = paddedImage[0].length;
		
		for (int x = 0; x < paddedWidth; x++) {
			for (int y = 0; y < paddedHeight; y++) {
					paddedImage[x][y] *= paddedKernel[x][y];
			}
		}
		
		fft.complexInverse(paddedImage, true);
		
		double[][] result = new double[imageWidth][imageHeight];
		for (int x = 0; x < imageWidth; x++) {
			for (int y = 0; y < imageHeight; y++) {
				if (isComplex) {
					result[x][y] = paddedImage[x+kernelWidth][y+kernelHeight];
				} else {
					result[x][y] = paddedImage[x*2+kernelWidth*2][y*2+kernelHeight*2];
				}
			}
		}		
		return result;
		
	}
		
	public double[][][] convolveFT(double[][][] volume, double[][] kernel, boolean isComplex, int dim) {
		if (dim == 0) volume = ds.shiftDim(volume, 2);
		if (dim == 1) volume = ds.shiftDim(volume, 1);
		if (dim > 2) throw new RuntimeException("Invalid dim");
		
		int volumeWidth = volume.length;
		
		for (int x = 0; x < volumeWidth; x++) {
				volume[x] = convolveFT(volume[x], kernel, isComplex);
		}
		return volume;
	}
	
	public double[][][] convolveFT(double[][][] volume, double[][] kernel, boolean isComplex) {
		return convolveFT(volume, kernel, isComplex, 0);
	}
	
	public double[][][] convolveFT(double[][][] volume, double[][][] kernel, boolean isComplex) {
		
		int volumeWidth = volume.length;
		int volumeHeight = volume[0].length;
		int volumeDepth = volume[0][0].length;
		
		int kernelWidth = kernel.length;
		int kernelHeight = kernel[0].length;
		int kernelDepth = kernel[0][0].length;
				
		DoubleFFT_3D fft = new DoubleFFT_3D(volumeWidth+2*kernelWidth, volumeHeight+2*kernelHeight, volumeDepth+2*kernelDepth);
		double[][][] paddedVolume;
		double[][][] paddedKernel;
		
		if (isComplex) {
			paddedVolume = new double[volumeWidth+2*kernelWidth][volumeHeight+2*kernelHeight][volumeDepth+2*kernelDepth];
			paddedKernel = new double[volumeWidth+2*kernelWidth][volumeHeight+2*kernelHeight][volumeDepth+2*kernelDepth];
		} else {
			paddedVolume = new double[2*(volumeWidth+2*kernelWidth)][2*(volumeHeight+2*kernelHeight)][2*(volumeDepth+2*kernelDepth)];
			paddedKernel = new double[2*(volumeWidth+2*kernelWidth)][2*(volumeHeight+2*kernelHeight)][2*(volumeDepth+2*kernelDepth)];
		}
		
		for (int x = 0; x < volumeWidth; x++) {
			for (int y = 0; y < volumeHeight; y++) {
				for (int z = 0; z < volumeDepth; z++) {
					if (isComplex) {
						paddedVolume[x+kernelWidth][y+kernelHeight][z+kernelDepth] = volume[x][y][z];
					} else 
						paddedVolume[x*2+kernelWidth*2][y*2+kernelHeight*2][z*2+kernelDepth*2] = volume[x][y][z];
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
		
		fft.complexForward(paddedVolume);
		fft.complexForward(paddedKernel);
		
		int paddedWidth = paddedVolume.length;
		int paddedHeight = paddedVolume[0].length;
		int paddedDepth = paddedVolume[0][0].length;
		
		for (int x = 0; x < paddedWidth; x++) {
			for (int y = 0; y < paddedHeight; y++) {
				for (int z = 0; z < paddedDepth; z++) {
					paddedVolume[x][y][z] *= paddedKernel[x][y][z];
				}
			}
		}
		
		fft.complexInverse(paddedVolume, true);
		
		double[][][] result = new double[volumeWidth][volumeHeight][volumeDepth];
		for (int x = 0; x < volumeWidth; x++) {
			for (int y = 0; y < volumeHeight; y++) {
				for (int z = 0; z < volumeDepth; z++) {
					if (isComplex) {
						result[x][y][z] = paddedVolume[x+kernelWidth][y+kernelHeight][z+kernelDepth];
					} else 
						result[x][y][z] = paddedVolume[x*2+kernelWidth*2][y*2+kernelHeight*2][z*2+kernelDepth*2];
				}
			}
		}		
		return result;
		
	}

}
