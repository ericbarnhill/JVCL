package com.ericbarnhill.jvcl;

abstract class ConvolverDouble extends Convolver<Double> {

    public ConvolverDouble() {
        super();
    }

    abstract double[] convolve(double[] f, double[] g);
    abstract double[][] convolve(double[][] f, double[] g);
    abstract double[][][] convolve(double[][][] f, double[] g);
    abstract double[][] convolve(double[][] f, double[][] g);
    abstract double[][] convolve(double[][][] f, double[][] g);
    abstract double[][][] convolve(double[][][] f, double[][][] g);

	public static double[] zeroPadBoundaries(double[] array, int paddingF, int paddingB) {
		int length = array.length;
		int newLength = length + paddingF + paddingB;
		double[] paddedArray = new double[newLength];
		for (int n = 0; n < length; n++) {
			paddedArray[paddingF+n] = array[n];
		}
		return paddedArray;
	}

	public static double[] zeroPadBoundaries(double[] array, int padding) {
		return zeroPadBoundaries(array, padding, padding);
	}

	public static double[][] zeroPadBoundaries(double[][] image, int paddingXF,
		int paddingXB, int paddingYF, int paddingYB) {
		int width = image.length;
		int height = image[0].length;
		int newWidth = width + paddingXF + paddingXB;
		int newHeight = height + paddingYF + paddingYB;
		double[][] paddedImage = new double[newWidth][newHeight];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				paddedImage[x+paddingXF][y+paddingYF] = image[x][y];
			}
		}
		return paddedImage;
	}

	public static double[][] zeroPadBoundaries(double[][] image, int paddingX, int paddingY) {
		return zeroPadBoundaries(image, paddingX, paddingX, paddingY, paddingY);
	}


	public static double[][] zeroPadBoundaries(double[][] image, int padding) {
		return zeroPadBoundaries(image, padding, padding);
	}

	public static double[][][] zeroPadBoundaries(double[][][] volume, int paddingXF,
			int paddingXB, int paddingYF, int paddingYB, int paddingZF, int paddingZB) {
		int width = volume.length;
		int height = volume[0].length;
		int depth = volume[0][0].length;
		int newWidth = width + paddingXF + paddingXB;
		int newHeight = height + paddingYF + paddingYB;
		int newDepth = depth + paddingZF + paddingZB;
		double[][][] paddedImage = new double[newWidth][newHeight][newDepth];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				for (int z = 0; z < depth; z++) {
					paddedImage[x+paddingXF][y+paddingYF][z+paddingZF] = volume[x][y][z];
				}
			}
		}
		return paddedImage;
	}

	public static double[][][] zeroPadBoundaries(double[][][] volume, int paddingX,
			int paddingY, int paddingZ) {
		return zeroPadBoundaries(volume, paddingX, paddingX, paddingY, paddingY, paddingZ, paddingZ);
	}

	public static double[][][] zeroPadBoundaries(double[][][] volume, int padding) {
		return zeroPadBoundaries(volume, padding, padding, padding);
	}

	public static double[][][] stripBorderPadding(double[][][] volume, int paddingX, int paddingY, int paddingZ) {
		int width = volume.length;
		int height = volume[0].length;
		int depth = volume[0][0].length;
		int newWidth = width - 2*paddingX;
		int newHeight = height - 2*paddingY;
		int newDepth = depth - 2*paddingZ;
		double[][][] paddedImage = new double[newWidth][newHeight][newDepth];
		for (int x = 0; x < newWidth; x++) {
			for (int y = 0; y < newHeight; y++) {
				for (int z = 0; z < newDepth; z++) {
					paddedImage[x][y][z] = volume[x+paddingX][y+paddingY][z+paddingZ];
				}
			}
		}
		return paddedImage;
	}

	public static double[][][] stripBorderPadding(double[][][] volume, int padding) {
		return stripBorderPadding(volume, padding, padding, padding);
	}

	public static double[] stripEndPadding(double[] array, int newLength) {
		double[] strippedArray = new double[newLength];
		if (newLength > array.length) {
			throw new RuntimeException("stripped length cannot be longer than array length");
		}
		for (int n = 0; n < newLength; n++) {
			strippedArray[n] = array[n];
		}
		return strippedArray;
	}


	public static double[][] stripEndPadding(double[][] array, int newWidth, int newHeight) {
		double[][] strippedArray = new double[newWidth][newHeight];
		for (int x = 0; x < newWidth; x++) {
			for (int y = 0; y < newHeight; y++) {
				strippedArray[x][y] = array[x][y];
			}
		}
		return strippedArray;
	}

	public static double[][][] stripEndPadding(double[][][] array, int newWidth, int newHeight, int newDepth) {
		int width = array.length;
		int height = array[0].length;
		int depth = array[0][0].length;
		double[][][] strippedArray = new double[newWidth][newHeight][newDepth];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				for (int z = 0; z < depth; z++) {
					strippedArray[x][y][z] = array[x][y][z];
				}
			}
		}
		return strippedArray;
	}

}
