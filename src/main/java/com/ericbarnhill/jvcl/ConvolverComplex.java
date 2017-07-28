package com.ericbarnhill.jvcl;

import org.apache.commons.numbers.complex.Complex;

abstract class ConvolverComplex<T> extends Convolver<T> {

    public ConvolverComplex() {
            super();
    }

    abstract Complex[] convolve(Complex[] f, Complex[] g);
    abstract Complex[][] convolve(Complex[][] f, Complex[] g);
    abstract Complex[][][] convolve(Complex[][][] f, Complex[] g);
    abstract Complex[][] convolve(Complex[][] f, Complex[][] g);
    abstract Complex[][] convolve(Complex[][][] f, Complex[][] g);
    abstract Complex[][][] convolve(Complex[][][] f, Complex[][][] g);

	public static Complex[] zeroPadBoundaries(Complex[] array, int paddingF, int paddingB) {
		int length = array.length;
		int newLength = length + paddingF + paddingB;
		Complex[] paddedArray = new Complex[newLength];
		for (int n = 0; n < length; n++) {
			paddedArray[paddingF+n] = array[n];
		}
		return paddedArray;
	}

	public static Complex[] zeroPadBoundaries(Complex[] array, int padding) {
		return zeroPadBoundaries(array, padding, padding);
	}

	public static Complex[][] zeroPadBoundaries(Complex[][] image, int paddingXF,
		int paddingXB, int paddingYF, int paddingYB) {
		int width = image.length;
		int height = image[0].length;
		int newWidth = width + paddingXF + paddingXB;
		int newHeight = height + paddingYF + paddingYB;
		Complex[][] paddedImage = new Complex[newWidth][newHeight];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				paddedImage[x+paddingXF][y+paddingYF] = image[x][y];
			}
		}
		return paddedImage;
	}

	public static Complex[][] zeroPadBoundaries(Complex[][] image, int paddingX, int paddingY) {
		return zeroPadBoundaries(image, paddingX, paddingX, paddingY, paddingY);
	}


	public static Complex[][] zeroPadBoundaries(Complex[][] image, int padding) {
		return zeroPadBoundaries(image, padding, padding);
	}

	public static Complex[][][] zeroPadBoundaries(Complex[][][] volume, int paddingXF,
			int paddingXB, int paddingYF, int paddingYB, int paddingZF, int paddingZB) {
		int width = volume.length;
		int height = volume[0].length;
		int depth = volume[0][0].length;
		int newWidth = width + paddingXF + paddingXB;
		int newHeight = height + paddingYF + paddingYB;
		int newDepth = depth + paddingZF + paddingZB;
		Complex[][][] paddedImage = new Complex[newWidth][newHeight][newDepth];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				for (int z = 0; z < depth; z++) {
					paddedImage[x+paddingXF][y+paddingYF][z+paddingZF] = volume[x][y][z];
				}
			}
		}
		return paddedImage;
	}

	public static Complex[][][] zeroPadBoundaries(Complex[][][] volume, int paddingX,
			int paddingY, int paddingZ) {
		return zeroPadBoundaries(volume, paddingX, paddingX, paddingY, paddingY, paddingZ, paddingZ);
	}

	public static Complex[][][] zeroPadBoundaries(Complex[][][] volume, int padding) {
		return zeroPadBoundaries(volume, padding, padding, padding);
	}

}
