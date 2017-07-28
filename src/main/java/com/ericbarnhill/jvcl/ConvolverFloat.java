package com.ericbarnhill.jvcl;

abstract class ConvolverFloat<T> extends Convolver<T> {

    public ConvolverFloat() {
        super();
    }

    abstract float[] convolve(float[] f, float[] g);
    abstract float[][] convolve(float[][] f, float[] g);
    abstract float[][][] convolve(float[][][] f, float[] g);
    abstract float[][] convolve(float[][] f, float[][] g);
    abstract float[][] convolve(float[][][] f, float[][] g);
    abstract float[][][] convolve(float[][][] f, float[][][] g);

	public static float[] zeroPadBoundaries(float[] array, int paddingF, int paddingB) {
		int length = array.length;
		int newLength = length + paddingF + paddingB;
		float[] paddedArray = new float[newLength];
		for (int n = 0; n < length; n++) {
			paddedArray[paddingF+n] = array[n];
		}
		return paddedArray;
	}

	public static float[] zeroPadBoundaries(float[] array, int padding) {
		return zeroPadBoundaries(array, padding, padding);
	}

	public static float[][] zeroPadBoundaries(float[][] image, int paddingXF,
		int paddingXB, int paddingYF, int paddingYB) {
		int width = image.length;
		int height = image[0].length;
		int newWidth = width + paddingXF + paddingXB;
		int newHeight = height + paddingYF + paddingYB;
		float[][] paddedImage = new float[newWidth][newHeight];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				paddedImage[x+paddingXF][y+paddingYF] = image[x][y];
			}
		}
		return paddedImage;
	}

	public static float[][] zeroPadBoundaries(float[][] image, int paddingX, int paddingY) {
		return zeroPadBoundaries(image, paddingX, paddingX, paddingY, paddingY);
	}


	public static float[][] zeroPadBoundaries(float[][] image, int padding) {
		return zeroPadBoundaries(image, padding, padding);
	}

	public static float[][][] zeroPadBoundaries(float[][][] volume, int paddingXF,
			int paddingXB, int paddingYF, int paddingYB, int paddingZF, int paddingZB) {
		int width = volume.length;
		int height = volume[0].length;
		int depth = volume[0][0].length;
		int newWidth = width + paddingXF + paddingXB;
		int newHeight = height + paddingYF + paddingYB;
		int newDepth = depth + paddingZF + paddingZB;
		float[][][] paddedImage = new float[newWidth][newHeight][newDepth];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				for (int z = 0; z < depth; z++) {
					paddedImage[x+paddingXF][y+paddingYF][z+paddingZF] = volume[x][y][z];
				}
			}
		}
		return paddedImage;
	}

	public static float[][][] zeroPadBoundaries(float[][][] volume, int paddingX,
			int paddingY, int paddingZ) {
		return zeroPadBoundaries(volume, paddingX, paddingX, paddingY, paddingY, paddingZ, paddingZ);
	}

	public static float[][][] zeroPadBoundaries(float[][][] volume, int padding) {
		return zeroPadBoundaries(volume, padding, padding, padding);
	}

	public static float[][][] stripBorderPadding(float[][][] volume, int paddingX, int paddingY, int paddingZ) {
		int width = volume.length;
		int height = volume[0].length;
		int depth = volume[0][0].length;
		int newWidth = width - 2*paddingX;
		int newHeight = height - 2*paddingY;
		int newDepth = depth - 2*paddingZ;
		float[][][] paddedImage = new float[newWidth][newHeight][newDepth];
		for (int x = 0; x < newWidth; x++) {
			for (int y = 0; y < newHeight; y++) {
				for (int z = 0; z < newDepth; z++) {
					paddedImage[x][y][z] = volume[x+paddingX][y+paddingY][z+paddingZ];
				}
			}
		}
		return paddedImage;
	}

	public static float[][][] stripBorderPadding(float[][][] volume, int padding) {
		return stripBorderPadding(volume, padding, padding, padding);
	}

	public static float[] stripEndPadding(float[] array, int newLength) {
		float[] strippedArray = new float[newLength];
		if (newLength > array.length) {
			throw new RuntimeException("stripped length cannot be longer than array length");
		}
		for (int n = 0; n < newLength; n++) {
			strippedArray[n] = array[n];
		}
		return strippedArray;
	}


	public static float[][] stripEndPadding(float[][] array, int newWidth, int newHeight) {
		float[][] strippedArray = new float[newWidth][newHeight];
		for (int x = 0; x < newWidth; x++) {
			for (int y = 0; y < newHeight; y++) {
				strippedArray[x][y] = array[x][y];
			}
		}
		return strippedArray;
	}

	public static float[][][] stripEndPadding(float[][][] array, int newWidth, int newHeight, int newDepth) {
		int width = array.length;
		int height = array[0].length;
		int depth = array[0][0].length;
		float[][][] strippedArray = new float[newWidth][newHeight][newDepth];
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
