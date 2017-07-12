/*
 * (c) Eric Barnhill 2016 All Rights Reserved.
 *
 * This file is part of the Java Volumetric Convolution Library (JVCL). JVCL is free software:
 * you can redistribute it and/or modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * JVCL is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details. You should have received a copy of
 * the GNU General Public License along with JVCL.  If not, see http://www.gnu.org/licenses/ .
 *
 * This code uses software from the Apache Software Foundation.
 * The Apache Software License can be found at: http://www.apache.org/licenses/LICENSE-2.0.txt .
 *
 * This code uses software from the JogAmp project.
 * Jogamp information and software license can be found at: https://jogamp.org/ .
 *
 * This code uses methods from the JTransforms package by Piotr Wendykier.
 * JTransforms information and software license can be found at: https://github.com/wendykierp/JTransforms .
 *
 */

package com.ericbarnhill.jvcl;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import org.apache.commons.numbers.complex.Complex;
import org.apache.commons.numbers.complex.ComplexUtils;

import com.ericbarnhill.arrayMath.ArrayMath;


/**
 * This collection of static utility methods contains several methods called by the convolution classes,
 * and several methods that may be convenient for the convolution calls of users. More documentation of these
 * methods can be added on request.
*/
public class JVCLUtils {

	/**
	 * Flips dimensions of 2D image without leaving 1D representation
	 * @param image {@code float[]} vectorized image
	 * @param width int first depth level dimension
	 * @return {@code float[]} with dimensions swapped
	 */
	public static float[] shiftVectorDim(float[] image, int width) {
		int length = image.length;
		int height = length / width;
		float[] result = new float[length];
		int x, y;
		for (int n = 0; n < length; n++) {
			x = n % width;
			y = n / width;
			result[x*height + y] = image[y*width + x];
		}
		return result;
	}

	/**
	 * A standard utility method for OpenCL thread and block allocation
	 * @param groupSize
	 * @param globalSize
	 * @return minimum required global size for group size
	 */
	public static int roundUp(int groupSize, int globalSize) {
        int r = globalSize % groupSize;
        if (r == 0) {
            return globalSize;
        } else {
            return globalSize + groupSize - r;
        }
    }

    public static String readFile(String fileName) {
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

	public static int nextPwr2(int length) {

		int pwr2Length = 1;
		while(pwr2Length < length) {
			pwr2Length *= 2;
		}
		return pwr2Length;
	}

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

	// FLOAT

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

	// COMPLEX

	public static Complex[] zeroPadBoundaries(Complex[] array, int paddingF, int paddingB) {
		int length = array.length;
		int newLength = length + paddingF + paddingB;
		Complex[] paddedArray = ComplexUtils.initialize(new Complex[newLength]);
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
		Complex[][] paddedImage = ComplexUtils.initialize(new Complex[newWidth][newHeight]);
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
		Complex[][][] paddedImage = ComplexUtils.initialize(new Complex[newWidth][newHeight][newDepth]);
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

	public static double[] stripBorderPadding(double[] array, int padding) {
		int length = array.length;
		int newLength = length - 2*padding;
		double[] strippedArray = new double[newLength];
		for (int n = 0; n < newLength; n++) {
			strippedArray[n] = array[padding+n];
		}
		return strippedArray;
	}

	public static Complex[] stripBorderPadding(Complex[] array, int padding) {
		int length = array.length;
		int newLength = length - 2*padding;
		Complex[] strippedArray = new Complex[newLength];
		for (int n = 0; n < newLength; n++) {
			strippedArray[n] = array[padding+n];
		}
		return strippedArray;
	}

	public static Complex[] stripBorderPadding(Complex[] array, int paddingF, int paddingB) {
		int length = array.length;
		int newLength = length - paddingF - paddingB;
		Complex[] strippedArray = new Complex[newLength];
		for (int n = 0; n < newLength; n++) {
			strippedArray[n] = array[paddingF+n];
		}
		return strippedArray;
	}

	public static double[][] stripBorderPadding(double[][] image, int paddingX, int paddingY) {
		int width = image.length;
		int height = image[0].length;
		int newWidth = width - 2*paddingX;
		int newHeight = height - 2*paddingY;
		double[][] paddedImage = new double[newWidth][newHeight];
		for (int x = 0; x < newWidth; x++) {
			for (int y = 0; y < newHeight; y++) {
				paddedImage[x][y] = image[x+paddingX][y+paddingY];
			}
		}
		return paddedImage;
	}

	public static double[][] stripBorderPadding(double[][] image, int padding) {
		return stripBorderPadding(image, padding, padding);
	}

	public static Complex[][] stripBorderPadding(Complex[][] image, int paddingX, int paddingY) {
		int width = image.length;
		int height = image[0].length;
		int newWidth = width - 2*paddingX;
		int newHeight = height - 2*paddingY;
		Complex[][] paddedImage = new Complex[newWidth][newHeight];
		for (int x = 0; x < newWidth; x++) {
			for (int y = 0; y < newHeight; y++) {
				paddedImage[x][y] = image[x+paddingX][y+paddingY];
			}
		}
		return paddedImage;
	}

	public static Complex[][] stripBorderPadding(Complex[][] image, int paddingFX, int paddingBX, int paddingFY, int paddingBY) {
		int width = image.length;
		int height = image[0].length;
		int newWidth = width - paddingFX - paddingBX;
		int newHeight = height - paddingFY - paddingBY;
		Complex[][] paddedImage = new Complex[newWidth][newHeight];
		for (int x = 0; x < newWidth; x++) {
			for (int y = 0; y < newHeight; y++) {
				paddedImage[x][y] = image[x+paddingFX][y+paddingFY];
			}
		}
		return paddedImage;
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

	public static Complex[] stripEndPadding(Complex[] array, int newLength) {
		Complex[] strippedArray = new Complex[newLength];
		if (newLength > array.length) {
			throw new RuntimeException("stripped length cannot be longer than array length");
		}
		for (int n = 0; n < newLength; n++) {
			strippedArray[n] = array[n];
		}
		return strippedArray;
	}

	public static Complex[][] stripEndPadding(Complex[][] array, int newWidth, int newHeight) {
		Complex[][] strippedArray = new Complex[newWidth][newHeight];
		for (int x = 0; x < newWidth; x++) {
			for (int y = 0; y < newHeight; y++) {
				strippedArray[x][y] = array[x][y];
			}
		}
		return strippedArray;
	}

	public static float[] stripEndPadding(float[] array, int newLength) {
		float[] strippedArray = new float[newLength];
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

	public static float[][] stripEndPadding(float[][] array, int newWidth, int newHeight) {
		float[][] strippedArray = new float[newWidth][newHeight];
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

	public static double[] deepCopy(double[] array) {
		int length = array.length;
		double[] copy = new double[length];
		for (int n = 0; n < length; n++) {
			copy[n] = array[n];
		}
		return copy;
	}

	public static Complex[] deepCopy(Complex[] array) {
		int length = array.length;
		Complex[] copy = new Complex[length];
		for (int n = 0; n < length; n++) {
			copy[n] = new Complex(array[n].getReal(), array[n].getImaginary());
		}
		return copy;
	}

	public static double[][] deepCopy(double[][] array) {
		int width = array.length;
		int height = array[0].length;
		double[][] copy = new double[width][height];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				copy[x][y] = array[x][y];
			}
		}
		return copy;
	}

	public static Complex[][] deepCopy(Complex[][] array) {
		int width = array.length;
		int height = array[0].length;
		Complex[][] copy = new Complex[width][height];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				copy[x][y] = new Complex(array[x][y].getReal(), array[x][y].getImaginary());
			}
		}
		return copy;
	}

	public static double[][][] deepCopy(double[][][] array) {
		int width = array.length;
		int height = array[0].length;
		int depth = array[0][0].length;
		double[][][] copy = new double[width][height][depth];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				for (int z = 0; z < depth; z++) {
					copy[x][y][z] = array[x][y][z];
				}
			}
		}
		return copy;
	}

	public static Complex[][][] deepCopy(Complex[][][] array) {
		int width = array.length;
		int height = array[0].length;
		int depth = array[0][0].length;
		Complex[][][] copy = new Complex[width][height][depth];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				for (int z = 0; z < depth; z++) {
					copy[x][y][z] = new Complex(array[x][y][z].getReal(), array[x][y][z].getImaginary());
				}
			}
		}
		return copy;
	}

	public static float[] deepCopy(float[] array) {
		int length = array.length;
		float[] copy = new float[length];
		for (int n = 0; n < length; n++) {
			copy[n] = array[n];
		}
		return copy;
	}

	public static double[] deepCopyToPadded(double[] array, int newSize) {
		int length = array.length;
		if (newSize < length) {
			throw new RuntimeException("Padded array cannot be smaller than original");
		}
		double[] copy = new double[newSize];
		for (int n = 0; n < length; n++) {
			copy[n] = array[n];
		}
		return copy;
	}

	public static double[][] deepCopyToPadded(double[][] image, int newWidth, int newHeight) {
		int width = image.length;
		int height = image[0].length;
		if (newWidth < width || newHeight < height) {
			throw new RuntimeException("Padded array cannot be smaller than original");
		}
		double[][] copy = new double[newWidth][newHeight];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				copy[x][y] = image[x][y];
			}
		}
		return copy;
	}

	public static double[][][] deepCopyToPadded(double[][][] image, int newWidth, int newHeight, int newDepth) {
		int width = image.length;
		int height = image[0].length;
		int depth = image[0][0].length;
		if (newWidth < width || newHeight < height || newDepth < depth) {
			throw new RuntimeException("Padded array cannot be smaller than original");
		}
		double[][][] copy = new double[newWidth][newHeight][newDepth];
		for (int x = 0; x < newWidth; x++) {
			for (int y = 0; y < newHeight; y++) {
				for (int z = 0; z < newDepth; z++) {
					copy[x][y][z] = image[x][y][z];
				}
			}
		}
		return copy;
	}

	public static float[] deepCopyToPadded(float[] array, int newSize) {
		int length = array.length;
		if (newSize < length) {
			throw new RuntimeException("Padded array cannot be smaller than original");
		}
		float[] copy = new float[newSize];
		for (int n = 0; n < length; n++) {
			copy[n] = array[n];
		}
		return copy;
	}

	public static float[][] deepCopyToPadded(float[][] image, int newWidth, int newHeight) {
		int width = image.length;
		int height = image[0].length;
		if (newWidth < width || newHeight < height) {
			throw new RuntimeException("Padded array cannot be smaller than original");
		}
		float[][] copy = new float[newWidth][newHeight];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				copy[x][y] = image[x][y];
			}
		}
		return copy;
	}

	public static float[][][] deepCopyToPadded(float[][][] image, int newWidth, int newHeight, int newDepth) {
		int width = image.length;
		int height = image[0].length;
		int depth = image[0][0].length;
		if (newWidth < width || newHeight < height || newDepth < depth) {
			throw new RuntimeException("Padded array cannot be smaller than original");
		}
		float[][][] copy = new float[newWidth][newHeight][newDepth];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				for (int z = 0; z < depth; z++) {
					copy[x][y][z] = image[x][y][z];
				}
			}
		}
		return copy;
	}

	/**
	 * Display contents of {@code array} in standard output, headed by {@code message} and breaking lines every {@code b} entries.
	 */
	public static void display(Complex[] array, String message, int b) {
		System.out.println(message);
		for (int n = 0; n < array.length; n++) {
			System.out.format("%.2f %.2f, ", Math.round(array[n].getReal()*100)/100.0, Math.round(array[n].getImaginary()*100)/100.0);
			if ((n+1) % b == 0) System.out.format("%n");
		}
	}

	/**
	 * Display contents of {@code array} in standard output, headed by {@code message} and breaking lines every {@code b} entries.
	 */
	public static void display(float[] array, String message, int b) {
		System.out.println("--");
		System.out.println(message);
		for (int n = 0; n < array.length; n++) {
			System.out.format("%.2f ,", Math.round(array[n]*100)/100.0);
			if ((n+1) % b == 0) System.out.format("%n");
		}
	}

	/**
	 * Display contents of {@code array} in standard output, headed by {@code message} and breaking lines every {@code b} entries.
	 */
	public static void display(double[] array, String message, int b) {
		System.out.println("--");
		System.out.println(message);
		for (int n = 0; n < array.length; n++) {
			System.out.format("%.2f ,", Math.round(array[n]*100)/100.0);
			if ((n+1) % b == 0) System.out.format("%n");
		}
	}

	/**
	 * Display contents of {@code array} in standard output, headed by {@code message} and breaking lines every {@code b} entries.
	 */
	public static void display(double[][] array, String message, int b) {
		display(ArrayMath.vectorize(array), message, b);
	}

	/**
	 * Display contents of {@code array} in standard output, headed by {@code message} and breaking lines every {@code b} entries.
	 */
	public static void display(double[][][] array, String message, int b) {
		display(ArrayMath.vectorize(array), message, b);
	}

 	public static double[] cat(double[] d1, double[] d2) {
		int l1 = d1.length;
		int l2 = d2.length;
		double[] cat = new double[l1+l2];
		for (int n = 0; n < l1; n++) {
			cat[n] = d1[n];
		}
		for (int n = 0; n < l2; n++) {
			cat[l1 + n] = d2[n];
		}
		return cat;
	}

	public static float[] cat(float[] d1, float[] d2) {
		int l1 = d1.length;
		int l2 = d2.length;
		float[] cat = new float[l1+l2];
		for (int n = 0; n < l1; n++) {
			cat[n] = d1[n];
		}
		for (int n = 0; n < l2; n++) {
			cat[l1 + n] = d2[n];
		}
		return cat;
	}

	public static ArrayList<double[]> split(double[] d, int l1, int l2) {
		double[] split1 = new double[l1];
		double[] split2 = new double[l2];
		for (int n = 0; n < l1; n++) {
			split1[n] = d[n];
		}
		for (int n = 0; n < l2; n++) {
			split2[n] = d[l1+n];
		}
		ArrayList<double[]> splits = new ArrayList<double[]>(2);
		splits.add(split1);
		splits.add(split2);
		return splits;
	}

	public static ArrayList<float[]> split(float[] d, int l1, int l2) {
		float[] split1 = new float[l1];
		float[] split2 = new float[l2];
		for (int n = 0; n < l1; n++) {
			split1[n] = d[n];
		}
		for (int n = 0; n < l2; n++) {
			split2[n] = d[l1+n];
		}
		ArrayList<float[]> splits = new ArrayList<float[]>(2);
		splits.add(split1);
		splits.add(split2);
		return splits;
	}

	public static double[] mirror(double[] vec) {
		int length = vec.length;
		double[] mirrorVec = new double[length*2];
		for (int n = 0; n < length; n++) {
			mirrorVec[n] = vec[n];
			mirrorVec[length-1-n] = vec[n];
		}
		return mirrorVec;
	}

	public static double[] mirrorComplexInterleaved(double[] vec) {
		int length = vec.length;
		double[] mirrorVec = new double[length*2];
		for (int n = 0; n < length/2; n++) {
			mirrorVec[n*2] = vec[n*2];
			mirrorVec[n*2+1] = vec[n*2+1];
			mirrorVec[length-2-n*2] = vec[n*2];
			mirrorVec[length-1-n*2] = vec[n*2+1];
		}
		return mirrorVec;
	}

	public static double[] interpolateZeros(double[] vec, int factor) {
		int length = vec.length;
		double[] interpVec = new double[length*factor-(factor-1)];
		for (int n = 0; n < length; n++) {
			interpVec[n*factor] = vec[n];
		}
		return interpVec;
	}

	public static Complex[] interpolateZeros(Complex[] vec, int factor) {
		int length = vec.length;
		Complex[] interpVec = new Complex[length*factor - (factor-1)];
		for (int n = 0; n < length; n++) {
			interpVec[n*factor] = vec[n];
		}
		return interpVec;
	}

	public static float[] interpolateZeros(float[] vec, int factor) {
		int length = vec.length;
		float[] interpVec = new float[length*factor - (factor-1)];
		for (int n = 0; n < length; n++) {
			interpVec[n*factor] = vec[n];
		}
		return interpVec;
	}

	public static double[] decimate(double[] vec, int factor) {
		int length = vec.length / factor + vec.length % factor;
		double[] deciVec = new double[length]; // off by one removed EB jan 2016
		for (int n = 0; n < length; n++) {
			deciVec[n] = vec[n*factor];
		}
		return deciVec;
	}

	public static Complex[] decimate(Complex[] vec, int factor) {
		int length = vec.length / factor + vec.length % factor;
		Complex[] deciVec = new Complex[length];
		for (int n = 0; n < length; n++) {
			deciVec[n] = vec[n*factor];
		}
		return deciVec;
	}

	public static float[] decimate(float[] vec, int factor) {
		int length = vec.length;
		float[] deciVec = new float[length];
		for (int n = 0; n < length; n++) {
			deciVec[n] = vec[n*factor];
		}
		return deciVec;
	}

	public static double[][] laplacian2() {
		return new double[][] { {0, 1, 0}, {1, -4, 1}, {0, 1, 0} };
	}

}

