/* Copyright (c) 2015 Eric Barnhill
*
*Permission is hereby granted, free of charge, to any person obtaining a copy
*of this software and associated documentation files (the "Software"), to deal
*in the Software without restriction, including without limitation the rights
*to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*copies of the Software, and to permit persons to whom the Software is
*furnished to do so, subject to the following conditions:
*
*The above copyright notice and this permission notice shall be included in all
*copies or substantial portions of the Software.
*
*THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*SOFTWARE.
*/

package jvcl;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;

import org.apache.commons.math4.complex.*;

import arrayMath.ArrayMath;

public class JVCLUtils {
	
	/** A utility class to shift dimensions. Equivalent to Matlab shiftDim. 
	*/
	public JVCLUtils() {};

	public static double[][] shiftDim(double[][] image) {
		int width = image.length;
		int height = image[0].length;
		double[][] result = new double[width][height];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				result[x][y] = image[y][x];
			}
		}
		return result;
	}

	public static float[][] shiftDim(float[][] image) {
		int width = image.length;
		int height = image[0].length;
		float[][] result = new float[width][height];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				result[x][y] = image[y][x];
			}
		}
		return result;
	}
	

	public static Complex[][] shiftDim(Complex[][] image) {
		int width = image.length;
		int height = image[0].length;
		Complex[][] result = new Complex[width][height];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				result[x][y] = image[y][x];
			}
		}
		return result;
	}

	public static double[][][] shiftDim(double[][][] image) {
		int width = image.length;
		int height = image[0].length;
		int depth = image[0][0].length;
		double[][][] result = new double[width][height][depth];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				for (int z = 0; z < depth; z++) {
					result[x][y][z] = image[y][z][x];
				}
			}
		}
		return result;
	}
	
	public static float[][][] shiftDim(float[][][] volume) {
		int width = volume.length;
		int height = volume[0].length;
		int depth = volume[0][0].length;
		float[][][] result = new float[width][height][depth];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				for (int z = 0; z < depth; z++) {
					result[x][y][z] = volume[y][z][x];
				}
			}
		}
		return result;
	}
	
	public static Complex[][][] shiftDim(Complex[][][] image) {
		int width = image.length;
		int height = image[0].length;
		int depth = image[0][0].length;
		Complex[][][] result = new Complex[width][height][depth];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				for (int z = 0; z < depth; z++) {
					result[x][y][z] = image[y][z][x];
				}
			}
		}
		return result;
	}
	
	/** Shifts dimensions of 3D volume.
    *  @param volume   3D array volume
    *  @param nDims    if [x][y][z] 1 = move y array to x position and z to y; 2 = move z array to x position and y to z 
    *  @return double[][][]
    */
	public static double[][][] shiftDim(double[][][] volume, int nDims) {
		int width = volume.length;
		int height = volume[0].length;
		int depth = volume[0][0].length;
		nDims = nDims % 3;
		double[][][] result;
		if (nDims == 1) {
			result = new double[height][depth][width];
			for (int x = 0; x < height; x++) {
				for (int y = 0; y < depth; y++) {
					for (int z = 0; z < width; z++) {
						result[x][y][z] = volume[y][z][x];
					}
				}
			}
		} else if (nDims == 2) {
			result = new double[depth][width][height];
			for (int x = 0; x < depth; x++) {
				for (int y = 0; y < width; y++) {
					for (int z = 0; z < height; z++) {
						result[x][y][z] = volume[z][x][y];
					}
				}
			}
		} else return volume;
		return result;
	}	
	
	/** Shifts dimensions of 3D volume.
    *  @param volume   3D array volume
    *  @param nDims    if [x][y][z] 1 = move y array to x position and z to y; 2 = move z array to x position and y to z 
    *  @return double[][][]
    */
	public static float[][][] shiftDim(float[][][] volume, int nDims) {
		int width = volume.length;
		int height = volume[0].length;
		int depth = volume[0][0].length;
		nDims = nDims % 3;
		float[][][] result;
		if (nDims == 1) {
			result = new float[height][depth][width];
			for (int x = 0; x < height; x++) {
				for (int y = 0; y < depth; y++) {
					for (int z = 0; z < width; z++) {
						result[x][y][z] = volume[y][z][x];
					}
				}
			}
		} else if (nDims == 2) {
			result = new float[depth][width][height];
			for (int x = 0; x < depth; x++) {
				for (int y = 0; y < width; y++) {
					for (int z = 0; z < height; z++) {
						result[x][y][z] = volume[z][x][y];
					}
				}
			}
		} else return volume;
		return result;
	}	
	/** Shifts dimensions of 3D volume.
    *  @param volume   3D array volume
    *  @param nDims    if [x][y][z] 1 = move y array to x position and z to y; 2 = move z array to x position and y to z 
    *  @return double[][][]
    */
	public static Complex[][][] shiftDim(Complex[][][] volume, int nDims) {
		int width = volume.length;
		int height = volume[0].length;
		int depth = volume[0][0].length;
		nDims = nDims % 3;
		Complex[][][] result;
		if (nDims == 1) {
			result = new Complex[height][depth][width];
			for (int x = 0; x < height; x++) {
				for (int y = 0; y < depth; y++) {
					for (int z = 0; z < width; z++) {
						result[x][y][z] = volume[y][z][x];
					}
				}
			}
		} else if (nDims == 2) {
			result = new Complex[depth][width][height];
			for (int x = 0; x < depth; x++) {
				for (int y = 0; y < width; y++) {
					for (int z = 0; z < height; z++) {
						result[x][y][z] = volume[z][x][y];
					}
				}
			}
		} else return volume;
		return result;
	}	
		
	public static float[] shiftDim(float[] image, int width) {
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
	
	public static Complex[] split2Complex(double[] real, double[] imag) {
		int length = real.length;
		Complex[] c = new Complex[length];
		for (int n = 0; n < length; n++) {
			c[n] = new Complex(real[n], imag[n]);
		}
		return c;
	}
	
	public static Complex[] split2Complex(float[] real, float[] imag) {
		int length = real.length;
		Complex[] c = new Complex[length];
		for (int n = 0; n < length; n++) {
			c[n] = new Complex(real[n], imag[n]);
		}
		return c;
	}
	
	public static double[][] complex2Split(Complex[] c) {
		int length = c.length;
		double[][] split  = new double[2][length];
		for (int n = 0; n < length; n++) {
			split[0][n] = c[n].getReal();
			split[1][n] = c[n].getImaginary();
		}
		return split;
	}
	
	public static float[][] complex2SplitFloat(Complex[] c) {
		int length = c.length;
		float[][] split  = new float[2][length];
		for (int n = 0; n < length; n++) {
			split[0][n] = (float)c[n].getReal();
			split[1][n] = (float)c[n].getImaginary();
		}
		return split;
	}
	
	public static double[] split2interleaved(double[] real, double[] imag) {
		int length = real.length;
		double[] interleaved = new double[length*2];
		for (int n = 0; n < length; n++) {
			interleaved[n*2] = real[n];
			interleaved[n*2+1] = imag[n];
		}
		return interleaved;
	}
	
	public static float[] split2interleaved(float[] real, float[] imag) {
		int length = real.length;
		float[] interleaved = new float[length*2];
		for (int n = 0; n < length; n++) {
			interleaved[n*2] = real[n];
			interleaved[n*2+1] = imag[n];
		}
		return interleaved;
	}
	
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

    public static float[] vectorise(float[][] image) {
    	int width = image.length;
    	int height = image[0].length;
    	int area = width*height;
    	float[] result = new float[area];
    	for (int x = 0; x < width; x++) {
    		for (int y = 0; y < height; y++) {
    			result[y*width + x] = image[x][y];
    		}
    	}
    	return result;
    }
    
    public static float[][] devectorise(float[] vector, int width) {
    	int length = vector.length;
    	int height = length / width;
    	float[][] image = new float[width][height];
    	for (int x = 0; x < width; x++) {
    		for (int y = 0; y < height; y++) {
    			 image[x][y] = vector[y*width + x];
    		}
    	}
    	return image;
    }
    
	static int nextPwr2(int length) {
		
		int pwr2Length = 1;
		while(pwr2Length < length) {
			pwr2Length *= 2;
		}
		return pwr2Length;
	}

	static double[] zeroPad(double[] array, int padding) {
		int length = array.length;
		int newLength = length + 2*padding;
		double[] paddedArray = new double[newLength];
		for (int n = 0; n < length; n++) {
			paddedArray[padding+n] = array[n];
		}
		return paddedArray;	
	}
	
	static double[][] zeroPad(double[][] image, int paddingX, int paddingY) {
		int width = image.length;
		int height = image[0].length;
		int newWidth = width + 2*paddingX;
		int newHeight = height + 2*paddingY;
		double[][] paddedImage = new double[newWidth][newHeight];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				paddedImage[x+paddingX][y+paddingY] = image[x][y];
			}
		}
		return paddedImage;	
	}
	
	static double[][][] zeroPad(double[][][] volume, int paddingX, int paddingY, int paddingZ) {
		int width = volume.length;
		int height = volume[0].length;
		int depth = volume[0][0].length;
		int newWidth = width + 2*paddingX;
		int newHeight = height + 2*paddingY;
		int newDepth = depth + 2*paddingZ;
		double[][][] paddedImage = new double[newWidth][newHeight][newDepth];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				for (int z = 0; z < depth; z++) {
					paddedImage[x+paddingX][y+paddingY][z+paddingZ] = volume[x][y][z];
				}
			}
		}
		return paddedImage;	
	}
	
	static Complex[] zeroPad(Complex[] array, int padding) {
		int length = array.length;
		int newLength = length + 2*padding;
		Complex[] paddedArray = new Complex[newLength];
		for (int n = 0; n < length; n++) {
			paddedArray[padding+n] = array[n];
		}
		return paddedArray;	
	}
	
	static Complex[][] zeroPad(Complex[][] image, int paddingX, int paddingY) {
		int width = image.length;
		int height = image[0].length;
		int newWidth = width + 2*paddingX;
		int newHeight = height + 2*paddingY;
		Complex[][] paddedImage = new Complex[newWidth][newHeight];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				paddedImage[x+paddingX][y+paddingY] = image[x][y];
			}
		}
		return paddedImage;	
	}
	
	static Complex[][][] zeroPad(Complex[][][] volume, int paddingX, int paddingY, int paddingZ) {
		int width = volume.length;
		int height = volume[0].length;
		int depth = volume[0][0].length;
		int newWidth = width + 2*paddingX;
		int newHeight = height + 2*paddingY;
		int newDepth = depth + 2*paddingZ;
		Complex[][][] paddedImage = new Complex[newWidth][newHeight][newDepth];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				for (int z = 0; z < depth; z++) {
					paddedImage[x+paddingX][y+paddingY][z+paddingZ] = volume[x][y][z];
				}
			}
		}
		return paddedImage;	
	}
	
	static double[] stripPadding(double[] array, int padding) {
		int length = array.length;
		int newLength = length - 2*padding;
		double[] strippedArray = new double[newLength];
		for (int n = 0; n < newLength; n++) {
			strippedArray[n] = array[padding+n];
		}
		return strippedArray;	
	}
	

	static double[][] stripPadding(double[][] image, int paddingX, int paddingY) {
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
	
	static double[][][] stripPadding(double[][][] volume, int paddingX, int paddingY, int paddingZ) {
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
	
	static double[] deepCopy(double[] array) {
		int length = array.length;
		double[] copy = new double[length];
		for (int n = 0; n < length; n++) {
			copy[n] = array[n];
		}
		return copy;
	}
	
	static float[] deepCopy(float[] array) {
		int length = array.length;
		float[] copy = new float[length];
		for (int n = 0; n < length; n++) {
			copy[n] = array[n];
		}
		return copy;
	}
	
	static double[] deepCopyToPadded(double[] array, int newSize) {
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
	
	static double[][] deepCopyToPadded(double[][] image, int newWidth, int newHeight) {
		int width = image.length;
		int height = image[0].length;
		if (newWidth < width || newHeight < height) {
			throw new RuntimeException("Padded array cannot be smaller than original");
		}
		double[][] copy = new double[newWidth][newHeight];
		for (int x = 0; x < newWidth; x++) {
			for (int y = 0; y < newHeight; y++) {
				copy[x][y] = image[x][y];
			}
		}
		return copy;
	}
	
	static double[][][] deepCopyToPadded(double[][][] image, int newWidth, int newHeight, int newDepth) {
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
	
	static double[] real2Interleaved(double[] array) {
		int length = array.length;
		double[] interleaved = new double[length*2];
		for (int n = 0; n < length; n++) {
			interleaved[n*2] = array[n];
		}
		return interleaved;
	}	
	
	static double[][] real2Interleaved(double[][] image) {
		int width = image.length;
		int height = image[0].length;
		double[][] interleaved = new double[width*2][height*2];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				interleaved[x*2][y*2] = image[x][y];
			}
		}
		return interleaved;
	}
	
	static double[][][] real2Interleaved(double[][][] image) {
		int width = image.length;
		int height = image[0].length;
		int depth = image[0][0].length;
		double[][][] interleaved = new double[width*2][height*2][depth*2];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				for (int z = 0; z < depth; z++) {
					interleaved[x*2][y*2][z*2] = image[x][y][z];
				}
			}
		}
		return interleaved;
	}
	
	static float[] double2Float(double[] array) {
		final int length = array.length;
		float[] floatArray = new float[length];
		for (int n = 0; n < length; n++) floatArray[n] = (float)array[n];
		return floatArray;
	}
	
	static double[] float2Double(float[] array) {
		final int length = array.length;
		double[] floatArray = new double[length];
		for (int n = 0; n < length; n++) floatArray[n] = array[n];
		return floatArray;
	}
	
	static float[] getInterleavedReal(float[] interleaved) {
		int length = interleaved.length;
		float[] real = new float[length/2];
		for (int n = 0; n < length/2; n++) {
			real[n] = interleaved[n*2];
		}
		return real;
	}
	
	static float[] getInterleavedImag(float[] interleaved) {
		int length = interleaved.length;
		float[] real = new float[length/2];
		for (int n = 0; n < length/2; n++) {
			real[n] = interleaved[n*2+1];
		}
		return real;
	}
	
	static void display2DArray(float[][] array, int width, int height, int subBreak, int decimalPlaces) {
		System.out.println("--");
		if (width == 0) width = array.length;
		if (height == 0) height = array[0].length;
		String f = "%." + Integer.toString(decimalPlaces) + "f ";
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				System.out.format(f, array[x][y]);
			}
			System.out.format("%n");
		}
		System.out.println("-");		
	}
	
	static void display2DArray(float[][] array) {
		display2DArray(array, array.length, array[0].length, 8, 2);	
	}
	

	static public void display(Complex[] array, String message) {
		System.out.println("--");
		System.out.println(message);System.out.println(Arrays.toString(
				ArrayMath.divide(
					ArrayMath.round(
						ArrayMath.multiply(ComplexUtils.complex2RealFloat(array), 100)
					)
				, 100)
			)
		);
	}
	
	static public void display(float[] array, String message, int b) {
		System.out.println("--");
		System.out.println(message);
		for (int n = 0; n < array.length; n++) {
			System.out.format("%.2f ,", Math.round(array[n]*100)/100.0);
			if ((n+1) % b == 0) System.out.format("%n");
		}
	}
	
}
