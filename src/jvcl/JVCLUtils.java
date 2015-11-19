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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math4.util.FastMath;
import org.apache.commons.math4.complex.*;

import arrayMath.ArrayMath;

public class JVCLUtils {
	
	/** A utility class to shift dimensions. Equivalent to Matlab shiftDim. 
	*/
	public JVCLUtils() {};

	public static double[][] shiftDim(double[][] image) {
		int width = image.length;
		int height = image[0].length;
		double[][] result = new double[height][width];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				result[y][x] = image[x][y];
			}
		}
		return result;
	}

	public static float[][] shiftDim(float[][] image) {
		int width = image.length;
		int height = image[0].length;
		float[][] result = new float[height][width];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				result[y][x] = image[x][y];
			}
		}
		return result;
	}
	
	public static Complex[][] shiftDim(Complex[][] image) {
		int width = image.length;
		int height = image[0].length;
		Complex[][] result = new Complex[height][width];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				result[y][x] = image[x][y];
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
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					for (int z = 0; z < depth; z++) {
						result[y][z][x] = volume[x][y][z];
					}
				}
			}
		} else if (nDims == 2) {
			result = new double[depth][width][height];
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					for (int z = 0; z < depth; z++) {
						result[z][x][y] = volume[x][y][z];
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
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					for (int z = 0; z < depth; z++) {
						result[y][z][x] = volume[x][y][z];
					}
				}
			}
		} else if (nDims == 2) {
			result = new Complex[depth][width][height];
			for (int x = 0; x < width; x++) {
				for (int y = 0; y < height; y++) {
					for (int z = 0; z < depth; z++) {
						result[z][x][y] = volume[x][y][z];
					}
				}
			}
		} else return volume;
		return result;
	}	
		
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


    public static double[] vectorise(double[][] image) {
    	int width = image.length;
    	int height = image[0].length;
    	int area = width*height;
    	double[] result = new double[area];
		for (int x = 0; x < width; x++) {
    		for (int y = 0; y < height; y++) {
    			result[y*width + x] = image[x][y];
    		}
    	}
    	return result;
    }
    
    public static double[] vectorise(double[][][] image) {
    	final int width = image.length;
    	final int height = image[0].length;
    	final int area = width*height;
    	final int depth = image[0][0].length;
    	int volume = width*height*depth;
    	double[] result = new double[volume];
    	for (int x = 0; x < width; x++) {
    		for (int y = 0; y < height; y++) {
    			for (int z = 0; z < depth; z++) {
    				result[z*area + y*width + x] = image[x][y][z];
    			}
    		}
    	}
    	return result;
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
    
    public static float[] vectorise(float[][][] image) {
    	final int width = image.length;
    	final int height = image[0].length;
    	final int area = width*height;
    	final int depth = image[0][0].length;
    	int volume = width*height*depth;
    	float[] result = new float[volume];
    	for (int x = 0; x < width; x++) {
    		for (int y = 0; y < height; y++) {
    			for (int z = 0; z < depth; z++) {
    				result[z*area + y*width + x] = image[x][y][z];
    			}
    		}
    	}
       	return result;
    }
    
    public static Complex[] vectorise(Complex[][] image) {
    	int width = image.length;
    	int height = image[0].length;
    	int area = width*height;
    	Complex[] result = new Complex[area];
		for (int x = 0; x < width; x++) {
    		for (int y = 0; y < height; y++) {
    			result[y*width + x] = image[x][y];
    		}
    	}
    	return result;
    }
    
    public static Complex[] vectorise(Complex[][][] image) {
    	final int width = image.length;
    	final int height = image[0].length;
    	final int depth = image[0][0].length;
    	final int area = width*height;
    	int volume = width*height*depth;
    	Complex[] result = new Complex[volume];
    	for (int x = 0; x < width; x++) {
    		for (int y = 0; y < height; y++) {
    			for (int z = 0; z < depth; z++) {
    				result[z*area + y*width + x] = image[x][y][z];
    			}
    		}
    	}
    	return result;
    }
    
    public static double[][] devectorise(double[] vector, int dim1) {
    	int length = vector.length;
    	int dim2 = length / dim1;
    	double[][] image = new double[dim1][dim2];
    	for (int x = 0; x < dim1; x++) {
    		for (int y = 0; y < dim2; y++) {
    			 image[x][y] = vector[y*dim1 + x];
    		}
    	}
    	return image;
    }
    
    public static double[][][] devectorise(double[] vector, int dim1, int dim2) {
    	int length = vector.length;
    	int dim3 = length / (dim1*dim2);
    	int dim12 = dim1*dim2;
    	double[][][] volume = new double[dim1][dim2][dim3];
		for (int x = 0; x < dim1; x++) {
			for (int y = 0; y < dim2; y++) {
				for (int z = 0; z < dim3; z++) {
        			volume[x][y][z] = vector[z*dim12 + y*dim1 + x];
				}
			}
		}
    	return volume;
    }
    
    public static float[][] devectorise(float[] vector, int dim1) {
    	int length = vector.length;
    	int dim2 = length / dim1;
    	float[][] image = new float[dim1][dim2];
    	for (int x = 0; x < dim1; x++) {
    		for (int y = 0; y < dim2; y++) {
    			 image[x][y] = vector[y*dim1 + x];
    		}
    	}
    	return image;
    }
    
    public static float[][][] devectorise(float[] vector, int dim1, int dim2) {
    	int length = vector.length;
    	int dim3 = length / (dim1*dim2);
    	int dim12 = dim1*dim2;
    	float[][][] volume = new float[dim1][dim2][dim3];
    	for (int x = 0; x < dim1; x++) {
			for (int y = 0; y < dim2; y++) {
				for (int z = 0; z < dim3; z++) {
        			volume[x][y][z] = vector[z*dim12 + y*dim1 + x];
				}
			}
		}
    	return volume;
    }
   

    public static Complex[][] devectorise(Complex[] vector, int dim1) {
    	int length = vector.length;
    	int dim2 = length / dim1;
    	Complex[][] image = new Complex[dim1][dim2];
    	for (int x = 0; x < dim1; x++) {
    		for (int y = 0; y < dim2; y++) {
    			 image[x][y] = vector[y*dim1 + x];
    		}
    	}
    	return image;
    }
    
    public static Complex[][][] devectorise(Complex[] vector, int dim1, int dim2) {
    	int length = vector.length;
    	int dim3 = length / (dim1*dim2);
    	int dim12 = dim1*dim2;
    	Complex[][][] volume = new Complex[dim1][dim2][dim3];
    	for (int x = 0; x < dim1; x++) {
			for (int y = 0; y < dim2; y++) {
				for (int z = 0; z < dim3; z++) {
        			volume[x][y][z] = vector[z*dim12 + y*dim1 + x];
				}
			}
		}
       	return volume;
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
	
	public static double[] real2Interleaved(double[] array) {
		int length = array.length;
		double[] interleaved = new double[length*2];
		for (int n = 0; n < length; n++) {
			interleaved[n*2] = array[n];
		}
		return interleaved;
	}	
	
	public static double[][] real2Interleaved(double[][] image) {
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
	
	public static double[][][] real2Interleaved(double[][][] image) {
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
	
	public static float[] double2Float(double[] array) {
		final int length = array.length;
		float[] floatArray = new float[length];
		for (int n = 0; n < length; n++) floatArray[n] = (float)array[n];
		return floatArray;
	}
	
	public static float[][] double2Float(double[][] array) {
		final int width = array.length;
		final int height = array[0].length;
		float[][] floatArray = new float[width][height];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				floatArray[x][y] = (float)array[x][y];
			}
		}
		return floatArray;
	}
	
	public static float[][][] double2Float(double[][][] array) {
		final int width = array.length;
		final int height = array[0].length;
		final int depth = array[0][0].length;
		float[][][] floatArray = new float[width][height][depth];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				for (int z = 0; z < depth; z++) {
					floatArray[x][y][z] = (float)array[x][y][z];
				}
			}
		}
		return floatArray;
	}
	
	public static double[] float2Double(float[] array) {
		final int length = array.length;
		double[] floatArray = new double[length];
		for (int n = 0; n < length; n++) floatArray[n] = array[n];
		return floatArray;
	}
	
	public static double[][] float2Double(float[][] array) {
		int width = array.length;
		int height = array[0].length;
		double[][] floatArray = new double[width][height];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				floatArray[x][y] = (float)array[x][y];
			}
		}
		return floatArray;
	}
	
	public static double[][][] float2Double(float[][][] array) {
		int width = array.length;
		int height = array[0].length;
		int depth = array[0][0].length;
		double[][][] doubleArray = new double[width][height][depth];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				for (int z = 0; z < depth; z++) {
					doubleArray[x][y][z] = array[x][y][z];
				}
			}
		}
		return doubleArray;
	}
	
	public static float[] getInterleavedReal(float[] interleaved) {
		int length = interleaved.length;
		float[] real = new float[length/2];
		for (int n = 0; n < length/2; n++) {
			real[n] = interleaved[n*2];
		}
		return real;
	}
	
	public static float[] getInterleavedImag(float[] interleaved) {
		int length = interleaved.length;
		float[] real = new float[length/2];
		for (int n = 0; n < length/2; n++) {
			real[n] = interleaved[n*2+1];
		}
		return real;
	}
	
	public static void display2DArray(float[][] array, int width, int height, int subBreak, int decimalPlaces) {
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
	
	public static void display2DArray(double[][] array, int width, int height, int subBreak, int decimalPlaces) {
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
	
	public static void display2DArray(float[][] array, int brk, int row) {
		display2DArray(array, array.length, array[0].length, brk, row);	
	}
	
	public static void display2DArray(double[][] array, int brk, int row) {
		display2DArray(array, array.length, array[0].length, brk, row);	
	}
	
	public static void display(Complex[] array, String message, int b) {
		System.out.println(message);
		for (int n = 0; n < array.length; n++) {
			System.out.format("%.2f %.2f, ", Math.round(array[n].getReal()*100)/100.0, Math.round(array[n].getImaginary()*100)/100.0);
			if ((n+1) % b == 0) System.out.format("%n");
		}
	}
	
	public static void display(float[] array, String message, int b) {
		System.out.println("--");
		System.out.println(message);
		for (int n = 0; n < array.length; n++) {
			System.out.format("%.2f ,", Math.round(array[n]*100)/100.0);
			if ((n+1) % b == 0) System.out.format("%n");
		}
	}

	public static void display(double[] array, String message, int b) {
		System.out.println("--");
		System.out.println(message);
		for (int n = 0; n < array.length; n++) {
			System.out.format("%.2f ,", Math.round(array[n]*100)/100.0);
			if ((n+1) % b == 0) System.out.format("%n");
		}
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
	
	public static void splitMultiply(double[] d) {
		final int length = d.length;
		for (int n = 0; n < length / 2; n++) {
			d[n] *= d[n+length/2];
		}
	}
	
	public static void splitMultiply(float[] d) {
		final int length = d.length;
		for (int n = 0; n < length / 2; n++) {
			d[n] *= d[n+length/2];
		}
	}

	public static Complex[][] initializeRandom(int dim) {
		Random r = new Random();
		Complex[][] c = new Complex[dim][dim];
		for (int x = 0; x < dim; x++) {
			for (int y = 0; y < dim; y++) {
				c[x][y] = new Complex(r.nextDouble(), r.nextDouble());
			}
		}
		return c;
	}
	
	public static double[] fillWithSecondOrder(int dim) {
		double[] c = new double[dim];
		for (int x = 0; x < dim; x++) {
				c[x] = x*x;
		}
		return c;
	}
	
	public static double[][] fillWithSecondOrder(int dim1, int dim2) {
		double[][] c = new double[dim1][dim2];
		for (int x = 0; x < dim1; x++) {
			for (int y = 0; y < dim2; y++) {
				c[x][y] = x*x + y*y;
			}
		}
		return c;
	}
	
	public static double[][][] fillWithSecondOrder(int dim1, int dim2, int dim3) {
		double[][][] c = new double[dim1][dim2][dim3];
		for (int x = 0; x < dim1; x++) {
			for (int y = 0; y < dim2; y++) {
				for (int z = 0; z < dim3; z++) {
					c[x][y][z] = x*x + y*y + z*z;
				}
			}
		}
		return c;
	}
	
	public static Complex[] fillWithSecondOrderComplex(int dim) {
		Complex[] c = new Complex[dim];
		for (int x = 0; x < dim; x++) {
				c[x] = new Complex(x*x);
		}
		return c;
	}
	
	public static Complex[][] fillWithSecondOrderComplex(int dim1, int dim2) {
		Complex[][] c = new Complex[dim1][dim2];
		for (int x = 0; x < dim1; x++) {
			for (int y = 0; y < dim2; y++) {
				c[x][y] = new Complex(x*x + y*y);
			}
		}
		return c;
	}
	
	public static Complex[][][] fillWithSecondOrderComplex(int dim1, int dim2, int dim3) {
		Complex[][][] c = new Complex[dim1][dim2][dim3];
		for (int x = 0; x < dim1; x++) {
			for (int y = 0; y < dim2; y++) {
				for (int z = 0; z < dim3; z++) {
					c[x][y][z] = new Complex(x*x + y*y + z*z);
				}
			}
		}
		return c;
	}
	
	public static double[] fillWithGradient(int dim) {
		double[] c = new double[dim];
		for (int x = 0; x < dim; x++) {
				c[x] = x+1;
		}
		return c;
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
		int vlength = vec.length;
		double[] deciVec = new double[length + 1];
		for (int n = 0; n < length; n++) {
			deciVec[n] = vec[n*factor];
		}
		deciVec[length] = vec[vlength-1];
		return deciVec;
	}
	
	public static Complex[] decimate(Complex[] vec, int factor) {
		int length = vec.length;
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
	
	
}

