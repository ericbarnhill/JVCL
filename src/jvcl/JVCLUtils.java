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

import org.apache.commons.math4.complex.*;

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

	
}
