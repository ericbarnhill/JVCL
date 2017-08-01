package com.ericbarnhill.jvcl;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

abstract class Convolver<T> {

    public Convolver() {}

    abstract T[] convolve(T[] data, T[] kernel);
    abstract T[][] convolve(T[][] data, T[] kernel);
    abstract T[][][] convolve(T[][][] data, T[] kernel);
    abstract T[][] convolve(T[][] data, T[][] kernel);
    abstract T[][][] convolve(T[][][] data, T[][] kernel);
    abstract T[][][] convolve(T[][][] data, T[][][] kernel);

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
            return globalSize + groupSize -r;
        }
	}


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
}
