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

//DOUBLE
    // BOOLEAN

	// FLOAT
	// COMPLEX

}

