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

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import java.util.prefs.Preferences;
import org.apache.commons.math4.complex.*;

public class Convolver {
	
	int boundaryConditions;
	
	final int ZERO_BOUNDARY = 0;
	final int MIRROR_BOUNDARY = 1;
	final int PERIODIC_BOUNDARY = 2;
	FDCPUNaive naive;
	FDCPUUnrolled unrolled;
	FTCPU ftcpu;
	FDGPU fdgpu;
	FTGPU ftgpu;
	byte[] convolverPrefs;
	Preferences p;
	
	/*
	 * Kernel Categories:
	 * 3, 5, 7, 9, 11, 13, 21, 31, 63, 127, 255, larger
	 * just handle with a switch statement
	 */
		
	public Convolver(int boundaryConditions) {
		this.boundaryConditions = boundaryConditions;
		naive = new FDCPUNaive(boundaryConditions);
		unrolled = new FDCPUUnrolled(boundaryConditions);
		ftcpu = new FTCPU();
		fdgpu = new FDGPU(boundaryConditions);
		ftgpu = new FTGPU();
	}
	
	public Convolver() {
		this.boundaryConditions = ZERO_BOUNDARY;
		naive = new FDCPUNaive(boundaryConditions);
		unrolled = new FDCPUUnrolled(boundaryConditions);
		ftcpu = new FTCPU();
		fdgpu = new FDGPU(boundaryConditions);
		ftgpu = new FTGPU();
		readPreferences();
	}	
	
	public double[] convolve(double[] vector, double[] kernel) {
		int pref = getPreferredConvolution(vector.length, kernel.length);
		switch (pref) {
		case 0:
			return naive.convolve(vector, kernel);
		case 1:
			return unrolled.convolve(vector, kernel);
		case 2:
			return fdgpu.convolve(vector, kernel);
		case 3:
			return ftcpu.convolve(vector, kernel, false);
		case 4:
			return ftgpu.convolve(vector, kernel, false);
		}
		throw new RuntimeException("Invalid preference value");
	}
	

	public Complex[] convolve(Complex[] vector, double[] kernel) {
		int pref = getPreferredConvolution(vector.length, kernel.length);
		switch (pref) {
		case 0:
			return ComplexUtils.split2Complex(
					naive.convolve(ComplexUtils.complex2Real(vector), kernel),
					naive.convolve(ComplexUtils.complex2Imaginary(vector), kernel)
			);
		case 1:
			return ComplexUtils.split2Complex(
					unrolled.convolve(ComplexUtils.complex2Real(vector), kernel),
					unrolled.convolve(ComplexUtils.complex2Imaginary(vector), kernel)
			);
		case 2:
			return ComplexUtils.split2Complex(
					fdgpu.convolve(ComplexUtils.complex2Real(vector), kernel),
					fdgpu.convolve(ComplexUtils.complex2Imaginary(vector), kernel)
			);
		case 3:
			return ftcpu.convolve(vector, kernel, true);
		case 4:
			return ftgpu.convolve(vector, kernel, true);
		}
		throw new RuntimeException("Invalid preference value");
	}
	
	
	public double[][] convolve(double[][] vector, double[][] kernel) {
		int pref = getPreferredConvolution(vector.length, kernel.length);
		switch (pref) {
		case 0:
			return naive.convolve(vector, kernel);
		case 1:
			return unrolled.convolve(vector, kernel);
		case 2:
			return fdgpu.convolve(vector, kernel);
		case 3:
			return ftcpu.convolve(vector, kernel, false);
		case 4:
			return ftgpu.convolve(vector, kernel, false);
		}
		throw new RuntimeException("Invalid preference value");
	}
	
	public Complex[][] convolve(Complex[][] vector, double[][] kernel) {
		int pref = getPreferredConvolution(vector.length, kernel.length);
		switch (pref) {
		case 0:
			return ComplexUtils.split2Complex(
					naive.convolve(ComplexUtils.complex2Real(vector), kernel),
					naive.convolve(ComplexUtils.complex2Imaginary(vector), kernel)
			);
		case 1:
			return ComplexUtils.split2Complex(
					unrolled.convolve(ComplexUtils.complex2Real(vector), kernel),
					unrolled.convolve(ComplexUtils.complex2Imaginary(vector), kernel)
			);
		case 2:
			return ComplexUtils.split2Complex(
					fdgpu.convolve(ComplexUtils.complex2Real(vector), kernel),
					fdgpu.convolve(ComplexUtils.complex2Imaginary(vector), kernel)
			);
		case 3:
			return ftcpu.convolve(vector, kernel, true);
		case 4:
			return ftgpu.convolve(vector, kernel, true);
		}
		throw new RuntimeException("Invalid preference value");
	}

	public double[][][] convolve(double[][][] vector, double[][][] kernel) {
		int pref = getPreferredConvolution(vector.length, kernel.length);
		switch (pref) {
		case 0:
			return naive.convolve(vector, kernel);
		case 1:
			return unrolled.convolve(vector, kernel);
		case 2:
			return fdgpu.convolve(vector, kernel);
		case 3:
			return ftcpu.convolve(vector, kernel, false);
		case 4:
			return ftgpu.convolve(vector, kernel, false);
		}
		throw new RuntimeException("Invalid preference value");
	}
	
	public Complex[][][] convolve(Complex[][][] vector, double[][][] kernel) {
		int pref = getPreferredConvolution(vector.length, kernel.length);
		switch (pref) {
		case 0:
			return ComplexUtils.split2Complex(
					naive.convolve(ComplexUtils.complex2Real(vector), kernel),
					naive.convolve(ComplexUtils.complex2Imaginary(vector), kernel)
			);
		case 1:
			return ComplexUtils.split2Complex(
					unrolled.convolve(ComplexUtils.complex2Real(vector), kernel),
					unrolled.convolve(ComplexUtils.complex2Imaginary(vector), kernel)
			);
		case 2:
			return ComplexUtils.split2Complex(
					fdgpu.convolve(ComplexUtils.complex2Real(vector), kernel),
					fdgpu.convolve(ComplexUtils.complex2Imaginary(vector), kernel)
			);
		case 3:
			return ftcpu.convolve(vector, kernel, true);
		case 4:
			return ftgpu.convolve(vector, kernel, true);
		}
		throw new RuntimeException("Invalid preference value");
	}
	
	public double[] convolveInterleaved(double[] vector, double[] kernel) {
		return ComplexUtils.complex2Interleaved(
				convolve(
						ComplexUtils.interleaved2Complex(vector), kernel
						)
				);
	}
	
	public double[][] convolveInterleaved(double[][] vector, double[][] kernel) {
		return ComplexUtils.complex2Interleaved(
				convolve(
						ComplexUtils.interleaved2Complex(vector), kernel
						)
				);
	}
	
	
	
	private int getPreferredConvolution(int vl, int kl) {
		int row = JVCLUtils.nextPwr2(vl) - 6;
		int col = -1;
		if (kl <= 3) {
			col = 0;
		} else if (kl <= 5) {
			col = 1;
		} else if (kl <= 7) {
			col = 2;
		} else if (kl <= 9) {
			col = 3;
		} else if (kl <= 13) {
			col = 4;
		} else if (kl <= 21) {
			col = 5;
		} else if (kl <= 31) {
			col = 6;
		} else if (kl <= 63) {
			col = 7;
		} else if (kl <= 127) {
			col = 8;
		} else if (kl <= 255) {
			col = 9;
		} else {
			col = 10;
		}
		return row*10 + col;
	}
	
	private void readPreferences() {
		p = Preferences.systemNodeForPackage(getClass());
		convolverPrefs = p.getByteArray("preferences", new byte[60]);
	}
		
}


