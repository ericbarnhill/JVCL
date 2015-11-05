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

import java.util.prefs.Preferences;

import org.apache.commons.math4.complex.Complex;
import org.apache.commons.math4.complex.ComplexUtils;
import org.apache.commons.math4.util.FastMath;

public class Convolver {
	
	int boundaryConditions;
	
	static final int FULL = 0;
	static final int SAME = 1;
	static final int VALID = 2;
	FDCPUNaive naive;
	FDCPUUnrolled unrolled;
	FTCPU ftcpu;
	FDGPU fdgpu;
	FTGPU ftgpu;
	byte[] convolverPrefs;
	boolean hasgpu;
	Preferences p;
	
	/*
	 * Kernel Categories:
	 * 3, 5, 7, 9, 11, 13, 21, 31, 63, 127, 255, larger
	 * just handle with a switch statement
	 */

	public Convolver() {
		naive = new FDCPUNaive();
		unrolled = new FDCPUUnrolled();
		hasgpu = true;
		ftcpu = new FTCPU();
		try {
			fdgpu = new FDGPU();
			ftgpu = new FTGPU();
		} catch (Exception e) {
			System.out.println("OpenCL not detected, using GPU methods");
			hasgpu = false;
		}
		readPreferences();
	}		
	
	public double[] convolve(double[] vector, double[] kernel) {
		int pref = getPreferredConvolution(vector.length, kernel.length);
		switch(convolverPrefs[pref]) {
		case 0:
			return FDCPUNaive.convolve(vector, kernel);
		case 1:
			return unrolled.convolve(vector, kernel);
		case 2:
			return fdgpu.convolve(vector, kernel);
		case 3:
			return ComplexUtils.complex2Real(ftcpu.convolve(ComplexUtils.real2Complex(vector), ComplexUtils.real2Complex(kernel)));
		case 4:
			return ComplexUtils.complex2Real(ftgpu.convolve(ComplexUtils.real2Complex(vector), ComplexUtils.real2Complex(kernel)));
		}
		throw new RuntimeException("Invalid preference value :"+pref);
	}
	
	public double[][] convolve(double[][] image, double[] kernel, int dim) {
		if (dim > 1) throw new RuntimeException("2D image, 1D kernel: Invalid dim specification");
		if (dim == 0) image = JVCLUtils.shiftDim(image);
		int height = image.length;
		for (int n = 0; n < height; n++) {
			image[n] = convolve(image[n], kernel);
		}
		if (dim == 0) {
			return JVCLUtils.shiftDim(image);
		} else {
			return image;
		}
	}
	
	public double[][] convolve(double[][] image, double[] kernel) {
		return convolve(image, kernel, 0);
	}

	public double[][][] convolve(double[][][] volume, double[] kernel, int dim) {
		if (dim == 0) volume = JVCLUtils.shiftDim(volume, 2);
		if (dim == 1) volume = JVCLUtils.shiftDim(volume, 1);
		if (dim > 2) throw new RuntimeException("Invalid dim");
		
		int volumeWidth = volume.length;
		int volumeHeight = volume[0].length;
		
		for (int x = 0; x < volumeWidth; x++) {
			for (int y = 0; y < volumeHeight; y++) {
				volume[x][y] = convolve(volume[x][y], kernel);
			}
		}
		return volume;
	}
	
	public double[][][] convolve(double[][][] volume, double[] kernel) {
		return convolve(volume, kernel, 0);
	}
	
	public Complex[] convolve(Complex[] vector, double[] kernel) {
		int pref = getPreferredConvolution(vector.length, kernel.length);
		switch(convolverPrefs[pref]) {
		case 0:
			return ComplexUtils.split2Complex(
					FDCPUNaive.convolve(ComplexUtils.complex2Real(vector), kernel),
					FDCPUNaive.convolve(ComplexUtils.complex2Imaginary(vector), kernel)
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
			return ftcpu.convolve(vector, ComplexUtils.real2Complex(kernel));
		case 4:
			return ftgpu.convolve(vector, ComplexUtils.real2Complex(kernel));
		}
		throw new RuntimeException("Invalid preference value");
	}
	
	public Complex[][] convolve(Complex[][] image, double[] kernel, int dim) {
		if (dim > 1) throw new RuntimeException("2D image, 1D kernel: Invalid dim specification");
		if (dim == 0) image = JVCLUtils.shiftDim(image);
		int height = image.length;
		for (int n = 0; n < height; n++) {
			image[n] = convolve(image[n], kernel);
		}
		if (dim == 0) {
			return JVCLUtils.shiftDim(image);
		} else {
			return image;
		}
	}
	
	public Complex[][] convolve(Complex[][] image, double[] kernel) {
		return convolve(image, kernel, 0);
	}
	
	public Complex[][][] convolve(Complex[][][] volume, double[] kernel, int dim) {
		if (dim == 0) volume = JVCLUtils.shiftDim(volume, 2);
		if (dim == 1) volume = JVCLUtils.shiftDim(volume, 1);
		if (dim > 2) throw new RuntimeException("Invalid dim");
		
		int volumeWidth = volume.length;
		int volumeHeight = volume[0].length;
		
		for (int x = 0; x < volumeWidth; x++) {
			for (int y = 0; y < volumeHeight; y++) {
				volume[x][y] = convolve(volume[x][y], kernel);
			}
		}
		return volume;
	}
	
	public Complex[][][] convolve(Complex[][][] volume, double[] kernel) {
		return convolve(volume, kernel, 0);
	}
	
	public Complex[] convolve(Complex[] vector, Complex[] kernel) {
		int pref = getPreferredConvolution(vector.length, kernel.length);
		switch(convolverPrefs[pref]) {
		case 3:
			return ftcpu.convolve(vector, kernel);
		case 4:
			return ftgpu.convolve(vector, kernel);
		}
		throw new RuntimeException("Invalid preference value");
	}
	
	public Complex[][] convolve(Complex[][] image, Complex[] kernel, int dim) {
		if (dim > 1) throw new RuntimeException("2D image, 1D kernel: Invalid dim specification");
		if (dim == 0) image = JVCLUtils.shiftDim(image);
		int height = image.length;
		for (int n = 0; n < height; n++) {
			image[n] = convolve(image[n], kernel);
		}
		if (dim == 0) {
			return JVCLUtils.shiftDim(image);
		} else {
			return image;
		}
	}
	
	public Complex[][] convolve(Complex[][] image, Complex[] kernel) {
		return convolve(image, kernel, 0);
	}
	
	public Complex[][][] convolve(Complex[][][] volume, Complex[] kernel, int dim) {
		if (dim == 0) volume = JVCLUtils.shiftDim(volume, 2);
		if (dim == 1) volume = JVCLUtils.shiftDim(volume, 1);
		if (dim > 2) throw new RuntimeException("Invalid dim");
		
		int volumeWidth = volume.length;
		int volumeHeight = volume[0].length;
		
		for (int x = 0; x < volumeWidth; x++) {
			for (int y = 0; y < volumeHeight; y++) {
				volume[x][y] = convolve(volume[x][y], kernel);
			}
		}
		return volume;
	}
	
	public Complex[][][] convolve(Complex[][][] volume, Complex[] kernel) {
		return convolve(volume, kernel, 0);
	}
	
	public double[][] convolve(double[][] vector, double[][] kernel) {
		int pref = getPreferredConvolution(vector.length, kernel.length);
		switch(convolverPrefs[pref]) {
		case 0:
			return FDCPUNaive.convolve(vector, kernel);
		case 1:
			return unrolled.convolve(vector, kernel);
		case 2:
			return fdgpu.convolve(vector, kernel);
		case 3:
			return ComplexUtils.complex2Real(ftcpu.convolve(ComplexUtils.real2Complex(vector), ComplexUtils.real2Complex(kernel)));
		case 4:
			return ComplexUtils.complex2Real(ftgpu.convolve(ComplexUtils.real2Complex(vector), ComplexUtils.real2Complex(kernel)));
		}
		throw new RuntimeException("Invalid preference value");
	}
	
	public Complex[][] convolve(Complex[][] vector, double[][] kernel) {
		int pref = getPreferredConvolution(vector.length, kernel.length);
		switch(convolverPrefs[pref]) {
		case 0:
			return ComplexUtils.split2Complex(
					FDCPUNaive.convolve(ComplexUtils.complex2Real(vector), kernel),
					FDCPUNaive.convolve(ComplexUtils.complex2Imaginary(vector), kernel)
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
			return ftcpu.convolve(vector, ComplexUtils.real2Complex(kernel));
		case 4:
			return ftgpu.convolve(vector, ComplexUtils.real2Complex(kernel));
		}
		throw new RuntimeException("Invalid preference value");
	}
	

	public Complex[][] convolve(Complex[][] vector, Complex[][] kernel) {
		int pref = getPreferredConvolution(vector.length, kernel.length);
		switch(convolverPrefs[pref]) {
		case 3:
			return ftcpu.convolve(vector, kernel);
		case 4:
			return ftgpu.convolve(vector, kernel);
		}
		throw new RuntimeException("Invalid preference value");
	}

	public double[][][] convolve(double[][][] vector, double[][][] kernel) {
		int pref = getPreferredConvolution(vector.length, kernel.length);
		switch(convolverPrefs[pref]) {
		case 0:
			return FDCPUNaive.convolve(vector, kernel);
		case 1:
			return unrolled.convolve(vector, kernel);
		case 2:
			return fdgpu.convolve(vector, kernel);
		case 3:
			return ComplexUtils.complex2Real(ftcpu.convolve(ComplexUtils.real2Complex(vector), ComplexUtils.real2Complex(kernel)));
		case 4:
			return ComplexUtils.complex2Real(ftgpu.convolve(ComplexUtils.real2Complex(vector), ComplexUtils.real2Complex(kernel)));
		}
		throw new RuntimeException("Invalid preference value");
	}
	
	public Complex[][][] convolve(Complex[][][] vector, double[][][] kernel) {
		int pref = getPreferredConvolution(vector.length, kernel.length);
		switch(convolverPrefs[pref]) {
		case 0:
			return ComplexUtils.split2Complex(
					FDCPUNaive.convolve(ComplexUtils.complex2Real(vector), kernel),
					FDCPUNaive.convolve(ComplexUtils.complex2Imaginary(vector), kernel)
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
			return ftcpu.convolve(vector, ComplexUtils.real2Complex(kernel));
		case 4:
			return ftgpu.convolve(vector, ComplexUtils.real2Complex(kernel));
		}
		throw new RuntimeException("Invalid preference value");
	}
	
	public Complex[][][] convolve(Complex[][][] vector, Complex[][][] kernel) {
		int pref = getPreferredConvolution(vector.length, kernel.length);
		switch(convolverPrefs[pref]) {
		case 3:
			return ftcpu.convolve(vector, kernel);
		case 4:
			return ftgpu.convolve(vector, kernel);
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
		int row = FastMath.max((int)(FastMath.log(JVCLUtils.nextPwr2(vl))/FastMath.log(2) - 6), 0);
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
		
	static public double[] applyBoundaries(double[] r, double[] g, int boundary) {
		final int ri = r.length;
		final int gi = g.length;
		final int hgi = (int)( gi / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi - 1 : hgi;
		switch (boundary) {
			case FULL:
				return r;
			case SAME:
				int rSameI = ri - hgi - hgie;
				double[] rSame = new double[rSameI];
				System.arraycopy(r, hgi, rSame, 0, rSameI);
				return rSame;
			case VALID:
				int rValidI = ri - 2*hgi - 2*hgie;
				double[] rValid = new double[rValidI];
				System.arraycopy(r, hgi + hgie, rValid, 0, rValidI);
				return rValid;
		}
		throw new RuntimeException("Invalid Boundary Condition");
	}
	
	static public double[][] applyBoundaries(double[][] r, double[][] g, int boundary) {
		final int ri = r.length;
		final int rj = r[0].length;
		final int gi = g.length;
		final int gj = g[0].length;
		final int hgi = (int)( (gi) / 2.0);
		final int hgj = (int)( (gj) / 2.0);
		//final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		//final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
		switch (boundary) {
			case FULL:
				return r;
			case SAME:
				int rSameI = ri - 2 * hgi;
				int rSameJ = rj - 2 * hgj;
				int sameStart = hgj;
				double[][] rSame = new double[rSameI][rSameJ];
				for (int i = 0; i < rSameI; i++) {
					System.arraycopy(r[i], sameStart, rSame[i], 0, rSameJ);
				}
				return rSame;
			case VALID:
				int rValidI = ri - 2 * (2 * hgi + 1);
				int rValidJ = rj - 2 * (2 * hgj + 1);
				int validStart = 2 * hgj + 1;
				double[][] rValid = new double[rValidI][rValidJ];
				for (int i = 0; i < rValidI; i++) {
					System.arraycopy(r, validStart, rValid, 0, rValidI);
				}
				return rValid;
		}
		throw new RuntimeException("Invalid Boundary Condition");
	}
	
	public static void main(String[] args) {
		int length = 16;
		Convolver c = new Convolver();
		double[] f = JVCLUtils.fillWithSecondOrder(length);
		double[] g = new double[] {-1, 1};
		JVCLUtils.display(
				applyBoundaries(
					c.convolve(f, g), 
				g, FULL),
			"Full", 8);
		JVCLUtils.display(
				applyBoundaries(
					c.convolve(f, g), 
				g, SAME),
			"Same", 8);
		JVCLUtils.display(
				applyBoundaries(
					c.convolve(f, g), 
				g, VALID),
			"Valid", 8);
	}
	
}


