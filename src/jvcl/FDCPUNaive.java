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

import java.util.Arrays;

import org.apache.commons.math4.complex.Complex;
import org.apache.commons.math4.complex.ComplexUtils;

public class FDCPUNaive {

	/** Default constructor sets zero boundary conditions 
	*/
	public FDCPUNaive() {
	}
	
	/*
	 * A glossary of variables
	 * f: array
	 * g: kernel
	 * r: result
	 * fi, fj, fk: array dimension
	 * gi, gj, gk: kernel dimension
	 * hgi, hgj, hgk = kernel half dimension
	 * ai, aj, ak = adjusted dimension
	 * i, j, k: array loops
	 * p, q, s: kernel loops
	 */

	public static double[] convolve(double[] f, double[] g) {
		
		final int fi = f.length;
		final int gi = g.length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		double[] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie);
		double[] r = JVCLUtils.zeroPadBoundaries(new double[fi], hgi, hgie);
		final int ri = r.length;
		int ai;
		for (int i = 0; i < ri; i++) {
			for (int p =  0; p < gi; p++) {
				ai = i + p - hgie;
				if (ai >= 0 && ai < ri) {
					r[i] += fPad[ai]*g[gi-1-p];
				}
			}
		}
		return r;
	}
	
	public static Complex[] convolve(Complex[] f, Complex[] g) {

		final int fi = f.length;
		final int gi = g.length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		Complex[] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie);
		Complex[] r = JVCLUtils.zeroPadBoundaries(ComplexUtils.initialize(new Complex[fi]), hgi, hgie);
		final int ri = r.length;
		int ai;
		for (int i = 0; i < ri; i++) {
			for (int p =  0; p < gi; p++) {
				ai = i + p - hgie;
				if (ai >= 0 && ai < ri) {
					r[i] = r[i].add(fPad[ai].multiply(g[gi-1-p]));
				}
			}
		}
		return r;
	}
	
	public static double[][] convolve(double[][] f, double[][] g) {
		final int fi = f.length;
		final int fj = f[0].length;
		final int gi = g.length;
		final int gj = g[0].length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgj = (int)( (gj - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
		double[][] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje);
		double[][] r = JVCLUtils.zeroPadBoundaries(new double[fi][fj], hgi, hgie, hgj, hgje);
		final int ri = r.length;
		final int rj = r[0].length;
		int ai, aj;
		for (int i = 0; i < ri; i++) {
			for (int j = 0; j < rj; j++) {
				for (int p = 0; p < gi; p++) {
					for (int q = 0; q < gj; q++) {
						ai = i + (p - hgie);
						aj = j + (q - hgje);
						if (ai >= 0 && ai < ri) {
							if (aj >= 0 && aj < rj) {
								r[i][j] += fPad[ai][aj]*g[gi-1-p][gj-1-q];
							}
						}
					}
				}
			}
		}
		return r;
	}
	
	public static Complex[][] convolve(Complex[][] f, Complex[][] g) {
		final int fi = f.length;
		final int fj = f[0].length;
		final int gi = g.length;
		final int gj = g[0].length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgj = (int)( (gj - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
		Complex[][] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje);
		Complex[][] r = JVCLUtils.zeroPadBoundaries(
							ComplexUtils.initialize(new Complex[fi][fj]),
						hgi, hgie, hgj, hgje);
		final int ri = r.length;
		final int rj = r[0].length;
		int ai, aj;
		for (int i = 0; i < ri; i++) {
			for (int j = 0; j < rj; j++) {
				for (int p = 0; p < gi; p++) {
					for (int q = 0; q < gj; q++) {
						ai = i + (p - hgie);
						aj = j + (q - hgje);
						if (ai >= 0 && ai < ri) {
							if (aj >= 0 && aj < rj) {
								r[i][j] = r[i][j].add(fPad[ai][aj].multiply(g[gi-1-p][gj-1-q]));
							}
						}
					}
				}
			}
		}
		return r;
	}
	
	public static double[][][] convolve(double[][][] f, double[][][] g) {
		final int fi = f.length;
		final int fj = f[0].length;
		final int fk = f[0][0].length;
		final int gi = g.length;
		final int gj = g[0].length;
		final int gk = g[0][0].length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgj = (int)( (gj - 1) / 2.0);
		final int hgk = (int)( (gk - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
		final int hgke = (gk % 2 == 0) ? hgk + 1 : hgk;
		double[][][] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje, hgk, hgke);
		double[][][] r = JVCLUtils.zeroPadBoundaries(new double[fi][fj][fk], hgi, hgie, hgj, hgje, hgk, hgke);
		final int ri = r.length;
		final int rj = r[0].length;
		final int rk = r[0][0].length;
		int ai, aj, ak;
		for (int i = 0; i < ri; i++) {
			for (int j = 0; j < rj; j++) {
				for (int k = 0; k < rk; k++) {
					for (int p = 0; p < gi; p++) {
						for (int q = 0; q < gj; q++) {
							for (int s = 0; s < gk; s++) {
								ai = i + (p - hgie);
								aj = j + (q - hgje);
								ak = k + (s - hgke);
								if (ai >= 0 && ai < ri) {
									if (aj >= 0 && aj < rj) {
										if (ak >= 0 && ak < rk) {
											r[i][j][k] += fPad[ai][aj][ak]*g[gi-1-p][gj-1-q][gk-1-s];
										}
									}
								}
							}
						}
					}
				}
			}
		}
		return r;
	}
	
	public static Complex[][][] convolve(Complex[][][] f, Complex[][][] g) {
		final int fi = f.length;
		final int fj = f[0].length;
		final int fk = f[0][0].length;
		final int gi = g.length;
		final int gj = g[0].length;
		final int gk = g[0][0].length;
		final int hgi = (int)( (gi - 1) / 2.0);
		final int hgj = (int)( (gj - 1) / 2.0);
		final int hgk = (int)( (gk - 1) / 2.0);
		final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
		final int hgke = (gk % 2 == 0) ? hgk + 1 : hgk;
		Complex[][][] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje, hgk, hgke);
		Complex[][][] r = JVCLUtils.zeroPadBoundaries(
								ComplexUtils.initialize(new Complex[fi][fj][fk]),
							hgi, hgie, hgj, hgje, hgk, hgke);
		final int ri = r.length;
		final int rj = r[0].length;
		final int rk = r[0][0].length;
		int ai, aj, ak;
		for (int i = 0; i < ri; i++) {
			for (int j = 0; j < rj; j++) {
				for (int k = 0; k < rk; k++) {
					for (int p = 0; p < gi; p++) {
						for (int q = 0; q < gj; q++) {
							for (int s = 0; s < gk; s++) {
								ai = i + (p - hgie);
								aj = j + (q - hgje);
								ak = k + (s - hgke);
								if (ai >= 0 && ai < ri) {
									if (aj >= 0 && aj < rj) {
										if (ak >= 0 && ak < rk) {
											r[i][j][k] = r[i][j][k].add(
													fPad[ai][aj][ak].multiply(g[gi-1-p][gj-1-q][gk-1-s])
												);
										}
									}
								}
							}
						}
					}
				}
			}
		}
		return r;
	}
	
	
	
	public static void main(String[] args) {
		double[] f1 = JVCLUtils.fillWithSecondOrder(16);
		double[] g1 = new double[] {1, -2, 1};
		JVCLUtils.display(convolve(f1, g1), "1D Double", 16);
		Complex[] f2 = JVCLUtils.fillWithSecondOrderComplex(16);
		Complex[] g2 = ComplexUtils.real2Complex(g1);
		JVCLUtils.display(convolve(f2, g2), "1D Complex", 16);
		double[][] f3 = JVCLUtils.fillWithSecondOrder(16, 16);
		JVCLUtils.display(JVCLUtils.vectorise(f3), "Second Order", f3.length);
		double[][] g3 = JVCLUtils.devectorise(new double[] {0,1,0, 0, 1,-4,1, 0, 0,1,0, 0}, 4);
		JVCLUtils.display(JVCLUtils.vectorise(g3), "Second Order Kernel", g3.length);
		double[][] r3 = convolve(f3, g3);
		JVCLUtils.display(JVCLUtils.vectorise(r3), "2D Double", r3.length);
		Complex[][] f4 = JVCLUtils.fillWithSecondOrderComplex(16, 16);
		Complex[][] g4 = ComplexUtils.real2Complex(g3);
		Complex[][] r4 = convolve(f4, g4);
		JVCLUtils.display(JVCLUtils.vectorise(r4), "2D Complex", r4.length);
		double[][][] f5 = JVCLUtils.fillWithSecondOrder(8, 8, 8);
		double[][][] g5 = JVCLUtils.devectorise(new double[] {
				0,0,0,0,1,0,0,0,0,0,1,0,1,-6,1,0,1,0,0,0,0,0,1,0,0,0,0}, 3, 3);
		JVCLUtils.display(JVCLUtils.vectorise(g5), "Third Order Kernel", g5.length);
		double[][][] r5 = convolve(f5, g5);
		JVCLUtils.display(JVCLUtils.vectorise(r5), "3D Double", r5.length);
		Complex[][][] f6 = JVCLUtils.fillWithSecondOrderComplex(8, 8, 8);
		Complex[][][] g6 = ComplexUtils.real2Complex(g5);
		Complex[][][] r6 = convolve(f6, g6);
		JVCLUtils.display(JVCLUtils.vectorise(r6), "3D Complex", r6.length);
	}

	
}
