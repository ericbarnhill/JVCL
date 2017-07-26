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
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * This class generates code for unrolled convolutions. At present, this code then needs to be pasted into
 * Unrolled.java and the code re-compiled. More streamlined approaches to adding unrolled kernels will be
 * developed in future releases. A commented out main method which contains some scratch work in this
 * direction has been left in the code.
 *
 * @author ericbarnhill
 * @since 0.1
 * @see Unrolled
 *
 */
public class Unroller {

	Path p;
	DecimalFormat fmtI2D, fmtJ2D, fmtI3D, fmtJ3D, fmtK3D;

	public Unroller() {
		fmtI2D = new DecimalFormat("+###;-###");
		fmtJ2D = new DecimalFormat("+####*ri;-####*ri");
		fmtI3D = new DecimalFormat("+###;-###");
		fmtJ3D = new DecimalFormat("+####*ri;-####*ri");
		fmtK3D = new DecimalFormat("+####*ri*rj;-####*ri*rj");
	}

	public Unroller(Path p) {
		fmtI2D = new DecimalFormat("+###;-###");
		fmtJ2D = new DecimalFormat("+####*ri;-####*ri");
		fmtI3D = new DecimalFormat("+###;-###");
		fmtJ3D = new DecimalFormat("+####*ri;-####*ri");
		fmtK3D = new DecimalFormat("+####*ri*rj;-####*ri*rj");
		this.p = p;
	}

	public void makeClassHead(Path p) {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("package jvcl;%n"));
		sb.append(String.format("%n"));
		sb.append(String.format("import org.apache.commons.numbers.complex.Complex;%n"));
		sb.append(String.format("public class Unrolled { %n"));
		sb.append(String.format("%n"));
		sb.append(String.format("}%n"));
		BufferedWriter w;
		try {
			w = new BufferedWriter(new FileWriter(p.toString(), false));
			w.write(sb.toString());
			w.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return;
	}

	public String makeMethodHead1dDouble(int gi) {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("// begin convolve_%s%n", Integer.toString(gi)));
		sb.append(
				String.format("    public static double[] convolve_%s(double[] f, double[] g) {%n", Integer.toString(gi))
			);
		sb.append(String.format("        final int fi = f.length;%n"));
		sb.append(String.format("        final int gi = g.length;%n"));
		sb.append(String.format("        final int hgi = (int)( (gi - 1) / 2.0);%n"));
		sb.append(String.format("        final int hgie = (gi %s 2 == 0) ? hgi + 1 : hgi;%n", "%"));
		sb.append(String.format("		 double[] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie);%n"));
		sb.append(String.format("		 double[] r = JVCLUtils.zeroPadBoundaries(new double[fi], hgi, hgie);%n"));
		sb.append(String.format("		 final int ri = r.length;%n"));
		sb.append(String.format("		 for (int i = hgi; i < ri-1-hgie; i++) {%n"));
		return sb.toString();
	}

	public String makeMethodHead1dComplex(int gi) {
		StringBuilder sb = new StringBuilder();
		sb.append(
				String.format("    public static Complex[] convolve_%s(Complex[] f, Complex[] g) {%n", Integer.toString(gi))
			);
		sb.append(String.format("        final int fi = f.length;%n"));
		sb.append(String.format("        final int gi = g.length;%n"));
		sb.append(String.format("        final int hgi = (int)( (gi - 1) / 2.0);%n"));
		sb.append(String.format("        final int hgie = (gi %s 2 == 0) ? hgi + 1 : hgi;%n", "%"));
		sb.append(String.format("		 Complex[] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie);%n"));
		sb.append(String.format("		 Complex[] r = JVCLUtils.zeroPadBoundaries(new Complex[fi], hgi, hgie);%n"));
		sb.append(String.format("		 final int ri = r.length;%n"));
		sb.append(String.format("		 for (int i = hgi; i < ri-1-hgie; i++) {%n"));
		return sb.toString();
	}

	public String makeMethodBody1dDouble(int gi) {
		int hgi = gi/2;
		StringBuilder sb = new StringBuilder();
		String iString = "";
		sb.append(String.format("                 r[i] = %n"));
		for (int i = 0; i < gi; i++) {
			int iIndex = i-hgi;
			int gIndex = gi-1-i;
			iString = fmtI2D.format(iIndex);
			sb.append(String.format("                 fPad[i%s]*g[%d]",iString, gIndex));
			if (i == gi-1) {
				sb.append(String.format(";%n"));
			} else {
				sb.append(String.format("+ %n"));
			}
		}
		return sb.toString();
	}

	public String makeMethodBody1dComplex(int gi) {
		int hgi = gi/2;
		StringBuilder sb = new StringBuilder();
		String iString = "";
		sb.append(String.format("                 r[i] %n"));
		for (int i = 0; i < gi; i++) {
			int iIndex = i-hgi;
			int gIndex = gi-1-i;
			iString = fmtI2D.format(iIndex);
			sb.append(String.format("                 .add(fPad[i%s].multiply(g[%d]))",iString, gIndex));
			if (i == gi-1) {
				sb.append(String.format(";%n"));
			} else {
				sb.append(String.format("%n"));
			}
		}
		return sb.toString();
	}

	public String makeMethodTail1dDouble() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("            }%n"));
		sb.append(String.format("        return Boundaries.finishBoundaries1d(fPad, g, r, gi,hgi, hgie,ri);%n"));
		sb.append(String.format("    };%n"));
		return sb.toString();
	}

	public String makeMethodTail1dComplex(int gi) {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("            }%n"));
		sb.append(String.format("        return Boundaries.finishBoundaries1d(fPad, g, r, gi, hgi, hgie, ri);%n"));
		sb.append(String.format("    };%n"));
		sb.append(String.format("// end convolve_%s%n", Integer.toString(gi)));
		return sb.toString();
	}

	public String makeMethodHead2dDouble(int gi, int gj) {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format(" // begin convolve_%s_%s%n", Integer.toString(gi), Integer.toString(gj)));
		sb.append(
				String.format("    public static double[][] convolve_%s_%s(double[][] f, double[][] g) {%n", Integer.toString(gi), Integer.toString(gj))
				//String.format("    public double[][] convolve_%s_%s() {%n", Integer.toString(gi), Integer.toString(gj))
			);
		sb.append(String.format("        final int fi = f.length;%n"));
		sb.append(String.format("        final int fj = f[0].length;%n"));
		sb.append(String.format("        final int gi = g.length;%n"));
		sb.append(String.format("        final int gj = g[0].length;%n"));
		sb.append(String.format("        final int hgi = (int)( (gi - 1) / 2.0);%n"));
		sb.append(String.format("        final int hgj = (int)( (gj - 1) / 2.0);%n"));
		sb.append(String.format("        final int hgie = (gi %s 2 == 0) ? hgi + 1 : hgi;%n", "%"));
		sb.append(String.format("        final int hgje = (gj %s 2 == 0) ? hgj + 1 : hgj;%n", "%"));
		sb.append(String.format("        double[] gg = JVCLUtils.vectorize(g);%n"));
		sb.append(String.format("        double[] fPad = JVCLUtils.vectorize(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje));%n"));
		sb.append(String.format("        double[][] rr = JVCLUtils.zeroPadBoundaries(new double[fi][fj], hgi, hgie, hgj, hgje);%n"));
		sb.append(String.format("        final int ri = rr.length;%n"));
		sb.append(String.format("        final int rj = rr[0].length;%n"));
		sb.append(String.format("        double[] r = JVCLUtils.vectorize(rr);%n"));
		sb.append(String.format("        int ij;%n"));
		sb.append(String.format("        for (int i = hgi; i < fi - 1 - hgie; i++) {%n"));
		sb.append(String.format("            for (int j = hgj; j < fj - 1 - hgje; j++) {%n"));
		sb.append(String.format("                ij = j*ri + i;%n"));
		return sb.toString();
	}

	public String makeMethodHead2dComplex(int gi, int gj) {
		StringBuilder sb = new StringBuilder();
		sb.append(
				String.format("    public static Complex[][] convolve_%s_%s(Complex[][] f, Complex[][] g) {%n", Integer.toString(gi), Integer.toString(gj))
				//String.format("    public Complex[][] convolve_%s_%s() {%n", Integer.toString(gi), Integer.toString(gj))
			);
		sb.append(String.format("        final int fi = f.length;%n"));
		sb.append(String.format("        final int fj = f[0].length;%n"));
		sb.append(String.format("        final int gi = g.length;%n"));
		sb.append(String.format("        final int gj = g[0].length;%n"));
		sb.append(String.format("        final int hgi = (int)( (gi - 1) / 2.0);%n"));
		sb.append(String.format("        final int hgj = (int)( (gj - 1) / 2.0);%n"));
		sb.append(String.format("        final int hgie = (gi %s 2 == 0) ? hgi + 1 : hgi;%n", "%"));
		sb.append(String.format("        final int hgje = (gj %s 2 == 0) ? hgj + 1 : hgj;%n", "%"));
		sb.append(String.format("        Complex[] gg = JVCLUtils.vectorize(g);%n"));
		sb.append(String.format("        Complex[] fPad = JVCLUtils.vectorize(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje));%n"));
		sb.append(String.format("        Complex[][] rr = JVCLUtils.zeroPadBoundaries(new Complex[fi][fj], hgi, hgie, hgj, hgje);%n"));
		sb.append(String.format("        final int ri = rr.length;%n"));
		sb.append(String.format("        final int rj = rr[0].length;%n"));
		sb.append(String.format("        Complex[] r = JVCLUtils.vectorize(rr);%n"));
		sb.append(String.format("        int ij;%n"));
		sb.append(String.format("        for (int i = hgi; i < fi - 1 - hgie; i++) {%n"));
		sb.append(String.format("            for (int j = hgj; j < fj - 1 - hgje; j++) {%n"));
		sb.append(String.format("                ij = j*ri + i;%n"));
		return sb.toString();
	}

	public String makeMethodBody2dDouble(int gi, int gj) {
		int hgi = gi/2;
		int hgj = gj/2;
		StringBuilder sb = new StringBuilder();
		String iString = ""; String jString = "";
		sb.append(String.format("                 r[ij] = %n"));
		for (int j = 0; j < gj; j++) {
			for (int i = 0; i < gi; i++) {
				int gIndex = j*gi + i;
				int iIndex = i-hgi;
				int jIndex = j-hgj;
				iString = fmtI2D.format(iIndex);
				jString = fmtJ2D.format(jIndex);
				sb.append(String.format("                 fPad[ij%s%s]*gg[%d]",jString,iString, gIndex));
				if (i == gi-1 && j == gj-1) {
					sb.append(String.format(";%n"));
				} else {
					sb.append(String.format("+ %n"));
				}
			}
		}

		return sb.toString();
	}

	public String makeMethodBody2dComplex(int gi, int gj) {
		int hgi = gi/2;
		int hgj = gj/2;
		StringBuilder sb = new StringBuilder();
		String iString = ""; String jString = "";
		sb.append(String.format("				r[ij] %n"));
		for (int j = 0; j < gj; j++) {
			for (int i = 0; i < gi; i++) {
				int gIndex = j*gi + i;
				int iIndex = i-hgi;
				int jIndex = j-hgj;
				iString = fmtI2D.format(iIndex);
				jString = fmtJ2D.format(jIndex);
				sb.append(String.format("                .add(fPad[ij%s%s].multiply(gg[%d]))",jString,iString, gIndex));
				if (i == gi-1 && j == gj-1) {
					sb.append(String.format(";%n"));
				} else {
					sb.append(String.format("%n"));
				}
			}
		}

		return sb.toString();
	}

	public String makeMethodTail2dDouble() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("                }%n"));
		sb.append(String.format("            }%n"));
		sb.append(String.format("        return Boundaries.finishBoundaries2d(fPad, gg, r, gi, gj, hgi, hgie, hgj, hgje, ri, rj);%n"));
		sb.append(String.format("    };%n"));
		return sb.toString();
	}

	public String makeMethodTail2dComplex(int gi, int gj) {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("                }%n"));
		sb.append(String.format("            }%n"));
		sb.append(String.format("    return Boundaries.finishBoundaries2d(fPad, gg, r, gi, gj, hgi, hgie, hgj, hgje, ri, rj)%n;"));
		sb.append(String.format("    };%n"));
		sb.append(String.format(" // end convolve_%s_%s%n", Integer.toString(gi), Integer.toString(gj)));
		return sb.toString();
	}

	public String makeMethodHead3dDouble(int gi, int gj, int gk) {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("// begin convolve_%s_%s_%s%n", Integer.toString(gi), Integer.toString(gj), Integer.toString(gk)));
		sb.append(String.format("    public static double[][][] convolve_%s_%s_%s(double[][][] f, double[][][] g) {%n", Integer.toString(gi), Integer.toString(gj),
				Integer.toString(gk)));
		sb.append(String.format("        final int fi = f.length;%n"));
		sb.append(String.format("        final int fj = f[0].length;%n"));
		sb.append(String.format("        final int fk = f[0][0].length;%n"));
		sb.append(String.format("        final int gi = g.length;%n"));
		sb.append(String.format("        final int gj = g[0].length;%n"));
		sb.append(String.format("        final int gk = g[0][0].length;%n"));
		sb.append(String.format("        final int hgi = (int)( (gi - 1) / 2.0);%n"));
		sb.append(String.format("        final int hgj = (int)( (gj - 1) / 2.0);%n"));
		sb.append(String.format("        final int hgk = (int)( (gk - 1) / 2.0);%n"));
		sb.append(String.format("        final int hgie = (gi %s 2 == 0) ? hgi + 1 : hgi;%n", "%"));
		sb.append(String.format("        final int hgje = (gj %s 2 == 0) ? hgj + 1 : hgj;%n", "%"));
		sb.append(String.format("        final int hgke = (gk %s 2 == 0) ? hgk + 1 : hgk;%n", "%"));
		sb.append(String.format("        double[] gg = JVCLUtils.vectorize(g);%n"));
		sb.append(String.format("        double[] fPad = JVCLUtils.vectorize(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje, hgk, hgke));%n"));
		sb.append(String.format("        double[][][] rr = JVCLUtils.zeroPadBoundaries(new double[fi][fj][fk], hgi, hgie, hgj, hgje, hgk, hgke);%n"));
		sb.append(String.format("        final int ri = rr.length;%n"));
		sb.append(String.format("        final int rj = rr[0].length;%n"));
		sb.append(String.format("        final int rk = rr[0][0].length;%n"));
		sb.append(String.format("        double[] r = JVCLUtils.vectorize(rr);%n"));
		sb.append(String.format("        int ijk;%n"));
		sb.append(String.format("        for (int i = hgi; i < fi - 1 - hgie; i++) {%n"));
		sb.append(String.format("            for (int j = hgj; j < fj - 1 - hgje; j++) {%n"));
		sb.append(String.format("            	for (int k = hgk; k < fk - 1 - hgke; k++) {%n"));
		sb.append(String.format("               	ijk = k*ri*rj + j*ri + i;%n"));
		return sb.toString();
	}


	public String makeMethodHead3dComplex(int gi, int gj, int gk) {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("    public static Complex[][][] convolve_%s_%s_%s(Complex[][][] f, Complex[][][] g) {%n", Integer.toString(gi), Integer.toString(gj), Integer.toString(gk)));
		sb.append(String.format("        final int fi = f.length;%n"));
		sb.append(String.format("        final int fj = f[0].length;%n"));
		sb.append(String.format("        final int fk = f[0][0].length;%n"));
		sb.append(String.format("        final int gi = g.length;%n"));
		sb.append(String.format("        final int gj = g[0].length;%n"));
		sb.append(String.format("        final int gk = g[0][0].length;%n"));
		sb.append(String.format("        final int hgi = (int)( (gi - 1) / 2.0);%n"));
		sb.append(String.format("        final int hgj = (int)( (gj - 1) / 2.0);%n"));
		sb.append(String.format("        final int hgk = (int)( (gk - 1) / 2.0);%n"));
		sb.append(String.format("        final int hgie = (gi %s 2 == 0) ? hgi + 1 : hgi;%n", "%"));
		sb.append(String.format("        final int hgje = (gj %s 2 == 0) ? hgj + 1 : hgj;%n", "%"));
		sb.append(String.format("        final int hgke = (gk %s 2 == 0) ? hgk + 1 : hgk;%n", "%"));
		sb.append(String.format("        Complex[] gg = JVCLUtils.vectorize(g);%n"));
		sb.append(String.format("        Complex[] fPad = JVCLUtils.vectorize(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje, hgk, hgke));%n"));
		sb.append(String.format("        Complex[][][] rr = JVCLUtils.zeroPadBoundaries(new Complex[fi][fj][fk], hgi, hgie, hgj, hgje, hgk, hgke);%n"));
		sb.append(String.format("        final int ri = rr.length;%n"));
		sb.append(String.format("        final int rj = rr[0].length;%n"));
		sb.append(String.format("        final int rk = rr[0][0].length;%n"));
		sb.append(String.format("        Complex[] r = JVCLUtils.vectorize(rr);%n"));
		sb.append(String.format("        int ijk;%n"));
		sb.append(String.format("        for (int i = hgi; i < fi - 1 - hgie; i++) {%n"));
		sb.append(String.format("            for (int j = hgj; j < fj - 1 - hgje; j++) {%n"));
		sb.append(String.format("            	for (int k = hgk; k < fk - 1 - hgke; k++) {%n"));
		sb.append(String.format("               	ijk = k*ri*rj + j*ri + i;%n"));
		return sb.toString();
	}

	public String makeMethodBody3dDouble(int gi, int gj, int gk) {
		int hgi = gi/2;
		int hgj = gj/2;
		int hgk = gk/2;
		StringBuilder sb = new StringBuilder();
		String iString = ""; String jString = "";
		@SuppressWarnings("unused")
		String kString="";
		sb.append(String.format("	                 r[ijk] = %n"));
		for (int k = 0; k < gk; k++) {
			for (int j = 0; j < gj; j++) {
				for (int i = 0; i < gi; i++) {
					int gIndex = k*gi*gj + j*gi + i;
					int iIndex = i-hgi;
					int jIndex = j-hgj;
					int kIndex = k-hgk;
					iString = fmtI3D.format(iIndex);
					jString = fmtJ3D.format(jIndex);
					kString = fmtK3D.format(kIndex);
					sb.append(String.format("    	             fPad[ijk%s%s]*gg[%d]",jString,iString, gIndex));
					if (i == gi-1 && j == gj-1 && k == gk-1) {
						sb.append(String.format(";%n"));
					} else {
						sb.append(String.format("+ %n"));
					}
				}
			}
		}

		return sb.toString();
	}

	public String makeMethodBody3dComplex(int gi, int gj, int gk) {
		int hgi = gi/2;
		int hgj = gj/2;
		int hgk = gk/2;
		StringBuilder sb = new StringBuilder();
		String iString = ""; String jString = "";
		@SuppressWarnings("unused")
		String kString="";
		sb.append(String.format("         	        r[ijk] %n"));
		for (int k = 0; k < gk; k++) {
			for (int j = 0; j < gj; j++) {
				for (int i = 0; i < gi; i++) {
					int gIndex = k*gi*gj + j*gi + i;
					int iIndex = i-hgi;
					int jIndex = j-hgj;
					int kIndex = k-hgk;
					iString = fmtI3D.format(iIndex);
					jString = fmtJ3D.format(jIndex);
					kString = fmtK3D.format(kIndex);
					sb.append(String.format("       	          .add(fPad[ijk%s%s].multiply(gg[%d]))",jString,iString, gIndex));
					if (i == gi-1 && j == gj-1 && k == gk-1) {
						sb.append(String.format(";%n"));
					} else {
						sb.append(String.format("%n"));
					}
				}
			}
		}

		return sb.toString();
	}

	public String makeMethodTail3dDouble() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("     		           }%n"));
		sb.append(String.format("                }%n"));
		sb.append(String.format("            }%n"));
		sb.append(String.format("        return Boundaries.finishBoundaries3d(fPad, gg, r, gi, gj, gk, hgi, hgie, hgj, hgje, hgk, hgke, ri, rj, rk);%n"));
		sb.append(String.format("    };%n"));
		return sb.toString();
	}

	public String makeMethodTail3dComplex(int gi, int gj, int gk) {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("     	        	   }%n"));
		sb.append(String.format("                }%n"));
		sb.append(String.format("            }%n"));
		sb.append(String.format("        return Boundaries.finishBoundaries3d(fPad, gg, r, gi, gj, gk, hgi, hgie, hgj, hgje, hgk, hgke, ri, rj, rk)%n;"));
		sb.append(String.format("    };%n"));
		sb.append(String.format("// end convolve_%s_%s_%s(double[][][] f, double[][][] g) {%n", Integer.toString(gi), Integer.toString(gj), Integer.toString(gk)));
		return sb.toString();
	}


	public void setPath() {
		try {
			System.getProperty("jvcl.unrolledsrcpath");
		} catch (SecurityException e) {
			System.out.println("Security exception, cannot create codegen file");
		} catch (IllegalArgumentException e) { // not found
			System.setProperty("jvcl.unrolledsrcpath", p.toString());
		}
	}

	public String makeMethodDouble(int i) {
		StringBuilder sb = new StringBuilder();
		sb.append(makeMethodHead1dDouble(i));
		sb.append(makeMethodBody1dDouble(i));
		sb.append(makeMethodTail1dDouble());
		return sb.toString();
	}

	public String makeMethodComplex(int i) {
		StringBuilder sb = new StringBuilder();
		sb.append(makeMethodHead1dComplex(i));
		sb.append(makeMethodBody1dComplex(i));
		sb.append(makeMethodTail1dComplex(i));
		return sb.toString();
	}

	public String makeMethodDouble(int i, int j) {
		StringBuilder sb = new StringBuilder();
		sb.append(makeMethodHead2dDouble(i, j));
		sb.append(makeMethodBody2dDouble(i, j));
		sb.append(makeMethodTail2dDouble());
		return sb.toString();
	}

	public String makeMethodComplex(int i, int j) {
		StringBuilder sb = new StringBuilder();
		sb.append(makeMethodHead2dComplex(i, j));
		sb.append(makeMethodBody2dComplex(i, j));
		sb.append(makeMethodTail2dComplex(i, j));
		return sb.toString();
	}

	public String makeMethodDouble(int i, int j, int k) {
		StringBuilder sb = new StringBuilder();
		sb.append(makeMethodHead3dDouble(i, j, k));
		sb.append(makeMethodBody3dDouble(i, j, k));
		sb.append(makeMethodTail3dDouble());
		return sb.toString();
	}

	public String makeMethodComplex(int i, int j, int k) {
		StringBuilder sb = new StringBuilder();
		sb.append(makeMethodHead3dComplex(i, j, k));
		sb.append(makeMethodBody3dComplex(i, j, k));
		sb.append(makeMethodTail3dComplex(i, j, k));
		return sb.toString();
	}


	public void addKernel(ArrayList<Integer> dims, Path kernelSrcPath) {
		switch(dims.size()) {
		case 1:
			try {
				StringBuilder currentFile = new StringBuilder();
				Scanner s = new Scanner(new BufferedReader(new FileReader(kernelSrcPath.toString())));
				String previousLine = "";
				while (s.hasNextLine()) {
					currentFile.append(previousLine);
					currentFile.append(String.format("%n"));
					previousLine = s.nextLine();
				} // drops final bracket
				s.close();
				currentFile.append(makeMethodDouble(dims.get(0)));
				currentFile.append(makeMethodComplex(dims.get(0)));
				currentFile.append(String.format("}%n"));
				BufferedWriter w = new BufferedWriter(new FileWriter(kernelSrcPath.toString(), false));
				w.write(currentFile.toString());
				w.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		break;
		case 2:
			try {
				StringBuilder currentFile = new StringBuilder();
				Scanner s = new Scanner(new BufferedReader(new FileReader(kernelSrcPath.toString())));
				String previousLine = "";
				while (s.hasNextLine()) {
					currentFile.append(previousLine);
					currentFile.append(String.format("%n"));
					previousLine = s.nextLine();
				} // drops final bracket
				s.close();
				currentFile.append(makeMethodDouble(dims.get(0), dims.get(1)));
				currentFile.append(makeMethodComplex(dims.get(0), dims.get(1)));
				currentFile.append(String.format("}%n"));
				BufferedWriter w = new BufferedWriter(new FileWriter(kernelSrcPath.toString(), false));
				w.write(currentFile.toString());
				w.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		break;
		case 3:
			try {
				StringBuilder currentFile = new StringBuilder();
				Scanner s = new Scanner(new BufferedReader(new FileReader(kernelSrcPath.toString())));
				String previousLine = "";
				while (s.hasNextLine()) {
					currentFile.append(previousLine);
					currentFile.append(String.format("%n"));
					previousLine = s.nextLine();
				} // drops final bracket
				s.close();
				currentFile.append(makeMethodDouble(dims.get(0), dims.get(1), dims.get(2)));
				currentFile.append(makeMethodComplex(dims.get(0), dims.get(1), dims.get(2)));
				currentFile.append(String.format("}%n"));
				BufferedWriter w = new BufferedWriter(new FileWriter(kernelSrcPath.toString(), false));
				w.write(currentFile.toString());
				w.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	/*
	public static void main(String[] args) {

		Path pp = Paths.get("/home/ericbarnhill/Documents/code/Unrolled.java");
		Unroller u = new Unroller(pp);
		int iSize = 7;
		int jSize = 7;
		System.out.println(unroll);
		StringBuilder currentFile = new StringBuilder();
		try {
			if (!Files.exists(pp)) {
				new FileOutputStream(pp.toString()).close();
			}
			Scanner s = new Scanner(new BufferedReader(new FileReader(pp.toString())));
			String previousLine = "";
			while (s.hasNextLine()) {
				currentFile.append(previousLine);
				currentFile.append(String.format("%n"));
				previousLine = s.nextLine();
			} // drops final bracket
			s.close();
			currentFile.append(unroll);
			currentFile.append(String.format("}%n"));
			BufferedWriter w = new BufferedWriter(new FileWriter(pp.toString(), false));
			w.write(currentFile.toString());
			w.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		int errorCode = com.sun.tools.javac.Main.compile(new String[] {
	            "-d", "/home/ericbarnhill/barnhill-eclipse-workspace/JVCL/bin",
	            "/home/ericbarnhill/Documents/code/Unrolled.java"});
		String newMethodName = "unr.convolve_"+Integer.toString(iSize)+"_"+Integer.toString(jSize)+"(f, g)";
		ClassLoader loader = Unroller.class.getClassLoader();
		Class<?> c;
		try {
			c = loader.loadClass("jvcl.Unrolled");
			Object cInst = c.newInstance();
			double[][] f = JVCLUtils.fillWithSecondOrder(64, 64);
			double[][] g = JVCLUtils.fillWithSecondOrder(7, 7);
			Method[] methods = c.getDeclaredMethods();
			Class<?>[] paramTypes = new Class<?>[2];
			paramTypes[0] = double[][].class;
			paramTypes[1] = double[][].class;
			Object[] arguments = new Object[2];
			arguments[0] = f;
			arguments[1] = g;
			Method convMethod = c.getDeclaredMethod("convolve_7_7", paramTypes);
			convMethod.invoke(cInst, arguments);

			/*
			// Create or retrieve a JexlEngine
            JexlEngine jexl = new JexlEngine();
            JexlContext jc = new ObjectContext<Unrolled>(jexl, cInst);
			//Unrolled unr = jexl.newInstance(Unrolled.class);
            // Create an expression object
            //String jexlExp = "foo.innerFoo.bar()";

            // Create a context and add data
			double[][] res;
			//jc.set("u", cInst);
            jc.set("f", JVCLUtils.fillWithSecondOrder(64, 64));
            jc.set("g", new double[][] {{0, 1, 0}, {1, -4, 1}, {0, 1, 0}});
            //String expr = "u.convolve_7_7(f, g)";
            //Expression e = jexl.createExpression(expr);
            //res = (double[][])e.evaluate(jc);
            String output = (String)jc.get("printF()");
            System.out.format("result test %s %n", output);
            res = (double[][])jc.get("convolve_7_7()");
            cInst.convolve_7_7();

		} catch (ClassNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (InstantiationException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IllegalAccessException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (NoSuchMethodException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (SecurityException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalArgumentException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvocationTargetException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	*/
}
