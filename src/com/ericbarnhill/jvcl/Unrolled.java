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

import org.apache.commons.math4.complex.Complex;

import com.ericbarnhill.arrayMath.ArrayMath;

/**
 * This class provides unrolled methods for many common convolutions.
 * Additional unrolled methods can be generated easily using the Unroller class, added to this class and the class recompiled.
 * This will be facilitated in future releases.
 *
 * @author ericbarnhill
 * @since 0.1
 * @see Unroller
 *
 */
public class Unrolled {

// begin convolve_3
    public static double[] convolve_3(double[] f, double[] g) {
        final int fi = f.length;
        final int gi = g.length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		 double[] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie);
		 double[] r = JVCLUtils.zeroPadBoundaries(new double[fi], hgi, hgie);
		 final int ri = r.length;
		 for (int i = hgi; i < ri-1-hgie; i++) {
                 r[i] =
                 fPad[i-1]*g[2]+
                 fPad[i+0]*g[1]+
                 fPad[i+1]*g[0];
            }
        return Boundaries.finishBoundaries1d(fPad, g, r, gi,hgi, hgie,ri);
    };
    public static Complex[] convolve_3(Complex[] f, Complex[] g) {
        final int fi = f.length;
        final int gi = g.length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		 Complex[] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie);
		 Complex[] r = JVCLUtils.zeroPadBoundaries(new Complex[fi], hgi, hgie);
		 final int ri = r.length;
		 for (int i = hgi; i < ri-1-hgie; i++) {
                 r[i]
                 .add(fPad[i-1].multiply(g[2]))
                 .add(fPad[i+0].multiply(g[1]))
                 .add(fPad[i+1].multiply(g[0]));
            }
        return Boundaries.finishBoundaries1d(fPad, g, r, gi, hgi, hgie, ri);
    };
// end convolve_3
 // begin convolve_3_3
    public static double[][] convolve_3_3(double[][] f, double[][] g) {
        final int fi = f.length;
        final int fj = f[0].length;
        final int gi = g.length;
        final int gj = g[0].length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgj = (int)( (gj - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
        final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
        double[] gg = ArrayMath.vectorise(g);
        double[] fPad = ArrayMath.vectorise(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje));
        double[][] rr = JVCLUtils.zeroPadBoundaries(new double[fi][fj], hgi, hgie, hgj, hgje);
        final int ri = rr.length;
        final int rj = rr[0].length;
        double[] r = ArrayMath.vectorise(rr);
        int ij;
        for (int i = hgi; i < fi - 1 - hgie; i++) {
            for (int j = hgj; j < fj - 1 - hgje; j++) {
                ij = j*ri + i;
                 r[ij] =
                 fPad[ij-1*ri-1]*gg[0]+
                 fPad[ij-1*ri+0]*gg[1]+
                 fPad[ij-1*ri+1]*gg[2]+
                 fPad[ij+0*ri-1]*gg[3]+
                 fPad[ij+0*ri+0]*gg[4]+
                 fPad[ij+0*ri+1]*gg[5]+
                 fPad[ij+1*ri-1]*gg[6]+
                 fPad[ij+1*ri+0]*gg[7]+
                 fPad[ij+1*ri+1]*gg[8];
                }
            }
        return Boundaries.finishBoundaries2d(fPad, gg, r, gi, gj, hgi, hgie, hgj, hgje, ri, rj);
    };
    public static Complex[][] convolve_3_3(Complex[][] f, Complex[][] g) {
        final int fi = f.length;
        final int fj = f[0].length;
        final int gi = g.length;
        final int gj = g[0].length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgj = (int)( (gj - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
        final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
        Complex[] gg = ArrayMath.vectorise(g);
        Complex[] fPad = ArrayMath.vectorise(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje));
        Complex[][] rr = JVCLUtils.zeroPadBoundaries(new Complex[fi][fj], hgi, hgie, hgj, hgje);
        final int ri = rr.length;
        final int rj = rr[0].length;
        Complex[] r = ArrayMath.vectorise(rr);
        int ij;
        for (int i = hgi; i < fi - 1 - hgie; i++) {
            for (int j = hgj; j < fj - 1 - hgje; j++) {
                ij = j*ri + i;
				r[ij]
                .add(fPad[ij-1*ri-1].multiply(gg[0]))
                .add(fPad[ij-1*ri+0].multiply(gg[1]))
                .add(fPad[ij-1*ri+1].multiply(gg[2]))
                .add(fPad[ij+0*ri-1].multiply(gg[3]))
                .add(fPad[ij+0*ri+0].multiply(gg[4]))
                .add(fPad[ij+0*ri+1].multiply(gg[5]))
                .add(fPad[ij+1*ri-1].multiply(gg[6]))
                .add(fPad[ij+1*ri+0].multiply(gg[7]))
                .add(fPad[ij+1*ri+1].multiply(gg[8]));
                }
            }
    return Boundaries.finishBoundaries2d(fPad, gg, r, gi, gj, hgi, hgie, hgj, hgje, ri, rj)
;    };
 // end convolve_3_3
// begin convolve_3_3_3
    public static double[][][] convolve_3_3_3(double[][][] f, double[][][] g) {
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
        double[] gg = ArrayMath.vectorise(g);
        double[] fPad = ArrayMath.vectorise(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje, hgk, hgke));
        double[][][] rr = JVCLUtils.zeroPadBoundaries(new double[fi][fj][fk], hgi, hgie, hgj, hgje, hgk, hgke);
        final int ri = rr.length;
        final int rj = rr[0].length;
        final int rk = rr[0][0].length;
        double[] r = ArrayMath.vectorise(rr);
        int ijk;
        for (int i = hgi; i < fi - 1 - hgie; i++) {
            for (int j = hgj; j < fj - 1 - hgje; j++) {
            	for (int k = hgk; k < fk - 1 - hgke; k++) {
               	ijk = k*ri*rj + j*ri + i;
	                 r[ijk] =
    	             fPad[ijk-1*ri-1]*gg[0]+
    	             fPad[ijk-1*ri+0]*gg[1]+
    	             fPad[ijk-1*ri+1]*gg[2]+
    	             fPad[ijk+0*ri-1]*gg[3]+
    	             fPad[ijk+0*ri+0]*gg[4]+
    	             fPad[ijk+0*ri+1]*gg[5]+
    	             fPad[ijk+1*ri-1]*gg[6]+
    	             fPad[ijk+1*ri+0]*gg[7]+
    	             fPad[ijk+1*ri+1]*gg[8]+
    	             fPad[ijk-1*ri-1]*gg[9]+
    	             fPad[ijk-1*ri+0]*gg[10]+
    	             fPad[ijk-1*ri+1]*gg[11]+
    	             fPad[ijk+0*ri-1]*gg[12]+
    	             fPad[ijk+0*ri+0]*gg[13]+
    	             fPad[ijk+0*ri+1]*gg[14]+
    	             fPad[ijk+1*ri-1]*gg[15]+
    	             fPad[ijk+1*ri+0]*gg[16]+
    	             fPad[ijk+1*ri+1]*gg[17]+
    	             fPad[ijk-1*ri-1]*gg[18]+
    	             fPad[ijk-1*ri+0]*gg[19]+
    	             fPad[ijk-1*ri+1]*gg[20]+
    	             fPad[ijk+0*ri-1]*gg[21]+
    	             fPad[ijk+0*ri+0]*gg[22]+
    	             fPad[ijk+0*ri+1]*gg[23]+
    	             fPad[ijk+1*ri-1]*gg[24]+
    	             fPad[ijk+1*ri+0]*gg[25]+
    	             fPad[ijk+1*ri+1]*gg[26];
     		           }
                }
            }
        return Boundaries.finishBoundaries3d(fPad, gg, r, gi, gj, gk, hgi, hgie, hgj, hgje, hgk, hgke, ri, rj, rk);
    };
    public static Complex[][][] convolve_3_3_3(Complex[][][] f, Complex[][][] g) {
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
        Complex[] gg = ArrayMath.vectorise(g);
        Complex[] fPad = ArrayMath.vectorise(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje, hgk, hgke));
        Complex[][][] rr = JVCLUtils.zeroPadBoundaries(new Complex[fi][fj][fk], hgi, hgie, hgj, hgje, hgk, hgke);
        final int ri = rr.length;
        final int rj = rr[0].length;
        final int rk = rr[0][0].length;
        Complex[] r = ArrayMath.vectorise(rr);
        int ijk;
        for (int i = hgi; i < fi - 1 - hgie; i++) {
            for (int j = hgj; j < fj - 1 - hgje; j++) {
            	for (int k = hgk; k < fk - 1 - hgke; k++) {
               	ijk = k*ri*rj + j*ri + i;
         	        r[ijk]
       	          .add(fPad[ijk-1*ri-1].multiply(gg[0]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[1]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[2]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[3]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[4]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[5]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[6]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[7]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[8]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[9]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[10]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[11]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[12]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[13]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[14]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[15]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[16]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[17]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[18]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[19]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[20]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[21]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[22]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[23]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[24]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[25]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[26]));
     	        	   }
                }
            }
        return Boundaries.finishBoundaries3d(fPad, gg, r, gi, gj, gk, hgi, hgie, hgj, hgje, hgk, hgke, ri, rj, rk)
;    };
// end convolve_3_3_3(double[][][] f, double[][][] g) {
// begin convolve_5
    public static double[] convolve_5(double[] f, double[] g) {
        final int fi = f.length;
        final int gi = g.length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		 double[] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie);
		 double[] r = JVCLUtils.zeroPadBoundaries(new double[fi], hgi, hgie);
		 final int ri = r.length;
		 for (int i = hgi; i < ri-1-hgie; i++) {
                 r[i] =
                 fPad[i-2]*g[4]+
                 fPad[i-1]*g[3]+
                 fPad[i+0]*g[2]+
                 fPad[i+1]*g[1]+
                 fPad[i+2]*g[0];
            }
        return Boundaries.finishBoundaries1d(fPad, g, r, gi,hgi, hgie,ri);
    };
    public static Complex[] convolve_5(Complex[] f, Complex[] g) {
        final int fi = f.length;
        final int gi = g.length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		 Complex[] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie);
		 Complex[] r = JVCLUtils.zeroPadBoundaries(new Complex[fi], hgi, hgie);
		 final int ri = r.length;
		 for (int i = hgi; i < ri-1-hgie; i++) {
                 r[i]
                 .add(fPad[i-2].multiply(g[4]))
                 .add(fPad[i-1].multiply(g[3]))
                 .add(fPad[i+0].multiply(g[2]))
                 .add(fPad[i+1].multiply(g[1]))
                 .add(fPad[i+2].multiply(g[0]));
            }
        return Boundaries.finishBoundaries1d(fPad, g, r, gi, hgi, hgie, ri);
    };
// end convolve_5
 // begin convolve_5_5
    public static double[][] convolve_5_5(double[][] f, double[][] g) {
        final int fi = f.length;
        final int fj = f[0].length;
        final int gi = g.length;
        final int gj = g[0].length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgj = (int)( (gj - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
        final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
        double[] gg = ArrayMath.vectorise(g);
        double[] fPad = ArrayMath.vectorise(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje));
        double[][] rr = JVCLUtils.zeroPadBoundaries(new double[fi][fj], hgi, hgie, hgj, hgje);
        final int ri = rr.length;
        final int rj = rr[0].length;
        double[] r = ArrayMath.vectorise(rr);
        int ij;
        for (int i = hgi; i < fi - 1 - hgie; i++) {
            for (int j = hgj; j < fj - 1 - hgje; j++) {
                ij = j*ri + i;
                 r[ij] =
                 fPad[ij-2*ri-2]*gg[0]+
                 fPad[ij-2*ri-1]*gg[1]+
                 fPad[ij-2*ri+0]*gg[2]+
                 fPad[ij-2*ri+1]*gg[3]+
                 fPad[ij-2*ri+2]*gg[4]+
                 fPad[ij-1*ri-2]*gg[5]+
                 fPad[ij-1*ri-1]*gg[6]+
                 fPad[ij-1*ri+0]*gg[7]+
                 fPad[ij-1*ri+1]*gg[8]+
                 fPad[ij-1*ri+2]*gg[9]+
                 fPad[ij+0*ri-2]*gg[10]+
                 fPad[ij+0*ri-1]*gg[11]+
                 fPad[ij+0*ri+0]*gg[12]+
                 fPad[ij+0*ri+1]*gg[13]+
                 fPad[ij+0*ri+2]*gg[14]+
                 fPad[ij+1*ri-2]*gg[15]+
                 fPad[ij+1*ri-1]*gg[16]+
                 fPad[ij+1*ri+0]*gg[17]+
                 fPad[ij+1*ri+1]*gg[18]+
                 fPad[ij+1*ri+2]*gg[19]+
                 fPad[ij+2*ri-2]*gg[20]+
                 fPad[ij+2*ri-1]*gg[21]+
                 fPad[ij+2*ri+0]*gg[22]+
                 fPad[ij+2*ri+1]*gg[23]+
                 fPad[ij+2*ri+2]*gg[24];
                }
            }
        return Boundaries.finishBoundaries2d(fPad, gg, r, gi, gj, hgi, hgie, hgj, hgje, ri, rj);
    };
    public static Complex[][] convolve_5_5(Complex[][] f, Complex[][] g) {
        final int fi = f.length;
        final int fj = f[0].length;
        final int gi = g.length;
        final int gj = g[0].length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgj = (int)( (gj - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
        final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
        Complex[] gg = ArrayMath.vectorise(g);
        Complex[] fPad = ArrayMath.vectorise(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje));
        Complex[][] rr = JVCLUtils.zeroPadBoundaries(new Complex[fi][fj], hgi, hgie, hgj, hgje);
        final int ri = rr.length;
        final int rj = rr[0].length;
        Complex[] r = ArrayMath.vectorise(rr);
        int ij;
        for (int i = hgi; i < fi - 1 - hgie; i++) {
            for (int j = hgj; j < fj - 1 - hgje; j++) {
                ij = j*ri + i;
				r[ij]
                .add(fPad[ij-2*ri-2].multiply(gg[0]))
                .add(fPad[ij-2*ri-1].multiply(gg[1]))
                .add(fPad[ij-2*ri+0].multiply(gg[2]))
                .add(fPad[ij-2*ri+1].multiply(gg[3]))
                .add(fPad[ij-2*ri+2].multiply(gg[4]))
                .add(fPad[ij-1*ri-2].multiply(gg[5]))
                .add(fPad[ij-1*ri-1].multiply(gg[6]))
                .add(fPad[ij-1*ri+0].multiply(gg[7]))
                .add(fPad[ij-1*ri+1].multiply(gg[8]))
                .add(fPad[ij-1*ri+2].multiply(gg[9]))
                .add(fPad[ij+0*ri-2].multiply(gg[10]))
                .add(fPad[ij+0*ri-1].multiply(gg[11]))
                .add(fPad[ij+0*ri+0].multiply(gg[12]))
                .add(fPad[ij+0*ri+1].multiply(gg[13]))
                .add(fPad[ij+0*ri+2].multiply(gg[14]))
                .add(fPad[ij+1*ri-2].multiply(gg[15]))
                .add(fPad[ij+1*ri-1].multiply(gg[16]))
                .add(fPad[ij+1*ri+0].multiply(gg[17]))
                .add(fPad[ij+1*ri+1].multiply(gg[18]))
                .add(fPad[ij+1*ri+2].multiply(gg[19]))
                .add(fPad[ij+2*ri-2].multiply(gg[20]))
                .add(fPad[ij+2*ri-1].multiply(gg[21]))
                .add(fPad[ij+2*ri+0].multiply(gg[22]))
                .add(fPad[ij+2*ri+1].multiply(gg[23]))
                .add(fPad[ij+2*ri+2].multiply(gg[24]));
                }
            }
    return Boundaries.finishBoundaries2d(fPad, gg, r, gi, gj, hgi, hgie, hgj, hgje, ri, rj)
;    };
 // end convolve_5_5
// begin convolve_5_5_5
    public static double[][][] convolve_5_5_5(double[][][] f, double[][][] g) {
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
        double[] gg = ArrayMath.vectorise(g);
        double[] fPad = ArrayMath.vectorise(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje, hgk, hgke));
        double[][][] rr = JVCLUtils.zeroPadBoundaries(new double[fi][fj][fk], hgi, hgie, hgj, hgje, hgk, hgke);
        final int ri = rr.length;
        final int rj = rr[0].length;
        final int rk = rr[0][0].length;
        double[] r = ArrayMath.vectorise(rr);
        int ijk;
        for (int i = hgi; i < fi - 1 - hgie; i++) {
            for (int j = hgj; j < fj - 1 - hgje; j++) {
            	for (int k = hgk; k < fk - 1 - hgke; k++) {
               	ijk = k*ri*rj + j*ri + i;
	                 r[ijk] =
    	             fPad[ijk-2*ri-2]*gg[0]+
    	             fPad[ijk-2*ri-1]*gg[1]+
    	             fPad[ijk-2*ri+0]*gg[2]+
    	             fPad[ijk-2*ri+1]*gg[3]+
    	             fPad[ijk-2*ri+2]*gg[4]+
    	             fPad[ijk-1*ri-2]*gg[5]+
    	             fPad[ijk-1*ri-1]*gg[6]+
    	             fPad[ijk-1*ri+0]*gg[7]+
    	             fPad[ijk-1*ri+1]*gg[8]+
    	             fPad[ijk-1*ri+2]*gg[9]+
    	             fPad[ijk+0*ri-2]*gg[10]+
    	             fPad[ijk+0*ri-1]*gg[11]+
    	             fPad[ijk+0*ri+0]*gg[12]+
    	             fPad[ijk+0*ri+1]*gg[13]+
    	             fPad[ijk+0*ri+2]*gg[14]+
    	             fPad[ijk+1*ri-2]*gg[15]+
    	             fPad[ijk+1*ri-1]*gg[16]+
    	             fPad[ijk+1*ri+0]*gg[17]+
    	             fPad[ijk+1*ri+1]*gg[18]+
    	             fPad[ijk+1*ri+2]*gg[19]+
    	             fPad[ijk+2*ri-2]*gg[20]+
    	             fPad[ijk+2*ri-1]*gg[21]+
    	             fPad[ijk+2*ri+0]*gg[22]+
    	             fPad[ijk+2*ri+1]*gg[23]+
    	             fPad[ijk+2*ri+2]*gg[24]+
    	             fPad[ijk-2*ri-2]*gg[25]+
    	             fPad[ijk-2*ri-1]*gg[26]+
    	             fPad[ijk-2*ri+0]*gg[27]+
    	             fPad[ijk-2*ri+1]*gg[28]+
    	             fPad[ijk-2*ri+2]*gg[29]+
    	             fPad[ijk-1*ri-2]*gg[30]+
    	             fPad[ijk-1*ri-1]*gg[31]+
    	             fPad[ijk-1*ri+0]*gg[32]+
    	             fPad[ijk-1*ri+1]*gg[33]+
    	             fPad[ijk-1*ri+2]*gg[34]+
    	             fPad[ijk+0*ri-2]*gg[35]+
    	             fPad[ijk+0*ri-1]*gg[36]+
    	             fPad[ijk+0*ri+0]*gg[37]+
    	             fPad[ijk+0*ri+1]*gg[38]+
    	             fPad[ijk+0*ri+2]*gg[39]+
    	             fPad[ijk+1*ri-2]*gg[40]+
    	             fPad[ijk+1*ri-1]*gg[41]+
    	             fPad[ijk+1*ri+0]*gg[42]+
    	             fPad[ijk+1*ri+1]*gg[43]+
    	             fPad[ijk+1*ri+2]*gg[44]+
    	             fPad[ijk+2*ri-2]*gg[45]+
    	             fPad[ijk+2*ri-1]*gg[46]+
    	             fPad[ijk+2*ri+0]*gg[47]+
    	             fPad[ijk+2*ri+1]*gg[48]+
    	             fPad[ijk+2*ri+2]*gg[49]+
    	             fPad[ijk-2*ri-2]*gg[50]+
    	             fPad[ijk-2*ri-1]*gg[51]+
    	             fPad[ijk-2*ri+0]*gg[52]+
    	             fPad[ijk-2*ri+1]*gg[53]+
    	             fPad[ijk-2*ri+2]*gg[54]+
    	             fPad[ijk-1*ri-2]*gg[55]+
    	             fPad[ijk-1*ri-1]*gg[56]+
    	             fPad[ijk-1*ri+0]*gg[57]+
    	             fPad[ijk-1*ri+1]*gg[58]+
    	             fPad[ijk-1*ri+2]*gg[59]+
    	             fPad[ijk+0*ri-2]*gg[60]+
    	             fPad[ijk+0*ri-1]*gg[61]+
    	             fPad[ijk+0*ri+0]*gg[62]+
    	             fPad[ijk+0*ri+1]*gg[63]+
    	             fPad[ijk+0*ri+2]*gg[64]+
    	             fPad[ijk+1*ri-2]*gg[65]+
    	             fPad[ijk+1*ri-1]*gg[66]+
    	             fPad[ijk+1*ri+0]*gg[67]+
    	             fPad[ijk+1*ri+1]*gg[68]+
    	             fPad[ijk+1*ri+2]*gg[69]+
    	             fPad[ijk+2*ri-2]*gg[70]+
    	             fPad[ijk+2*ri-1]*gg[71]+
    	             fPad[ijk+2*ri+0]*gg[72]+
    	             fPad[ijk+2*ri+1]*gg[73]+
    	             fPad[ijk+2*ri+2]*gg[74]+
    	             fPad[ijk-2*ri-2]*gg[75]+
    	             fPad[ijk-2*ri-1]*gg[76]+
    	             fPad[ijk-2*ri+0]*gg[77]+
    	             fPad[ijk-2*ri+1]*gg[78]+
    	             fPad[ijk-2*ri+2]*gg[79]+
    	             fPad[ijk-1*ri-2]*gg[80]+
    	             fPad[ijk-1*ri-1]*gg[81]+
    	             fPad[ijk-1*ri+0]*gg[82]+
    	             fPad[ijk-1*ri+1]*gg[83]+
    	             fPad[ijk-1*ri+2]*gg[84]+
    	             fPad[ijk+0*ri-2]*gg[85]+
    	             fPad[ijk+0*ri-1]*gg[86]+
    	             fPad[ijk+0*ri+0]*gg[87]+
    	             fPad[ijk+0*ri+1]*gg[88]+
    	             fPad[ijk+0*ri+2]*gg[89]+
    	             fPad[ijk+1*ri-2]*gg[90]+
    	             fPad[ijk+1*ri-1]*gg[91]+
    	             fPad[ijk+1*ri+0]*gg[92]+
    	             fPad[ijk+1*ri+1]*gg[93]+
    	             fPad[ijk+1*ri+2]*gg[94]+
    	             fPad[ijk+2*ri-2]*gg[95]+
    	             fPad[ijk+2*ri-1]*gg[96]+
    	             fPad[ijk+2*ri+0]*gg[97]+
    	             fPad[ijk+2*ri+1]*gg[98]+
    	             fPad[ijk+2*ri+2]*gg[99]+
    	             fPad[ijk-2*ri-2]*gg[100]+
    	             fPad[ijk-2*ri-1]*gg[101]+
    	             fPad[ijk-2*ri+0]*gg[102]+
    	             fPad[ijk-2*ri+1]*gg[103]+
    	             fPad[ijk-2*ri+2]*gg[104]+
    	             fPad[ijk-1*ri-2]*gg[105]+
    	             fPad[ijk-1*ri-1]*gg[106]+
    	             fPad[ijk-1*ri+0]*gg[107]+
    	             fPad[ijk-1*ri+1]*gg[108]+
    	             fPad[ijk-1*ri+2]*gg[109]+
    	             fPad[ijk+0*ri-2]*gg[110]+
    	             fPad[ijk+0*ri-1]*gg[111]+
    	             fPad[ijk+0*ri+0]*gg[112]+
    	             fPad[ijk+0*ri+1]*gg[113]+
    	             fPad[ijk+0*ri+2]*gg[114]+
    	             fPad[ijk+1*ri-2]*gg[115]+
    	             fPad[ijk+1*ri-1]*gg[116]+
    	             fPad[ijk+1*ri+0]*gg[117]+
    	             fPad[ijk+1*ri+1]*gg[118]+
    	             fPad[ijk+1*ri+2]*gg[119]+
    	             fPad[ijk+2*ri-2]*gg[120]+
    	             fPad[ijk+2*ri-1]*gg[121]+
    	             fPad[ijk+2*ri+0]*gg[122]+
    	             fPad[ijk+2*ri+1]*gg[123]+
    	             fPad[ijk+2*ri+2]*gg[124];
     		           }
                }
            }
        return Boundaries.finishBoundaries3d(fPad, gg, r, gi, gj, gk, hgi, hgie, hgj, hgje, hgk, hgke, ri, rj, rk);
    };
    public static Complex[][][] convolve_5_5_5(Complex[][][] f, Complex[][][] g) {
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
        Complex[] gg = ArrayMath.vectorise(g);
        Complex[] fPad = ArrayMath.vectorise(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje, hgk, hgke));
        Complex[][][] rr = JVCLUtils.zeroPadBoundaries(new Complex[fi][fj][fk], hgi, hgie, hgj, hgje, hgk, hgke);
        final int ri = rr.length;
        final int rj = rr[0].length;
        final int rk = rr[0][0].length;
        Complex[] r = ArrayMath.vectorise(rr);
        int ijk;
        for (int i = hgi; i < fi - 1 - hgie; i++) {
            for (int j = hgj; j < fj - 1 - hgje; j++) {
            	for (int k = hgk; k < fk - 1 - hgke; k++) {
               	ijk = k*ri*rj + j*ri + i;
         	        r[ijk]
       	          .add(fPad[ijk-2*ri-2].multiply(gg[0]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[1]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[2]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[3]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[4]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[5]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[6]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[7]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[8]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[9]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[10]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[11]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[12]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[13]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[14]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[15]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[16]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[17]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[18]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[19]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[20]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[21]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[22]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[23]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[24]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[25]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[26]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[27]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[28]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[29]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[30]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[31]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[32]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[33]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[34]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[35]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[36]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[37]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[38]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[39]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[40]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[41]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[42]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[43]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[44]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[45]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[46]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[47]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[48]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[49]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[50]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[51]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[52]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[53]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[54]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[55]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[56]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[57]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[58]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[59]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[60]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[61]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[62]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[63]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[64]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[65]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[66]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[67]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[68]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[69]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[70]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[71]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[72]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[73]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[74]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[75]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[76]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[77]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[78]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[79]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[80]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[81]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[82]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[83]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[84]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[85]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[86]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[87]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[88]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[89]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[90]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[91]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[92]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[93]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[94]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[95]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[96]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[97]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[98]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[99]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[100]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[101]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[102]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[103]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[104]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[105]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[106]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[107]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[108]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[109]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[110]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[111]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[112]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[113]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[114]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[115]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[116]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[117]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[118]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[119]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[120]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[121]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[122]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[123]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[124]));
     	        	   }
                }
            }
        return Boundaries.finishBoundaries3d(fPad, gg, r, gi, gj, gk, hgi, hgie, hgj, hgje, hgk, hgke, ri, rj, rk)
;    };
// end convolve_5_5_5(double[][][] f, double[][][] g) {
// begin convolve_7
    public static double[] convolve_7(double[] f, double[] g) {
        final int fi = f.length;
        final int gi = g.length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		 double[] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie);
		 double[] r = JVCLUtils.zeroPadBoundaries(new double[fi], hgi, hgie);
		 final int ri = r.length;
		 for (int i = hgi; i < ri-1-hgie; i++) {
                 r[i] =
                 fPad[i-3]*g[6]+
                 fPad[i-2]*g[5]+
                 fPad[i-1]*g[4]+
                 fPad[i+0]*g[3]+
                 fPad[i+1]*g[2]+
                 fPad[i+2]*g[1]+
                 fPad[i+3]*g[0];
            }
        return Boundaries.finishBoundaries1d(fPad, g, r, gi,hgi, hgie,ri);
    };
    public static Complex[] convolve_7(Complex[] f, Complex[] g) {
        final int fi = f.length;
        final int gi = g.length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		 Complex[] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie);
		 Complex[] r = JVCLUtils.zeroPadBoundaries(new Complex[fi], hgi, hgie);
		 final int ri = r.length;
		 for (int i = hgi; i < ri-1-hgie; i++) {
                 r[i]
                 .add(fPad[i-3].multiply(g[6]))
                 .add(fPad[i-2].multiply(g[5]))
                 .add(fPad[i-1].multiply(g[4]))
                 .add(fPad[i+0].multiply(g[3]))
                 .add(fPad[i+1].multiply(g[2]))
                 .add(fPad[i+2].multiply(g[1]))
                 .add(fPad[i+3].multiply(g[0]));
            }
        return Boundaries.finishBoundaries1d(fPad, g, r, gi, hgi, hgie, ri);
    };
// end convolve_7
 // begin convolve_7_7
    public static double[][] convolve_7_7(double[][] f, double[][] g) {
        final int fi = f.length;
        final int fj = f[0].length;
        final int gi = g.length;
        final int gj = g[0].length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgj = (int)( (gj - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
        final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
        double[] gg = ArrayMath.vectorise(g);
        double[] fPad = ArrayMath.vectorise(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje));
        double[][] rr = JVCLUtils.zeroPadBoundaries(new double[fi][fj], hgi, hgie, hgj, hgje);
        final int ri = rr.length;
        final int rj = rr[0].length;
        double[] r = ArrayMath.vectorise(rr);
        int ij;
        for (int i = hgi; i < fi - 1 - hgie; i++) {
            for (int j = hgj; j < fj - 1 - hgje; j++) {
                ij = j*ri + i;
                 r[ij] =
                 fPad[ij-3*ri-3]*gg[0]+
                 fPad[ij-3*ri-2]*gg[1]+
                 fPad[ij-3*ri-1]*gg[2]+
                 fPad[ij-3*ri+0]*gg[3]+
                 fPad[ij-3*ri+1]*gg[4]+
                 fPad[ij-3*ri+2]*gg[5]+
                 fPad[ij-3*ri+3]*gg[6]+
                 fPad[ij-2*ri-3]*gg[7]+
                 fPad[ij-2*ri-2]*gg[8]+
                 fPad[ij-2*ri-1]*gg[9]+
                 fPad[ij-2*ri+0]*gg[10]+
                 fPad[ij-2*ri+1]*gg[11]+
                 fPad[ij-2*ri+2]*gg[12]+
                 fPad[ij-2*ri+3]*gg[13]+
                 fPad[ij-1*ri-3]*gg[14]+
                 fPad[ij-1*ri-2]*gg[15]+
                 fPad[ij-1*ri-1]*gg[16]+
                 fPad[ij-1*ri+0]*gg[17]+
                 fPad[ij-1*ri+1]*gg[18]+
                 fPad[ij-1*ri+2]*gg[19]+
                 fPad[ij-1*ri+3]*gg[20]+
                 fPad[ij+0*ri-3]*gg[21]+
                 fPad[ij+0*ri-2]*gg[22]+
                 fPad[ij+0*ri-1]*gg[23]+
                 fPad[ij+0*ri+0]*gg[24]+
                 fPad[ij+0*ri+1]*gg[25]+
                 fPad[ij+0*ri+2]*gg[26]+
                 fPad[ij+0*ri+3]*gg[27]+
                 fPad[ij+1*ri-3]*gg[28]+
                 fPad[ij+1*ri-2]*gg[29]+
                 fPad[ij+1*ri-1]*gg[30]+
                 fPad[ij+1*ri+0]*gg[31]+
                 fPad[ij+1*ri+1]*gg[32]+
                 fPad[ij+1*ri+2]*gg[33]+
                 fPad[ij+1*ri+3]*gg[34]+
                 fPad[ij+2*ri-3]*gg[35]+
                 fPad[ij+2*ri-2]*gg[36]+
                 fPad[ij+2*ri-1]*gg[37]+
                 fPad[ij+2*ri+0]*gg[38]+
                 fPad[ij+2*ri+1]*gg[39]+
                 fPad[ij+2*ri+2]*gg[40]+
                 fPad[ij+2*ri+3]*gg[41]+
                 fPad[ij+3*ri-3]*gg[42]+
                 fPad[ij+3*ri-2]*gg[43]+
                 fPad[ij+3*ri-1]*gg[44]+
                 fPad[ij+3*ri+0]*gg[45]+
                 fPad[ij+3*ri+1]*gg[46]+
                 fPad[ij+3*ri+2]*gg[47]+
                 fPad[ij+3*ri+3]*gg[48];
                }
            }
        return Boundaries.finishBoundaries2d(fPad, gg, r, gi, gj, hgi, hgie, hgj, hgje, ri, rj);
    };
    public static Complex[][] convolve_7_7(Complex[][] f, Complex[][] g) {
        final int fi = f.length;
        final int fj = f[0].length;
        final int gi = g.length;
        final int gj = g[0].length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgj = (int)( (gj - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
        final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
        Complex[] gg = ArrayMath.vectorise(g);
        Complex[] fPad = ArrayMath.vectorise(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje));
        Complex[][] rr = JVCLUtils.zeroPadBoundaries(new Complex[fi][fj], hgi, hgie, hgj, hgje);
        final int ri = rr.length;
        final int rj = rr[0].length;
        Complex[] r = ArrayMath.vectorise(rr);
        int ij;
        for (int i = hgi; i < fi - 1 - hgie; i++) {
            for (int j = hgj; j < fj - 1 - hgje; j++) {
                ij = j*ri + i;
				r[ij]
                .add(fPad[ij-3*ri-3].multiply(gg[0]))
                .add(fPad[ij-3*ri-2].multiply(gg[1]))
                .add(fPad[ij-3*ri-1].multiply(gg[2]))
                .add(fPad[ij-3*ri+0].multiply(gg[3]))
                .add(fPad[ij-3*ri+1].multiply(gg[4]))
                .add(fPad[ij-3*ri+2].multiply(gg[5]))
                .add(fPad[ij-3*ri+3].multiply(gg[6]))
                .add(fPad[ij-2*ri-3].multiply(gg[7]))
                .add(fPad[ij-2*ri-2].multiply(gg[8]))
                .add(fPad[ij-2*ri-1].multiply(gg[9]))
                .add(fPad[ij-2*ri+0].multiply(gg[10]))
                .add(fPad[ij-2*ri+1].multiply(gg[11]))
                .add(fPad[ij-2*ri+2].multiply(gg[12]))
                .add(fPad[ij-2*ri+3].multiply(gg[13]))
                .add(fPad[ij-1*ri-3].multiply(gg[14]))
                .add(fPad[ij-1*ri-2].multiply(gg[15]))
                .add(fPad[ij-1*ri-1].multiply(gg[16]))
                .add(fPad[ij-1*ri+0].multiply(gg[17]))
                .add(fPad[ij-1*ri+1].multiply(gg[18]))
                .add(fPad[ij-1*ri+2].multiply(gg[19]))
                .add(fPad[ij-1*ri+3].multiply(gg[20]))
                .add(fPad[ij+0*ri-3].multiply(gg[21]))
                .add(fPad[ij+0*ri-2].multiply(gg[22]))
                .add(fPad[ij+0*ri-1].multiply(gg[23]))
                .add(fPad[ij+0*ri+0].multiply(gg[24]))
                .add(fPad[ij+0*ri+1].multiply(gg[25]))
                .add(fPad[ij+0*ri+2].multiply(gg[26]))
                .add(fPad[ij+0*ri+3].multiply(gg[27]))
                .add(fPad[ij+1*ri-3].multiply(gg[28]))
                .add(fPad[ij+1*ri-2].multiply(gg[29]))
                .add(fPad[ij+1*ri-1].multiply(gg[30]))
                .add(fPad[ij+1*ri+0].multiply(gg[31]))
                .add(fPad[ij+1*ri+1].multiply(gg[32]))
                .add(fPad[ij+1*ri+2].multiply(gg[33]))
                .add(fPad[ij+1*ri+3].multiply(gg[34]))
                .add(fPad[ij+2*ri-3].multiply(gg[35]))
                .add(fPad[ij+2*ri-2].multiply(gg[36]))
                .add(fPad[ij+2*ri-1].multiply(gg[37]))
                .add(fPad[ij+2*ri+0].multiply(gg[38]))
                .add(fPad[ij+2*ri+1].multiply(gg[39]))
                .add(fPad[ij+2*ri+2].multiply(gg[40]))
                .add(fPad[ij+2*ri+3].multiply(gg[41]))
                .add(fPad[ij+3*ri-3].multiply(gg[42]))
                .add(fPad[ij+3*ri-2].multiply(gg[43]))
                .add(fPad[ij+3*ri-1].multiply(gg[44]))
                .add(fPad[ij+3*ri+0].multiply(gg[45]))
                .add(fPad[ij+3*ri+1].multiply(gg[46]))
                .add(fPad[ij+3*ri+2].multiply(gg[47]))
                .add(fPad[ij+3*ri+3].multiply(gg[48]));
                }
            }
    return Boundaries.finishBoundaries2d(fPad, gg, r, gi, gj, hgi, hgie, hgj, hgje, ri, rj)
;    };
 // end convolve_7_7
// begin convolve_7_7_7
    public static double[][][] convolve_7_7_7(double[][][] f, double[][][] g) {
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
        double[] gg = ArrayMath.vectorise(g);
        double[] fPad = ArrayMath.vectorise(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje, hgk, hgke));
        double[][][] rr = JVCLUtils.zeroPadBoundaries(new double[fi][fj][fk], hgi, hgie, hgj, hgje, hgk, hgke);
        final int ri = rr.length;
        final int rj = rr[0].length;
        final int rk = rr[0][0].length;
        double[] r = ArrayMath.vectorise(rr);
        int ijk;
        for (int i = hgi; i < fi - 1 - hgie; i++) {
            for (int j = hgj; j < fj - 1 - hgje; j++) {
            	for (int k = hgk; k < fk - 1 - hgke; k++) {
               	ijk = k*ri*rj + j*ri + i;
	                 r[ijk] =
    	             fPad[ijk-3*ri-3]*gg[0]+
    	             fPad[ijk-3*ri-2]*gg[1]+
    	             fPad[ijk-3*ri-1]*gg[2]+
    	             fPad[ijk-3*ri+0]*gg[3]+
    	             fPad[ijk-3*ri+1]*gg[4]+
    	             fPad[ijk-3*ri+2]*gg[5]+
    	             fPad[ijk-3*ri+3]*gg[6]+
    	             fPad[ijk-2*ri-3]*gg[7]+
    	             fPad[ijk-2*ri-2]*gg[8]+
    	             fPad[ijk-2*ri-1]*gg[9]+
    	             fPad[ijk-2*ri+0]*gg[10]+
    	             fPad[ijk-2*ri+1]*gg[11]+
    	             fPad[ijk-2*ri+2]*gg[12]+
    	             fPad[ijk-2*ri+3]*gg[13]+
    	             fPad[ijk-1*ri-3]*gg[14]+
    	             fPad[ijk-1*ri-2]*gg[15]+
    	             fPad[ijk-1*ri-1]*gg[16]+
    	             fPad[ijk-1*ri+0]*gg[17]+
    	             fPad[ijk-1*ri+1]*gg[18]+
    	             fPad[ijk-1*ri+2]*gg[19]+
    	             fPad[ijk-1*ri+3]*gg[20]+
    	             fPad[ijk+0*ri-3]*gg[21]+
    	             fPad[ijk+0*ri-2]*gg[22]+
    	             fPad[ijk+0*ri-1]*gg[23]+
    	             fPad[ijk+0*ri+0]*gg[24]+
    	             fPad[ijk+0*ri+1]*gg[25]+
    	             fPad[ijk+0*ri+2]*gg[26]+
    	             fPad[ijk+0*ri+3]*gg[27]+
    	             fPad[ijk+1*ri-3]*gg[28]+
    	             fPad[ijk+1*ri-2]*gg[29]+
    	             fPad[ijk+1*ri-1]*gg[30]+
    	             fPad[ijk+1*ri+0]*gg[31]+
    	             fPad[ijk+1*ri+1]*gg[32]+
    	             fPad[ijk+1*ri+2]*gg[33]+
    	             fPad[ijk+1*ri+3]*gg[34]+
    	             fPad[ijk+2*ri-3]*gg[35]+
    	             fPad[ijk+2*ri-2]*gg[36]+
    	             fPad[ijk+2*ri-1]*gg[37]+
    	             fPad[ijk+2*ri+0]*gg[38]+
    	             fPad[ijk+2*ri+1]*gg[39]+
    	             fPad[ijk+2*ri+2]*gg[40]+
    	             fPad[ijk+2*ri+3]*gg[41]+
    	             fPad[ijk+3*ri-3]*gg[42]+
    	             fPad[ijk+3*ri-2]*gg[43]+
    	             fPad[ijk+3*ri-1]*gg[44]+
    	             fPad[ijk+3*ri+0]*gg[45]+
    	             fPad[ijk+3*ri+1]*gg[46]+
    	             fPad[ijk+3*ri+2]*gg[47]+
    	             fPad[ijk+3*ri+3]*gg[48]+
    	             fPad[ijk-3*ri-3]*gg[49]+
    	             fPad[ijk-3*ri-2]*gg[50]+
    	             fPad[ijk-3*ri-1]*gg[51]+
    	             fPad[ijk-3*ri+0]*gg[52]+
    	             fPad[ijk-3*ri+1]*gg[53]+
    	             fPad[ijk-3*ri+2]*gg[54]+
    	             fPad[ijk-3*ri+3]*gg[55]+
    	             fPad[ijk-2*ri-3]*gg[56]+
    	             fPad[ijk-2*ri-2]*gg[57]+
    	             fPad[ijk-2*ri-1]*gg[58]+
    	             fPad[ijk-2*ri+0]*gg[59]+
    	             fPad[ijk-2*ri+1]*gg[60]+
    	             fPad[ijk-2*ri+2]*gg[61]+
    	             fPad[ijk-2*ri+3]*gg[62]+
    	             fPad[ijk-1*ri-3]*gg[63]+
    	             fPad[ijk-1*ri-2]*gg[64]+
    	             fPad[ijk-1*ri-1]*gg[65]+
    	             fPad[ijk-1*ri+0]*gg[66]+
    	             fPad[ijk-1*ri+1]*gg[67]+
    	             fPad[ijk-1*ri+2]*gg[68]+
    	             fPad[ijk-1*ri+3]*gg[69]+
    	             fPad[ijk+0*ri-3]*gg[70]+
    	             fPad[ijk+0*ri-2]*gg[71]+
    	             fPad[ijk+0*ri-1]*gg[72]+
    	             fPad[ijk+0*ri+0]*gg[73]+
    	             fPad[ijk+0*ri+1]*gg[74]+
    	             fPad[ijk+0*ri+2]*gg[75]+
    	             fPad[ijk+0*ri+3]*gg[76]+
    	             fPad[ijk+1*ri-3]*gg[77]+
    	             fPad[ijk+1*ri-2]*gg[78]+
    	             fPad[ijk+1*ri-1]*gg[79]+
    	             fPad[ijk+1*ri+0]*gg[80]+
    	             fPad[ijk+1*ri+1]*gg[81]+
    	             fPad[ijk+1*ri+2]*gg[82]+
    	             fPad[ijk+1*ri+3]*gg[83]+
    	             fPad[ijk+2*ri-3]*gg[84]+
    	             fPad[ijk+2*ri-2]*gg[85]+
    	             fPad[ijk+2*ri-1]*gg[86]+
    	             fPad[ijk+2*ri+0]*gg[87]+
    	             fPad[ijk+2*ri+1]*gg[88]+
    	             fPad[ijk+2*ri+2]*gg[89]+
    	             fPad[ijk+2*ri+3]*gg[90]+
    	             fPad[ijk+3*ri-3]*gg[91]+
    	             fPad[ijk+3*ri-2]*gg[92]+
    	             fPad[ijk+3*ri-1]*gg[93]+
    	             fPad[ijk+3*ri+0]*gg[94]+
    	             fPad[ijk+3*ri+1]*gg[95]+
    	             fPad[ijk+3*ri+2]*gg[96]+
    	             fPad[ijk+3*ri+3]*gg[97]+
    	             fPad[ijk-3*ri-3]*gg[98]+
    	             fPad[ijk-3*ri-2]*gg[99]+
    	             fPad[ijk-3*ri-1]*gg[100]+
    	             fPad[ijk-3*ri+0]*gg[101]+
    	             fPad[ijk-3*ri+1]*gg[102]+
    	             fPad[ijk-3*ri+2]*gg[103]+
    	             fPad[ijk-3*ri+3]*gg[104]+
    	             fPad[ijk-2*ri-3]*gg[105]+
    	             fPad[ijk-2*ri-2]*gg[106]+
    	             fPad[ijk-2*ri-1]*gg[107]+
    	             fPad[ijk-2*ri+0]*gg[108]+
    	             fPad[ijk-2*ri+1]*gg[109]+
    	             fPad[ijk-2*ri+2]*gg[110]+
    	             fPad[ijk-2*ri+3]*gg[111]+
    	             fPad[ijk-1*ri-3]*gg[112]+
    	             fPad[ijk-1*ri-2]*gg[113]+
    	             fPad[ijk-1*ri-1]*gg[114]+
    	             fPad[ijk-1*ri+0]*gg[115]+
    	             fPad[ijk-1*ri+1]*gg[116]+
    	             fPad[ijk-1*ri+2]*gg[117]+
    	             fPad[ijk-1*ri+3]*gg[118]+
    	             fPad[ijk+0*ri-3]*gg[119]+
    	             fPad[ijk+0*ri-2]*gg[120]+
    	             fPad[ijk+0*ri-1]*gg[121]+
    	             fPad[ijk+0*ri+0]*gg[122]+
    	             fPad[ijk+0*ri+1]*gg[123]+
    	             fPad[ijk+0*ri+2]*gg[124]+
    	             fPad[ijk+0*ri+3]*gg[125]+
    	             fPad[ijk+1*ri-3]*gg[126]+
    	             fPad[ijk+1*ri-2]*gg[127]+
    	             fPad[ijk+1*ri-1]*gg[128]+
    	             fPad[ijk+1*ri+0]*gg[129]+
    	             fPad[ijk+1*ri+1]*gg[130]+
    	             fPad[ijk+1*ri+2]*gg[131]+
    	             fPad[ijk+1*ri+3]*gg[132]+
    	             fPad[ijk+2*ri-3]*gg[133]+
    	             fPad[ijk+2*ri-2]*gg[134]+
    	             fPad[ijk+2*ri-1]*gg[135]+
    	             fPad[ijk+2*ri+0]*gg[136]+
    	             fPad[ijk+2*ri+1]*gg[137]+
    	             fPad[ijk+2*ri+2]*gg[138]+
    	             fPad[ijk+2*ri+3]*gg[139]+
    	             fPad[ijk+3*ri-3]*gg[140]+
    	             fPad[ijk+3*ri-2]*gg[141]+
    	             fPad[ijk+3*ri-1]*gg[142]+
    	             fPad[ijk+3*ri+0]*gg[143]+
    	             fPad[ijk+3*ri+1]*gg[144]+
    	             fPad[ijk+3*ri+2]*gg[145]+
    	             fPad[ijk+3*ri+3]*gg[146]+
    	             fPad[ijk-3*ri-3]*gg[147]+
    	             fPad[ijk-3*ri-2]*gg[148]+
    	             fPad[ijk-3*ri-1]*gg[149]+
    	             fPad[ijk-3*ri+0]*gg[150]+
    	             fPad[ijk-3*ri+1]*gg[151]+
    	             fPad[ijk-3*ri+2]*gg[152]+
    	             fPad[ijk-3*ri+3]*gg[153]+
    	             fPad[ijk-2*ri-3]*gg[154]+
    	             fPad[ijk-2*ri-2]*gg[155]+
    	             fPad[ijk-2*ri-1]*gg[156]+
    	             fPad[ijk-2*ri+0]*gg[157]+
    	             fPad[ijk-2*ri+1]*gg[158]+
    	             fPad[ijk-2*ri+2]*gg[159]+
    	             fPad[ijk-2*ri+3]*gg[160]+
    	             fPad[ijk-1*ri-3]*gg[161]+
    	             fPad[ijk-1*ri-2]*gg[162]+
    	             fPad[ijk-1*ri-1]*gg[163]+
    	             fPad[ijk-1*ri+0]*gg[164]+
    	             fPad[ijk-1*ri+1]*gg[165]+
    	             fPad[ijk-1*ri+2]*gg[166]+
    	             fPad[ijk-1*ri+3]*gg[167]+
    	             fPad[ijk+0*ri-3]*gg[168]+
    	             fPad[ijk+0*ri-2]*gg[169]+
    	             fPad[ijk+0*ri-1]*gg[170]+
    	             fPad[ijk+0*ri+0]*gg[171]+
    	             fPad[ijk+0*ri+1]*gg[172]+
    	             fPad[ijk+0*ri+2]*gg[173]+
    	             fPad[ijk+0*ri+3]*gg[174]+
    	             fPad[ijk+1*ri-3]*gg[175]+
    	             fPad[ijk+1*ri-2]*gg[176]+
    	             fPad[ijk+1*ri-1]*gg[177]+
    	             fPad[ijk+1*ri+0]*gg[178]+
    	             fPad[ijk+1*ri+1]*gg[179]+
    	             fPad[ijk+1*ri+2]*gg[180]+
    	             fPad[ijk+1*ri+3]*gg[181]+
    	             fPad[ijk+2*ri-3]*gg[182]+
    	             fPad[ijk+2*ri-2]*gg[183]+
    	             fPad[ijk+2*ri-1]*gg[184]+
    	             fPad[ijk+2*ri+0]*gg[185]+
    	             fPad[ijk+2*ri+1]*gg[186]+
    	             fPad[ijk+2*ri+2]*gg[187]+
    	             fPad[ijk+2*ri+3]*gg[188]+
    	             fPad[ijk+3*ri-3]*gg[189]+
    	             fPad[ijk+3*ri-2]*gg[190]+
    	             fPad[ijk+3*ri-1]*gg[191]+
    	             fPad[ijk+3*ri+0]*gg[192]+
    	             fPad[ijk+3*ri+1]*gg[193]+
    	             fPad[ijk+3*ri+2]*gg[194]+
    	             fPad[ijk+3*ri+3]*gg[195]+
    	             fPad[ijk-3*ri-3]*gg[196]+
    	             fPad[ijk-3*ri-2]*gg[197]+
    	             fPad[ijk-3*ri-1]*gg[198]+
    	             fPad[ijk-3*ri+0]*gg[199]+
    	             fPad[ijk-3*ri+1]*gg[200]+
    	             fPad[ijk-3*ri+2]*gg[201]+
    	             fPad[ijk-3*ri+3]*gg[202]+
    	             fPad[ijk-2*ri-3]*gg[203]+
    	             fPad[ijk-2*ri-2]*gg[204]+
    	             fPad[ijk-2*ri-1]*gg[205]+
    	             fPad[ijk-2*ri+0]*gg[206]+
    	             fPad[ijk-2*ri+1]*gg[207]+
    	             fPad[ijk-2*ri+2]*gg[208]+
    	             fPad[ijk-2*ri+3]*gg[209]+
    	             fPad[ijk-1*ri-3]*gg[210]+
    	             fPad[ijk-1*ri-2]*gg[211]+
    	             fPad[ijk-1*ri-1]*gg[212]+
    	             fPad[ijk-1*ri+0]*gg[213]+
    	             fPad[ijk-1*ri+1]*gg[214]+
    	             fPad[ijk-1*ri+2]*gg[215]+
    	             fPad[ijk-1*ri+3]*gg[216]+
    	             fPad[ijk+0*ri-3]*gg[217]+
    	             fPad[ijk+0*ri-2]*gg[218]+
    	             fPad[ijk+0*ri-1]*gg[219]+
    	             fPad[ijk+0*ri+0]*gg[220]+
    	             fPad[ijk+0*ri+1]*gg[221]+
    	             fPad[ijk+0*ri+2]*gg[222]+
    	             fPad[ijk+0*ri+3]*gg[223]+
    	             fPad[ijk+1*ri-3]*gg[224]+
    	             fPad[ijk+1*ri-2]*gg[225]+
    	             fPad[ijk+1*ri-1]*gg[226]+
    	             fPad[ijk+1*ri+0]*gg[227]+
    	             fPad[ijk+1*ri+1]*gg[228]+
    	             fPad[ijk+1*ri+2]*gg[229]+
    	             fPad[ijk+1*ri+3]*gg[230]+
    	             fPad[ijk+2*ri-3]*gg[231]+
    	             fPad[ijk+2*ri-2]*gg[232]+
    	             fPad[ijk+2*ri-1]*gg[233]+
    	             fPad[ijk+2*ri+0]*gg[234]+
    	             fPad[ijk+2*ri+1]*gg[235]+
    	             fPad[ijk+2*ri+2]*gg[236]+
    	             fPad[ijk+2*ri+3]*gg[237]+
    	             fPad[ijk+3*ri-3]*gg[238]+
    	             fPad[ijk+3*ri-2]*gg[239]+
    	             fPad[ijk+3*ri-1]*gg[240]+
    	             fPad[ijk+3*ri+0]*gg[241]+
    	             fPad[ijk+3*ri+1]*gg[242]+
    	             fPad[ijk+3*ri+2]*gg[243]+
    	             fPad[ijk+3*ri+3]*gg[244]+
    	             fPad[ijk-3*ri-3]*gg[245]+
    	             fPad[ijk-3*ri-2]*gg[246]+
    	             fPad[ijk-3*ri-1]*gg[247]+
    	             fPad[ijk-3*ri+0]*gg[248]+
    	             fPad[ijk-3*ri+1]*gg[249]+
    	             fPad[ijk-3*ri+2]*gg[250]+
    	             fPad[ijk-3*ri+3]*gg[251]+
    	             fPad[ijk-2*ri-3]*gg[252]+
    	             fPad[ijk-2*ri-2]*gg[253]+
    	             fPad[ijk-2*ri-1]*gg[254]+
    	             fPad[ijk-2*ri+0]*gg[255]+
    	             fPad[ijk-2*ri+1]*gg[256]+
    	             fPad[ijk-2*ri+2]*gg[257]+
    	             fPad[ijk-2*ri+3]*gg[258]+
    	             fPad[ijk-1*ri-3]*gg[259]+
    	             fPad[ijk-1*ri-2]*gg[260]+
    	             fPad[ijk-1*ri-1]*gg[261]+
    	             fPad[ijk-1*ri+0]*gg[262]+
    	             fPad[ijk-1*ri+1]*gg[263]+
    	             fPad[ijk-1*ri+2]*gg[264]+
    	             fPad[ijk-1*ri+3]*gg[265]+
    	             fPad[ijk+0*ri-3]*gg[266]+
    	             fPad[ijk+0*ri-2]*gg[267]+
    	             fPad[ijk+0*ri-1]*gg[268]+
    	             fPad[ijk+0*ri+0]*gg[269]+
    	             fPad[ijk+0*ri+1]*gg[270]+
    	             fPad[ijk+0*ri+2]*gg[271]+
    	             fPad[ijk+0*ri+3]*gg[272]+
    	             fPad[ijk+1*ri-3]*gg[273]+
    	             fPad[ijk+1*ri-2]*gg[274]+
    	             fPad[ijk+1*ri-1]*gg[275]+
    	             fPad[ijk+1*ri+0]*gg[276]+
    	             fPad[ijk+1*ri+1]*gg[277]+
    	             fPad[ijk+1*ri+2]*gg[278]+
    	             fPad[ijk+1*ri+3]*gg[279]+
    	             fPad[ijk+2*ri-3]*gg[280]+
    	             fPad[ijk+2*ri-2]*gg[281]+
    	             fPad[ijk+2*ri-1]*gg[282]+
    	             fPad[ijk+2*ri+0]*gg[283]+
    	             fPad[ijk+2*ri+1]*gg[284]+
    	             fPad[ijk+2*ri+2]*gg[285]+
    	             fPad[ijk+2*ri+3]*gg[286]+
    	             fPad[ijk+3*ri-3]*gg[287]+
    	             fPad[ijk+3*ri-2]*gg[288]+
    	             fPad[ijk+3*ri-1]*gg[289]+
    	             fPad[ijk+3*ri+0]*gg[290]+
    	             fPad[ijk+3*ri+1]*gg[291]+
    	             fPad[ijk+3*ri+2]*gg[292]+
    	             fPad[ijk+3*ri+3]*gg[293]+
    	             fPad[ijk-3*ri-3]*gg[294]+
    	             fPad[ijk-3*ri-2]*gg[295]+
    	             fPad[ijk-3*ri-1]*gg[296]+
    	             fPad[ijk-3*ri+0]*gg[297]+
    	             fPad[ijk-3*ri+1]*gg[298]+
    	             fPad[ijk-3*ri+2]*gg[299]+
    	             fPad[ijk-3*ri+3]*gg[300]+
    	             fPad[ijk-2*ri-3]*gg[301]+
    	             fPad[ijk-2*ri-2]*gg[302]+
    	             fPad[ijk-2*ri-1]*gg[303]+
    	             fPad[ijk-2*ri+0]*gg[304]+
    	             fPad[ijk-2*ri+1]*gg[305]+
    	             fPad[ijk-2*ri+2]*gg[306]+
    	             fPad[ijk-2*ri+3]*gg[307]+
    	             fPad[ijk-1*ri-3]*gg[308]+
    	             fPad[ijk-1*ri-2]*gg[309]+
    	             fPad[ijk-1*ri-1]*gg[310]+
    	             fPad[ijk-1*ri+0]*gg[311]+
    	             fPad[ijk-1*ri+1]*gg[312]+
    	             fPad[ijk-1*ri+2]*gg[313]+
    	             fPad[ijk-1*ri+3]*gg[314]+
    	             fPad[ijk+0*ri-3]*gg[315]+
    	             fPad[ijk+0*ri-2]*gg[316]+
    	             fPad[ijk+0*ri-1]*gg[317]+
    	             fPad[ijk+0*ri+0]*gg[318]+
    	             fPad[ijk+0*ri+1]*gg[319]+
    	             fPad[ijk+0*ri+2]*gg[320]+
    	             fPad[ijk+0*ri+3]*gg[321]+
    	             fPad[ijk+1*ri-3]*gg[322]+
    	             fPad[ijk+1*ri-2]*gg[323]+
    	             fPad[ijk+1*ri-1]*gg[324]+
    	             fPad[ijk+1*ri+0]*gg[325]+
    	             fPad[ijk+1*ri+1]*gg[326]+
    	             fPad[ijk+1*ri+2]*gg[327]+
    	             fPad[ijk+1*ri+3]*gg[328]+
    	             fPad[ijk+2*ri-3]*gg[329]+
    	             fPad[ijk+2*ri-2]*gg[330]+
    	             fPad[ijk+2*ri-1]*gg[331]+
    	             fPad[ijk+2*ri+0]*gg[332]+
    	             fPad[ijk+2*ri+1]*gg[333]+
    	             fPad[ijk+2*ri+2]*gg[334]+
    	             fPad[ijk+2*ri+3]*gg[335]+
    	             fPad[ijk+3*ri-3]*gg[336]+
    	             fPad[ijk+3*ri-2]*gg[337]+
    	             fPad[ijk+3*ri-1]*gg[338]+
    	             fPad[ijk+3*ri+0]*gg[339]+
    	             fPad[ijk+3*ri+1]*gg[340]+
    	             fPad[ijk+3*ri+2]*gg[341]+
    	             fPad[ijk+3*ri+3]*gg[342];
     		           }
                }
            }
        return Boundaries.finishBoundaries3d(fPad, gg, r, gi, gj, gk, hgi, hgie, hgj, hgje, hgk, hgke, ri, rj, rk);
    };
    public static Complex[][][] convolve_7_7_7(Complex[][][] f, Complex[][][] g) {
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
        Complex[] gg = ArrayMath.vectorise(g);
        Complex[] fPad = ArrayMath.vectorise(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje, hgk, hgke));
        Complex[][][] rr = JVCLUtils.zeroPadBoundaries(new Complex[fi][fj][fk], hgi, hgie, hgj, hgje, hgk, hgke);
        final int ri = rr.length;
        final int rj = rr[0].length;
        final int rk = rr[0][0].length;
        Complex[] r = ArrayMath.vectorise(rr);
        int ijk;
        for (int i = hgi; i < fi - 1 - hgie; i++) {
            for (int j = hgj; j < fj - 1 - hgje; j++) {
            	for (int k = hgk; k < fk - 1 - hgke; k++) {
               	ijk = k*ri*rj + j*ri + i;
         	        r[ijk]
       	          .add(fPad[ijk-3*ri-3].multiply(gg[0]))
       	          .add(fPad[ijk-3*ri-2].multiply(gg[1]))
       	          .add(fPad[ijk-3*ri-1].multiply(gg[2]))
       	          .add(fPad[ijk-3*ri+0].multiply(gg[3]))
       	          .add(fPad[ijk-3*ri+1].multiply(gg[4]))
       	          .add(fPad[ijk-3*ri+2].multiply(gg[5]))
       	          .add(fPad[ijk-3*ri+3].multiply(gg[6]))
       	          .add(fPad[ijk-2*ri-3].multiply(gg[7]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[8]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[9]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[10]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[11]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[12]))
       	          .add(fPad[ijk-2*ri+3].multiply(gg[13]))
       	          .add(fPad[ijk-1*ri-3].multiply(gg[14]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[15]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[16]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[17]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[18]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[19]))
       	          .add(fPad[ijk-1*ri+3].multiply(gg[20]))
       	          .add(fPad[ijk+0*ri-3].multiply(gg[21]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[22]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[23]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[24]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[25]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[26]))
       	          .add(fPad[ijk+0*ri+3].multiply(gg[27]))
       	          .add(fPad[ijk+1*ri-3].multiply(gg[28]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[29]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[30]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[31]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[32]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[33]))
       	          .add(fPad[ijk+1*ri+3].multiply(gg[34]))
       	          .add(fPad[ijk+2*ri-3].multiply(gg[35]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[36]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[37]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[38]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[39]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[40]))
       	          .add(fPad[ijk+2*ri+3].multiply(gg[41]))
       	          .add(fPad[ijk+3*ri-3].multiply(gg[42]))
       	          .add(fPad[ijk+3*ri-2].multiply(gg[43]))
       	          .add(fPad[ijk+3*ri-1].multiply(gg[44]))
       	          .add(fPad[ijk+3*ri+0].multiply(gg[45]))
       	          .add(fPad[ijk+3*ri+1].multiply(gg[46]))
       	          .add(fPad[ijk+3*ri+2].multiply(gg[47]))
       	          .add(fPad[ijk+3*ri+3].multiply(gg[48]))
       	          .add(fPad[ijk-3*ri-3].multiply(gg[49]))
       	          .add(fPad[ijk-3*ri-2].multiply(gg[50]))
       	          .add(fPad[ijk-3*ri-1].multiply(gg[51]))
       	          .add(fPad[ijk-3*ri+0].multiply(gg[52]))
       	          .add(fPad[ijk-3*ri+1].multiply(gg[53]))
       	          .add(fPad[ijk-3*ri+2].multiply(gg[54]))
       	          .add(fPad[ijk-3*ri+3].multiply(gg[55]))
       	          .add(fPad[ijk-2*ri-3].multiply(gg[56]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[57]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[58]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[59]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[60]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[61]))
       	          .add(fPad[ijk-2*ri+3].multiply(gg[62]))
       	          .add(fPad[ijk-1*ri-3].multiply(gg[63]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[64]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[65]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[66]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[67]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[68]))
       	          .add(fPad[ijk-1*ri+3].multiply(gg[69]))
       	          .add(fPad[ijk+0*ri-3].multiply(gg[70]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[71]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[72]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[73]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[74]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[75]))
       	          .add(fPad[ijk+0*ri+3].multiply(gg[76]))
       	          .add(fPad[ijk+1*ri-3].multiply(gg[77]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[78]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[79]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[80]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[81]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[82]))
       	          .add(fPad[ijk+1*ri+3].multiply(gg[83]))
       	          .add(fPad[ijk+2*ri-3].multiply(gg[84]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[85]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[86]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[87]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[88]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[89]))
       	          .add(fPad[ijk+2*ri+3].multiply(gg[90]))
       	          .add(fPad[ijk+3*ri-3].multiply(gg[91]))
       	          .add(fPad[ijk+3*ri-2].multiply(gg[92]))
       	          .add(fPad[ijk+3*ri-1].multiply(gg[93]))
       	          .add(fPad[ijk+3*ri+0].multiply(gg[94]))
       	          .add(fPad[ijk+3*ri+1].multiply(gg[95]))
       	          .add(fPad[ijk+3*ri+2].multiply(gg[96]))
       	          .add(fPad[ijk+3*ri+3].multiply(gg[97]))
       	          .add(fPad[ijk-3*ri-3].multiply(gg[98]))
       	          .add(fPad[ijk-3*ri-2].multiply(gg[99]))
       	          .add(fPad[ijk-3*ri-1].multiply(gg[100]))
       	          .add(fPad[ijk-3*ri+0].multiply(gg[101]))
       	          .add(fPad[ijk-3*ri+1].multiply(gg[102]))
       	          .add(fPad[ijk-3*ri+2].multiply(gg[103]))
       	          .add(fPad[ijk-3*ri+3].multiply(gg[104]))
       	          .add(fPad[ijk-2*ri-3].multiply(gg[105]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[106]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[107]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[108]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[109]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[110]))
       	          .add(fPad[ijk-2*ri+3].multiply(gg[111]))
       	          .add(fPad[ijk-1*ri-3].multiply(gg[112]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[113]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[114]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[115]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[116]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[117]))
       	          .add(fPad[ijk-1*ri+3].multiply(gg[118]))
       	          .add(fPad[ijk+0*ri-3].multiply(gg[119]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[120]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[121]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[122]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[123]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[124]))
       	          .add(fPad[ijk+0*ri+3].multiply(gg[125]))
       	          .add(fPad[ijk+1*ri-3].multiply(gg[126]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[127]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[128]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[129]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[130]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[131]))
       	          .add(fPad[ijk+1*ri+3].multiply(gg[132]))
       	          .add(fPad[ijk+2*ri-3].multiply(gg[133]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[134]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[135]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[136]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[137]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[138]))
       	          .add(fPad[ijk+2*ri+3].multiply(gg[139]))
       	          .add(fPad[ijk+3*ri-3].multiply(gg[140]))
       	          .add(fPad[ijk+3*ri-2].multiply(gg[141]))
       	          .add(fPad[ijk+3*ri-1].multiply(gg[142]))
       	          .add(fPad[ijk+3*ri+0].multiply(gg[143]))
       	          .add(fPad[ijk+3*ri+1].multiply(gg[144]))
       	          .add(fPad[ijk+3*ri+2].multiply(gg[145]))
       	          .add(fPad[ijk+3*ri+3].multiply(gg[146]))
       	          .add(fPad[ijk-3*ri-3].multiply(gg[147]))
       	          .add(fPad[ijk-3*ri-2].multiply(gg[148]))
       	          .add(fPad[ijk-3*ri-1].multiply(gg[149]))
       	          .add(fPad[ijk-3*ri+0].multiply(gg[150]))
       	          .add(fPad[ijk-3*ri+1].multiply(gg[151]))
       	          .add(fPad[ijk-3*ri+2].multiply(gg[152]))
       	          .add(fPad[ijk-3*ri+3].multiply(gg[153]))
       	          .add(fPad[ijk-2*ri-3].multiply(gg[154]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[155]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[156]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[157]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[158]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[159]))
       	          .add(fPad[ijk-2*ri+3].multiply(gg[160]))
       	          .add(fPad[ijk-1*ri-3].multiply(gg[161]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[162]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[163]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[164]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[165]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[166]))
       	          .add(fPad[ijk-1*ri+3].multiply(gg[167]))
       	          .add(fPad[ijk+0*ri-3].multiply(gg[168]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[169]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[170]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[171]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[172]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[173]))
       	          .add(fPad[ijk+0*ri+3].multiply(gg[174]))
       	          .add(fPad[ijk+1*ri-3].multiply(gg[175]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[176]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[177]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[178]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[179]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[180]))
       	          .add(fPad[ijk+1*ri+3].multiply(gg[181]))
       	          .add(fPad[ijk+2*ri-3].multiply(gg[182]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[183]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[184]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[185]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[186]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[187]))
       	          .add(fPad[ijk+2*ri+3].multiply(gg[188]))
       	          .add(fPad[ijk+3*ri-3].multiply(gg[189]))
       	          .add(fPad[ijk+3*ri-2].multiply(gg[190]))
       	          .add(fPad[ijk+3*ri-1].multiply(gg[191]))
       	          .add(fPad[ijk+3*ri+0].multiply(gg[192]))
       	          .add(fPad[ijk+3*ri+1].multiply(gg[193]))
       	          .add(fPad[ijk+3*ri+2].multiply(gg[194]))
       	          .add(fPad[ijk+3*ri+3].multiply(gg[195]))
       	          .add(fPad[ijk-3*ri-3].multiply(gg[196]))
       	          .add(fPad[ijk-3*ri-2].multiply(gg[197]))
       	          .add(fPad[ijk-3*ri-1].multiply(gg[198]))
       	          .add(fPad[ijk-3*ri+0].multiply(gg[199]))
       	          .add(fPad[ijk-3*ri+1].multiply(gg[200]))
       	          .add(fPad[ijk-3*ri+2].multiply(gg[201]))
       	          .add(fPad[ijk-3*ri+3].multiply(gg[202]))
       	          .add(fPad[ijk-2*ri-3].multiply(gg[203]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[204]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[205]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[206]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[207]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[208]))
       	          .add(fPad[ijk-2*ri+3].multiply(gg[209]))
       	          .add(fPad[ijk-1*ri-3].multiply(gg[210]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[211]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[212]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[213]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[214]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[215]))
       	          .add(fPad[ijk-1*ri+3].multiply(gg[216]))
       	          .add(fPad[ijk+0*ri-3].multiply(gg[217]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[218]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[219]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[220]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[221]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[222]))
       	          .add(fPad[ijk+0*ri+3].multiply(gg[223]))
       	          .add(fPad[ijk+1*ri-3].multiply(gg[224]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[225]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[226]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[227]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[228]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[229]))
       	          .add(fPad[ijk+1*ri+3].multiply(gg[230]))
       	          .add(fPad[ijk+2*ri-3].multiply(gg[231]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[232]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[233]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[234]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[235]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[236]))
       	          .add(fPad[ijk+2*ri+3].multiply(gg[237]))
       	          .add(fPad[ijk+3*ri-3].multiply(gg[238]))
       	          .add(fPad[ijk+3*ri-2].multiply(gg[239]))
       	          .add(fPad[ijk+3*ri-1].multiply(gg[240]))
       	          .add(fPad[ijk+3*ri+0].multiply(gg[241]))
       	          .add(fPad[ijk+3*ri+1].multiply(gg[242]))
       	          .add(fPad[ijk+3*ri+2].multiply(gg[243]))
       	          .add(fPad[ijk+3*ri+3].multiply(gg[244]))
       	          .add(fPad[ijk-3*ri-3].multiply(gg[245]))
       	          .add(fPad[ijk-3*ri-2].multiply(gg[246]))
       	          .add(fPad[ijk-3*ri-1].multiply(gg[247]))
       	          .add(fPad[ijk-3*ri+0].multiply(gg[248]))
       	          .add(fPad[ijk-3*ri+1].multiply(gg[249]))
       	          .add(fPad[ijk-3*ri+2].multiply(gg[250]))
       	          .add(fPad[ijk-3*ri+3].multiply(gg[251]))
       	          .add(fPad[ijk-2*ri-3].multiply(gg[252]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[253]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[254]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[255]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[256]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[257]))
       	          .add(fPad[ijk-2*ri+3].multiply(gg[258]))
       	          .add(fPad[ijk-1*ri-3].multiply(gg[259]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[260]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[261]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[262]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[263]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[264]))
       	          .add(fPad[ijk-1*ri+3].multiply(gg[265]))
       	          .add(fPad[ijk+0*ri-3].multiply(gg[266]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[267]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[268]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[269]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[270]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[271]))
       	          .add(fPad[ijk+0*ri+3].multiply(gg[272]))
       	          .add(fPad[ijk+1*ri-3].multiply(gg[273]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[274]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[275]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[276]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[277]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[278]))
       	          .add(fPad[ijk+1*ri+3].multiply(gg[279]))
       	          .add(fPad[ijk+2*ri-3].multiply(gg[280]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[281]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[282]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[283]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[284]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[285]))
       	          .add(fPad[ijk+2*ri+3].multiply(gg[286]))
       	          .add(fPad[ijk+3*ri-3].multiply(gg[287]))
       	          .add(fPad[ijk+3*ri-2].multiply(gg[288]))
       	          .add(fPad[ijk+3*ri-1].multiply(gg[289]))
       	          .add(fPad[ijk+3*ri+0].multiply(gg[290]))
       	          .add(fPad[ijk+3*ri+1].multiply(gg[291]))
       	          .add(fPad[ijk+3*ri+2].multiply(gg[292]))
       	          .add(fPad[ijk+3*ri+3].multiply(gg[293]))
       	          .add(fPad[ijk-3*ri-3].multiply(gg[294]))
       	          .add(fPad[ijk-3*ri-2].multiply(gg[295]))
       	          .add(fPad[ijk-3*ri-1].multiply(gg[296]))
       	          .add(fPad[ijk-3*ri+0].multiply(gg[297]))
       	          .add(fPad[ijk-3*ri+1].multiply(gg[298]))
       	          .add(fPad[ijk-3*ri+2].multiply(gg[299]))
       	          .add(fPad[ijk-3*ri+3].multiply(gg[300]))
       	          .add(fPad[ijk-2*ri-3].multiply(gg[301]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[302]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[303]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[304]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[305]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[306]))
       	          .add(fPad[ijk-2*ri+3].multiply(gg[307]))
       	          .add(fPad[ijk-1*ri-3].multiply(gg[308]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[309]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[310]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[311]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[312]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[313]))
       	          .add(fPad[ijk-1*ri+3].multiply(gg[314]))
       	          .add(fPad[ijk+0*ri-3].multiply(gg[315]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[316]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[317]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[318]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[319]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[320]))
       	          .add(fPad[ijk+0*ri+3].multiply(gg[321]))
       	          .add(fPad[ijk+1*ri-3].multiply(gg[322]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[323]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[324]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[325]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[326]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[327]))
       	          .add(fPad[ijk+1*ri+3].multiply(gg[328]))
       	          .add(fPad[ijk+2*ri-3].multiply(gg[329]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[330]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[331]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[332]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[333]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[334]))
       	          .add(fPad[ijk+2*ri+3].multiply(gg[335]))
       	          .add(fPad[ijk+3*ri-3].multiply(gg[336]))
       	          .add(fPad[ijk+3*ri-2].multiply(gg[337]))
       	          .add(fPad[ijk+3*ri-1].multiply(gg[338]))
       	          .add(fPad[ijk+3*ri+0].multiply(gg[339]))
       	          .add(fPad[ijk+3*ri+1].multiply(gg[340]))
       	          .add(fPad[ijk+3*ri+2].multiply(gg[341]))
       	          .add(fPad[ijk+3*ri+3].multiply(gg[342]));
     	        	   }
                }
            }
        return Boundaries.finishBoundaries3d(fPad, gg, r, gi, gj, gk, hgi, hgie, hgj, hgje, hgk, hgke, ri, rj, rk)
;    };
// end convolve_7_7_7(double[][][] f, double[][][] g) {
// begin convolve_9
    public static double[] convolve_9(double[] f, double[] g) {
        final int fi = f.length;
        final int gi = g.length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		 double[] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie);
		 double[] r = JVCLUtils.zeroPadBoundaries(new double[fi], hgi, hgie);
		 final int ri = r.length;
		 for (int i = hgi; i < ri-1-hgie; i++) {
                 r[i] =
                 fPad[i-4]*g[8]+
                 fPad[i-3]*g[7]+
                 fPad[i-2]*g[6]+
                 fPad[i-1]*g[5]+
                 fPad[i+0]*g[4]+
                 fPad[i+1]*g[3]+
                 fPad[i+2]*g[2]+
                 fPad[i+3]*g[1]+
                 fPad[i+4]*g[0];
            }
        return Boundaries.finishBoundaries1d(fPad, g, r, gi,hgi, hgie,ri);
    };
    public static Complex[] convolve_9(Complex[] f, Complex[] g) {
        final int fi = f.length;
        final int gi = g.length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		 Complex[] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie);
		 Complex[] r = JVCLUtils.zeroPadBoundaries(new Complex[fi], hgi, hgie);
		 final int ri = r.length;
		 for (int i = hgi; i < ri-1-hgie; i++) {
                 r[i]
                 .add(fPad[i-4].multiply(g[8]))
                 .add(fPad[i-3].multiply(g[7]))
                 .add(fPad[i-2].multiply(g[6]))
                 .add(fPad[i-1].multiply(g[5]))
                 .add(fPad[i+0].multiply(g[4]))
                 .add(fPad[i+1].multiply(g[3]))
                 .add(fPad[i+2].multiply(g[2]))
                 .add(fPad[i+3].multiply(g[1]))
                 .add(fPad[i+4].multiply(g[0]));
            }
        return Boundaries.finishBoundaries1d(fPad, g, r, gi, hgi, hgie, ri);
    };
// end convolve_9
 // begin convolve_9_9
    public static double[][] convolve_9_9(double[][] f, double[][] g) {
        final int fi = f.length;
        final int fj = f[0].length;
        final int gi = g.length;
        final int gj = g[0].length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgj = (int)( (gj - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
        final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
        double[] gg = ArrayMath.vectorise(g);
        double[] fPad = ArrayMath.vectorise(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje));
        double[][] rr = JVCLUtils.zeroPadBoundaries(new double[fi][fj], hgi, hgie, hgj, hgje);
        final int ri = rr.length;
        final int rj = rr[0].length;
        double[] r = ArrayMath.vectorise(rr);
        int ij;
        for (int i = hgi; i < fi - 1 - hgie; i++) {
            for (int j = hgj; j < fj - 1 - hgje; j++) {
                ij = j*ri + i;
                 r[ij] =
                 fPad[ij-4*ri-4]*gg[0]+
                 fPad[ij-4*ri-3]*gg[1]+
                 fPad[ij-4*ri-2]*gg[2]+
                 fPad[ij-4*ri-1]*gg[3]+
                 fPad[ij-4*ri+0]*gg[4]+
                 fPad[ij-4*ri+1]*gg[5]+
                 fPad[ij-4*ri+2]*gg[6]+
                 fPad[ij-4*ri+3]*gg[7]+
                 fPad[ij-4*ri+4]*gg[8]+
                 fPad[ij-3*ri-4]*gg[9]+
                 fPad[ij-3*ri-3]*gg[10]+
                 fPad[ij-3*ri-2]*gg[11]+
                 fPad[ij-3*ri-1]*gg[12]+
                 fPad[ij-3*ri+0]*gg[13]+
                 fPad[ij-3*ri+1]*gg[14]+
                 fPad[ij-3*ri+2]*gg[15]+
                 fPad[ij-3*ri+3]*gg[16]+
                 fPad[ij-3*ri+4]*gg[17]+
                 fPad[ij-2*ri-4]*gg[18]+
                 fPad[ij-2*ri-3]*gg[19]+
                 fPad[ij-2*ri-2]*gg[20]+
                 fPad[ij-2*ri-1]*gg[21]+
                 fPad[ij-2*ri+0]*gg[22]+
                 fPad[ij-2*ri+1]*gg[23]+
                 fPad[ij-2*ri+2]*gg[24]+
                 fPad[ij-2*ri+3]*gg[25]+
                 fPad[ij-2*ri+4]*gg[26]+
                 fPad[ij-1*ri-4]*gg[27]+
                 fPad[ij-1*ri-3]*gg[28]+
                 fPad[ij-1*ri-2]*gg[29]+
                 fPad[ij-1*ri-1]*gg[30]+
                 fPad[ij-1*ri+0]*gg[31]+
                 fPad[ij-1*ri+1]*gg[32]+
                 fPad[ij-1*ri+2]*gg[33]+
                 fPad[ij-1*ri+3]*gg[34]+
                 fPad[ij-1*ri+4]*gg[35]+
                 fPad[ij+0*ri-4]*gg[36]+
                 fPad[ij+0*ri-3]*gg[37]+
                 fPad[ij+0*ri-2]*gg[38]+
                 fPad[ij+0*ri-1]*gg[39]+
                 fPad[ij+0*ri+0]*gg[40]+
                 fPad[ij+0*ri+1]*gg[41]+
                 fPad[ij+0*ri+2]*gg[42]+
                 fPad[ij+0*ri+3]*gg[43]+
                 fPad[ij+0*ri+4]*gg[44]+
                 fPad[ij+1*ri-4]*gg[45]+
                 fPad[ij+1*ri-3]*gg[46]+
                 fPad[ij+1*ri-2]*gg[47]+
                 fPad[ij+1*ri-1]*gg[48]+
                 fPad[ij+1*ri+0]*gg[49]+
                 fPad[ij+1*ri+1]*gg[50]+
                 fPad[ij+1*ri+2]*gg[51]+
                 fPad[ij+1*ri+3]*gg[52]+
                 fPad[ij+1*ri+4]*gg[53]+
                 fPad[ij+2*ri-4]*gg[54]+
                 fPad[ij+2*ri-3]*gg[55]+
                 fPad[ij+2*ri-2]*gg[56]+
                 fPad[ij+2*ri-1]*gg[57]+
                 fPad[ij+2*ri+0]*gg[58]+
                 fPad[ij+2*ri+1]*gg[59]+
                 fPad[ij+2*ri+2]*gg[60]+
                 fPad[ij+2*ri+3]*gg[61]+
                 fPad[ij+2*ri+4]*gg[62]+
                 fPad[ij+3*ri-4]*gg[63]+
                 fPad[ij+3*ri-3]*gg[64]+
                 fPad[ij+3*ri-2]*gg[65]+
                 fPad[ij+3*ri-1]*gg[66]+
                 fPad[ij+3*ri+0]*gg[67]+
                 fPad[ij+3*ri+1]*gg[68]+
                 fPad[ij+3*ri+2]*gg[69]+
                 fPad[ij+3*ri+3]*gg[70]+
                 fPad[ij+3*ri+4]*gg[71]+
                 fPad[ij+4*ri-4]*gg[72]+
                 fPad[ij+4*ri-3]*gg[73]+
                 fPad[ij+4*ri-2]*gg[74]+
                 fPad[ij+4*ri-1]*gg[75]+
                 fPad[ij+4*ri+0]*gg[76]+
                 fPad[ij+4*ri+1]*gg[77]+
                 fPad[ij+4*ri+2]*gg[78]+
                 fPad[ij+4*ri+3]*gg[79]+
                 fPad[ij+4*ri+4]*gg[80];
                }
            }
        return Boundaries.finishBoundaries2d(fPad, gg, r, gi, gj, hgi, hgie, hgj, hgje, ri, rj);
    };
    public static Complex[][] convolve_9_9(Complex[][] f, Complex[][] g) {
        final int fi = f.length;
        final int fj = f[0].length;
        final int gi = g.length;
        final int gj = g[0].length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgj = (int)( (gj - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
        final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
        Complex[] gg = ArrayMath.vectorise(g);
        Complex[] fPad = ArrayMath.vectorise(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje));
        Complex[][] rr = JVCLUtils.zeroPadBoundaries(new Complex[fi][fj], hgi, hgie, hgj, hgje);
        final int ri = rr.length;
        final int rj = rr[0].length;
        Complex[] r = ArrayMath.vectorise(rr);
        int ij;
        for (int i = hgi; i < fi - 1 - hgie; i++) {
            for (int j = hgj; j < fj - 1 - hgje; j++) {
                ij = j*ri + i;
				r[ij]
                .add(fPad[ij-4*ri-4].multiply(gg[0]))
                .add(fPad[ij-4*ri-3].multiply(gg[1]))
                .add(fPad[ij-4*ri-2].multiply(gg[2]))
                .add(fPad[ij-4*ri-1].multiply(gg[3]))
                .add(fPad[ij-4*ri+0].multiply(gg[4]))
                .add(fPad[ij-4*ri+1].multiply(gg[5]))
                .add(fPad[ij-4*ri+2].multiply(gg[6]))
                .add(fPad[ij-4*ri+3].multiply(gg[7]))
                .add(fPad[ij-4*ri+4].multiply(gg[8]))
                .add(fPad[ij-3*ri-4].multiply(gg[9]))
                .add(fPad[ij-3*ri-3].multiply(gg[10]))
                .add(fPad[ij-3*ri-2].multiply(gg[11]))
                .add(fPad[ij-3*ri-1].multiply(gg[12]))
                .add(fPad[ij-3*ri+0].multiply(gg[13]))
                .add(fPad[ij-3*ri+1].multiply(gg[14]))
                .add(fPad[ij-3*ri+2].multiply(gg[15]))
                .add(fPad[ij-3*ri+3].multiply(gg[16]))
                .add(fPad[ij-3*ri+4].multiply(gg[17]))
                .add(fPad[ij-2*ri-4].multiply(gg[18]))
                .add(fPad[ij-2*ri-3].multiply(gg[19]))
                .add(fPad[ij-2*ri-2].multiply(gg[20]))
                .add(fPad[ij-2*ri-1].multiply(gg[21]))
                .add(fPad[ij-2*ri+0].multiply(gg[22]))
                .add(fPad[ij-2*ri+1].multiply(gg[23]))
                .add(fPad[ij-2*ri+2].multiply(gg[24]))
                .add(fPad[ij-2*ri+3].multiply(gg[25]))
                .add(fPad[ij-2*ri+4].multiply(gg[26]))
                .add(fPad[ij-1*ri-4].multiply(gg[27]))
                .add(fPad[ij-1*ri-3].multiply(gg[28]))
                .add(fPad[ij-1*ri-2].multiply(gg[29]))
                .add(fPad[ij-1*ri-1].multiply(gg[30]))
                .add(fPad[ij-1*ri+0].multiply(gg[31]))
                .add(fPad[ij-1*ri+1].multiply(gg[32]))
                .add(fPad[ij-1*ri+2].multiply(gg[33]))
                .add(fPad[ij-1*ri+3].multiply(gg[34]))
                .add(fPad[ij-1*ri+4].multiply(gg[35]))
                .add(fPad[ij+0*ri-4].multiply(gg[36]))
                .add(fPad[ij+0*ri-3].multiply(gg[37]))
                .add(fPad[ij+0*ri-2].multiply(gg[38]))
                .add(fPad[ij+0*ri-1].multiply(gg[39]))
                .add(fPad[ij+0*ri+0].multiply(gg[40]))
                .add(fPad[ij+0*ri+1].multiply(gg[41]))
                .add(fPad[ij+0*ri+2].multiply(gg[42]))
                .add(fPad[ij+0*ri+3].multiply(gg[43]))
                .add(fPad[ij+0*ri+4].multiply(gg[44]))
                .add(fPad[ij+1*ri-4].multiply(gg[45]))
                .add(fPad[ij+1*ri-3].multiply(gg[46]))
                .add(fPad[ij+1*ri-2].multiply(gg[47]))
                .add(fPad[ij+1*ri-1].multiply(gg[48]))
                .add(fPad[ij+1*ri+0].multiply(gg[49]))
                .add(fPad[ij+1*ri+1].multiply(gg[50]))
                .add(fPad[ij+1*ri+2].multiply(gg[51]))
                .add(fPad[ij+1*ri+3].multiply(gg[52]))
                .add(fPad[ij+1*ri+4].multiply(gg[53]))
                .add(fPad[ij+2*ri-4].multiply(gg[54]))
                .add(fPad[ij+2*ri-3].multiply(gg[55]))
                .add(fPad[ij+2*ri-2].multiply(gg[56]))
                .add(fPad[ij+2*ri-1].multiply(gg[57]))
                .add(fPad[ij+2*ri+0].multiply(gg[58]))
                .add(fPad[ij+2*ri+1].multiply(gg[59]))
                .add(fPad[ij+2*ri+2].multiply(gg[60]))
                .add(fPad[ij+2*ri+3].multiply(gg[61]))
                .add(fPad[ij+2*ri+4].multiply(gg[62]))
                .add(fPad[ij+3*ri-4].multiply(gg[63]))
                .add(fPad[ij+3*ri-3].multiply(gg[64]))
                .add(fPad[ij+3*ri-2].multiply(gg[65]))
                .add(fPad[ij+3*ri-1].multiply(gg[66]))
                .add(fPad[ij+3*ri+0].multiply(gg[67]))
                .add(fPad[ij+3*ri+1].multiply(gg[68]))
                .add(fPad[ij+3*ri+2].multiply(gg[69]))
                .add(fPad[ij+3*ri+3].multiply(gg[70]))
                .add(fPad[ij+3*ri+4].multiply(gg[71]))
                .add(fPad[ij+4*ri-4].multiply(gg[72]))
                .add(fPad[ij+4*ri-3].multiply(gg[73]))
                .add(fPad[ij+4*ri-2].multiply(gg[74]))
                .add(fPad[ij+4*ri-1].multiply(gg[75]))
                .add(fPad[ij+4*ri+0].multiply(gg[76]))
                .add(fPad[ij+4*ri+1].multiply(gg[77]))
                .add(fPad[ij+4*ri+2].multiply(gg[78]))
                .add(fPad[ij+4*ri+3].multiply(gg[79]))
                .add(fPad[ij+4*ri+4].multiply(gg[80]));
                }
            }
    return Boundaries.finishBoundaries2d(fPad, gg, r, gi, gj, hgi, hgie, hgj, hgje, ri, rj)
;    };
 // end convolve_9_9
// begin convolve_9_9_9
    public static double[][][] convolve_9_9_9(double[][][] f, double[][][] g) {
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
        double[] gg = ArrayMath.vectorise(g);
        double[] fPad = ArrayMath.vectorise(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje, hgk, hgke));
        double[][][] rr = JVCLUtils.zeroPadBoundaries(new double[fi][fj][fk], hgi, hgie, hgj, hgje, hgk, hgke);
        final int ri = rr.length;
        final int rj = rr[0].length;
        final int rk = rr[0][0].length;
        double[] r = ArrayMath.vectorise(rr);
        int ijk;
        for (int i = hgi; i < fi - 1 - hgie; i++) {
            for (int j = hgj; j < fj - 1 - hgje; j++) {
            	for (int k = hgk; k < fk - 1 - hgke; k++) {
               	ijk = k*ri*rj + j*ri + i;
	                 r[ijk] =
    	             fPad[ijk-4*ri-4]*gg[0]+
    	             fPad[ijk-4*ri-3]*gg[1]+
    	             fPad[ijk-4*ri-2]*gg[2]+
    	             fPad[ijk-4*ri-1]*gg[3]+
    	             fPad[ijk-4*ri+0]*gg[4]+
    	             fPad[ijk-4*ri+1]*gg[5]+
    	             fPad[ijk-4*ri+2]*gg[6]+
    	             fPad[ijk-4*ri+3]*gg[7]+
    	             fPad[ijk-4*ri+4]*gg[8]+
    	             fPad[ijk-3*ri-4]*gg[9]+
    	             fPad[ijk-3*ri-3]*gg[10]+
    	             fPad[ijk-3*ri-2]*gg[11]+
    	             fPad[ijk-3*ri-1]*gg[12]+
    	             fPad[ijk-3*ri+0]*gg[13]+
    	             fPad[ijk-3*ri+1]*gg[14]+
    	             fPad[ijk-3*ri+2]*gg[15]+
    	             fPad[ijk-3*ri+3]*gg[16]+
    	             fPad[ijk-3*ri+4]*gg[17]+
    	             fPad[ijk-2*ri-4]*gg[18]+
    	             fPad[ijk-2*ri-3]*gg[19]+
    	             fPad[ijk-2*ri-2]*gg[20]+
    	             fPad[ijk-2*ri-1]*gg[21]+
    	             fPad[ijk-2*ri+0]*gg[22]+
    	             fPad[ijk-2*ri+1]*gg[23]+
    	             fPad[ijk-2*ri+2]*gg[24]+
    	             fPad[ijk-2*ri+3]*gg[25]+
    	             fPad[ijk-2*ri+4]*gg[26]+
    	             fPad[ijk-1*ri-4]*gg[27]+
    	             fPad[ijk-1*ri-3]*gg[28]+
    	             fPad[ijk-1*ri-2]*gg[29]+
    	             fPad[ijk-1*ri-1]*gg[30]+
    	             fPad[ijk-1*ri+0]*gg[31]+
    	             fPad[ijk-1*ri+1]*gg[32]+
    	             fPad[ijk-1*ri+2]*gg[33]+
    	             fPad[ijk-1*ri+3]*gg[34]+
    	             fPad[ijk-1*ri+4]*gg[35]+
    	             fPad[ijk+0*ri-4]*gg[36]+
    	             fPad[ijk+0*ri-3]*gg[37]+
    	             fPad[ijk+0*ri-2]*gg[38]+
    	             fPad[ijk+0*ri-1]*gg[39]+
    	             fPad[ijk+0*ri+0]*gg[40]+
    	             fPad[ijk+0*ri+1]*gg[41]+
    	             fPad[ijk+0*ri+2]*gg[42]+
    	             fPad[ijk+0*ri+3]*gg[43]+
    	             fPad[ijk+0*ri+4]*gg[44]+
    	             fPad[ijk+1*ri-4]*gg[45]+
    	             fPad[ijk+1*ri-3]*gg[46]+
    	             fPad[ijk+1*ri-2]*gg[47]+
    	             fPad[ijk+1*ri-1]*gg[48]+
    	             fPad[ijk+1*ri+0]*gg[49]+
    	             fPad[ijk+1*ri+1]*gg[50]+
    	             fPad[ijk+1*ri+2]*gg[51]+
    	             fPad[ijk+1*ri+3]*gg[52]+
    	             fPad[ijk+1*ri+4]*gg[53]+
    	             fPad[ijk+2*ri-4]*gg[54]+
    	             fPad[ijk+2*ri-3]*gg[55]+
    	             fPad[ijk+2*ri-2]*gg[56]+
    	             fPad[ijk+2*ri-1]*gg[57]+
    	             fPad[ijk+2*ri+0]*gg[58]+
    	             fPad[ijk+2*ri+1]*gg[59]+
    	             fPad[ijk+2*ri+2]*gg[60]+
    	             fPad[ijk+2*ri+3]*gg[61]+
    	             fPad[ijk+2*ri+4]*gg[62]+
    	             fPad[ijk+3*ri-4]*gg[63]+
    	             fPad[ijk+3*ri-3]*gg[64]+
    	             fPad[ijk+3*ri-2]*gg[65]+
    	             fPad[ijk+3*ri-1]*gg[66]+
    	             fPad[ijk+3*ri+0]*gg[67]+
    	             fPad[ijk+3*ri+1]*gg[68]+
    	             fPad[ijk+3*ri+2]*gg[69]+
    	             fPad[ijk+3*ri+3]*gg[70]+
    	             fPad[ijk+3*ri+4]*gg[71]+
    	             fPad[ijk+4*ri-4]*gg[72]+
    	             fPad[ijk+4*ri-3]*gg[73]+
    	             fPad[ijk+4*ri-2]*gg[74]+
    	             fPad[ijk+4*ri-1]*gg[75]+
    	             fPad[ijk+4*ri+0]*gg[76]+
    	             fPad[ijk+4*ri+1]*gg[77]+
    	             fPad[ijk+4*ri+2]*gg[78]+
    	             fPad[ijk+4*ri+3]*gg[79]+
    	             fPad[ijk+4*ri+4]*gg[80]+
    	             fPad[ijk-4*ri-4]*gg[81]+
    	             fPad[ijk-4*ri-3]*gg[82]+
    	             fPad[ijk-4*ri-2]*gg[83]+
    	             fPad[ijk-4*ri-1]*gg[84]+
    	             fPad[ijk-4*ri+0]*gg[85]+
    	             fPad[ijk-4*ri+1]*gg[86]+
    	             fPad[ijk-4*ri+2]*gg[87]+
    	             fPad[ijk-4*ri+3]*gg[88]+
    	             fPad[ijk-4*ri+4]*gg[89]+
    	             fPad[ijk-3*ri-4]*gg[90]+
    	             fPad[ijk-3*ri-3]*gg[91]+
    	             fPad[ijk-3*ri-2]*gg[92]+
    	             fPad[ijk-3*ri-1]*gg[93]+
    	             fPad[ijk-3*ri+0]*gg[94]+
    	             fPad[ijk-3*ri+1]*gg[95]+
    	             fPad[ijk-3*ri+2]*gg[96]+
    	             fPad[ijk-3*ri+3]*gg[97]+
    	             fPad[ijk-3*ri+4]*gg[98]+
    	             fPad[ijk-2*ri-4]*gg[99]+
    	             fPad[ijk-2*ri-3]*gg[100]+
    	             fPad[ijk-2*ri-2]*gg[101]+
    	             fPad[ijk-2*ri-1]*gg[102]+
    	             fPad[ijk-2*ri+0]*gg[103]+
    	             fPad[ijk-2*ri+1]*gg[104]+
    	             fPad[ijk-2*ri+2]*gg[105]+
    	             fPad[ijk-2*ri+3]*gg[106]+
    	             fPad[ijk-2*ri+4]*gg[107]+
    	             fPad[ijk-1*ri-4]*gg[108]+
    	             fPad[ijk-1*ri-3]*gg[109]+
    	             fPad[ijk-1*ri-2]*gg[110]+
    	             fPad[ijk-1*ri-1]*gg[111]+
    	             fPad[ijk-1*ri+0]*gg[112]+
    	             fPad[ijk-1*ri+1]*gg[113]+
    	             fPad[ijk-1*ri+2]*gg[114]+
    	             fPad[ijk-1*ri+3]*gg[115]+
    	             fPad[ijk-1*ri+4]*gg[116]+
    	             fPad[ijk+0*ri-4]*gg[117]+
    	             fPad[ijk+0*ri-3]*gg[118]+
    	             fPad[ijk+0*ri-2]*gg[119]+
    	             fPad[ijk+0*ri-1]*gg[120]+
    	             fPad[ijk+0*ri+0]*gg[121]+
    	             fPad[ijk+0*ri+1]*gg[122]+
    	             fPad[ijk+0*ri+2]*gg[123]+
    	             fPad[ijk+0*ri+3]*gg[124]+
    	             fPad[ijk+0*ri+4]*gg[125]+
    	             fPad[ijk+1*ri-4]*gg[126]+
    	             fPad[ijk+1*ri-3]*gg[127]+
    	             fPad[ijk+1*ri-2]*gg[128]+
    	             fPad[ijk+1*ri-1]*gg[129]+
    	             fPad[ijk+1*ri+0]*gg[130]+
    	             fPad[ijk+1*ri+1]*gg[131]+
    	             fPad[ijk+1*ri+2]*gg[132]+
    	             fPad[ijk+1*ri+3]*gg[133]+
    	             fPad[ijk+1*ri+4]*gg[134]+
    	             fPad[ijk+2*ri-4]*gg[135]+
    	             fPad[ijk+2*ri-3]*gg[136]+
    	             fPad[ijk+2*ri-2]*gg[137]+
    	             fPad[ijk+2*ri-1]*gg[138]+
    	             fPad[ijk+2*ri+0]*gg[139]+
    	             fPad[ijk+2*ri+1]*gg[140]+
    	             fPad[ijk+2*ri+2]*gg[141]+
    	             fPad[ijk+2*ri+3]*gg[142]+
    	             fPad[ijk+2*ri+4]*gg[143]+
    	             fPad[ijk+3*ri-4]*gg[144]+
    	             fPad[ijk+3*ri-3]*gg[145]+
    	             fPad[ijk+3*ri-2]*gg[146]+
    	             fPad[ijk+3*ri-1]*gg[147]+
    	             fPad[ijk+3*ri+0]*gg[148]+
    	             fPad[ijk+3*ri+1]*gg[149]+
    	             fPad[ijk+3*ri+2]*gg[150]+
    	             fPad[ijk+3*ri+3]*gg[151]+
    	             fPad[ijk+3*ri+4]*gg[152]+
    	             fPad[ijk+4*ri-4]*gg[153]+
    	             fPad[ijk+4*ri-3]*gg[154]+
    	             fPad[ijk+4*ri-2]*gg[155]+
    	             fPad[ijk+4*ri-1]*gg[156]+
    	             fPad[ijk+4*ri+0]*gg[157]+
    	             fPad[ijk+4*ri+1]*gg[158]+
    	             fPad[ijk+4*ri+2]*gg[159]+
    	             fPad[ijk+4*ri+3]*gg[160]+
    	             fPad[ijk+4*ri+4]*gg[161]+
    	             fPad[ijk-4*ri-4]*gg[162]+
    	             fPad[ijk-4*ri-3]*gg[163]+
    	             fPad[ijk-4*ri-2]*gg[164]+
    	             fPad[ijk-4*ri-1]*gg[165]+
    	             fPad[ijk-4*ri+0]*gg[166]+
    	             fPad[ijk-4*ri+1]*gg[167]+
    	             fPad[ijk-4*ri+2]*gg[168]+
    	             fPad[ijk-4*ri+3]*gg[169]+
    	             fPad[ijk-4*ri+4]*gg[170]+
    	             fPad[ijk-3*ri-4]*gg[171]+
    	             fPad[ijk-3*ri-3]*gg[172]+
    	             fPad[ijk-3*ri-2]*gg[173]+
    	             fPad[ijk-3*ri-1]*gg[174]+
    	             fPad[ijk-3*ri+0]*gg[175]+
    	             fPad[ijk-3*ri+1]*gg[176]+
    	             fPad[ijk-3*ri+2]*gg[177]+
    	             fPad[ijk-3*ri+3]*gg[178]+
    	             fPad[ijk-3*ri+4]*gg[179]+
    	             fPad[ijk-2*ri-4]*gg[180]+
    	             fPad[ijk-2*ri-3]*gg[181]+
    	             fPad[ijk-2*ri-2]*gg[182]+
    	             fPad[ijk-2*ri-1]*gg[183]+
    	             fPad[ijk-2*ri+0]*gg[184]+
    	             fPad[ijk-2*ri+1]*gg[185]+
    	             fPad[ijk-2*ri+2]*gg[186]+
    	             fPad[ijk-2*ri+3]*gg[187]+
    	             fPad[ijk-2*ri+4]*gg[188]+
    	             fPad[ijk-1*ri-4]*gg[189]+
    	             fPad[ijk-1*ri-3]*gg[190]+
    	             fPad[ijk-1*ri-2]*gg[191]+
    	             fPad[ijk-1*ri-1]*gg[192]+
    	             fPad[ijk-1*ri+0]*gg[193]+
    	             fPad[ijk-1*ri+1]*gg[194]+
    	             fPad[ijk-1*ri+2]*gg[195]+
    	             fPad[ijk-1*ri+3]*gg[196]+
    	             fPad[ijk-1*ri+4]*gg[197]+
    	             fPad[ijk+0*ri-4]*gg[198]+
    	             fPad[ijk+0*ri-3]*gg[199]+
    	             fPad[ijk+0*ri-2]*gg[200]+
    	             fPad[ijk+0*ri-1]*gg[201]+
    	             fPad[ijk+0*ri+0]*gg[202]+
    	             fPad[ijk+0*ri+1]*gg[203]+
    	             fPad[ijk+0*ri+2]*gg[204]+
    	             fPad[ijk+0*ri+3]*gg[205]+
    	             fPad[ijk+0*ri+4]*gg[206]+
    	             fPad[ijk+1*ri-4]*gg[207]+
    	             fPad[ijk+1*ri-3]*gg[208]+
    	             fPad[ijk+1*ri-2]*gg[209]+
    	             fPad[ijk+1*ri-1]*gg[210]+
    	             fPad[ijk+1*ri+0]*gg[211]+
    	             fPad[ijk+1*ri+1]*gg[212]+
    	             fPad[ijk+1*ri+2]*gg[213]+
    	             fPad[ijk+1*ri+3]*gg[214]+
    	             fPad[ijk+1*ri+4]*gg[215]+
    	             fPad[ijk+2*ri-4]*gg[216]+
    	             fPad[ijk+2*ri-3]*gg[217]+
    	             fPad[ijk+2*ri-2]*gg[218]+
    	             fPad[ijk+2*ri-1]*gg[219]+
    	             fPad[ijk+2*ri+0]*gg[220]+
    	             fPad[ijk+2*ri+1]*gg[221]+
    	             fPad[ijk+2*ri+2]*gg[222]+
    	             fPad[ijk+2*ri+3]*gg[223]+
    	             fPad[ijk+2*ri+4]*gg[224]+
    	             fPad[ijk+3*ri-4]*gg[225]+
    	             fPad[ijk+3*ri-3]*gg[226]+
    	             fPad[ijk+3*ri-2]*gg[227]+
    	             fPad[ijk+3*ri-1]*gg[228]+
    	             fPad[ijk+3*ri+0]*gg[229]+
    	             fPad[ijk+3*ri+1]*gg[230]+
    	             fPad[ijk+3*ri+2]*gg[231]+
    	             fPad[ijk+3*ri+3]*gg[232]+
    	             fPad[ijk+3*ri+4]*gg[233]+
    	             fPad[ijk+4*ri-4]*gg[234]+
    	             fPad[ijk+4*ri-3]*gg[235]+
    	             fPad[ijk+4*ri-2]*gg[236]+
    	             fPad[ijk+4*ri-1]*gg[237]+
    	             fPad[ijk+4*ri+0]*gg[238]+
    	             fPad[ijk+4*ri+1]*gg[239]+
    	             fPad[ijk+4*ri+2]*gg[240]+
    	             fPad[ijk+4*ri+3]*gg[241]+
    	             fPad[ijk+4*ri+4]*gg[242]+
    	             fPad[ijk-4*ri-4]*gg[243]+
    	             fPad[ijk-4*ri-3]*gg[244]+
    	             fPad[ijk-4*ri-2]*gg[245]+
    	             fPad[ijk-4*ri-1]*gg[246]+
    	             fPad[ijk-4*ri+0]*gg[247]+
    	             fPad[ijk-4*ri+1]*gg[248]+
    	             fPad[ijk-4*ri+2]*gg[249]+
    	             fPad[ijk-4*ri+3]*gg[250]+
    	             fPad[ijk-4*ri+4]*gg[251]+
    	             fPad[ijk-3*ri-4]*gg[252]+
    	             fPad[ijk-3*ri-3]*gg[253]+
    	             fPad[ijk-3*ri-2]*gg[254]+
    	             fPad[ijk-3*ri-1]*gg[255]+
    	             fPad[ijk-3*ri+0]*gg[256]+
    	             fPad[ijk-3*ri+1]*gg[257]+
    	             fPad[ijk-3*ri+2]*gg[258]+
    	             fPad[ijk-3*ri+3]*gg[259]+
    	             fPad[ijk-3*ri+4]*gg[260]+
    	             fPad[ijk-2*ri-4]*gg[261]+
    	             fPad[ijk-2*ri-3]*gg[262]+
    	             fPad[ijk-2*ri-2]*gg[263]+
    	             fPad[ijk-2*ri-1]*gg[264]+
    	             fPad[ijk-2*ri+0]*gg[265]+
    	             fPad[ijk-2*ri+1]*gg[266]+
    	             fPad[ijk-2*ri+2]*gg[267]+
    	             fPad[ijk-2*ri+3]*gg[268]+
    	             fPad[ijk-2*ri+4]*gg[269]+
    	             fPad[ijk-1*ri-4]*gg[270]+
    	             fPad[ijk-1*ri-3]*gg[271]+
    	             fPad[ijk-1*ri-2]*gg[272]+
    	             fPad[ijk-1*ri-1]*gg[273]+
    	             fPad[ijk-1*ri+0]*gg[274]+
    	             fPad[ijk-1*ri+1]*gg[275]+
    	             fPad[ijk-1*ri+2]*gg[276]+
    	             fPad[ijk-1*ri+3]*gg[277]+
    	             fPad[ijk-1*ri+4]*gg[278]+
    	             fPad[ijk+0*ri-4]*gg[279]+
    	             fPad[ijk+0*ri-3]*gg[280]+
    	             fPad[ijk+0*ri-2]*gg[281]+
    	             fPad[ijk+0*ri-1]*gg[282]+
    	             fPad[ijk+0*ri+0]*gg[283]+
    	             fPad[ijk+0*ri+1]*gg[284]+
    	             fPad[ijk+0*ri+2]*gg[285]+
    	             fPad[ijk+0*ri+3]*gg[286]+
    	             fPad[ijk+0*ri+4]*gg[287]+
    	             fPad[ijk+1*ri-4]*gg[288]+
    	             fPad[ijk+1*ri-3]*gg[289]+
    	             fPad[ijk+1*ri-2]*gg[290]+
    	             fPad[ijk+1*ri-1]*gg[291]+
    	             fPad[ijk+1*ri+0]*gg[292]+
    	             fPad[ijk+1*ri+1]*gg[293]+
    	             fPad[ijk+1*ri+2]*gg[294]+
    	             fPad[ijk+1*ri+3]*gg[295]+
    	             fPad[ijk+1*ri+4]*gg[296]+
    	             fPad[ijk+2*ri-4]*gg[297]+
    	             fPad[ijk+2*ri-3]*gg[298]+
    	             fPad[ijk+2*ri-2]*gg[299]+
    	             fPad[ijk+2*ri-1]*gg[300]+
    	             fPad[ijk+2*ri+0]*gg[301]+
    	             fPad[ijk+2*ri+1]*gg[302]+
    	             fPad[ijk+2*ri+2]*gg[303]+
    	             fPad[ijk+2*ri+3]*gg[304]+
    	             fPad[ijk+2*ri+4]*gg[305]+
    	             fPad[ijk+3*ri-4]*gg[306]+
    	             fPad[ijk+3*ri-3]*gg[307]+
    	             fPad[ijk+3*ri-2]*gg[308]+
    	             fPad[ijk+3*ri-1]*gg[309]+
    	             fPad[ijk+3*ri+0]*gg[310]+
    	             fPad[ijk+3*ri+1]*gg[311]+
    	             fPad[ijk+3*ri+2]*gg[312]+
    	             fPad[ijk+3*ri+3]*gg[313]+
    	             fPad[ijk+3*ri+4]*gg[314]+
    	             fPad[ijk+4*ri-4]*gg[315]+
    	             fPad[ijk+4*ri-3]*gg[316]+
    	             fPad[ijk+4*ri-2]*gg[317]+
    	             fPad[ijk+4*ri-1]*gg[318]+
    	             fPad[ijk+4*ri+0]*gg[319]+
    	             fPad[ijk+4*ri+1]*gg[320]+
    	             fPad[ijk+4*ri+2]*gg[321]+
    	             fPad[ijk+4*ri+3]*gg[322]+
    	             fPad[ijk+4*ri+4]*gg[323]+
    	             fPad[ijk-4*ri-4]*gg[324]+
    	             fPad[ijk-4*ri-3]*gg[325]+
    	             fPad[ijk-4*ri-2]*gg[326]+
    	             fPad[ijk-4*ri-1]*gg[327]+
    	             fPad[ijk-4*ri+0]*gg[328]+
    	             fPad[ijk-4*ri+1]*gg[329]+
    	             fPad[ijk-4*ri+2]*gg[330]+
    	             fPad[ijk-4*ri+3]*gg[331]+
    	             fPad[ijk-4*ri+4]*gg[332]+
    	             fPad[ijk-3*ri-4]*gg[333]+
    	             fPad[ijk-3*ri-3]*gg[334]+
    	             fPad[ijk-3*ri-2]*gg[335]+
    	             fPad[ijk-3*ri-1]*gg[336]+
    	             fPad[ijk-3*ri+0]*gg[337]+
    	             fPad[ijk-3*ri+1]*gg[338]+
    	             fPad[ijk-3*ri+2]*gg[339]+
    	             fPad[ijk-3*ri+3]*gg[340]+
    	             fPad[ijk-3*ri+4]*gg[341]+
    	             fPad[ijk-2*ri-4]*gg[342]+
    	             fPad[ijk-2*ri-3]*gg[343]+
    	             fPad[ijk-2*ri-2]*gg[344]+
    	             fPad[ijk-2*ri-1]*gg[345]+
    	             fPad[ijk-2*ri+0]*gg[346]+
    	             fPad[ijk-2*ri+1]*gg[347]+
    	             fPad[ijk-2*ri+2]*gg[348]+
    	             fPad[ijk-2*ri+3]*gg[349]+
    	             fPad[ijk-2*ri+4]*gg[350]+
    	             fPad[ijk-1*ri-4]*gg[351]+
    	             fPad[ijk-1*ri-3]*gg[352]+
    	             fPad[ijk-1*ri-2]*gg[353]+
    	             fPad[ijk-1*ri-1]*gg[354]+
    	             fPad[ijk-1*ri+0]*gg[355]+
    	             fPad[ijk-1*ri+1]*gg[356]+
    	             fPad[ijk-1*ri+2]*gg[357]+
    	             fPad[ijk-1*ri+3]*gg[358]+
    	             fPad[ijk-1*ri+4]*gg[359]+
    	             fPad[ijk+0*ri-4]*gg[360]+
    	             fPad[ijk+0*ri-3]*gg[361]+
    	             fPad[ijk+0*ri-2]*gg[362]+
    	             fPad[ijk+0*ri-1]*gg[363]+
    	             fPad[ijk+0*ri+0]*gg[364]+
    	             fPad[ijk+0*ri+1]*gg[365]+
    	             fPad[ijk+0*ri+2]*gg[366]+
    	             fPad[ijk+0*ri+3]*gg[367]+
    	             fPad[ijk+0*ri+4]*gg[368]+
    	             fPad[ijk+1*ri-4]*gg[369]+
    	             fPad[ijk+1*ri-3]*gg[370]+
    	             fPad[ijk+1*ri-2]*gg[371]+
    	             fPad[ijk+1*ri-1]*gg[372]+
    	             fPad[ijk+1*ri+0]*gg[373]+
    	             fPad[ijk+1*ri+1]*gg[374]+
    	             fPad[ijk+1*ri+2]*gg[375]+
    	             fPad[ijk+1*ri+3]*gg[376]+
    	             fPad[ijk+1*ri+4]*gg[377]+
    	             fPad[ijk+2*ri-4]*gg[378]+
    	             fPad[ijk+2*ri-3]*gg[379]+
    	             fPad[ijk+2*ri-2]*gg[380]+
    	             fPad[ijk+2*ri-1]*gg[381]+
    	             fPad[ijk+2*ri+0]*gg[382]+
    	             fPad[ijk+2*ri+1]*gg[383]+
    	             fPad[ijk+2*ri+2]*gg[384]+
    	             fPad[ijk+2*ri+3]*gg[385]+
    	             fPad[ijk+2*ri+4]*gg[386]+
    	             fPad[ijk+3*ri-4]*gg[387]+
    	             fPad[ijk+3*ri-3]*gg[388]+
    	             fPad[ijk+3*ri-2]*gg[389]+
    	             fPad[ijk+3*ri-1]*gg[390]+
    	             fPad[ijk+3*ri+0]*gg[391]+
    	             fPad[ijk+3*ri+1]*gg[392]+
    	             fPad[ijk+3*ri+2]*gg[393]+
    	             fPad[ijk+3*ri+3]*gg[394]+
    	             fPad[ijk+3*ri+4]*gg[395]+
    	             fPad[ijk+4*ri-4]*gg[396]+
    	             fPad[ijk+4*ri-3]*gg[397]+
    	             fPad[ijk+4*ri-2]*gg[398]+
    	             fPad[ijk+4*ri-1]*gg[399]+
    	             fPad[ijk+4*ri+0]*gg[400]+
    	             fPad[ijk+4*ri+1]*gg[401]+
    	             fPad[ijk+4*ri+2]*gg[402]+
    	             fPad[ijk+4*ri+3]*gg[403]+
    	             fPad[ijk+4*ri+4]*gg[404]+
    	             fPad[ijk-4*ri-4]*gg[405]+
    	             fPad[ijk-4*ri-3]*gg[406]+
    	             fPad[ijk-4*ri-2]*gg[407]+
    	             fPad[ijk-4*ri-1]*gg[408]+
    	             fPad[ijk-4*ri+0]*gg[409]+
    	             fPad[ijk-4*ri+1]*gg[410]+
    	             fPad[ijk-4*ri+2]*gg[411]+
    	             fPad[ijk-4*ri+3]*gg[412]+
    	             fPad[ijk-4*ri+4]*gg[413]+
    	             fPad[ijk-3*ri-4]*gg[414]+
    	             fPad[ijk-3*ri-3]*gg[415]+
    	             fPad[ijk-3*ri-2]*gg[416]+
    	             fPad[ijk-3*ri-1]*gg[417]+
    	             fPad[ijk-3*ri+0]*gg[418]+
    	             fPad[ijk-3*ri+1]*gg[419]+
    	             fPad[ijk-3*ri+2]*gg[420]+
    	             fPad[ijk-3*ri+3]*gg[421]+
    	             fPad[ijk-3*ri+4]*gg[422]+
    	             fPad[ijk-2*ri-4]*gg[423]+
    	             fPad[ijk-2*ri-3]*gg[424]+
    	             fPad[ijk-2*ri-2]*gg[425]+
    	             fPad[ijk-2*ri-1]*gg[426]+
    	             fPad[ijk-2*ri+0]*gg[427]+
    	             fPad[ijk-2*ri+1]*gg[428]+
    	             fPad[ijk-2*ri+2]*gg[429]+
    	             fPad[ijk-2*ri+3]*gg[430]+
    	             fPad[ijk-2*ri+4]*gg[431]+
    	             fPad[ijk-1*ri-4]*gg[432]+
    	             fPad[ijk-1*ri-3]*gg[433]+
    	             fPad[ijk-1*ri-2]*gg[434]+
    	             fPad[ijk-1*ri-1]*gg[435]+
    	             fPad[ijk-1*ri+0]*gg[436]+
    	             fPad[ijk-1*ri+1]*gg[437]+
    	             fPad[ijk-1*ri+2]*gg[438]+
    	             fPad[ijk-1*ri+3]*gg[439]+
    	             fPad[ijk-1*ri+4]*gg[440]+
    	             fPad[ijk+0*ri-4]*gg[441]+
    	             fPad[ijk+0*ri-3]*gg[442]+
    	             fPad[ijk+0*ri-2]*gg[443]+
    	             fPad[ijk+0*ri-1]*gg[444]+
    	             fPad[ijk+0*ri+0]*gg[445]+
    	             fPad[ijk+0*ri+1]*gg[446]+
    	             fPad[ijk+0*ri+2]*gg[447]+
    	             fPad[ijk+0*ri+3]*gg[448]+
    	             fPad[ijk+0*ri+4]*gg[449]+
    	             fPad[ijk+1*ri-4]*gg[450]+
    	             fPad[ijk+1*ri-3]*gg[451]+
    	             fPad[ijk+1*ri-2]*gg[452]+
    	             fPad[ijk+1*ri-1]*gg[453]+
    	             fPad[ijk+1*ri+0]*gg[454]+
    	             fPad[ijk+1*ri+1]*gg[455]+
    	             fPad[ijk+1*ri+2]*gg[456]+
    	             fPad[ijk+1*ri+3]*gg[457]+
    	             fPad[ijk+1*ri+4]*gg[458]+
    	             fPad[ijk+2*ri-4]*gg[459]+
    	             fPad[ijk+2*ri-3]*gg[460]+
    	             fPad[ijk+2*ri-2]*gg[461]+
    	             fPad[ijk+2*ri-1]*gg[462]+
    	             fPad[ijk+2*ri+0]*gg[463]+
    	             fPad[ijk+2*ri+1]*gg[464]+
    	             fPad[ijk+2*ri+2]*gg[465]+
    	             fPad[ijk+2*ri+3]*gg[466]+
    	             fPad[ijk+2*ri+4]*gg[467]+
    	             fPad[ijk+3*ri-4]*gg[468]+
    	             fPad[ijk+3*ri-3]*gg[469]+
    	             fPad[ijk+3*ri-2]*gg[470]+
    	             fPad[ijk+3*ri-1]*gg[471]+
    	             fPad[ijk+3*ri+0]*gg[472]+
    	             fPad[ijk+3*ri+1]*gg[473]+
    	             fPad[ijk+3*ri+2]*gg[474]+
    	             fPad[ijk+3*ri+3]*gg[475]+
    	             fPad[ijk+3*ri+4]*gg[476]+
    	             fPad[ijk+4*ri-4]*gg[477]+
    	             fPad[ijk+4*ri-3]*gg[478]+
    	             fPad[ijk+4*ri-2]*gg[479]+
    	             fPad[ijk+4*ri-1]*gg[480]+
    	             fPad[ijk+4*ri+0]*gg[481]+
    	             fPad[ijk+4*ri+1]*gg[482]+
    	             fPad[ijk+4*ri+2]*gg[483]+
    	             fPad[ijk+4*ri+3]*gg[484]+
    	             fPad[ijk+4*ri+4]*gg[485]+
    	             fPad[ijk-4*ri-4]*gg[486]+
    	             fPad[ijk-4*ri-3]*gg[487]+
    	             fPad[ijk-4*ri-2]*gg[488]+
    	             fPad[ijk-4*ri-1]*gg[489]+
    	             fPad[ijk-4*ri+0]*gg[490]+
    	             fPad[ijk-4*ri+1]*gg[491]+
    	             fPad[ijk-4*ri+2]*gg[492]+
    	             fPad[ijk-4*ri+3]*gg[493]+
    	             fPad[ijk-4*ri+4]*gg[494]+
    	             fPad[ijk-3*ri-4]*gg[495]+
    	             fPad[ijk-3*ri-3]*gg[496]+
    	             fPad[ijk-3*ri-2]*gg[497]+
    	             fPad[ijk-3*ri-1]*gg[498]+
    	             fPad[ijk-3*ri+0]*gg[499]+
    	             fPad[ijk-3*ri+1]*gg[500]+
    	             fPad[ijk-3*ri+2]*gg[501]+
    	             fPad[ijk-3*ri+3]*gg[502]+
    	             fPad[ijk-3*ri+4]*gg[503]+
    	             fPad[ijk-2*ri-4]*gg[504]+
    	             fPad[ijk-2*ri-3]*gg[505]+
    	             fPad[ijk-2*ri-2]*gg[506]+
    	             fPad[ijk-2*ri-1]*gg[507]+
    	             fPad[ijk-2*ri+0]*gg[508]+
    	             fPad[ijk-2*ri+1]*gg[509]+
    	             fPad[ijk-2*ri+2]*gg[510]+
    	             fPad[ijk-2*ri+3]*gg[511]+
    	             fPad[ijk-2*ri+4]*gg[512]+
    	             fPad[ijk-1*ri-4]*gg[513]+
    	             fPad[ijk-1*ri-3]*gg[514]+
    	             fPad[ijk-1*ri-2]*gg[515]+
    	             fPad[ijk-1*ri-1]*gg[516]+
    	             fPad[ijk-1*ri+0]*gg[517]+
    	             fPad[ijk-1*ri+1]*gg[518]+
    	             fPad[ijk-1*ri+2]*gg[519]+
    	             fPad[ijk-1*ri+3]*gg[520]+
    	             fPad[ijk-1*ri+4]*gg[521]+
    	             fPad[ijk+0*ri-4]*gg[522]+
    	             fPad[ijk+0*ri-3]*gg[523]+
    	             fPad[ijk+0*ri-2]*gg[524]+
    	             fPad[ijk+0*ri-1]*gg[525]+
    	             fPad[ijk+0*ri+0]*gg[526]+
    	             fPad[ijk+0*ri+1]*gg[527]+
    	             fPad[ijk+0*ri+2]*gg[528]+
    	             fPad[ijk+0*ri+3]*gg[529]+
    	             fPad[ijk+0*ri+4]*gg[530]+
    	             fPad[ijk+1*ri-4]*gg[531]+
    	             fPad[ijk+1*ri-3]*gg[532]+
    	             fPad[ijk+1*ri-2]*gg[533]+
    	             fPad[ijk+1*ri-1]*gg[534]+
    	             fPad[ijk+1*ri+0]*gg[535]+
    	             fPad[ijk+1*ri+1]*gg[536]+
    	             fPad[ijk+1*ri+2]*gg[537]+
    	             fPad[ijk+1*ri+3]*gg[538]+
    	             fPad[ijk+1*ri+4]*gg[539]+
    	             fPad[ijk+2*ri-4]*gg[540]+
    	             fPad[ijk+2*ri-3]*gg[541]+
    	             fPad[ijk+2*ri-2]*gg[542]+
    	             fPad[ijk+2*ri-1]*gg[543]+
    	             fPad[ijk+2*ri+0]*gg[544]+
    	             fPad[ijk+2*ri+1]*gg[545]+
    	             fPad[ijk+2*ri+2]*gg[546]+
    	             fPad[ijk+2*ri+3]*gg[547]+
    	             fPad[ijk+2*ri+4]*gg[548]+
    	             fPad[ijk+3*ri-4]*gg[549]+
    	             fPad[ijk+3*ri-3]*gg[550]+
    	             fPad[ijk+3*ri-2]*gg[551]+
    	             fPad[ijk+3*ri-1]*gg[552]+
    	             fPad[ijk+3*ri+0]*gg[553]+
    	             fPad[ijk+3*ri+1]*gg[554]+
    	             fPad[ijk+3*ri+2]*gg[555]+
    	             fPad[ijk+3*ri+3]*gg[556]+
    	             fPad[ijk+3*ri+4]*gg[557]+
    	             fPad[ijk+4*ri-4]*gg[558]+
    	             fPad[ijk+4*ri-3]*gg[559]+
    	             fPad[ijk+4*ri-2]*gg[560]+
    	             fPad[ijk+4*ri-1]*gg[561]+
    	             fPad[ijk+4*ri+0]*gg[562]+
    	             fPad[ijk+4*ri+1]*gg[563]+
    	             fPad[ijk+4*ri+2]*gg[564]+
    	             fPad[ijk+4*ri+3]*gg[565]+
    	             fPad[ijk+4*ri+4]*gg[566]+
    	             fPad[ijk-4*ri-4]*gg[567]+
    	             fPad[ijk-4*ri-3]*gg[568]+
    	             fPad[ijk-4*ri-2]*gg[569]+
    	             fPad[ijk-4*ri-1]*gg[570]+
    	             fPad[ijk-4*ri+0]*gg[571]+
    	             fPad[ijk-4*ri+1]*gg[572]+
    	             fPad[ijk-4*ri+2]*gg[573]+
    	             fPad[ijk-4*ri+3]*gg[574]+
    	             fPad[ijk-4*ri+4]*gg[575]+
    	             fPad[ijk-3*ri-4]*gg[576]+
    	             fPad[ijk-3*ri-3]*gg[577]+
    	             fPad[ijk-3*ri-2]*gg[578]+
    	             fPad[ijk-3*ri-1]*gg[579]+
    	             fPad[ijk-3*ri+0]*gg[580]+
    	             fPad[ijk-3*ri+1]*gg[581]+
    	             fPad[ijk-3*ri+2]*gg[582]+
    	             fPad[ijk-3*ri+3]*gg[583]+
    	             fPad[ijk-3*ri+4]*gg[584]+
    	             fPad[ijk-2*ri-4]*gg[585]+
    	             fPad[ijk-2*ri-3]*gg[586]+
    	             fPad[ijk-2*ri-2]*gg[587]+
    	             fPad[ijk-2*ri-1]*gg[588]+
    	             fPad[ijk-2*ri+0]*gg[589]+
    	             fPad[ijk-2*ri+1]*gg[590]+
    	             fPad[ijk-2*ri+2]*gg[591]+
    	             fPad[ijk-2*ri+3]*gg[592]+
    	             fPad[ijk-2*ri+4]*gg[593]+
    	             fPad[ijk-1*ri-4]*gg[594]+
    	             fPad[ijk-1*ri-3]*gg[595]+
    	             fPad[ijk-1*ri-2]*gg[596]+
    	             fPad[ijk-1*ri-1]*gg[597]+
    	             fPad[ijk-1*ri+0]*gg[598]+
    	             fPad[ijk-1*ri+1]*gg[599]+
    	             fPad[ijk-1*ri+2]*gg[600]+
    	             fPad[ijk-1*ri+3]*gg[601]+
    	             fPad[ijk-1*ri+4]*gg[602]+
    	             fPad[ijk+0*ri-4]*gg[603]+
    	             fPad[ijk+0*ri-3]*gg[604]+
    	             fPad[ijk+0*ri-2]*gg[605]+
    	             fPad[ijk+0*ri-1]*gg[606]+
    	             fPad[ijk+0*ri+0]*gg[607]+
    	             fPad[ijk+0*ri+1]*gg[608]+
    	             fPad[ijk+0*ri+2]*gg[609]+
    	             fPad[ijk+0*ri+3]*gg[610]+
    	             fPad[ijk+0*ri+4]*gg[611]+
    	             fPad[ijk+1*ri-4]*gg[612]+
    	             fPad[ijk+1*ri-3]*gg[613]+
    	             fPad[ijk+1*ri-2]*gg[614]+
    	             fPad[ijk+1*ri-1]*gg[615]+
    	             fPad[ijk+1*ri+0]*gg[616]+
    	             fPad[ijk+1*ri+1]*gg[617]+
    	             fPad[ijk+1*ri+2]*gg[618]+
    	             fPad[ijk+1*ri+3]*gg[619]+
    	             fPad[ijk+1*ri+4]*gg[620]+
    	             fPad[ijk+2*ri-4]*gg[621]+
    	             fPad[ijk+2*ri-3]*gg[622]+
    	             fPad[ijk+2*ri-2]*gg[623]+
    	             fPad[ijk+2*ri-1]*gg[624]+
    	             fPad[ijk+2*ri+0]*gg[625]+
    	             fPad[ijk+2*ri+1]*gg[626]+
    	             fPad[ijk+2*ri+2]*gg[627]+
    	             fPad[ijk+2*ri+3]*gg[628]+
    	             fPad[ijk+2*ri+4]*gg[629]+
    	             fPad[ijk+3*ri-4]*gg[630]+
    	             fPad[ijk+3*ri-3]*gg[631]+
    	             fPad[ijk+3*ri-2]*gg[632]+
    	             fPad[ijk+3*ri-1]*gg[633]+
    	             fPad[ijk+3*ri+0]*gg[634]+
    	             fPad[ijk+3*ri+1]*gg[635]+
    	             fPad[ijk+3*ri+2]*gg[636]+
    	             fPad[ijk+3*ri+3]*gg[637]+
    	             fPad[ijk+3*ri+4]*gg[638]+
    	             fPad[ijk+4*ri-4]*gg[639]+
    	             fPad[ijk+4*ri-3]*gg[640]+
    	             fPad[ijk+4*ri-2]*gg[641]+
    	             fPad[ijk+4*ri-1]*gg[642]+
    	             fPad[ijk+4*ri+0]*gg[643]+
    	             fPad[ijk+4*ri+1]*gg[644]+
    	             fPad[ijk+4*ri+2]*gg[645]+
    	             fPad[ijk+4*ri+3]*gg[646]+
    	             fPad[ijk+4*ri+4]*gg[647]+
    	             fPad[ijk-4*ri-4]*gg[648]+
    	             fPad[ijk-4*ri-3]*gg[649]+
    	             fPad[ijk-4*ri-2]*gg[650]+
    	             fPad[ijk-4*ri-1]*gg[651]+
    	             fPad[ijk-4*ri+0]*gg[652]+
    	             fPad[ijk-4*ri+1]*gg[653]+
    	             fPad[ijk-4*ri+2]*gg[654]+
    	             fPad[ijk-4*ri+3]*gg[655]+
    	             fPad[ijk-4*ri+4]*gg[656]+
    	             fPad[ijk-3*ri-4]*gg[657]+
    	             fPad[ijk-3*ri-3]*gg[658]+
    	             fPad[ijk-3*ri-2]*gg[659]+
    	             fPad[ijk-3*ri-1]*gg[660]+
    	             fPad[ijk-3*ri+0]*gg[661]+
    	             fPad[ijk-3*ri+1]*gg[662]+
    	             fPad[ijk-3*ri+2]*gg[663]+
    	             fPad[ijk-3*ri+3]*gg[664]+
    	             fPad[ijk-3*ri+4]*gg[665]+
    	             fPad[ijk-2*ri-4]*gg[666]+
    	             fPad[ijk-2*ri-3]*gg[667]+
    	             fPad[ijk-2*ri-2]*gg[668]+
    	             fPad[ijk-2*ri-1]*gg[669]+
    	             fPad[ijk-2*ri+0]*gg[670]+
    	             fPad[ijk-2*ri+1]*gg[671]+
    	             fPad[ijk-2*ri+2]*gg[672]+
    	             fPad[ijk-2*ri+3]*gg[673]+
    	             fPad[ijk-2*ri+4]*gg[674]+
    	             fPad[ijk-1*ri-4]*gg[675]+
    	             fPad[ijk-1*ri-3]*gg[676]+
    	             fPad[ijk-1*ri-2]*gg[677]+
    	             fPad[ijk-1*ri-1]*gg[678]+
    	             fPad[ijk-1*ri+0]*gg[679]+
    	             fPad[ijk-1*ri+1]*gg[680]+
    	             fPad[ijk-1*ri+2]*gg[681]+
    	             fPad[ijk-1*ri+3]*gg[682]+
    	             fPad[ijk-1*ri+4]*gg[683]+
    	             fPad[ijk+0*ri-4]*gg[684]+
    	             fPad[ijk+0*ri-3]*gg[685]+
    	             fPad[ijk+0*ri-2]*gg[686]+
    	             fPad[ijk+0*ri-1]*gg[687]+
    	             fPad[ijk+0*ri+0]*gg[688]+
    	             fPad[ijk+0*ri+1]*gg[689]+
    	             fPad[ijk+0*ri+2]*gg[690]+
    	             fPad[ijk+0*ri+3]*gg[691]+
    	             fPad[ijk+0*ri+4]*gg[692]+
    	             fPad[ijk+1*ri-4]*gg[693]+
    	             fPad[ijk+1*ri-3]*gg[694]+
    	             fPad[ijk+1*ri-2]*gg[695]+
    	             fPad[ijk+1*ri-1]*gg[696]+
    	             fPad[ijk+1*ri+0]*gg[697]+
    	             fPad[ijk+1*ri+1]*gg[698]+
    	             fPad[ijk+1*ri+2]*gg[699]+
    	             fPad[ijk+1*ri+3]*gg[700]+
    	             fPad[ijk+1*ri+4]*gg[701]+
    	             fPad[ijk+2*ri-4]*gg[702]+
    	             fPad[ijk+2*ri-3]*gg[703]+
    	             fPad[ijk+2*ri-2]*gg[704]+
    	             fPad[ijk+2*ri-1]*gg[705]+
    	             fPad[ijk+2*ri+0]*gg[706]+
    	             fPad[ijk+2*ri+1]*gg[707]+
    	             fPad[ijk+2*ri+2]*gg[708]+
    	             fPad[ijk+2*ri+3]*gg[709]+
    	             fPad[ijk+2*ri+4]*gg[710]+
    	             fPad[ijk+3*ri-4]*gg[711]+
    	             fPad[ijk+3*ri-3]*gg[712]+
    	             fPad[ijk+3*ri-2]*gg[713]+
    	             fPad[ijk+3*ri-1]*gg[714]+
    	             fPad[ijk+3*ri+0]*gg[715]+
    	             fPad[ijk+3*ri+1]*gg[716]+
    	             fPad[ijk+3*ri+2]*gg[717]+
    	             fPad[ijk+3*ri+3]*gg[718]+
    	             fPad[ijk+3*ri+4]*gg[719]+
    	             fPad[ijk+4*ri-4]*gg[720]+
    	             fPad[ijk+4*ri-3]*gg[721]+
    	             fPad[ijk+4*ri-2]*gg[722]+
    	             fPad[ijk+4*ri-1]*gg[723]+
    	             fPad[ijk+4*ri+0]*gg[724]+
    	             fPad[ijk+4*ri+1]*gg[725]+
    	             fPad[ijk+4*ri+2]*gg[726]+
    	             fPad[ijk+4*ri+3]*gg[727]+
    	             fPad[ijk+4*ri+4]*gg[728];
     		           }
                }
            }
        return Boundaries.finishBoundaries3d(fPad, gg, r, gi, gj, gk, hgi, hgie, hgj, hgje, hgk, hgke, ri, rj, rk);
    };
    public static Complex[][][] convolve_9_9_9(Complex[][][] f, Complex[][][] g) {
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
        Complex[] gg = ArrayMath.vectorise(g);
        Complex[] fPad = ArrayMath.vectorise(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje, hgk, hgke));
        Complex[][][] rr = JVCLUtils.zeroPadBoundaries(new Complex[fi][fj][fk], hgi, hgie, hgj, hgje, hgk, hgke);
        final int ri = rr.length;
        final int rj = rr[0].length;
        final int rk = rr[0][0].length;
        Complex[] r = ArrayMath.vectorise(rr);
        int ijk;
        for (int i = hgi; i < fi - 1 - hgie; i++) {
            for (int j = hgj; j < fj - 1 - hgje; j++) {
            	for (int k = hgk; k < fk - 1 - hgke; k++) {
               	ijk = k*ri*rj + j*ri + i;
         	        r[ijk]
       	          .add(fPad[ijk-4*ri-4].multiply(gg[0]))
       	          .add(fPad[ijk-4*ri-3].multiply(gg[1]))
       	          .add(fPad[ijk-4*ri-2].multiply(gg[2]))
       	          .add(fPad[ijk-4*ri-1].multiply(gg[3]))
       	          .add(fPad[ijk-4*ri+0].multiply(gg[4]))
       	          .add(fPad[ijk-4*ri+1].multiply(gg[5]))
       	          .add(fPad[ijk-4*ri+2].multiply(gg[6]))
       	          .add(fPad[ijk-4*ri+3].multiply(gg[7]))
       	          .add(fPad[ijk-4*ri+4].multiply(gg[8]))
       	          .add(fPad[ijk-3*ri-4].multiply(gg[9]))
       	          .add(fPad[ijk-3*ri-3].multiply(gg[10]))
       	          .add(fPad[ijk-3*ri-2].multiply(gg[11]))
       	          .add(fPad[ijk-3*ri-1].multiply(gg[12]))
       	          .add(fPad[ijk-3*ri+0].multiply(gg[13]))
       	          .add(fPad[ijk-3*ri+1].multiply(gg[14]))
       	          .add(fPad[ijk-3*ri+2].multiply(gg[15]))
       	          .add(fPad[ijk-3*ri+3].multiply(gg[16]))
       	          .add(fPad[ijk-3*ri+4].multiply(gg[17]))
       	          .add(fPad[ijk-2*ri-4].multiply(gg[18]))
       	          .add(fPad[ijk-2*ri-3].multiply(gg[19]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[20]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[21]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[22]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[23]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[24]))
       	          .add(fPad[ijk-2*ri+3].multiply(gg[25]))
       	          .add(fPad[ijk-2*ri+4].multiply(gg[26]))
       	          .add(fPad[ijk-1*ri-4].multiply(gg[27]))
       	          .add(fPad[ijk-1*ri-3].multiply(gg[28]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[29]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[30]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[31]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[32]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[33]))
       	          .add(fPad[ijk-1*ri+3].multiply(gg[34]))
       	          .add(fPad[ijk-1*ri+4].multiply(gg[35]))
       	          .add(fPad[ijk+0*ri-4].multiply(gg[36]))
       	          .add(fPad[ijk+0*ri-3].multiply(gg[37]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[38]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[39]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[40]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[41]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[42]))
       	          .add(fPad[ijk+0*ri+3].multiply(gg[43]))
       	          .add(fPad[ijk+0*ri+4].multiply(gg[44]))
       	          .add(fPad[ijk+1*ri-4].multiply(gg[45]))
       	          .add(fPad[ijk+1*ri-3].multiply(gg[46]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[47]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[48]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[49]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[50]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[51]))
       	          .add(fPad[ijk+1*ri+3].multiply(gg[52]))
       	          .add(fPad[ijk+1*ri+4].multiply(gg[53]))
       	          .add(fPad[ijk+2*ri-4].multiply(gg[54]))
       	          .add(fPad[ijk+2*ri-3].multiply(gg[55]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[56]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[57]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[58]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[59]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[60]))
       	          .add(fPad[ijk+2*ri+3].multiply(gg[61]))
       	          .add(fPad[ijk+2*ri+4].multiply(gg[62]))
       	          .add(fPad[ijk+3*ri-4].multiply(gg[63]))
       	          .add(fPad[ijk+3*ri-3].multiply(gg[64]))
       	          .add(fPad[ijk+3*ri-2].multiply(gg[65]))
       	          .add(fPad[ijk+3*ri-1].multiply(gg[66]))
       	          .add(fPad[ijk+3*ri+0].multiply(gg[67]))
       	          .add(fPad[ijk+3*ri+1].multiply(gg[68]))
       	          .add(fPad[ijk+3*ri+2].multiply(gg[69]))
       	          .add(fPad[ijk+3*ri+3].multiply(gg[70]))
       	          .add(fPad[ijk+3*ri+4].multiply(gg[71]))
       	          .add(fPad[ijk+4*ri-4].multiply(gg[72]))
       	          .add(fPad[ijk+4*ri-3].multiply(gg[73]))
       	          .add(fPad[ijk+4*ri-2].multiply(gg[74]))
       	          .add(fPad[ijk+4*ri-1].multiply(gg[75]))
       	          .add(fPad[ijk+4*ri+0].multiply(gg[76]))
       	          .add(fPad[ijk+4*ri+1].multiply(gg[77]))
       	          .add(fPad[ijk+4*ri+2].multiply(gg[78]))
       	          .add(fPad[ijk+4*ri+3].multiply(gg[79]))
       	          .add(fPad[ijk+4*ri+4].multiply(gg[80]))
       	          .add(fPad[ijk-4*ri-4].multiply(gg[81]))
       	          .add(fPad[ijk-4*ri-3].multiply(gg[82]))
       	          .add(fPad[ijk-4*ri-2].multiply(gg[83]))
       	          .add(fPad[ijk-4*ri-1].multiply(gg[84]))
       	          .add(fPad[ijk-4*ri+0].multiply(gg[85]))
       	          .add(fPad[ijk-4*ri+1].multiply(gg[86]))
       	          .add(fPad[ijk-4*ri+2].multiply(gg[87]))
       	          .add(fPad[ijk-4*ri+3].multiply(gg[88]))
       	          .add(fPad[ijk-4*ri+4].multiply(gg[89]))
       	          .add(fPad[ijk-3*ri-4].multiply(gg[90]))
       	          .add(fPad[ijk-3*ri-3].multiply(gg[91]))
       	          .add(fPad[ijk-3*ri-2].multiply(gg[92]))
       	          .add(fPad[ijk-3*ri-1].multiply(gg[93]))
       	          .add(fPad[ijk-3*ri+0].multiply(gg[94]))
       	          .add(fPad[ijk-3*ri+1].multiply(gg[95]))
       	          .add(fPad[ijk-3*ri+2].multiply(gg[96]))
       	          .add(fPad[ijk-3*ri+3].multiply(gg[97]))
       	          .add(fPad[ijk-3*ri+4].multiply(gg[98]))
       	          .add(fPad[ijk-2*ri-4].multiply(gg[99]))
       	          .add(fPad[ijk-2*ri-3].multiply(gg[100]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[101]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[102]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[103]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[104]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[105]))
       	          .add(fPad[ijk-2*ri+3].multiply(gg[106]))
       	          .add(fPad[ijk-2*ri+4].multiply(gg[107]))
       	          .add(fPad[ijk-1*ri-4].multiply(gg[108]))
       	          .add(fPad[ijk-1*ri-3].multiply(gg[109]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[110]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[111]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[112]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[113]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[114]))
       	          .add(fPad[ijk-1*ri+3].multiply(gg[115]))
       	          .add(fPad[ijk-1*ri+4].multiply(gg[116]))
       	          .add(fPad[ijk+0*ri-4].multiply(gg[117]))
       	          .add(fPad[ijk+0*ri-3].multiply(gg[118]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[119]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[120]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[121]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[122]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[123]))
       	          .add(fPad[ijk+0*ri+3].multiply(gg[124]))
       	          .add(fPad[ijk+0*ri+4].multiply(gg[125]))
       	          .add(fPad[ijk+1*ri-4].multiply(gg[126]))
       	          .add(fPad[ijk+1*ri-3].multiply(gg[127]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[128]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[129]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[130]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[131]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[132]))
       	          .add(fPad[ijk+1*ri+3].multiply(gg[133]))
       	          .add(fPad[ijk+1*ri+4].multiply(gg[134]))
       	          .add(fPad[ijk+2*ri-4].multiply(gg[135]))
       	          .add(fPad[ijk+2*ri-3].multiply(gg[136]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[137]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[138]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[139]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[140]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[141]))
       	          .add(fPad[ijk+2*ri+3].multiply(gg[142]))
       	          .add(fPad[ijk+2*ri+4].multiply(gg[143]))
       	          .add(fPad[ijk+3*ri-4].multiply(gg[144]))
       	          .add(fPad[ijk+3*ri-3].multiply(gg[145]))
       	          .add(fPad[ijk+3*ri-2].multiply(gg[146]))
       	          .add(fPad[ijk+3*ri-1].multiply(gg[147]))
       	          .add(fPad[ijk+3*ri+0].multiply(gg[148]))
       	          .add(fPad[ijk+3*ri+1].multiply(gg[149]))
       	          .add(fPad[ijk+3*ri+2].multiply(gg[150]))
       	          .add(fPad[ijk+3*ri+3].multiply(gg[151]))
       	          .add(fPad[ijk+3*ri+4].multiply(gg[152]))
       	          .add(fPad[ijk+4*ri-4].multiply(gg[153]))
       	          .add(fPad[ijk+4*ri-3].multiply(gg[154]))
       	          .add(fPad[ijk+4*ri-2].multiply(gg[155]))
       	          .add(fPad[ijk+4*ri-1].multiply(gg[156]))
       	          .add(fPad[ijk+4*ri+0].multiply(gg[157]))
       	          .add(fPad[ijk+4*ri+1].multiply(gg[158]))
       	          .add(fPad[ijk+4*ri+2].multiply(gg[159]))
       	          .add(fPad[ijk+4*ri+3].multiply(gg[160]))
       	          .add(fPad[ijk+4*ri+4].multiply(gg[161]))
       	          .add(fPad[ijk-4*ri-4].multiply(gg[162]))
       	          .add(fPad[ijk-4*ri-3].multiply(gg[163]))
       	          .add(fPad[ijk-4*ri-2].multiply(gg[164]))
       	          .add(fPad[ijk-4*ri-1].multiply(gg[165]))
       	          .add(fPad[ijk-4*ri+0].multiply(gg[166]))
       	          .add(fPad[ijk-4*ri+1].multiply(gg[167]))
       	          .add(fPad[ijk-4*ri+2].multiply(gg[168]))
       	          .add(fPad[ijk-4*ri+3].multiply(gg[169]))
       	          .add(fPad[ijk-4*ri+4].multiply(gg[170]))
       	          .add(fPad[ijk-3*ri-4].multiply(gg[171]))
       	          .add(fPad[ijk-3*ri-3].multiply(gg[172]))
       	          .add(fPad[ijk-3*ri-2].multiply(gg[173]))
       	          .add(fPad[ijk-3*ri-1].multiply(gg[174]))
       	          .add(fPad[ijk-3*ri+0].multiply(gg[175]))
       	          .add(fPad[ijk-3*ri+1].multiply(gg[176]))
       	          .add(fPad[ijk-3*ri+2].multiply(gg[177]))
       	          .add(fPad[ijk-3*ri+3].multiply(gg[178]))
       	          .add(fPad[ijk-3*ri+4].multiply(gg[179]))
       	          .add(fPad[ijk-2*ri-4].multiply(gg[180]))
       	          .add(fPad[ijk-2*ri-3].multiply(gg[181]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[182]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[183]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[184]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[185]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[186]))
       	          .add(fPad[ijk-2*ri+3].multiply(gg[187]))
       	          .add(fPad[ijk-2*ri+4].multiply(gg[188]))
       	          .add(fPad[ijk-1*ri-4].multiply(gg[189]))
       	          .add(fPad[ijk-1*ri-3].multiply(gg[190]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[191]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[192]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[193]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[194]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[195]))
       	          .add(fPad[ijk-1*ri+3].multiply(gg[196]))
       	          .add(fPad[ijk-1*ri+4].multiply(gg[197]))
       	          .add(fPad[ijk+0*ri-4].multiply(gg[198]))
       	          .add(fPad[ijk+0*ri-3].multiply(gg[199]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[200]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[201]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[202]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[203]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[204]))
       	          .add(fPad[ijk+0*ri+3].multiply(gg[205]))
       	          .add(fPad[ijk+0*ri+4].multiply(gg[206]))
       	          .add(fPad[ijk+1*ri-4].multiply(gg[207]))
       	          .add(fPad[ijk+1*ri-3].multiply(gg[208]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[209]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[210]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[211]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[212]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[213]))
       	          .add(fPad[ijk+1*ri+3].multiply(gg[214]))
       	          .add(fPad[ijk+1*ri+4].multiply(gg[215]))
       	          .add(fPad[ijk+2*ri-4].multiply(gg[216]))
       	          .add(fPad[ijk+2*ri-3].multiply(gg[217]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[218]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[219]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[220]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[221]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[222]))
       	          .add(fPad[ijk+2*ri+3].multiply(gg[223]))
       	          .add(fPad[ijk+2*ri+4].multiply(gg[224]))
       	          .add(fPad[ijk+3*ri-4].multiply(gg[225]))
       	          .add(fPad[ijk+3*ri-3].multiply(gg[226]))
       	          .add(fPad[ijk+3*ri-2].multiply(gg[227]))
       	          .add(fPad[ijk+3*ri-1].multiply(gg[228]))
       	          .add(fPad[ijk+3*ri+0].multiply(gg[229]))
       	          .add(fPad[ijk+3*ri+1].multiply(gg[230]))
       	          .add(fPad[ijk+3*ri+2].multiply(gg[231]))
       	          .add(fPad[ijk+3*ri+3].multiply(gg[232]))
       	          .add(fPad[ijk+3*ri+4].multiply(gg[233]))
       	          .add(fPad[ijk+4*ri-4].multiply(gg[234]))
       	          .add(fPad[ijk+4*ri-3].multiply(gg[235]))
       	          .add(fPad[ijk+4*ri-2].multiply(gg[236]))
       	          .add(fPad[ijk+4*ri-1].multiply(gg[237]))
       	          .add(fPad[ijk+4*ri+0].multiply(gg[238]))
       	          .add(fPad[ijk+4*ri+1].multiply(gg[239]))
       	          .add(fPad[ijk+4*ri+2].multiply(gg[240]))
       	          .add(fPad[ijk+4*ri+3].multiply(gg[241]))
       	          .add(fPad[ijk+4*ri+4].multiply(gg[242]))
       	          .add(fPad[ijk-4*ri-4].multiply(gg[243]))
       	          .add(fPad[ijk-4*ri-3].multiply(gg[244]))
       	          .add(fPad[ijk-4*ri-2].multiply(gg[245]))
       	          .add(fPad[ijk-4*ri-1].multiply(gg[246]))
       	          .add(fPad[ijk-4*ri+0].multiply(gg[247]))
       	          .add(fPad[ijk-4*ri+1].multiply(gg[248]))
       	          .add(fPad[ijk-4*ri+2].multiply(gg[249]))
       	          .add(fPad[ijk-4*ri+3].multiply(gg[250]))
       	          .add(fPad[ijk-4*ri+4].multiply(gg[251]))
       	          .add(fPad[ijk-3*ri-4].multiply(gg[252]))
       	          .add(fPad[ijk-3*ri-3].multiply(gg[253]))
       	          .add(fPad[ijk-3*ri-2].multiply(gg[254]))
       	          .add(fPad[ijk-3*ri-1].multiply(gg[255]))
       	          .add(fPad[ijk-3*ri+0].multiply(gg[256]))
       	          .add(fPad[ijk-3*ri+1].multiply(gg[257]))
       	          .add(fPad[ijk-3*ri+2].multiply(gg[258]))
       	          .add(fPad[ijk-3*ri+3].multiply(gg[259]))
       	          .add(fPad[ijk-3*ri+4].multiply(gg[260]))
       	          .add(fPad[ijk-2*ri-4].multiply(gg[261]))
       	          .add(fPad[ijk-2*ri-3].multiply(gg[262]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[263]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[264]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[265]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[266]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[267]))
       	          .add(fPad[ijk-2*ri+3].multiply(gg[268]))
       	          .add(fPad[ijk-2*ri+4].multiply(gg[269]))
       	          .add(fPad[ijk-1*ri-4].multiply(gg[270]))
       	          .add(fPad[ijk-1*ri-3].multiply(gg[271]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[272]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[273]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[274]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[275]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[276]))
       	          .add(fPad[ijk-1*ri+3].multiply(gg[277]))
       	          .add(fPad[ijk-1*ri+4].multiply(gg[278]))
       	          .add(fPad[ijk+0*ri-4].multiply(gg[279]))
       	          .add(fPad[ijk+0*ri-3].multiply(gg[280]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[281]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[282]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[283]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[284]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[285]))
       	          .add(fPad[ijk+0*ri+3].multiply(gg[286]))
       	          .add(fPad[ijk+0*ri+4].multiply(gg[287]))
       	          .add(fPad[ijk+1*ri-4].multiply(gg[288]))
       	          .add(fPad[ijk+1*ri-3].multiply(gg[289]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[290]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[291]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[292]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[293]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[294]))
       	          .add(fPad[ijk+1*ri+3].multiply(gg[295]))
       	          .add(fPad[ijk+1*ri+4].multiply(gg[296]))
       	          .add(fPad[ijk+2*ri-4].multiply(gg[297]))
       	          .add(fPad[ijk+2*ri-3].multiply(gg[298]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[299]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[300]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[301]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[302]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[303]))
       	          .add(fPad[ijk+2*ri+3].multiply(gg[304]))
       	          .add(fPad[ijk+2*ri+4].multiply(gg[305]))
       	          .add(fPad[ijk+3*ri-4].multiply(gg[306]))
       	          .add(fPad[ijk+3*ri-3].multiply(gg[307]))
       	          .add(fPad[ijk+3*ri-2].multiply(gg[308]))
       	          .add(fPad[ijk+3*ri-1].multiply(gg[309]))
       	          .add(fPad[ijk+3*ri+0].multiply(gg[310]))
       	          .add(fPad[ijk+3*ri+1].multiply(gg[311]))
       	          .add(fPad[ijk+3*ri+2].multiply(gg[312]))
       	          .add(fPad[ijk+3*ri+3].multiply(gg[313]))
       	          .add(fPad[ijk+3*ri+4].multiply(gg[314]))
       	          .add(fPad[ijk+4*ri-4].multiply(gg[315]))
       	          .add(fPad[ijk+4*ri-3].multiply(gg[316]))
       	          .add(fPad[ijk+4*ri-2].multiply(gg[317]))
       	          .add(fPad[ijk+4*ri-1].multiply(gg[318]))
       	          .add(fPad[ijk+4*ri+0].multiply(gg[319]))
       	          .add(fPad[ijk+4*ri+1].multiply(gg[320]))
       	          .add(fPad[ijk+4*ri+2].multiply(gg[321]))
       	          .add(fPad[ijk+4*ri+3].multiply(gg[322]))
       	          .add(fPad[ijk+4*ri+4].multiply(gg[323]))
       	          .add(fPad[ijk-4*ri-4].multiply(gg[324]))
       	          .add(fPad[ijk-4*ri-3].multiply(gg[325]))
       	          .add(fPad[ijk-4*ri-2].multiply(gg[326]))
       	          .add(fPad[ijk-4*ri-1].multiply(gg[327]))
       	          .add(fPad[ijk-4*ri+0].multiply(gg[328]))
       	          .add(fPad[ijk-4*ri+1].multiply(gg[329]))
       	          .add(fPad[ijk-4*ri+2].multiply(gg[330]))
       	          .add(fPad[ijk-4*ri+3].multiply(gg[331]))
       	          .add(fPad[ijk-4*ri+4].multiply(gg[332]))
       	          .add(fPad[ijk-3*ri-4].multiply(gg[333]))
       	          .add(fPad[ijk-3*ri-3].multiply(gg[334]))
       	          .add(fPad[ijk-3*ri-2].multiply(gg[335]))
       	          .add(fPad[ijk-3*ri-1].multiply(gg[336]))
       	          .add(fPad[ijk-3*ri+0].multiply(gg[337]))
       	          .add(fPad[ijk-3*ri+1].multiply(gg[338]))
       	          .add(fPad[ijk-3*ri+2].multiply(gg[339]))
       	          .add(fPad[ijk-3*ri+3].multiply(gg[340]))
       	          .add(fPad[ijk-3*ri+4].multiply(gg[341]))
       	          .add(fPad[ijk-2*ri-4].multiply(gg[342]))
       	          .add(fPad[ijk-2*ri-3].multiply(gg[343]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[344]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[345]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[346]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[347]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[348]))
       	          .add(fPad[ijk-2*ri+3].multiply(gg[349]))
       	          .add(fPad[ijk-2*ri+4].multiply(gg[350]))
       	          .add(fPad[ijk-1*ri-4].multiply(gg[351]))
       	          .add(fPad[ijk-1*ri-3].multiply(gg[352]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[353]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[354]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[355]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[356]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[357]))
       	          .add(fPad[ijk-1*ri+3].multiply(gg[358]))
       	          .add(fPad[ijk-1*ri+4].multiply(gg[359]))
       	          .add(fPad[ijk+0*ri-4].multiply(gg[360]))
       	          .add(fPad[ijk+0*ri-3].multiply(gg[361]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[362]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[363]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[364]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[365]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[366]))
       	          .add(fPad[ijk+0*ri+3].multiply(gg[367]))
       	          .add(fPad[ijk+0*ri+4].multiply(gg[368]))
       	          .add(fPad[ijk+1*ri-4].multiply(gg[369]))
       	          .add(fPad[ijk+1*ri-3].multiply(gg[370]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[371]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[372]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[373]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[374]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[375]))
       	          .add(fPad[ijk+1*ri+3].multiply(gg[376]))
       	          .add(fPad[ijk+1*ri+4].multiply(gg[377]))
       	          .add(fPad[ijk+2*ri-4].multiply(gg[378]))
       	          .add(fPad[ijk+2*ri-3].multiply(gg[379]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[380]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[381]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[382]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[383]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[384]))
       	          .add(fPad[ijk+2*ri+3].multiply(gg[385]))
       	          .add(fPad[ijk+2*ri+4].multiply(gg[386]))
       	          .add(fPad[ijk+3*ri-4].multiply(gg[387]))
       	          .add(fPad[ijk+3*ri-3].multiply(gg[388]))
       	          .add(fPad[ijk+3*ri-2].multiply(gg[389]))
       	          .add(fPad[ijk+3*ri-1].multiply(gg[390]))
       	          .add(fPad[ijk+3*ri+0].multiply(gg[391]))
       	          .add(fPad[ijk+3*ri+1].multiply(gg[392]))
       	          .add(fPad[ijk+3*ri+2].multiply(gg[393]))
       	          .add(fPad[ijk+3*ri+3].multiply(gg[394]))
       	          .add(fPad[ijk+3*ri+4].multiply(gg[395]))
       	          .add(fPad[ijk+4*ri-4].multiply(gg[396]))
       	          .add(fPad[ijk+4*ri-3].multiply(gg[397]))
       	          .add(fPad[ijk+4*ri-2].multiply(gg[398]))
       	          .add(fPad[ijk+4*ri-1].multiply(gg[399]))
       	          .add(fPad[ijk+4*ri+0].multiply(gg[400]))
       	          .add(fPad[ijk+4*ri+1].multiply(gg[401]))
       	          .add(fPad[ijk+4*ri+2].multiply(gg[402]))
       	          .add(fPad[ijk+4*ri+3].multiply(gg[403]))
       	          .add(fPad[ijk+4*ri+4].multiply(gg[404]))
       	          .add(fPad[ijk-4*ri-4].multiply(gg[405]))
       	          .add(fPad[ijk-4*ri-3].multiply(gg[406]))
       	          .add(fPad[ijk-4*ri-2].multiply(gg[407]))
       	          .add(fPad[ijk-4*ri-1].multiply(gg[408]))
       	          .add(fPad[ijk-4*ri+0].multiply(gg[409]))
       	          .add(fPad[ijk-4*ri+1].multiply(gg[410]))
       	          .add(fPad[ijk-4*ri+2].multiply(gg[411]))
       	          .add(fPad[ijk-4*ri+3].multiply(gg[412]))
       	          .add(fPad[ijk-4*ri+4].multiply(gg[413]))
       	          .add(fPad[ijk-3*ri-4].multiply(gg[414]))
       	          .add(fPad[ijk-3*ri-3].multiply(gg[415]))
       	          .add(fPad[ijk-3*ri-2].multiply(gg[416]))
       	          .add(fPad[ijk-3*ri-1].multiply(gg[417]))
       	          .add(fPad[ijk-3*ri+0].multiply(gg[418]))
       	          .add(fPad[ijk-3*ri+1].multiply(gg[419]))
       	          .add(fPad[ijk-3*ri+2].multiply(gg[420]))
       	          .add(fPad[ijk-3*ri+3].multiply(gg[421]))
       	          .add(fPad[ijk-3*ri+4].multiply(gg[422]))
       	          .add(fPad[ijk-2*ri-4].multiply(gg[423]))
       	          .add(fPad[ijk-2*ri-3].multiply(gg[424]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[425]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[426]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[427]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[428]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[429]))
       	          .add(fPad[ijk-2*ri+3].multiply(gg[430]))
       	          .add(fPad[ijk-2*ri+4].multiply(gg[431]))
       	          .add(fPad[ijk-1*ri-4].multiply(gg[432]))
       	          .add(fPad[ijk-1*ri-3].multiply(gg[433]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[434]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[435]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[436]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[437]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[438]))
       	          .add(fPad[ijk-1*ri+3].multiply(gg[439]))
       	          .add(fPad[ijk-1*ri+4].multiply(gg[440]))
       	          .add(fPad[ijk+0*ri-4].multiply(gg[441]))
       	          .add(fPad[ijk+0*ri-3].multiply(gg[442]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[443]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[444]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[445]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[446]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[447]))
       	          .add(fPad[ijk+0*ri+3].multiply(gg[448]))
       	          .add(fPad[ijk+0*ri+4].multiply(gg[449]))
       	          .add(fPad[ijk+1*ri-4].multiply(gg[450]))
       	          .add(fPad[ijk+1*ri-3].multiply(gg[451]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[452]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[453]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[454]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[455]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[456]))
       	          .add(fPad[ijk+1*ri+3].multiply(gg[457]))
       	          .add(fPad[ijk+1*ri+4].multiply(gg[458]))
       	          .add(fPad[ijk+2*ri-4].multiply(gg[459]))
       	          .add(fPad[ijk+2*ri-3].multiply(gg[460]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[461]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[462]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[463]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[464]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[465]))
       	          .add(fPad[ijk+2*ri+3].multiply(gg[466]))
       	          .add(fPad[ijk+2*ri+4].multiply(gg[467]))
       	          .add(fPad[ijk+3*ri-4].multiply(gg[468]))
       	          .add(fPad[ijk+3*ri-3].multiply(gg[469]))
       	          .add(fPad[ijk+3*ri-2].multiply(gg[470]))
       	          .add(fPad[ijk+3*ri-1].multiply(gg[471]))
       	          .add(fPad[ijk+3*ri+0].multiply(gg[472]))
       	          .add(fPad[ijk+3*ri+1].multiply(gg[473]))
       	          .add(fPad[ijk+3*ri+2].multiply(gg[474]))
       	          .add(fPad[ijk+3*ri+3].multiply(gg[475]))
       	          .add(fPad[ijk+3*ri+4].multiply(gg[476]))
       	          .add(fPad[ijk+4*ri-4].multiply(gg[477]))
       	          .add(fPad[ijk+4*ri-3].multiply(gg[478]))
       	          .add(fPad[ijk+4*ri-2].multiply(gg[479]))
       	          .add(fPad[ijk+4*ri-1].multiply(gg[480]))
       	          .add(fPad[ijk+4*ri+0].multiply(gg[481]))
       	          .add(fPad[ijk+4*ri+1].multiply(gg[482]))
       	          .add(fPad[ijk+4*ri+2].multiply(gg[483]))
       	          .add(fPad[ijk+4*ri+3].multiply(gg[484]))
       	          .add(fPad[ijk+4*ri+4].multiply(gg[485]))
       	          .add(fPad[ijk-4*ri-4].multiply(gg[486]))
       	          .add(fPad[ijk-4*ri-3].multiply(gg[487]))
       	          .add(fPad[ijk-4*ri-2].multiply(gg[488]))
       	          .add(fPad[ijk-4*ri-1].multiply(gg[489]))
       	          .add(fPad[ijk-4*ri+0].multiply(gg[490]))
       	          .add(fPad[ijk-4*ri+1].multiply(gg[491]))
       	          .add(fPad[ijk-4*ri+2].multiply(gg[492]))
       	          .add(fPad[ijk-4*ri+3].multiply(gg[493]))
       	          .add(fPad[ijk-4*ri+4].multiply(gg[494]))
       	          .add(fPad[ijk-3*ri-4].multiply(gg[495]))
       	          .add(fPad[ijk-3*ri-3].multiply(gg[496]))
       	          .add(fPad[ijk-3*ri-2].multiply(gg[497]))
       	          .add(fPad[ijk-3*ri-1].multiply(gg[498]))
       	          .add(fPad[ijk-3*ri+0].multiply(gg[499]))
       	          .add(fPad[ijk-3*ri+1].multiply(gg[500]))
       	          .add(fPad[ijk-3*ri+2].multiply(gg[501]))
       	          .add(fPad[ijk-3*ri+3].multiply(gg[502]))
       	          .add(fPad[ijk-3*ri+4].multiply(gg[503]))
       	          .add(fPad[ijk-2*ri-4].multiply(gg[504]))
       	          .add(fPad[ijk-2*ri-3].multiply(gg[505]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[506]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[507]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[508]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[509]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[510]))
       	          .add(fPad[ijk-2*ri+3].multiply(gg[511]))
       	          .add(fPad[ijk-2*ri+4].multiply(gg[512]))
       	          .add(fPad[ijk-1*ri-4].multiply(gg[513]))
       	          .add(fPad[ijk-1*ri-3].multiply(gg[514]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[515]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[516]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[517]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[518]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[519]))
       	          .add(fPad[ijk-1*ri+3].multiply(gg[520]))
       	          .add(fPad[ijk-1*ri+4].multiply(gg[521]))
       	          .add(fPad[ijk+0*ri-4].multiply(gg[522]))
       	          .add(fPad[ijk+0*ri-3].multiply(gg[523]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[524]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[525]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[526]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[527]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[528]))
       	          .add(fPad[ijk+0*ri+3].multiply(gg[529]))
       	          .add(fPad[ijk+0*ri+4].multiply(gg[530]))
       	          .add(fPad[ijk+1*ri-4].multiply(gg[531]))
       	          .add(fPad[ijk+1*ri-3].multiply(gg[532]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[533]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[534]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[535]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[536]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[537]))
       	          .add(fPad[ijk+1*ri+3].multiply(gg[538]))
       	          .add(fPad[ijk+1*ri+4].multiply(gg[539]))
       	          .add(fPad[ijk+2*ri-4].multiply(gg[540]))
       	          .add(fPad[ijk+2*ri-3].multiply(gg[541]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[542]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[543]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[544]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[545]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[546]))
       	          .add(fPad[ijk+2*ri+3].multiply(gg[547]))
       	          .add(fPad[ijk+2*ri+4].multiply(gg[548]))
       	          .add(fPad[ijk+3*ri-4].multiply(gg[549]))
       	          .add(fPad[ijk+3*ri-3].multiply(gg[550]))
       	          .add(fPad[ijk+3*ri-2].multiply(gg[551]))
       	          .add(fPad[ijk+3*ri-1].multiply(gg[552]))
       	          .add(fPad[ijk+3*ri+0].multiply(gg[553]))
       	          .add(fPad[ijk+3*ri+1].multiply(gg[554]))
       	          .add(fPad[ijk+3*ri+2].multiply(gg[555]))
       	          .add(fPad[ijk+3*ri+3].multiply(gg[556]))
       	          .add(fPad[ijk+3*ri+4].multiply(gg[557]))
       	          .add(fPad[ijk+4*ri-4].multiply(gg[558]))
       	          .add(fPad[ijk+4*ri-3].multiply(gg[559]))
       	          .add(fPad[ijk+4*ri-2].multiply(gg[560]))
       	          .add(fPad[ijk+4*ri-1].multiply(gg[561]))
       	          .add(fPad[ijk+4*ri+0].multiply(gg[562]))
       	          .add(fPad[ijk+4*ri+1].multiply(gg[563]))
       	          .add(fPad[ijk+4*ri+2].multiply(gg[564]))
       	          .add(fPad[ijk+4*ri+3].multiply(gg[565]))
       	          .add(fPad[ijk+4*ri+4].multiply(gg[566]))
       	          .add(fPad[ijk-4*ri-4].multiply(gg[567]))
       	          .add(fPad[ijk-4*ri-3].multiply(gg[568]))
       	          .add(fPad[ijk-4*ri-2].multiply(gg[569]))
       	          .add(fPad[ijk-4*ri-1].multiply(gg[570]))
       	          .add(fPad[ijk-4*ri+0].multiply(gg[571]))
       	          .add(fPad[ijk-4*ri+1].multiply(gg[572]))
       	          .add(fPad[ijk-4*ri+2].multiply(gg[573]))
       	          .add(fPad[ijk-4*ri+3].multiply(gg[574]))
       	          .add(fPad[ijk-4*ri+4].multiply(gg[575]))
       	          .add(fPad[ijk-3*ri-4].multiply(gg[576]))
       	          .add(fPad[ijk-3*ri-3].multiply(gg[577]))
       	          .add(fPad[ijk-3*ri-2].multiply(gg[578]))
       	          .add(fPad[ijk-3*ri-1].multiply(gg[579]))
       	          .add(fPad[ijk-3*ri+0].multiply(gg[580]))
       	          .add(fPad[ijk-3*ri+1].multiply(gg[581]))
       	          .add(fPad[ijk-3*ri+2].multiply(gg[582]))
       	          .add(fPad[ijk-3*ri+3].multiply(gg[583]))
       	          .add(fPad[ijk-3*ri+4].multiply(gg[584]))
       	          .add(fPad[ijk-2*ri-4].multiply(gg[585]))
       	          .add(fPad[ijk-2*ri-3].multiply(gg[586]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[587]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[588]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[589]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[590]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[591]))
       	          .add(fPad[ijk-2*ri+3].multiply(gg[592]))
       	          .add(fPad[ijk-2*ri+4].multiply(gg[593]))
       	          .add(fPad[ijk-1*ri-4].multiply(gg[594]))
       	          .add(fPad[ijk-1*ri-3].multiply(gg[595]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[596]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[597]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[598]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[599]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[600]))
       	          .add(fPad[ijk-1*ri+3].multiply(gg[601]))
       	          .add(fPad[ijk-1*ri+4].multiply(gg[602]))
       	          .add(fPad[ijk+0*ri-4].multiply(gg[603]))
       	          .add(fPad[ijk+0*ri-3].multiply(gg[604]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[605]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[606]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[607]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[608]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[609]))
       	          .add(fPad[ijk+0*ri+3].multiply(gg[610]))
       	          .add(fPad[ijk+0*ri+4].multiply(gg[611]))
       	          .add(fPad[ijk+1*ri-4].multiply(gg[612]))
       	          .add(fPad[ijk+1*ri-3].multiply(gg[613]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[614]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[615]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[616]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[617]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[618]))
       	          .add(fPad[ijk+1*ri+3].multiply(gg[619]))
       	          .add(fPad[ijk+1*ri+4].multiply(gg[620]))
       	          .add(fPad[ijk+2*ri-4].multiply(gg[621]))
       	          .add(fPad[ijk+2*ri-3].multiply(gg[622]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[623]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[624]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[625]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[626]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[627]))
       	          .add(fPad[ijk+2*ri+3].multiply(gg[628]))
       	          .add(fPad[ijk+2*ri+4].multiply(gg[629]))
       	          .add(fPad[ijk+3*ri-4].multiply(gg[630]))
       	          .add(fPad[ijk+3*ri-3].multiply(gg[631]))
       	          .add(fPad[ijk+3*ri-2].multiply(gg[632]))
       	          .add(fPad[ijk+3*ri-1].multiply(gg[633]))
       	          .add(fPad[ijk+3*ri+0].multiply(gg[634]))
       	          .add(fPad[ijk+3*ri+1].multiply(gg[635]))
       	          .add(fPad[ijk+3*ri+2].multiply(gg[636]))
       	          .add(fPad[ijk+3*ri+3].multiply(gg[637]))
       	          .add(fPad[ijk+3*ri+4].multiply(gg[638]))
       	          .add(fPad[ijk+4*ri-4].multiply(gg[639]))
       	          .add(fPad[ijk+4*ri-3].multiply(gg[640]))
       	          .add(fPad[ijk+4*ri-2].multiply(gg[641]))
       	          .add(fPad[ijk+4*ri-1].multiply(gg[642]))
       	          .add(fPad[ijk+4*ri+0].multiply(gg[643]))
       	          .add(fPad[ijk+4*ri+1].multiply(gg[644]))
       	          .add(fPad[ijk+4*ri+2].multiply(gg[645]))
       	          .add(fPad[ijk+4*ri+3].multiply(gg[646]))
       	          .add(fPad[ijk+4*ri+4].multiply(gg[647]))
       	          .add(fPad[ijk-4*ri-4].multiply(gg[648]))
       	          .add(fPad[ijk-4*ri-3].multiply(gg[649]))
       	          .add(fPad[ijk-4*ri-2].multiply(gg[650]))
       	          .add(fPad[ijk-4*ri-1].multiply(gg[651]))
       	          .add(fPad[ijk-4*ri+0].multiply(gg[652]))
       	          .add(fPad[ijk-4*ri+1].multiply(gg[653]))
       	          .add(fPad[ijk-4*ri+2].multiply(gg[654]))
       	          .add(fPad[ijk-4*ri+3].multiply(gg[655]))
       	          .add(fPad[ijk-4*ri+4].multiply(gg[656]))
       	          .add(fPad[ijk-3*ri-4].multiply(gg[657]))
       	          .add(fPad[ijk-3*ri-3].multiply(gg[658]))
       	          .add(fPad[ijk-3*ri-2].multiply(gg[659]))
       	          .add(fPad[ijk-3*ri-1].multiply(gg[660]))
       	          .add(fPad[ijk-3*ri+0].multiply(gg[661]))
       	          .add(fPad[ijk-3*ri+1].multiply(gg[662]))
       	          .add(fPad[ijk-3*ri+2].multiply(gg[663]))
       	          .add(fPad[ijk-3*ri+3].multiply(gg[664]))
       	          .add(fPad[ijk-3*ri+4].multiply(gg[665]))
       	          .add(fPad[ijk-2*ri-4].multiply(gg[666]))
       	          .add(fPad[ijk-2*ri-3].multiply(gg[667]))
       	          .add(fPad[ijk-2*ri-2].multiply(gg[668]))
       	          .add(fPad[ijk-2*ri-1].multiply(gg[669]))
       	          .add(fPad[ijk-2*ri+0].multiply(gg[670]))
       	          .add(fPad[ijk-2*ri+1].multiply(gg[671]))
       	          .add(fPad[ijk-2*ri+2].multiply(gg[672]))
       	          .add(fPad[ijk-2*ri+3].multiply(gg[673]))
       	          .add(fPad[ijk-2*ri+4].multiply(gg[674]))
       	          .add(fPad[ijk-1*ri-4].multiply(gg[675]))
       	          .add(fPad[ijk-1*ri-3].multiply(gg[676]))
       	          .add(fPad[ijk-1*ri-2].multiply(gg[677]))
       	          .add(fPad[ijk-1*ri-1].multiply(gg[678]))
       	          .add(fPad[ijk-1*ri+0].multiply(gg[679]))
       	          .add(fPad[ijk-1*ri+1].multiply(gg[680]))
       	          .add(fPad[ijk-1*ri+2].multiply(gg[681]))
       	          .add(fPad[ijk-1*ri+3].multiply(gg[682]))
       	          .add(fPad[ijk-1*ri+4].multiply(gg[683]))
       	          .add(fPad[ijk+0*ri-4].multiply(gg[684]))
       	          .add(fPad[ijk+0*ri-3].multiply(gg[685]))
       	          .add(fPad[ijk+0*ri-2].multiply(gg[686]))
       	          .add(fPad[ijk+0*ri-1].multiply(gg[687]))
       	          .add(fPad[ijk+0*ri+0].multiply(gg[688]))
       	          .add(fPad[ijk+0*ri+1].multiply(gg[689]))
       	          .add(fPad[ijk+0*ri+2].multiply(gg[690]))
       	          .add(fPad[ijk+0*ri+3].multiply(gg[691]))
       	          .add(fPad[ijk+0*ri+4].multiply(gg[692]))
       	          .add(fPad[ijk+1*ri-4].multiply(gg[693]))
       	          .add(fPad[ijk+1*ri-3].multiply(gg[694]))
       	          .add(fPad[ijk+1*ri-2].multiply(gg[695]))
       	          .add(fPad[ijk+1*ri-1].multiply(gg[696]))
       	          .add(fPad[ijk+1*ri+0].multiply(gg[697]))
       	          .add(fPad[ijk+1*ri+1].multiply(gg[698]))
       	          .add(fPad[ijk+1*ri+2].multiply(gg[699]))
       	          .add(fPad[ijk+1*ri+3].multiply(gg[700]))
       	          .add(fPad[ijk+1*ri+4].multiply(gg[701]))
       	          .add(fPad[ijk+2*ri-4].multiply(gg[702]))
       	          .add(fPad[ijk+2*ri-3].multiply(gg[703]))
       	          .add(fPad[ijk+2*ri-2].multiply(gg[704]))
       	          .add(fPad[ijk+2*ri-1].multiply(gg[705]))
       	          .add(fPad[ijk+2*ri+0].multiply(gg[706]))
       	          .add(fPad[ijk+2*ri+1].multiply(gg[707]))
       	          .add(fPad[ijk+2*ri+2].multiply(gg[708]))
       	          .add(fPad[ijk+2*ri+3].multiply(gg[709]))
       	          .add(fPad[ijk+2*ri+4].multiply(gg[710]))
       	          .add(fPad[ijk+3*ri-4].multiply(gg[711]))
       	          .add(fPad[ijk+3*ri-3].multiply(gg[712]))
       	          .add(fPad[ijk+3*ri-2].multiply(gg[713]))
       	          .add(fPad[ijk+3*ri-1].multiply(gg[714]))
       	          .add(fPad[ijk+3*ri+0].multiply(gg[715]))
       	          .add(fPad[ijk+3*ri+1].multiply(gg[716]))
       	          .add(fPad[ijk+3*ri+2].multiply(gg[717]))
       	          .add(fPad[ijk+3*ri+3].multiply(gg[718]))
       	          .add(fPad[ijk+3*ri+4].multiply(gg[719]))
       	          .add(fPad[ijk+4*ri-4].multiply(gg[720]))
       	          .add(fPad[ijk+4*ri-3].multiply(gg[721]))
       	          .add(fPad[ijk+4*ri-2].multiply(gg[722]))
       	          .add(fPad[ijk+4*ri-1].multiply(gg[723]))
       	          .add(fPad[ijk+4*ri+0].multiply(gg[724]))
       	          .add(fPad[ijk+4*ri+1].multiply(gg[725]))
       	          .add(fPad[ijk+4*ri+2].multiply(gg[726]))
       	          .add(fPad[ijk+4*ri+3].multiply(gg[727]))
       	          .add(fPad[ijk+4*ri+4].multiply(gg[728]));
     	        	   }
                }
            }
        return Boundaries.finishBoundaries3d(fPad, gg, r, gi, gj, gk, hgi, hgie, hgj, hgje, hgk, hgke, ri, rj, rk)
;    };
// end convolve_9_9_9(double[][][] f, double[][][] g) {
// begin convolve_10
    public static double[] convolve_10(double[] f, double[] g) {
        final int fi = f.length;
        final int gi = g.length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		 double[] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie);
		 double[] r = JVCLUtils.zeroPadBoundaries(new double[fi], hgi, hgie);
		 final int ri = r.length;
		 for (int i = hgie; i < ri-1-hgi; i++) {
                 r[i] =
                 fPad[i-5]*g[9]+
                 fPad[i-4]*g[8]+
                 fPad[i-3]*g[7]+
                 fPad[i-2]*g[6]+
                 fPad[i-1]*g[5]+
                 fPad[i+0]*g[4]+
                 fPad[i+1]*g[3]+
                 fPad[i+2]*g[2]+
                 fPad[i+3]*g[1]+
                 fPad[i+4]*g[0];
            }
        return Boundaries.finishBoundaries1d(fPad, g, r, gi,hgi, hgie,ri);
    };
    public static Complex[] convolve_10(Complex[] f, Complex[] g) {
        final int fi = f.length;
        final int gi = g.length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
		 Complex[] fPad = JVCLUtils.zeroPadBoundaries(f, hgi, hgie);
		 Complex[] r = JVCLUtils.zeroPadBoundaries(new Complex[fi], hgi, hgie);
		 final int ri = r.length;
		 for (int i = hgi; i < ri-1-hgie; i++) {
                 r[i]
                 .add(fPad[i-5].multiply(g[9]))
                 .add(fPad[i-4].multiply(g[8]))
                 .add(fPad[i-3].multiply(g[7]))
                 .add(fPad[i-2].multiply(g[6]))
                 .add(fPad[i-1].multiply(g[5]))
                 .add(fPad[i+0].multiply(g[4]))
                 .add(fPad[i+1].multiply(g[3]))
                 .add(fPad[i+2].multiply(g[2]))
                 .add(fPad[i+3].multiply(g[1]))
                 .add(fPad[i+4].multiply(g[0]));
            }
        return Boundaries.finishBoundaries1d(fPad, g, r, gi, hgi, hgie, ri);
    };
// end convolve_10
 // begin convolve_12_12
    public static double[][] convolve_12_12(double[][] f, double[][] g) {
        final int fi = f.length;
        final int fj = f[0].length;
        final int gi = g.length;
        final int gj = g[0].length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgj = (int)( (gj - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
        final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
        double[] gg = ArrayMath.vectorise(g);
        double[] fPad = ArrayMath.vectorise(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje));
        double[][] rr = JVCLUtils.zeroPadBoundaries(new double[fi][fj], hgi, hgie, hgj, hgje);
        final int ri = rr.length;
        final int rj = rr[0].length;
        double[] r = ArrayMath.vectorise(rr);
        int ij;
        for (int i = hgi; i < fi - 1 - hgie; i++) {
            for (int j = hgj; j < fj - 1 - hgje; j++) {
                ij = j*ri + i;
                 r[ij] =
                 fPad[ij-6*ri-6]*gg[0]+
                 fPad[ij-6*ri-5]*gg[1]+
                 fPad[ij-6*ri-4]*gg[2]+
                 fPad[ij-6*ri-3]*gg[3]+
                 fPad[ij-6*ri-2]*gg[4]+
                 fPad[ij-6*ri-1]*gg[5]+
                 fPad[ij-6*ri+0]*gg[6]+
                 fPad[ij-6*ri+1]*gg[7]+
                 fPad[ij-6*ri+2]*gg[8]+
                 fPad[ij-6*ri+3]*gg[9]+
                 fPad[ij-6*ri+4]*gg[10]+
                 fPad[ij-6*ri+5]*gg[11]+
                 fPad[ij-5*ri-6]*gg[12]+
                 fPad[ij-5*ri-5]*gg[13]+
                 fPad[ij-5*ri-4]*gg[14]+
                 fPad[ij-5*ri-3]*gg[15]+
                 fPad[ij-5*ri-2]*gg[16]+
                 fPad[ij-5*ri-1]*gg[17]+
                 fPad[ij-5*ri+0]*gg[18]+
                 fPad[ij-5*ri+1]*gg[19]+
                 fPad[ij-5*ri+2]*gg[20]+
                 fPad[ij-5*ri+3]*gg[21]+
                 fPad[ij-5*ri+4]*gg[22]+
                 fPad[ij-5*ri+5]*gg[23]+
                 fPad[ij-4*ri-6]*gg[24]+
                 fPad[ij-4*ri-5]*gg[25]+
                 fPad[ij-4*ri-4]*gg[26]+
                 fPad[ij-4*ri-3]*gg[27]+
                 fPad[ij-4*ri-2]*gg[28]+
                 fPad[ij-4*ri-1]*gg[29]+
                 fPad[ij-4*ri+0]*gg[30]+
                 fPad[ij-4*ri+1]*gg[31]+
                 fPad[ij-4*ri+2]*gg[32]+
                 fPad[ij-4*ri+3]*gg[33]+
                 fPad[ij-4*ri+4]*gg[34]+
                 fPad[ij-4*ri+5]*gg[35]+
                 fPad[ij-3*ri-6]*gg[36]+
                 fPad[ij-3*ri-5]*gg[37]+
                 fPad[ij-3*ri-4]*gg[38]+
                 fPad[ij-3*ri-3]*gg[39]+
                 fPad[ij-3*ri-2]*gg[40]+
                 fPad[ij-3*ri-1]*gg[41]+
                 fPad[ij-3*ri+0]*gg[42]+
                 fPad[ij-3*ri+1]*gg[43]+
                 fPad[ij-3*ri+2]*gg[44]+
                 fPad[ij-3*ri+3]*gg[45]+
                 fPad[ij-3*ri+4]*gg[46]+
                 fPad[ij-3*ri+5]*gg[47]+
                 fPad[ij-2*ri-6]*gg[48]+
                 fPad[ij-2*ri-5]*gg[49]+
                 fPad[ij-2*ri-4]*gg[50]+
                 fPad[ij-2*ri-3]*gg[51]+
                 fPad[ij-2*ri-2]*gg[52]+
                 fPad[ij-2*ri-1]*gg[53]+
                 fPad[ij-2*ri+0]*gg[54]+
                 fPad[ij-2*ri+1]*gg[55]+
                 fPad[ij-2*ri+2]*gg[56]+
                 fPad[ij-2*ri+3]*gg[57]+
                 fPad[ij-2*ri+4]*gg[58]+
                 fPad[ij-2*ri+5]*gg[59]+
                 fPad[ij-1*ri-6]*gg[60]+
                 fPad[ij-1*ri-5]*gg[61]+
                 fPad[ij-1*ri-4]*gg[62]+
                 fPad[ij-1*ri-3]*gg[63]+
                 fPad[ij-1*ri-2]*gg[64]+
                 fPad[ij-1*ri-1]*gg[65]+
                 fPad[ij-1*ri+0]*gg[66]+
                 fPad[ij-1*ri+1]*gg[67]+
                 fPad[ij-1*ri+2]*gg[68]+
                 fPad[ij-1*ri+3]*gg[69]+
                 fPad[ij-1*ri+4]*gg[70]+
                 fPad[ij-1*ri+5]*gg[71]+
                 fPad[ij+0*ri-6]*gg[72]+
                 fPad[ij+0*ri-5]*gg[73]+
                 fPad[ij+0*ri-4]*gg[74]+
                 fPad[ij+0*ri-3]*gg[75]+
                 fPad[ij+0*ri-2]*gg[76]+
                 fPad[ij+0*ri-1]*gg[77]+
                 fPad[ij+0*ri+0]*gg[78]+
                 fPad[ij+0*ri+1]*gg[79]+
                 fPad[ij+0*ri+2]*gg[80]+
                 fPad[ij+0*ri+3]*gg[81]+
                 fPad[ij+0*ri+4]*gg[82]+
                 fPad[ij+0*ri+5]*gg[83]+
                 fPad[ij+1*ri-6]*gg[84]+
                 fPad[ij+1*ri-5]*gg[85]+
                 fPad[ij+1*ri-4]*gg[86]+
                 fPad[ij+1*ri-3]*gg[87]+
                 fPad[ij+1*ri-2]*gg[88]+
                 fPad[ij+1*ri-1]*gg[89]+
                 fPad[ij+1*ri+0]*gg[90]+
                 fPad[ij+1*ri+1]*gg[91]+
                 fPad[ij+1*ri+2]*gg[92]+
                 fPad[ij+1*ri+3]*gg[93]+
                 fPad[ij+1*ri+4]*gg[94]+
                 fPad[ij+1*ri+5]*gg[95]+
                 fPad[ij+2*ri-6]*gg[96]+
                 fPad[ij+2*ri-5]*gg[97]+
                 fPad[ij+2*ri-4]*gg[98]+
                 fPad[ij+2*ri-3]*gg[99]+
                 fPad[ij+2*ri-2]*gg[100]+
                 fPad[ij+2*ri-1]*gg[101]+
                 fPad[ij+2*ri+0]*gg[102]+
                 fPad[ij+2*ri+1]*gg[103]+
                 fPad[ij+2*ri+2]*gg[104]+
                 fPad[ij+2*ri+3]*gg[105]+
                 fPad[ij+2*ri+4]*gg[106]+
                 fPad[ij+2*ri+5]*gg[107]+
                 fPad[ij+3*ri-6]*gg[108]+
                 fPad[ij+3*ri-5]*gg[109]+
                 fPad[ij+3*ri-4]*gg[110]+
                 fPad[ij+3*ri-3]*gg[111]+
                 fPad[ij+3*ri-2]*gg[112]+
                 fPad[ij+3*ri-1]*gg[113]+
                 fPad[ij+3*ri+0]*gg[114]+
                 fPad[ij+3*ri+1]*gg[115]+
                 fPad[ij+3*ri+2]*gg[116]+
                 fPad[ij+3*ri+3]*gg[117]+
                 fPad[ij+3*ri+4]*gg[118]+
                 fPad[ij+3*ri+5]*gg[119]+
                 fPad[ij+4*ri-6]*gg[120]+
                 fPad[ij+4*ri-5]*gg[121]+
                 fPad[ij+4*ri-4]*gg[122]+
                 fPad[ij+4*ri-3]*gg[123]+
                 fPad[ij+4*ri-2]*gg[124]+
                 fPad[ij+4*ri-1]*gg[125]+
                 fPad[ij+4*ri+0]*gg[126]+
                 fPad[ij+4*ri+1]*gg[127]+
                 fPad[ij+4*ri+2]*gg[128]+
                 fPad[ij+4*ri+3]*gg[129]+
                 fPad[ij+4*ri+4]*gg[130]+
                 fPad[ij+4*ri+5]*gg[131]+
                 fPad[ij+5*ri-6]*gg[132]+
                 fPad[ij+5*ri-5]*gg[133]+
                 fPad[ij+5*ri-4]*gg[134]+
                 fPad[ij+5*ri-3]*gg[135]+
                 fPad[ij+5*ri-2]*gg[136]+
                 fPad[ij+5*ri-1]*gg[137]+
                 fPad[ij+5*ri+0]*gg[138]+
                 fPad[ij+5*ri+1]*gg[139]+
                 fPad[ij+5*ri+2]*gg[140]+
                 fPad[ij+5*ri+3]*gg[141]+
                 fPad[ij+5*ri+4]*gg[142]+
                 fPad[ij+5*ri+5]*gg[143];
                }
            }
        return Boundaries.finishBoundaries2d(fPad, gg, r, gi, gj, hgi, hgie, hgj, hgje, ri, rj);
    };
    public static Complex[][] convolve_12_12(Complex[][] f, Complex[][] g) {
        final int fi = f.length;
        final int fj = f[0].length;
        final int gi = g.length;
        final int gj = g[0].length;
        final int hgi = (int)( (gi - 1) / 2.0);
        final int hgj = (int)( (gj - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
        final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
        Complex[] gg = ArrayMath.vectorise(g);
        Complex[] fPad = ArrayMath.vectorise(JVCLUtils.zeroPadBoundaries(f, hgi, hgie, hgj, hgje));
        Complex[][] rr = JVCLUtils.zeroPadBoundaries(new Complex[fi][fj], hgi, hgie, hgj, hgje);
        final int ri = rr.length;
        final int rj = rr[0].length;
        Complex[] r = ArrayMath.vectorise(rr);
        int ij;
        for (int i = hgi; i < fi - 1 - hgie; i++) {
            for (int j = hgj; j < fj - 1 - hgje; j++) {
                ij = j*ri + i;
				r[ij]
                .add(fPad[ij-6*ri-6].multiply(gg[0]))
                .add(fPad[ij-6*ri-5].multiply(gg[1]))
                .add(fPad[ij-6*ri-4].multiply(gg[2]))
                .add(fPad[ij-6*ri-3].multiply(gg[3]))
                .add(fPad[ij-6*ri-2].multiply(gg[4]))
                .add(fPad[ij-6*ri-1].multiply(gg[5]))
                .add(fPad[ij-6*ri+0].multiply(gg[6]))
                .add(fPad[ij-6*ri+1].multiply(gg[7]))
                .add(fPad[ij-6*ri+2].multiply(gg[8]))
                .add(fPad[ij-6*ri+3].multiply(gg[9]))
                .add(fPad[ij-6*ri+4].multiply(gg[10]))
                .add(fPad[ij-6*ri+5].multiply(gg[11]))
                .add(fPad[ij-5*ri-6].multiply(gg[12]))
                .add(fPad[ij-5*ri-5].multiply(gg[13]))
                .add(fPad[ij-5*ri-4].multiply(gg[14]))
                .add(fPad[ij-5*ri-3].multiply(gg[15]))
                .add(fPad[ij-5*ri-2].multiply(gg[16]))
                .add(fPad[ij-5*ri-1].multiply(gg[17]))
                .add(fPad[ij-5*ri+0].multiply(gg[18]))
                .add(fPad[ij-5*ri+1].multiply(gg[19]))
                .add(fPad[ij-5*ri+2].multiply(gg[20]))
                .add(fPad[ij-5*ri+3].multiply(gg[21]))
                .add(fPad[ij-5*ri+4].multiply(gg[22]))
                .add(fPad[ij-5*ri+5].multiply(gg[23]))
                .add(fPad[ij-4*ri-6].multiply(gg[24]))
                .add(fPad[ij-4*ri-5].multiply(gg[25]))
                .add(fPad[ij-4*ri-4].multiply(gg[26]))
                .add(fPad[ij-4*ri-3].multiply(gg[27]))
                .add(fPad[ij-4*ri-2].multiply(gg[28]))
                .add(fPad[ij-4*ri-1].multiply(gg[29]))
                .add(fPad[ij-4*ri+0].multiply(gg[30]))
                .add(fPad[ij-4*ri+1].multiply(gg[31]))
                .add(fPad[ij-4*ri+2].multiply(gg[32]))
                .add(fPad[ij-4*ri+3].multiply(gg[33]))
                .add(fPad[ij-4*ri+4].multiply(gg[34]))
                .add(fPad[ij-4*ri+5].multiply(gg[35]))
                .add(fPad[ij-3*ri-6].multiply(gg[36]))
                .add(fPad[ij-3*ri-5].multiply(gg[37]))
                .add(fPad[ij-3*ri-4].multiply(gg[38]))
                .add(fPad[ij-3*ri-3].multiply(gg[39]))
                .add(fPad[ij-3*ri-2].multiply(gg[40]))
                .add(fPad[ij-3*ri-1].multiply(gg[41]))
                .add(fPad[ij-3*ri+0].multiply(gg[42]))
                .add(fPad[ij-3*ri+1].multiply(gg[43]))
                .add(fPad[ij-3*ri+2].multiply(gg[44]))
                .add(fPad[ij-3*ri+3].multiply(gg[45]))
                .add(fPad[ij-3*ri+4].multiply(gg[46]))
                .add(fPad[ij-3*ri+5].multiply(gg[47]))
                .add(fPad[ij-2*ri-6].multiply(gg[48]))
                .add(fPad[ij-2*ri-5].multiply(gg[49]))
                .add(fPad[ij-2*ri-4].multiply(gg[50]))
                .add(fPad[ij-2*ri-3].multiply(gg[51]))
                .add(fPad[ij-2*ri-2].multiply(gg[52]))
                .add(fPad[ij-2*ri-1].multiply(gg[53]))
                .add(fPad[ij-2*ri+0].multiply(gg[54]))
                .add(fPad[ij-2*ri+1].multiply(gg[55]))
                .add(fPad[ij-2*ri+2].multiply(gg[56]))
                .add(fPad[ij-2*ri+3].multiply(gg[57]))
                .add(fPad[ij-2*ri+4].multiply(gg[58]))
                .add(fPad[ij-2*ri+5].multiply(gg[59]))
                .add(fPad[ij-1*ri-6].multiply(gg[60]))
                .add(fPad[ij-1*ri-5].multiply(gg[61]))
                .add(fPad[ij-1*ri-4].multiply(gg[62]))
                .add(fPad[ij-1*ri-3].multiply(gg[63]))
                .add(fPad[ij-1*ri-2].multiply(gg[64]))
                .add(fPad[ij-1*ri-1].multiply(gg[65]))
                .add(fPad[ij-1*ri+0].multiply(gg[66]))
                .add(fPad[ij-1*ri+1].multiply(gg[67]))
                .add(fPad[ij-1*ri+2].multiply(gg[68]))
                .add(fPad[ij-1*ri+3].multiply(gg[69]))
                .add(fPad[ij-1*ri+4].multiply(gg[70]))
                .add(fPad[ij-1*ri+5].multiply(gg[71]))
                .add(fPad[ij+0*ri-6].multiply(gg[72]))
                .add(fPad[ij+0*ri-5].multiply(gg[73]))
                .add(fPad[ij+0*ri-4].multiply(gg[74]))
                .add(fPad[ij+0*ri-3].multiply(gg[75]))
                .add(fPad[ij+0*ri-2].multiply(gg[76]))
                .add(fPad[ij+0*ri-1].multiply(gg[77]))
                .add(fPad[ij+0*ri+0].multiply(gg[78]))
                .add(fPad[ij+0*ri+1].multiply(gg[79]))
                .add(fPad[ij+0*ri+2].multiply(gg[80]))
                .add(fPad[ij+0*ri+3].multiply(gg[81]))
                .add(fPad[ij+0*ri+4].multiply(gg[82]))
                .add(fPad[ij+0*ri+5].multiply(gg[83]))
                .add(fPad[ij+1*ri-6].multiply(gg[84]))
                .add(fPad[ij+1*ri-5].multiply(gg[85]))
                .add(fPad[ij+1*ri-4].multiply(gg[86]))
                .add(fPad[ij+1*ri-3].multiply(gg[87]))
                .add(fPad[ij+1*ri-2].multiply(gg[88]))
                .add(fPad[ij+1*ri-1].multiply(gg[89]))
                .add(fPad[ij+1*ri+0].multiply(gg[90]))
                .add(fPad[ij+1*ri+1].multiply(gg[91]))
                .add(fPad[ij+1*ri+2].multiply(gg[92]))
                .add(fPad[ij+1*ri+3].multiply(gg[93]))
                .add(fPad[ij+1*ri+4].multiply(gg[94]))
                .add(fPad[ij+1*ri+5].multiply(gg[95]))
                .add(fPad[ij+2*ri-6].multiply(gg[96]))
                .add(fPad[ij+2*ri-5].multiply(gg[97]))
                .add(fPad[ij+2*ri-4].multiply(gg[98]))
                .add(fPad[ij+2*ri-3].multiply(gg[99]))
                .add(fPad[ij+2*ri-2].multiply(gg[100]))
                .add(fPad[ij+2*ri-1].multiply(gg[101]))
                .add(fPad[ij+2*ri+0].multiply(gg[102]))
                .add(fPad[ij+2*ri+1].multiply(gg[103]))
                .add(fPad[ij+2*ri+2].multiply(gg[104]))
                .add(fPad[ij+2*ri+3].multiply(gg[105]))
                .add(fPad[ij+2*ri+4].multiply(gg[106]))
                .add(fPad[ij+2*ri+5].multiply(gg[107]))
                .add(fPad[ij+3*ri-6].multiply(gg[108]))
                .add(fPad[ij+3*ri-5].multiply(gg[109]))
                .add(fPad[ij+3*ri-4].multiply(gg[110]))
                .add(fPad[ij+3*ri-3].multiply(gg[111]))
                .add(fPad[ij+3*ri-2].multiply(gg[112]))
                .add(fPad[ij+3*ri-1].multiply(gg[113]))
                .add(fPad[ij+3*ri+0].multiply(gg[114]))
                .add(fPad[ij+3*ri+1].multiply(gg[115]))
                .add(fPad[ij+3*ri+2].multiply(gg[116]))
                .add(fPad[ij+3*ri+3].multiply(gg[117]))
                .add(fPad[ij+3*ri+4].multiply(gg[118]))
                .add(fPad[ij+3*ri+5].multiply(gg[119]))
                .add(fPad[ij+4*ri-6].multiply(gg[120]))
                .add(fPad[ij+4*ri-5].multiply(gg[121]))
                .add(fPad[ij+4*ri-4].multiply(gg[122]))
                .add(fPad[ij+4*ri-3].multiply(gg[123]))
                .add(fPad[ij+4*ri-2].multiply(gg[124]))
                .add(fPad[ij+4*ri-1].multiply(gg[125]))
                .add(fPad[ij+4*ri+0].multiply(gg[126]))
                .add(fPad[ij+4*ri+1].multiply(gg[127]))
                .add(fPad[ij+4*ri+2].multiply(gg[128]))
                .add(fPad[ij+4*ri+3].multiply(gg[129]))
                .add(fPad[ij+4*ri+4].multiply(gg[130]))
                .add(fPad[ij+4*ri+5].multiply(gg[131]))
                .add(fPad[ij+5*ri-6].multiply(gg[132]))
                .add(fPad[ij+5*ri-5].multiply(gg[133]))
                .add(fPad[ij+5*ri-4].multiply(gg[134]))
                .add(fPad[ij+5*ri-3].multiply(gg[135]))
                .add(fPad[ij+5*ri-2].multiply(gg[136]))
                .add(fPad[ij+5*ri-1].multiply(gg[137]))
                .add(fPad[ij+5*ri+0].multiply(gg[138]))
                .add(fPad[ij+5*ri+1].multiply(gg[139]))
                .add(fPad[ij+5*ri+2].multiply(gg[140]))
                .add(fPad[ij+5*ri+3].multiply(gg[141]))
                .add(fPad[ij+5*ri+4].multiply(gg[142]))
                .add(fPad[ij+5*ri+5].multiply(gg[143]));
                }
            }
    return Boundaries.finishBoundaries2d(fPad, gg, r, gi, gj, hgi, hgie, hgj, hgje, ri, rj)
;    };
 // end convolve_12_12
}
