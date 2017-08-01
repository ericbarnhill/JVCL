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

import org.apache.commons.math4.exception.OutOfRangeException;

import com.ericbarnhill.arrayMath.ArrayMath;

/**
 * This class performs naive Finite-Differences convolutions on the CPU.
 *
 * @author ericbarnhill
 * @since 0.1
 *
 */
public class ConvolverFloatFDCPU extends ConvolverFloat{
    /**
     * Convolve 1D {@code float[]} array with 1D {@code float[]} kernel
     * 
     * @param f
     *            {@code float[]} array
     * @param g
     *            {@code float[]} kernel
     * @return {@code float[]}
     */
    public float[] convolve(float[] f, float[] g) {
        final int fi = f.length;
        final int gi = g.length;
        final int hgi = (int) ((gi - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
        float[] fPad = ArrayMath.zeroPadBoundaries(f, hgi, hgie);
        float[] r = ArrayMath.zeroPadBoundaries(new float[fi], hgi, hgie);
        final int ri = r.length;
        int ai;
        for (int i = 0; i < ri; i++) {
            for (int p = 0; p < gi; p++) {
                ai = i + p - hgie;
                if (ai >= 0 && ai < ri) {
                    r[i] += fPad[ai] * g[gi - 1 - p];
                }
            }
        }
        return r;
    }

    /**
     * Convolve 2D {@code float[][]} array with 1D {@code float[]} kernel.
     * 
     * @param f
     *            {@code float[][]} array
     * @param g
     *            {@code float[]} kernel
     * @param dim
     *            orientation of kernel(0 or 1)
     * @return {@code float[][]}
     */
    public float[][] convolve(float[][] f, float[] g, int dim) {
        if (dim < 0 || dim > 1) {
            throw new OutOfRangeException(dim, 0, 1);
        }
        if (dim == 1)
            f = ArrayMath.shiftDim(f);
        final int fi = f.length;
        for (int i = 0; i < fi; i++) {
            f[i] = convolve(f[i], g);
        }
        if (dim == 1)
            f = ArrayMath.shiftDim(f);
        return f;
    }

    /**
     * Convolve 2D {@code float[][]} array with 1D {@code float[]} kernel.
     * Default orientation of 0 (first depth level)
     * 
     * @param f
     *            {@code float[][]} array
     * @param g
     *            {@code float[]} kernel
     * @return {@code float[][]}
     */
    public float[][] convolve(float[][] f, float[] g) {
        return convolve(f, g, 0);
    }

    /**
     * Convolve 3D {@code float[][][]} array with 1D {@code float[]} kernel
     * 
     * @param f
     *            {@code float[][][]} array
     * @param g
     *            {@code float[]} kernel
     * @param dim
     *            orientation of kernel (0, 1 or 2)
     * @return {@code float[][][]}
     */
    public float[][][] convolve(float[][][] f, float[] g, int dim) {
        if (dim < 0 || dim > 2) {
            throw new OutOfRangeException(dim, 0, 2);
        }
        if (dim > 0)
            f = ArrayMath.shiftDim(f, dim);
        final int fi = f.length;
        for (int i = 0; i < fi; i++) {
            f[i] = convolve(f[i], g);
        }
        if (dim > 0)
            f = ArrayMath.shiftDim(f, 3 - dim);
        return f;
    }

    /**
     * Convolve 3D {@code float[][][]} array with 1D {@code float[]} kernel
     * Default orientation of 0 (first depth level)
     * 
     * @param f
     *            {@code float[][][]} array
     * @param g
     *            {@code float[]} kernel
     * @return {@code float[][][]}
     */
    public float[][][] convolve(float[][][] f, float[] g) {
        return convolve(f, g, 0);
    }

    /**
     * Convolve 2D {@code float[][]} array with 2D {@code float[][]} kernel
     * 
     * @param f
     *            {@code float[][]} array
     * @param g
     *            {@code float[][]} kernel
     * @return {@code float[][]}
     */
    public float[][] convolve(float[][] f, float[][] g) {
        final int fi = f.length;
        final int fj = f[0].length;
        final int gi = g.length;
        final int gj = g[0].length;
        final int hgi = (int) ((gi - 1) / 2.0);
        final int hgj = (int) ((gj - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
        final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
        float[][] fPad = ArrayMath.zeroPadBoundaries(f, hgi, hgie, hgj, hgje);
        float[][] r = ArrayMath.zeroPadBoundaries(new float[fi][fj], hgi, hgie, hgj, hgje);
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
                                r[i][j] += fPad[ai][aj] * g[gi - 1 - p][gj - 1 - q];
                            }
                        }
                    }
                }
            }
        }
        return r;
    }

    /**
     * Convolve 3D {@code float[][][]} array with 2D {@code float[][]} kernel
     * 
     * @param f
     *            {@code float[][][]} array
     * @param g
     *            {@code float[][]} kernel
     * @param dim
     *            orientation of kernel (0, 1 or 2)
     * @return {@code float[][][]}
     */
    public float[][][] convolve(float[][][] f, float[][] g, int dim) {
        if (dim < 0 || dim > 2) {
            throw new OutOfRangeException(dim, 0, 2);
        }
        if (dim > 0)
            f = ArrayMath.shiftDim(f, dim);
        final int fi = f.length;
        for (int i = 0; i < fi; i++) {
            f[i] = convolve(f[i], g);
        }
        if (dim > 0)
            f = ArrayMath.shiftDim(f, 3 - dim);
        return f;
    }

    /**
     * Convolve 3D {@code float[][][]} array with 2D {@code float[][]} kernel
     * Default kernel orientation/depth level of 0.
     * 
     * @param f
     *            {@code float[][][]} array
     * @param g
     *            {@code float[][]} kernel
     * @return {@code float[][][]}
     */
    public float[][][] convolve(float[][][] f, float[][] g) {
        return convolve(f, g, 0);
    }

    /**
     * Convolve 3D {@code float[][][]} array with 3D {@code float[][][]}
     * kernel
     * 
     * @param f
     *            {@code float[][][]} array
     * @param g
     *            {@code float[][][]} kernel
     * @return {@code float[][][]}
     */
    public float[][][] convolve(float[][][] f, float[][][] g) {
        final int fi = f.length;
        final int fj = f[0].length;
        final int fk = f[0][0].length;
        final int gi = g.length;
        final int gj = g[0].length;
        final int gk = g[0][0].length;
        final int hgi = (int) ((gi - 1) / 2.0);
        final int hgj = (int) ((gj - 1) / 2.0);
        final int hgk = (int) ((gk - 1) / 2.0);
        final int hgie = (gi % 2 == 0) ? hgi + 1 : hgi;
        final int hgje = (gj % 2 == 0) ? hgj + 1 : hgj;
        final int hgke = (gk % 2 == 0) ? hgk + 1 : hgk;
        float[][][] fPad = ArrayMath.zeroPadBoundaries(f, hgi, hgie, hgj, hgje, hgk, hgke);
        float[][][] r = ArrayMath.zeroPadBoundaries(new float[fi][fj][fk], hgi, hgie, hgj, hgje, hgk, hgke);
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
                                            r[i][j][k] += fPad[ai][aj][ak] * g[gi - 1 - p][gj - 1 - q][gk - 1 - s];
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



    public Float[] convolve(Float[] f, Float[] g) {
        return ArrayMath.box(convolve(ArrayMath.unbox(f), ArrayMath.unbox(g)));
    }

    public Float[][] convolve(Float[][] f, Float[] g) {
        return ArrayMath.box(convolve(ArrayMath.unbox(f), ArrayMath.unbox(g)));
    }

    public Float[][] convolve(Float[][] f, Float[][] g) {
        return ArrayMath.box(convolve(ArrayMath.unbox(f), ArrayMath.unbox(g)));
    }

    public Float[][][] convolve(Float[][][] f, Float[] g) {
        return ArrayMath.box(convolve(ArrayMath.unbox(f), ArrayMath.unbox(g)));
    }

    public Float[][][] convolve(Float[][][] f, Float[][] g) {
        return ArrayMath.box(convolve(ArrayMath.unbox(f), ArrayMath.unbox(g)));
    }

    public Float[][][] convolve(Float[][][] f, Float[][][] g) {
        return ArrayMath.box(convolve(ArrayMath.unbox(f), ArrayMath.unbox(g)));
    }

}
