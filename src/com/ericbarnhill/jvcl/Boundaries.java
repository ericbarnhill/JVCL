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
 * This class finishes boundaries for Unrolled and GPU methods. This allows the methods to run straightforwardly
 * with boundaries cleaned up in a final step, according to preferred boundary condition specifications.
 * @author ericbarnhill
 * @see Unrolled
 * @see FDGPU
 * @since 0.1
 *
 */
public class Boundaries {

	/**
	 * Adds convolutions at boundaries for 1D {@code double[]}.
	 * @param fPad padded original array
	 * @param g kernel
	 * @param r convolved result array
	 * @param gi kernel width
	 * @param hgi left kernel radius
	 * @param hgie right kernel raduis
	 * @param ri result length
	 * @return r with completed boundaries
	 */
	public static double[] finishBoundaries1d(double[] fPad, double[] g, double[] r,
			int gi, int hgi, int hgie, int ri) {
		int ai;
		for (int i = 0; i < hgie; i++) {
			for (int p = 0; p < gi; p++) {
				ai = i + (p - hgie);
				if (ai >= 0 && ai < ri) {
					r[i] += fPad[ai]*g[(gi-1-p)];
				}
			}
		}
		for (int i = ri-hgi-1; i < ri; i++) {
			for (int p = 0; p < gi; p++) {
				ai = i + (p - hgie);
				if (ai >= 0 && ai < ri) {
					r[i] += fPad[ai]*g[(gi-1-p)];
				}
			}
		}
		return r;
	}

	/**
	 * Adds convolutions at boundaries for 1D {@code Complex[]}.
	 * @param fPad padded original array
	 * @param g kernel
	 * @param r convolved result array
	 * @param gi kernel width
	 * @param hgi left kernel radius
	 * @param hgie right kernel raduis
	 * @param ri result length
	 * @return r with completed boundaries
	 */
	public static Complex[] finishBoundaries1d(Complex[] fPad, Complex[] g, Complex[] r,
			int gi, int hgi, int hgie, int ri) {
		int ai;
		for (int i = 0; i < hgi; i++) {
			for (int p = 0; p < gi; p++) {
				ai = i + (p - hgie);
				if (ai >= 0 && ai < ri) {
					r[i].add(fPad[ai].multiply(g[(gi-1-p)]));
				}
			}
		}
		for (int i = ri-hgie-1; i < ri-1; i++) {
			for (int p = 0; p < gi; p++) {
				ai = i + (p - hgie);
				if (ai >= 0 && ai < ri) {
					r[i].add(fPad[ai].multiply(g[(gi-1-p)]));
				}
			}
		}
		return r;
	}

	/**
	 * Adds convolutions at boundaries for 2D {@code double[][]} vectorised to a 1D {@code double[]}.
	 * @param fPad padded original array
	 * @param g kernel
	 * @param r convolved result array
	 * @param gi kernel dim1 length
	 * @param gj kernel dim2 length
	 * @param hgi kernel dim1 start padding
	 * @param hgie kernel dim1 end padding
	 * @param hgj kernel dim2 start padding
	 * @param hgje kernel dim2 end padding
	 * @param ri result array dim 1
	 * @param rj result array dim 2
	 * @return reference to r
	 */
	public static double[][] finishBoundaries2d(double[] fPad, double[] g, double[] r,
			int gi, int gj, int hgi, int hgie, int hgj, int hgje, int ri, int rj) {
		int ai, aj;
		for (int i = 0; i < hgi; i++) {
			for (int j = 0; j < rj; j++) {
				for (int p = 0; p < gi; p++) {
					for (int q = 0; q < gj; q++) {
						ai = i + (p - hgie);
						aj = j + (q - hgje);
						if (ai >= 0 && ai < ri) {
							if (aj >= 0 && aj < rj) {
								r[i + j*ri] += fPad[ai + aj*ri]*g[(gi-1-p) + (gj-1-q)*gi];
							}
						}
					}
				}
			}
		}
		for (int i = ri-hgie-1; i < ri-1; i++) {
			for (int j = 0; j < rj; j++) {
				for (int p = 0; p < gi; p++) {
					for (int q = 0; q < gj; q++) {
						ai = i + (p - hgie);
						aj = j + (q - hgje);
						if (ai >= 0 && ai < ri) {
							if (aj >= 0 && aj < rj) {
								r[i + j*ri] += fPad[ai + aj*ri]*g[(gi-1-p) + (gj-1-q)*gi];
							}
						}
					}
				}
			}
		}
		for (int i = hgi; i < ri-hgie; i++) {
			for (int j = rj-hgje; j < rj-1; j++) {
				for (int p = 0; p < gi; p++) {
					for (int q = 0; q < gj; q++) {
						ai = i + (p - hgie);
						aj = j + (q - hgje);
						if (ai >= 0 && ai < ri) {
							if (aj >= 0 && aj < rj) {
								r[i + j*ri] += fPad[ai + aj*ri]*g[(gi-1-p) + (gj-1-q)*gi];
							}
						}
					}
				}
			}
		}
		for (int i = hgi; i < ri-hgie; i++) {
			for (int j = rj-hgje; j < rj-1; j++) {
				for (int p = 0; p < gi; p++) {
					for (int q = 0; q < gj; q++) {
						ai = i + (p - hgie);
						aj = j + (q - hgje);
						if (ai >= 0 && ai < ri) {
							if (aj >= 0 && aj < rj) {
								r[i + j*ri] += fPad[ai + aj*ri]*g[(gi-1-p) + (gj-1-q)*gi];
							}
						}
					}
				}
			}
		}
		return ArrayMath.devectorise(r, ri);
	}

	/**
	 * Adds convolutions at boundaries for 2D {@code Complex[][]} vectorised to a 1D {@code Complex[]}.
	 * @param fPad padded original array
	 * @param g kernel
	 * @param r convolved result array
	 * @param gi kernel dim1 length
	 * @param gj kernel dim2 length
	 * @param hgi kernel dim1 left radius
	 * @param hgie kernel dim1 right radius
	 * @param hgj kernel dim2 left radius
	 * @param hgje kernel dim2 right radius
	 * @param ri result array dim 1
	 * @param rj result array dim 2
	 * @return reference to r
	 */
	public static Complex[][] finishBoundaries2d(Complex[] fPad, Complex[] g, Complex[] r,
			int gi, int gj, int hgi, int hgie, int hgj, int hgje, int ri, int rj) {
		int ai, aj;
		for (int i = 0; i < hgi; i++) {
			for (int j = 0; j < rj; j++) {
				for (int p = 0; p < gi; p++) {
					for (int q = 0; q < gj; q++) {
						ai = i + (p - hgie);
						aj = j + (q - hgje);
						if (ai >= 0 && ai < ri) {
							if (aj >= 0 && aj < rj) {
								r[i + j*ri].add(fPad[ai + aj*ri].multiply(g[(gi-1-p) + (gj-1-q)*gi]));
							}
						}
					}
				}
			}
		}
		for (int i = ri-hgie-1; i < ri-1; i++) {
			for (int j = 0; j < rj; j++) {
				for (int p = 0; p < gi; p++) {
					for (int q = 0; q < gj; q++) {
						ai = i + (p - hgie);
						aj = j + (q - hgje);
						if (ai >= 0 && ai < ri) {
							if (aj >= 0 && aj < rj) {
								r[i + j*ri].add(fPad[ai + aj*ri].multiply(g[(gi-1-p) + (gj-1-q)*gi]));
							}
						}
					}
				}
			}
		}
		for (int i = hgi; i < ri-hgie; i++) {
			for (int j = rj-hgje; j < rj-1; j++) {
				for (int p = 0; p < gi; p++) {
					for (int q = 0; q < gj; q++) {
						ai = i + (p - hgie);
						aj = j + (q - hgje);
						if (ai >= 0 && ai < ri) {
							if (aj >= 0 && aj < rj) {
								r[i + j*ri].add(fPad[ai + aj*ri].multiply(g[(gi-1-p) + (gj-1-q)*gi]));
							}
						}
					}
				}
			}
		}
		for (int i = hgi; i < ri-hgie; i++) {
			for (int j = rj-hgje; j < rj-1; j++) {
				for (int p = 0; p < gi; p++) {
					for (int q = 0; q < gj; q++) {
						ai = i + (p - hgie);
						aj = j + (q - hgje);
						if (ai >= 0 && ai < ri) {
							if (aj >= 0 && aj < rj) {
								r[i + j*ri].add(fPad[ai + aj*ri].multiply(g[(gi-1-p) + (gj-1-q)*gi]));
							}
						}
					}
				}
			}
		}
		return ArrayMath.devectorise(r, ri);
	}

	/**
	 * Adds convolutions at boundaries for 3D {@code double[][][]} vectorised to a 1D {@code double[]}.
	 * @param fPad padded original array
	 * @param g kernel
	 * @param r convolved result array
	 * @param gi kernel dim1 length
	 * @param gj kernel dim2 length
	 * @param gk kernel dim3 length
	 * @param hgi kernel dim1 left radius
	 * @param hgie kernel dim1 right radius
	 * @param hgj kernel dim2 left radius
	 * @param hgje kernel dim2 right radius
	 * @param hgk kernel dim3 left radius
	 * @param hgke kernel dim3 right radius
	 * @param ri result array dim 1
	 * @param rj result array dim 2
	 * @param rk result array dim 3
	 * @return reference to r
	 */
	public static double[][][] finishBoundaries3d(double[] fPad, double[] g, double[] r,
			int gi, int gj, int gk, int hgi, int hgie, int hgj, int hgje, int hgk, int hgke,
			int ri, int rj, int rk) {
		boolean isBoundary = true;
		int ai, aj, ak;
		for (int i = 0; i < ri; i++) {
			for (int j = 0; j < rj; j++) {
				for (int k = 0; k < rk; k++) {
					isBoundary = (i > hgi && i < ri-hgie-1 && j > hgj && j < rj-hgje-1 &&
							k > hgk && k < rk - hgke - 1) ? false : true;
					if (!isBoundary) break;
					for (int p = 0; p < gi; p++) {
						for (int q = 0; q < gj; q++) {
							for (int s = 0; s < gk; s++) {
								ai = i + (p - hgie);
								aj = j + (q - hgje);
								ak = k + (s - hgke);
								if (ai >= 0 && ai < ri) {
									if (aj >= 0 && aj < rj) {
										if (ak >= 0 && ak < rk) {
											r[i + j*ri + k*ri*rj] += fPad[ai + aj*ri + ak*ri*rj]*g[gi-1-p + (gj-1-q) *gi + (gk-1-s)*gi*gj];
										}
									}
								}
							}
						}
					}
				}
			}
		}

		return ArrayMath.devectorise(r, ri, rj);
	}

	/**
	 * Adds convolutions at boundaries for 3D {@code Complex[][][]} vectorised to a 1D {@code Complex[]}.
	 * @param fPad padded original array
	 * @param g kernel
	 * @param r convolved result array
	 * @param gi kernel dim1 length
	 * @param gj kernel dim2 length
	 * @param gk kernel dim3 length
	 * @param hgi kernel dim1 left radius
	 * @param hgie kernel dim1 right radius
	 * @param hgj kernel dim2 left radius
	 * @param hgje kernel dim2 right radius
	 * @param hgk kernel dim3 left radius
	 * @param hgke kernel dim3 right radius
	 * @param ri result array dim 1
	 * @param rj result array dim 2
	 * @param rk result array dim 3
	 * @return reference to r
	 */
	public static Complex[][][] finishBoundaries3d(Complex[] fPad, Complex[] g, Complex[] r,
			int gi, int gj, int gk, int hgi, int hgie, int hgj, int hgje, int hgk, int hgke,
			int ri, int rj, int rk) {
		boolean isBoundary = true;
		int ai, aj, ak;
		for (int i = 0; i < ri; i++) {
			for (int j = 0; j < rj; j++) {
				for (int k = 0; k < rk; k++) {
					isBoundary = (i > hgi && i < ri-hgie-1 && j > hgj && j < rj-hgje-1 &&
							k > hgk && k < rk - hgke - 1) ? false : true;
					if (!isBoundary) break;
					for (int p = 0; p < gi; p++) {
						for (int q = 0; q < gj; q++) {
							for (int s = 0; s < gk; s++) {
								ai = i + (p - hgie);
								aj = j + (q - hgje);
								ak = k + (s - hgke);
								if (ai >= 0 && ai < ri) {
									if (aj >= 0 && aj < rj) {
										if (ak >= 0 && ak < rk) {
											r[i + j*ri + k*ri*rj].add(fPad[ai + aj*ri + ak*ri*rj].multiply(g[gi-1-p + (gj-1-q) *gi + (gk-1-s)*gi*gj]));
										}
									}
								}
							}
						}
					}
				}
			}
		}

		return ArrayMath.devectorise(r, ri, rj);
	}




}
